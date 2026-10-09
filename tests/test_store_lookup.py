"""Resolving against the store without loading it (#114, DECISIONS #31).

The resolver asks a `StoreLookup` for candidates and judges them by its own
rules: a proof re-keys the incoming facts onto the store's key and leaves the
stored entity as it was; anything weaker is a `SIMILAR` link. `MemoryLookup` is
the store in memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openodke import (
    Entity,
    EntityLink,
    Fact,
    LinkKind,
    Resolution,
    StoreLookup,
)
from openodke.corroborate import (
    MemoryLookup,
    NativeResolver,
    block_keys,
    candidate_pairs,
)

ACME = Entity(
    key="c:acme",
    type="Company",
    label="Acme Corporation",
    aliases=("acme.com",),
    external_id="wikidata:Q1",
    resolution=Resolution(method="caller"),
    attributes={"country": "GB", "tenant": "t1"},
)
WIDGETS = Entity(key="c:widgets", type="Company", label="Acme Widgets", attributes={"tenant": "t1"})
# The same name and domain as ACME, but a person: never a candidate for a company.
NAMESAKE = Entity(
    key="p:acme", type="Person", label="Acme Corporation", aliases=("acme.com",), external_id="X1"
)
STORE = {e.key: e for e in (ACME, WIDGETS, NAMESAKE)}


def _facts(*entities: Entity) -> list[Fact]:
    return [Fact(subject=e, predicate="name", object_value=e.label) for e in entities]


def _resolve(
    store: Mapping[str, Entity] | StoreLookup, *entities: Entity, **options: Any
) -> tuple[list[Fact], list[EntityLink], NativeResolver]:
    lookup = store if isinstance(store, StoreLookup) else MemoryLookup(store)
    resolver = NativeResolver(lookup=lookup, **options)
    resolved, links = resolver.resolve(_facts(*entities), {})
    return list(resolved), list(links), resolver


def _pairs(links: Sequence[EntityLink]) -> list[tuple[str, str, LinkKind]]:
    return [(link.source_key, link.target_key, link.kind) for link in links]


# --------------------------------------------------------------------------- #
# The resolver against a store
# --------------------------------------------------------------------------- #


def test_a_proof_re_keys_the_incoming_facts_onto_the_stored_entity_as_it_is() -> None:
    by_domain = Entity(
        key="c:acme-inc", type="Company", label="ACME Inc.", aliases=("https://www.acme.com/",)
    )
    by_id = Entity(key="c:q1", type="Company", label="Acme Holdings", external_id="WIKIDATA:q1")
    facts, links, resolver = _resolve(STORE, by_domain, by_id)

    # The stored entity, field for field: writing it again changes nothing on the node.
    assert [f.subject for f in facts] == [ACME, ACME]
    assert sorted(_pairs(links)) == [
        ("c:acme-inc", "c:acme", LinkKind.SAME_AS),
        ("c:q1", "c:acme", LinkKind.SAME_AS),
    ]
    reasons = {link.source_key: link.reason for link in links}
    assert reasons == {
        "c:acme-inc": "domain match: acme.com",
        "c:q1": "external_id match: WIKIDATA:q1",
    }
    assert resolver.stats is not None
    assert resolver.stats["store"] == {
        "looked_up": 2,
        "candidates": 2,
        "rekeyed": 2,
        "same_as": 2,
        "similar": 0,
        "different": 0,
    }


def test_weak_evidence_is_a_similar_link_with_a_score_and_no_key_moves() -> None:
    widget = Entity(key="c:widget", type="Company", label="Acme Widget")
    facts, links, _ = _resolve(STORE, widget)
    (link,) = links
    assert (link.source_key, link.target_key, link.kind) == ("c:widget", "c:widgets", "similar")
    assert link.score is not None and 0.9 <= link.score < 1.0
    assert facts[0].subject.key == "c:widget"
    assert facts[0].subject.resolution == Resolution(method="linker", linker="odke.native")

    # Below the bar, the same pair is nothing at all.
    _, links, _ = _resolve(STORE, widget, threshold=0.99)
    assert links == []


def test_a_different_type_never_links_whatever_the_lookup_returns() -> None:
    person = Entity(key="p:acme-2", type="Person", label="Acme Corporation", aliases=("acme.com",))
    company = Entity(key="c:other", type="Company", label="Acme Corporation", external_id="X1")

    class Careless:
        """Returns every stored entity, whatever its type."""

        def candidates(self, entities: Sequence[Entity]) -> Mapping[str, Sequence[Entity]]:
            return {e.key: list(STORE.values()) for e in entities}

    for lookup in (MemoryLookup(STORE), Careless()):
        facts, links, _ = _resolve(lookup, company)
        # Never to NAMESAKE, though the names and the id match exactly.
        assert all(link.target_key != "p:acme" for link in links)
        assert facts[0].subject.key == "c:other"
    facts, links, _ = _resolve(STORE, person)
    # Not to ACME, though the name and the domain match exactly; NAMESAKE is a person.
    assert _pairs(links) == [("p:acme-2", "p:acme", LinkKind.SAME_AS)]
    assert facts[0].subject == NAMESAKE


def test_an_id_that_disagrees_with_the_store_is_a_different_link() -> None:
    rival = Entity(key="c:acme-de", type="Company", label="Acme Corporation", external_id="Q2")
    rival = rival.model_copy(update={"external_id": "wikidata:Q2"})
    facts, links, resolver = _resolve(STORE, rival)
    (link,) = links
    assert (link.kind, link.target_key) == (LinkKind.DIFFERENT, "c:acme")
    assert link.reason == "external_id mismatch: wikidata:Q2 vs wikidata:Q1"
    assert facts[0].subject.key == "c:acme-de"
    assert resolver.stats is not None and resolver.stats["store"]["different"] == 1


def test_store_entities_are_never_compared_with_each_other() -> None:
    twin = Entity(
        key="c:acme-twin", type="Company", label="Acme Corporation", aliases=("acme.com",)
    )
    store = {**STORE, twin.key: twin}
    probe = Entity(key="c:new", type="Company", label="Acme Corp")
    _, links, _ = _resolve(store, probe)
    # Both stored companies are candidates, and only the incoming one is linked.
    assert {link.source_key for link in links} == {"c:new"}
    assert {link.target_key for link in links} == {"c:acme", "c:acme-twin"}


def test_a_batch_that_states_the_stored_key_or_a_caller_key_keeps_its_own_entity() -> None:
    restated = ACME.model_copy(update={"label": "Acme Corp", "resolution": None})
    alias = Entity(key="c:acme-inc", type="Company", label="ACME Inc.", aliases=("acme.com",))
    facts, links, _ = _resolve(STORE, restated, alias)
    # The batch states c:acme itself, so its own statement wins, with the alias merged in,
    # exactly as a batch with no store would merge it.
    assert {f.subject.key for f in facts} == {"c:acme"}
    assert facts[0].subject.label == "Acme Corp"
    assert "c:acme-inc" in facts[0].subject.aliases

    keyed = alias.model_copy(update={"resolution": Resolution(method="caller")})
    facts, links, _ = _resolve(STORE, keyed)
    assert facts[0].subject == keyed
    assert _pairs(links) == [("c:acme-inc", "c:acme", LinkKind.SAME_AS)]


def test_an_empty_store_changes_nothing() -> None:
    batch = [
        Entity(key="a", type="Company", label="Acme Widgets", aliases=("acme.com",)),
        Entity(key="b", type="Company", label="Acme Widget", aliases=("www.acme.com",)),
        Entity(key="c", type="Company", label="Acme Widgets", external_id="Q9"),
    ]
    alone, alone_links = NativeResolver().resolve(_facts(*batch), {})
    looked, looked_links, _ = _resolve({}, *batch)
    assert [f.subject for f in alone] == [f.subject for f in looked]
    assert _pairs(alone_links) == _pairs(looked_links)
    assert NativeResolver().lookup is None and NativeResolver().stats is None


def test_the_in_memory_store_returns_exactly_the_pairs_blocking_would_compare() -> None:
    """Same block keys as the resolver: what the lookup finds is what a loaded index would."""
    stored = [
        ACME,
        WIDGETS,
        NAMESAKE,
        Entity(key="c:bank", type="Company", label="Bank of Leeds", attributes={"tenant": "t2"}),
        Entity(key="c:north", type="Company", label="Northern Bank"),
        Entity(key="c:q7", type="Company", label="Seven", external_id="wikidata:q7"),
    ]
    incoming = [
        Entity(key="i1", type="Company", label="Leeds Bank"),
        Entity(key="i2", type="Company", label="Something", external_id="wikidata:Q7"),
        Entity(key="i3", type="Company", label="Other", aliases=("http://acme.com",)),
        Entity(key="i4", type="Person", label="Acme Person"),
    ]
    found = MemoryLookup({e.key: e for e in stored}).candidates(incoming)
    looked_up = {(e.key, c.key) for e in incoming for c in found[e.key]}
    keys = {e.key for e in incoming}
    blocked = {
        (a, b) if a in keys else (b, a)
        for a, b in candidate_pairs([*stored, *incoming])
        if (a in keys) != (b in keys)
    }
    assert looked_up == blocked
    assert ("i1", "c:bank") in looked_up and ("i2", "c:q7") in looked_up


def test_the_in_memory_store_is_scoped_to_a_tenant_and_caps_a_crowded_token() -> None:
    probe = Entity(key="c:new", type="Company", label="Acme Corp")
    assert {c.key for c in MemoryLookup(STORE, tenant="t1").candidates([probe])["c:new"]} == {
        "c:acme",
        "c:widgets",
    }
    assert MemoryLookup(STORE, tenant="t2").candidates([probe])["c:new"] == []

    crowd = {f"c:{i}": Entity(key=f"c:{i}", type="Company", label=f"Bank {i}") for i in range(9)}
    near = Entity(key="c:near", type="Company", label="Bank 3")
    nearest = MemoryLookup(crowd, limit=2).candidates([near])["c:near"]
    assert [c.key for c in nearest] == ["c:3", "c:0"]


def _stand_in_embed(calls: list[list[str]]) -> Callable[[Sequence[str]], list[list[float]]]:
    """Not a model: a text's vector is its letter counts, so near spellings are near."""

    def embed(texts: Sequence[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [
            [float(text.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] for text in texts
        ]

    return embed


def test_a_vector_lookup_adds_the_nearest_of_the_type_and_judges_them_like_any_other() -> None:
    calls: list[list[str]] = []
    store = {
        "c:widgets": WIDGETS,
        "c:zed": Entity(key="c:zed", type="Company", label="Zed"),
        "p:widgets": Entity(key="p:widgets", type="Person", label="Acme Widgets"),
    }
    lookup = MemoryLookup(store, embed=_stand_in_embed(calls), vector_k=1)
    probe = Entity(key="c:new", type="Company", label="Acmee Widget")
    # A typo at each end: no first or last name token in common, so only the vector finds it.
    assert block_keys(probe).tokens.isdisjoint(block_keys(WIDGETS).tokens)
    assert MemoryLookup(store).candidates([probe])["c:new"] == []
    assert [c.key for c in lookup.candidates([probe])["c:new"]] == ["c:widgets"]
    # The store is embedded once, and each batch in one call.
    lookup.candidates([probe])
    assert len(calls) == 3 and len(calls[0]) == 3

    # Found by the vector, judged by the names, as any candidate is.
    _, links, _ = _resolve(lookup, probe)
    assert _pairs(links) == [("c:new", "c:widgets", LinkKind.SIMILAR)]
