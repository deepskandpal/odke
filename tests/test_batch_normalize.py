"""Normalising mentions in a batch (#148): look-alikes that mean one thing are one entity.

Look-alikes the batch itself introduces merge when nothing in the names or the
context keeps them apart; a store's entity is only ever linked. Every embedding
is a scripted function and every judge answer a recorded one: no model.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from openodke import (
    Document,
    Entity,
    EntityLink,
    Evidence,
    Fact,
    LinkKind,
    Ontology,
    Resolution,
    Span,
)
from openodke.corroborate import MemoryLookup, NativeResolver, PairJudge, legal_form
from openodke.llm import RecordedClient
from openodke.run import build, execute, parse_config
from openodke.validator import Validator

TEXT = (
    "Mercury is the planet closest to the Sun. "
    "Mercury is a metal that stays liquid at room temperature. "
    "Mercury takes 88 days to orbit the Sun once."
)
DOC = Document(id="d1", text=TEXT)


def _fact(entity: Entity, quote: str, text: str = TEXT, doc: str = "d1") -> Fact:
    start = text.index(quote)
    span = Span(doc_id=doc, start=start, end=start + len(quote), quote=quote)
    return Fact(
        subject=entity,
        predicate="mentioned",
        object_value=quote,
        evidence=(Evidence(doc_id=doc, span=span),),
    )


PLANET = Entity(key="t:mercury-1", type="Thing", label="Mercury")
METAL = Entity(key="t:mercury-2", type="Thing", label="Mercury")
ORBIT = Entity(key="t:mercury-3", type="Thing", label="MERCURY")
FACTS = [
    _fact(PLANET, "Mercury is the planet closest to the Sun."),
    _fact(METAL, "Mercury is a metal that stays liquid at room temperature."),
    _fact(ORBIT, "Mercury takes 88 days to orbit the Sun once."),
]


class Embed:
    """A scripted embedding: a sentence about orbits or about metals, and the texts it was given."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float("planet" in t or "orbit" in t), float("metal" in t)] for t in texts]


def _resolver(**options: Any) -> NativeResolver:
    resolver = NativeResolver(**{"normalize_batch": True, **options})
    if resolver.documents is not None:
        resolver.documents[DOC.id] = DOC
    return resolver


def _keys(facts: Sequence[Fact]) -> list[str]:
    return [f.subject.key for f in facts]


def _kinds(links: Sequence[EntityLink]) -> list[tuple[str, str, LinkKind]]:
    return sorted((link.source_key, link.target_key, link.kind) for link in links)


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


def test_look_alikes_with_one_context_merge_and_those_with_another_stay_apart() -> None:
    """Mercury the planet twice is one entity; Mercury the metal is another, by its sentence."""
    embed = Embed()
    resolver = _resolver(embed=embed)
    facts, links = resolver.resolve(FACTS, {})

    assert _keys(facts) == ["t:mercury-1", "t:mercury-2", "t:mercury-1"]
    assert _kinds(links) == [
        ("t:mercury-1", "t:mercury-2", LinkKind.SIMILAR),
        ("t:mercury-1", "t:mercury-3", LinkKind.SAME_AS),
        ("t:mercury-2", "t:mercury-3", LinkKind.SIMILAR),
    ]
    merged = next(link for link in links if link.kind is LinkKind.SAME_AS)
    assert merged.score == 1.0
    assert merged.reason == "in-batch merge: names 1.0, contexts 1.0000"
    apart = [link.reason for link in links if link.kind is LinkKind.SIMILAR]
    assert apart == ["not merged: the contexts differ (0.0000 < 0.5)"] * 2
    # Each mention's own sentence, embedded once, in one call.
    assert embed.calls == [sorted(s + "." for s in TEXT.split(". ")[:2]) + [TEXT.split(". ")[2]]]
    assert resolver.stats is not None and resolver.stats["batch"] == {
        "alike": 3,
        "merged": 1,
        "groups": 1,
        "mentions": 2,
        "numbers": 0,
        "forms": 0,
        "context": 2,
        "ambiguous": 0,
        "refused": 0,
        "embedded": 3,
    }


def test_without_an_embedding_the_names_decide_and_a_chain_cannot_join_a_pair_kept_apart() -> None:
    """Judge-free and embedding-free, the three look-alikes are one: rules and strings alone."""
    facts, links = _resolver().resolve(FACTS, {})
    assert set(_keys(facts)) == {"t:mercury-1"}
    assert {link.kind for link in links} == {LinkKind.SAME_AS}

    # With the planet and the metal kept apart by context, a third sentence
    # alike to both could be either, so it merges with neither.
    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors = {"planet": [1.0, 0.0], "metal": [0.0, 1.0]}
        return [next((v for w, v in vectors.items() if w in t), [0.6, 0.8]) for t in texts]

    resolver = _resolver(embed=embed)
    facts, links = resolver.resolve(FACTS, {})
    assert _keys(facts) == ["t:mercury-1", "t:mercury-2", "t:mercury-3"]
    torn = [link.reason for link in links if link.reason and "alike to both" in link.reason]
    assert torn == ["not merged: alike to both t:mercury-1 and t:mercury-2, kept apart"] * 2
    assert resolver.stats is not None and resolver.stats["batch"]["ambiguous"] == 2


def test_a_merge_that_would_join_a_pair_kept_apart_through_a_chain_is_refused() -> None:
    """A is alike to B, and B to C, but A and C carry different numbers: B joins one of them."""
    names = ["Apollo 11", "Apollo", "Apollo 13"]
    entities = [Entity(key=f"m:{i}", type="Mission", label=n) for i, n in enumerate(names)]
    resolver = _resolver(threshold=0.6)
    facts, links = resolver.resolve(
        [Fact(subject=e, predicate="p", object_value=1) for e in entities], {}
    )
    # "Apollo" is alike to both, and they are kept apart: it is torn, not merged.
    assert _keys(facts) == ["m:0", "m:1", "m:2"]
    assert resolver.stats is not None and resolver.stats["batch"]["ambiguous"] == 2


def test_a_mention_without_a_sentence_is_decided_on_its_names() -> None:
    embed = Embed()
    resolver = _resolver(embed=embed)
    bare = [Fact(subject=e, predicate="mentioned", object_value="x") for e in (PLANET, METAL)]
    facts, _ = resolver.resolve(bare, {})
    assert set(_keys(facts)) == {"t:mercury-1"} and embed.calls == []


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def test_every_surface_form_survives_as_an_alias_on_the_one_entity() -> None:
    forms = [
        Entity(key="Company:acme corp.", type="Company", label="Acme Corp.", aliases=("ACME",)),
        Entity(key="Company:acme corporation", type="Company", label="Acme Corporation"),
        Entity(key="Company:the acme corporation", type="Company", label="The ACME Corporation"),
    ]
    facts, links = _resolver().resolve(
        [Fact(subject=e, predicate="hq", object_value="Leeds") for e in forms], {}
    )
    entity = facts[0].subject
    assert all(f.subject == entity for f in facts)
    assert entity.key == "Company:acme corp." and entity.label == "Acme Corp."
    assert entity.aliases == (
        "ACME",
        "Company:acme corporation",
        "Acme Corporation",
        "Company:the acme corporation",
        "The ACME Corporation",
    )
    assert entity.resolution == Resolution(method="linker", linker="odke.native", score=1.0)
    # Three pairs alike, each a SAME_AS: the third joins two already one.
    assert [link.kind for link in links] == [LinkKind.SAME_AS] * 3


def test_names_with_different_numbers_or_legal_forms_are_linked_never_merged() -> None:
    assert legal_form("ACME Limited") == legal_form("Acme Corporation Ltd") == "ltd"
    assert legal_form("Acme") is None
    pairs = [
        ("1900 County Championship", "1902 County Championship", "1900 vs 1902"),
        ("Robert II", "Robert III", "ii vs iii"),
        ("Acme Corporation GmbH", "Acme Corporation Ltd", "gmbh vs ltd"),
    ]
    for left, right, said in pairs:
        a = Entity(key=f"x:{left}", type="Thing", label=left)
        b = Entity(key=f"x:{right}", type="Thing", label=right)
        facts, (link,) = _resolver().resolve(
            [Fact(subject=e, predicate="p", object_value=1) for e in (a, b)], {}
        )
        assert link.kind is LinkKind.SIMILAR and link.reason is not None and said in link.reason
        assert _keys(facts) == [a.key, b.key]
    # A name with no number may be either: "Robert" and "Robert II" are not kept apart.
    one = [Entity(key=f"x:{n}", type="Thing", label=n) for n in ("Acme Ltd", "Acme")]
    stated = [Fact(subject=e, predicate="p", object_value=1) for e in one]
    facts, _ = _resolver().resolve(stated, {})
    assert set(_keys(facts)) == {"x:Acme"}


def test_an_ontology_types_aliases_are_that_type_when_blocking() -> None:
    ontology = Ontology.from_dict(
        {"types": {"Company": {"aliases": ["Organisation", "Firm"]}, "Person": {}}}
    )
    a = Entity(key="o:acme", type="organisation", label="Acme Corp")
    b = Entity(key="c:acme", type="Company", label="ACME Corporation")
    facts = [Fact(subject=e, predicate="p", object_value=1) for e in (a, b)]
    merged, _ = _resolver(ontology=ontology).resolve(facts, {})
    # The ontology's own type name is preferred for the canonical entity.
    assert {f.subject.key for f in merged} == {"c:acme"} and merged[0].subject.type == "Company"
    apart, links = _resolver().resolve(facts, {})
    assert _keys(apart) == ["o:acme", "c:acme"] and links == []


# --------------------------------------------------------------------------- #
# The judge, in the band only
# --------------------------------------------------------------------------- #

BIO = (
    "Ada Lovelace wrote the first published program. "
    "Lovelace died in London in 1852. "
    "Ada Lovelace was born in 1815. "
    "Charles Babbage designed the engine."
)
BIO_DOC = Document(id="bio", text=BIO)


def _bio(key: str, label: str, quote: str) -> Fact:
    return _fact(Entity(key=key, type="Person", label=label), quote, BIO, "bio")


BIO_FACTS = [
    _bio("p:ada", "Ada Lovelace", "Ada Lovelace wrote the first published program."),
    _bio("p:lovelace", "Lovelace", "Lovelace died in London in 1852."),
    _bio("p:ada-2", "Ada  Lovelace", "Ada Lovelace was born in 1815."),
    _bio("p:charles", "Charles Babbage", "Charles Babbage designed the engine."),
]


def _answers(decision: str) -> RecordedClient:
    reply = {"because": "her surname", "decision": decision}
    return RecordedClient([{"match": "Mention A", "response": reply}])


def test_the_judge_is_asked_only_in_the_band_and_its_same_merges_the_batch() -> None:
    client = _answers("same")
    judge = PairJudge(client=client, max_workers=1)
    judge.documents[BIO_DOC.id] = BIO_DOC
    resolver = NativeResolver(judge=judge, normalize_batch=True)
    facts, links = resolver.resolve(BIO_FACTS, {})

    # Two pairs are in the band (each Ada with the surname, 0.8); the two Adas
    # are above it, and Babbage below it. Two calls a pair, in both orders.
    assert judge.stats["pairs"] == 2 and len(client.calls) == judge.stats["calls"] == 4
    assert _keys(facts) == ["p:ada", "p:ada", "p:ada", "p:charles"]
    reasons = sorted(link.reason or "" for link in links if link.kind is LinkKind.SAME_AS)
    assert reasons[0] == "in-batch merge: names 1.0"
    assert all(r.startswith("in-batch merge: pair judge (pair@1") for r in reasons[1:])
    # The group's identity was decided at its weakest join, a score in the band.
    assert facts[0].subject.resolution == Resolution(
        method="linker", linker="odke.native", score=0.8
    )
    assert facts[0].subject.aliases == ("p:ada-2", "Ada  Lovelace", "p:lovelace", "Lovelace")


def test_the_judges_different_keeps_a_pair_apart_through_any_chain() -> None:
    judge = PairJudge(client=_answers("different"), max_workers=1)
    judge.documents[BIO_DOC.id] = BIO_DOC
    facts, links = NativeResolver(judge=judge, normalize_batch=True).resolve(BIO_FACTS, {})
    assert _keys(facts) == ["p:ada", "p:lovelace", "p:ada", "p:charles"]
    assert [link.kind for link in links].count(LinkKind.DIFFERENT) == 2


def test_without_normalizing_the_judges_same_is_a_link_as_before() -> None:
    judge = PairJudge(client=_answers("same"), max_workers=1)
    judge.documents[BIO_DOC.id] = BIO_DOC
    resolver = NativeResolver(judge=judge, normalize_batch=False)
    facts, links = resolver.resolve(BIO_FACTS, {})
    assert _keys(facts) == ["p:ada", "p:lovelace", "p:ada-2", "p:charles"]
    assert LinkKind.SAME_AS not in {link.kind for link in links}
    assert resolver.stats is not None and "batch" not in resolver.stats


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


def test_store_nodes_are_never_touched_only_linked() -> None:
    stored = Entity(
        key="s:acme",
        type="Company",
        label="Acme Corporation",
        aliases=("Acme",),
        attributes={"tenant": "t1"},
    )
    held = Entity(key="s:globex", type="Company", label="Globex")
    incoming = [
        Entity(key="b:acme-1", type="Company", label="Acme Corp."),
        Entity(key="b:acme-2", type="Company", label="ACME Corporation"),
        # The batch restates a stored key, and a look-alike of it.
        Entity(key="s:globex", type="Company", label="Globex"),
        Entity(key="b:globex", type="Company", label="Globex Inc"),
    ]
    store = {e.key: e for e in (stored, held)}
    before = {k: e.model_copy(deep=True) for k, e in store.items()}
    resolver = _resolver(lookup=MemoryLookup(store))
    facts, links = resolver.resolve(
        [Fact(subject=e, predicate="hq", object_value="Leeds") for e in incoming], {}
    )

    assert store == before
    # The two batch mentions are one entity, and the store's is only linked.
    assert _keys(facts) == ["b:acme-1", "b:acme-1", "s:globex", "b:globex"]
    assert facts[0].subject.aliases == ("b:acme-2", "ACME Corporation")
    to_store = {(k.source_key, k.target_key): k.kind for k in links if k.target_key in store}
    assert to_store == {
        ("b:acme-1", "s:acme"): LinkKind.SIMILAR,
        ("b:acme-2", "s:acme"): LinkKind.SIMILAR,
        ("b:globex", "s:globex"): LinkKind.SIMILAR,
    }
    # The restated key keeps its own entity, and nothing merged into it.
    assert facts[2].subject == incoming[2].model_copy(
        update={"resolution": Resolution(method="linker", linker="odke.native")}
    )


def test_a_key_a_caller_chose_is_kept_and_two_are_never_merged() -> None:
    chosen = Resolution(method="caller")
    a = Entity(key="mine-1", type="Thing", label="Mercury", resolution=chosen)
    b = Entity(key="mine-2", type="Thing", label="Mercury", resolution=chosen)
    c = Entity(key="theirs", type="Thing", label="mercury")
    facts, links = _resolver().resolve(
        [Fact(subject=e, predicate="p", object_value=1) for e in (a, b, c)], {}
    )
    # The caller's two are two, so the third, alike to both, could be either.
    assert _keys(facts) == ["mine-1", "mine-2", "theirs"]
    reasons = sorted(link.reason or "" for link in links)
    assert reasons[-1] == "not merged: two keys a caller chose"
    assert all("alike to both mine-1 and mine-2" in r for r in reasons[:-1])

    # With one of them, the look-alike takes the caller's entity.
    facts, _ = _resolver().resolve(
        [Fact(subject=e, predicate="p", object_value=1) for e in (c, b)], {}
    )
    assert _keys(facts) == ["mine-2", "mine-2"] and facts[0].subject.resolution == chosen


# --------------------------------------------------------------------------- #
# Determinism, options and wiring
# --------------------------------------------------------------------------- #


def test_the_same_batch_in_any_order_gives_the_same_entities_and_links() -> None:
    names = ["Acme Corp", "ACME Corporation", "Acme Widgets", "Acme Widget", "Globex", "Globex Inc"]
    entities = [Entity(key=f"c{i}", type="Company", label=n) for i, n in enumerate(names)]
    facts = [Fact(subject=e, predicate="p", object_value=e.key) for e in entities]

    def outcome(order: list[Fact]) -> tuple[Any, ...]:
        resolved, links = _resolver().resolve(order, {})
        by_value = sorted((f.object_value, f.subject.key, f.subject.aliases) for f in resolved)
        made = (k.model_dump_json(exclude={"created_at"}) for k in links)
        return tuple(by_value), tuple(sorted(made))

    first = outcome(facts)
    for seed in range(5):
        shuffled = list(facts)
        random.Random(seed).shuffle(shuffled)
        assert outcome(shuffled) == first


def test_an_embedding_needs_the_batch_normalised_and_a_floor_is_a_cosine() -> None:
    with pytest.raises(ValueError, match="normalize_batch=True"):
        NativeResolver(normalize_batch=False, embed=Embed())
    with pytest.raises(ValueError, match="context_floor is a cosine"):
        NativeResolver(context_floor=1.5)
    assert NativeResolver().documents is None and not NativeResolver().normalize_batch
    assert NativeResolver(normalize_batch=True, embed=Embed()).documents == {}


ROWS = [
    {"doc": "d1", "subject": name, "subject_type": "Company", "predicate": "hq", "object": "Leeds"}
    for name in ("Acme Corp.", "ACME Corporation")
]
TYPES = Ontology.from_dict({"types": {"Company": {}}, "predicates": {"hq": {"range": "string"}}})
HQ = Document(id="d1", text="Acme Corp. is in Leeds. ACME Corporation has its office in Leeds.")


def test_the_validator_normalises_the_batch_when_asked_and_reports_it() -> None:
    kg, report = Validator(TYPES, normalize_batch=True).validate(ROWS, [HQ], dry_run=True)
    assert len(kg.entities) == 1 and report.facts_out == 1
    assert report.batch == {
        "alike": 1,
        "merged": 1,
        "groups": 1,
        "mentions": 2,
        "numbers": 0,
        "forms": 0,
        "context": 0,
        "ambiguous": 0,
        "refused": 0,
        "embedded": 0,
    }
    assert "batch         2 mentions merged into 1 entity" in report.render()

    # Off by default (DECISIONS #43): the two spellings stay two entities.
    kg, report = Validator(TYPES).validate(ROWS, [HQ], dry_run=True)
    assert len(kg.entities) == 2 and report.batch is None
    with pytest.raises(ValueError, match=r"NativeResolver\(normalize_batch=..., embed=...\)"):
        Validator(resolver=NativeResolver(), normalize_batch=True)


def test_a_run_config_turns_it_on_and_hands_the_resolver_the_ontology(tmp_path: Path) -> None:
    (tmp_path / "ontology.json").write_text(TYPES.model_dump_json(), encoding="utf-8")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "d1.txt").write_text(HQ.text, encoding="utf-8")
    rows = [{**row, "doc": "d1"} for row in ROWS]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def config(resolver: Any) -> dict[str, Any]:
        return {
            "ontology": "ontology.json",
            "inputs": ["corpus"],
            "stages": {
                "extractor": {"use": "triples", "path": "rows.jsonl"},
                "grounder": "passthrough",
                "resolver": resolver,
            },
        }

    on = {"use": "native", "normalize_batch": True}
    built = build(parse_config(config(on), base_dir=tmp_path))
    resolver = built.stages["resolver"]
    assert resolver.normalize_batch and resolver.ontology is built.ontology
    result = execute(parse_config(config(on), base_dir=tmp_path), dry_run=True)
    assert result.stats["stages"]["resolver"]["batch"]["merged"] == 1
    assert len(result.graph.entities) == 1

    off = execute(parse_config(config("native"), base_dir=tmp_path), dry_run=True)
    assert len(off.graph.entities) == 2
