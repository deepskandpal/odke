"""Resolving against the store without loading it (#114, #149, DECISIONS #31).

The resolver asks a `StoreLookup` for candidates and judges them by its own
rules: a proof re-keys the incoming facts onto the store's key and leaves the
stored node as it was; anything weaker is a `SIMILAR` link. `MemoryLookup` is
the store in memory. `Neo4jLookup` is checked here against the recording
driver from `test_neo4j_sink` (one read transaction a batch, every statement
through an index, scoped by type and tenant, nothing written), and against a
real server in `test_a_batch_resolves_against_a_live_neo4j` when `NEO4J_URI`
is set.
"""

from __future__ import annotations

import os
import re
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from openodke import (
    Document,
    Entity,
    EntityLink,
    Fact,
    LinkKind,
    Ontology,
    Pipeline,
    Resolution,
    StoreLookup,
)
from openodke.corroborate import (
    MemoryLookup,
    NativeResolver,
    block_keys,
    candidate_pairs,
)
from openodke.sinks.neo4j import (
    SHOW_INDEXES,
    Neo4jConstrainer,
    Neo4jLookup,
    Neo4jSink,
    _entity_row,
    id_forms,
    store_indexes,
    stored_entity,
)
from test_neo4j_sink import FakeDriver

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


# --------------------------------------------------------------------------- #
# Neo4j, on the recording driver
# --------------------------------------------------------------------------- #


def _index(name: str, kind: str, label: str | None, *props: str) -> dict[str, Any]:
    """One `SHOW INDEXES` row."""
    return {
        "name": name,
        "type": kind,
        "labelsOrTypes": [label] if label else None,
        "properties": list(props) or None,
    }


INDEXES = [
    _index("odke_entity_key", "RANGE", "Entity", "key"),
    _index("odke_key_Company", "RANGE", "Company", "key"),
    _index("odke_external_id_Company", "RANGE", "Company", "external_id"),
    _index("odke_names_Company", "FULLTEXT", "Company", "label", "aliases"),
    _index("odke_key_Person", "RANGE", "Person", "key"),
    _index("odke_names_Person", "FULLTEXT", "Person", "label", "aliases"),
    _index("index_343aff4e", "LOOKUP", None),
]
# How a statement reads the store: always through one of these, never by scanning.
THROUGH_AN_INDEX = (
    "MATCH (n:`Company` {key: row.value})",
    "MATCH (n:`Person` {key: row.value})",
    "MATCH (n:`Entity` {key: row.value})",
    "MATCH (n:`Company` {external_id: row.value})",
    "CALL db.index.fulltext.queryNodes($index, row.query, {limit: $limit})",
    "CALL db.index.vector.queryNodes($index, $k, row.vector)",
)
WRITES = ("CREATE", "MERGE", "SET ", "DELETE", "REMOVE", "DETACH")


class ReadCountingDriver(FakeDriver):
    """The recording driver, also counting read transactions."""

    def __init__(self, answers: dict[str, list[dict[str, Any]]] | None = None) -> None:
        super().__init__({"SHOW INDEXES": INDEXES, **(answers or {})})
        self.reads = 0

    def session(self, **config: Any) -> Any:
        session = super().session(**config)
        execute_read = session.execute_read

        def counted(fn: Callable[..., Any], *args: Any) -> Any:
            self.reads += 1
            return execute_read(fn, *args)

        session.execute_read = counted  # type: ignore[method-assign]
        return session

    @property
    def statements(self) -> list[tuple[str, dict[str, Any]]]:
        return [(cypher, params) for mode, cypher, params in self.calls if mode == "read"]


def _node(entity: Entity) -> dict[str, Any]:
    """A node's properties as the sink wrote them."""
    row = _entity_row(entity)
    return {**{k: v for k, v in row.items() if k != "attributes"}, **row["attributes"]}


def test_a_batch_is_one_read_transaction_of_index_queries_scoped_by_type_and_tenant() -> None:
    hits = [{"block": 0, "node": _node(ACME)}]
    driver = ReadCountingDriver({"external_id": hits})
    lookup = Neo4jLookup(driver=driver, tenant="t1", database="graph")
    batch = [
        Entity(key="c:1", type="Company", label="Acme Widgets", external_id="wikidata:Q1"),
        Entity(key="c:2", type="Company", label="Acme Corp", aliases=("https://acme.com/",)),
        Entity(key="p:1", type="Person", label="Ada Acme"),
    ]
    found = lookup.candidates(batch)

    assert driver.reads == 1 and lookup.stats["transactions"] == 1
    assert [mode for mode, _, _ in driver.calls] == ["auto"] + ["read"] * 5
    assert driver.calls[0][1] == SHOW_INDEXES
    assert {session["database"] for session in driver.sessions} == {"graph"}
    for cypher, params in driver.statements:
        assert cypher.startswith("UNWIND $rows AS row\n")
        assert sum(way in cypher for way in THROUGH_AN_INDEX) == 1, cypher
        assert not any(word in cypher.upper() for word in WRITES), cypher
        assert "n.`tenant` = $tenant" in cypher and params["tenant"] == "t1"
        assert "MATCH (n)" not in cypher and "IN n.aliases" not in cypher

    by_kind = {(q.type, q.kind): q for q in lookup.statements(batch)}
    assert sorted(by_kind) == [
        ("Company", "id"),
        ("Company", "key"),
        ("Company", "names"),
        ("Person", "key"),
        ("Person", "names"),
    ]
    # Batched by block key: "acme" is a token of both companies, and is asked once.
    names = by_kind[("Company", "names")].params
    assert [r["query"] for r in names["rows"]] == [
        '"acme"',
        '"widgets"',
        '"acme.com" OR "www.acme.com"',
    ]
    assert (names["index"], names["limit"]) == ("odke_names_Company", 100)
    assert [r["value"] for r in by_kind[("Company", "id")].params["rows"]] == id_forms(
        "wikidata:Q1"
    )
    assert "WHERE n:`Person`" in by_kind[("Person", "names")].cypher
    # A hit for a block goes to every entity that asked for it, rebuilt as it was written.
    assert found["c:1"] == [ACME] and found["c:2"] == [] and found["p:1"] == []

    # The indexes are read once per lookup, not once per batch.
    lookup.candidates(batch[:1])
    assert [mode for mode, _, _ in driver.calls].count("auto") == 1
    assert driver.reads == 2 and driver.writes == []
    assert lookup.candidates([]) == {} and driver.reads == 2


def test_a_type_without_an_index_is_not_read_and_says_so() -> None:
    driver = ReadCountingDriver({"SHOW INDEXES": INDEXES[:1]})
    lookup = Neo4jLookup(driver=driver)
    batch = [Entity(key="c:1", type="Company", label="Acme", external_id="Q1")]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lookup.candidates(batch)
    assert [str(w.message).split(",")[0] for w in caught] == [
        "the store has no index for Company.external_id",
        "the store has no index for Company.names",
    ]
    ((cypher, _),) = driver.statements
    # The key still has the sink's :Entity(key) index; names and the id have none.
    assert "MATCH (n:`Entity` {key: row.value})\nWHERE n:`Company`" in cypher
    assert lookup.stats["unindexed"] == ["Company.external_id", "Company.names"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lookup.candidates(batch)  # said once


def test_the_lookup_shares_the_sinks_connection_and_reads_its_indexes_by_shape() -> None:
    driver = ReadCountingDriver()
    ontology = Ontology.from_dict({"types": {"Company": {}}, "predicates": {"founded": {}}})
    sink = Neo4jSink(driver=driver, database="graph", ontology=ontology)
    lookup = sink.lookup(tenant="t1", limit=5)
    assert (lookup.database, lookup.tenant, lookup.limit) == ("graph", "t1", 5)
    assert lookup.ontology is ontology
    lookup.close()
    assert not driver.closed  # the sink's to close

    found = store_indexes([*INDEXES, {**INDEXES[3], "name": "custom_names"}])
    assert found.names == {"Company": "odke_names_Company", "Person": "odke_names_Person"}
    assert (found.keys, found.ids, found.entity_key) == (
        frozenset({"Entity", "Company", "Person"}),
        frozenset({"Company"}),
        True,
    )


def test_an_accented_name_is_asked_for_as_written_too() -> None:
    """The full-text analyzer keeps accents and a name key drops them: ask both ways."""
    lookup = Neo4jLookup(driver=ReadCountingDriver())
    batch = [Entity(key="p:1", type="Person", label="José Smith", aliases=("Acme Inc",))]
    (names,) = [q for q in lookup.statements(batch) if q.kind == "names"]
    assert [r["query"] for r in names.params["rows"]] == ['"acme"', '"jose"', '"josé"', '"smith"']


def test_a_vector_lookup_needs_its_function_and_its_index() -> None:
    calls: list[list[str]] = []
    with pytest.raises(ValueError, match="both embed and vector_index"):
        Neo4jLookup(driver=FakeDriver(), embed=_stand_in_embed(calls))
    lookup = Neo4jLookup(
        driver=ReadCountingDriver(), embed=_stand_in_embed(calls), vector_index="names_vec"
    )
    batch = [
        Entity(key="c:1", type="Company", label="Acme"),
        Entity(key="c:2", type="Company", label="Acme"),
    ]
    vector = next(q for q in lookup.statements(batch) if q.kind == "vector")
    assert (vector.params["index"], vector.params["k"]) == ("names_vec", 5)
    assert len(vector.params["rows"]) == 1  # one text, asked once
    assert "WHERE n:`Company`" in vector.cypher
    assert calls == [["Acme"]]


def test_a_node_read_back_is_the_entity_the_sink_wrote() -> None:
    written = Entity(
        key="c:1",
        type="Company",
        label="Acme",
        aliases=("acme.com", "Acme Ltd"),
        external_id="Q1",
        resolution=Resolution(method="linker", score=0.95, linker="odke.native"),
        attributes={"country": "GB", "label": "a caller's own 'label'", "odke.name_key": "acme"},
    )
    assert stored_entity(_node(written), "Company") == written
    # A value projected from a fact is not an attribute, when the ontology says so.
    projected = {**_node(written), "founded": 2014}
    ontology = Ontology.from_dict({"types": {"Company": {}}, "predicates": {"founded": {}}})
    assert stored_entity(projected, "Company", ontology=ontology) == written


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


def _operators(plan: Mapping[str, Any] | None) -> list[str]:
    if not plan:
        return []
    found = [str(plan.get("operatorType", "")).split("@")[0]]
    for child in plan.get("children", ()):
        found += _operators(child)
    return found


class _Replay:
    def __init__(self, facts: list[Fact]) -> None:
        self.facts = facts

    def extract(self, chunk: Any, ontology: Ontology) -> list[Fact]:
        return self.facts


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_a_batch_resolves_against_a_live_neo4j() -> None:
    """Write, then resolve a new batch against the store: it links, and the node is unchanged.

    Types, keys and the predicate are suffixed, so a shared server is left as found.
    """
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    company, person, employer = f"Company_{suffix}", f"Person_{suffix}", f"employer_{suffix}"
    ontology = Ontology.from_dict(
        {
            "types": {company: {}, person: {}},
            "predicates": {employer: {"domain": [person], "range": company}},
        }
    )

    def mine(entity: Entity, name: str) -> Entity:
        return entity.model_copy(update={"type": company, "key": f"{name}:{suffix}"})

    acme, widgets = mine(ACME, "c:acme"), mine(WIDGETS, "c:widgets")
    ada = Entity(key=f"p:ada:{suffix}", type=person, label="Ada")
    grace = Entity(key=f"p:grace:{suffix}", type=person, label="Grace")
    incoming = mine(
        Entity(key="", type="", label="ACME Inc.", aliases=("https://www.acme.com/",)),
        "c:acme-inc",
    )
    widget = mine(Entity(key="", type="", label="Acme Widget"), "c:widget")
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(ontology)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            sink.bootstrap(ontology)
            driver.execute_query("CALL db.awaitIndexes(300)")
            first = [
                Fact(subject=ada, predicate=employer, object_entity=acme),
                Fact(subject=ada, predicate=employer, object_entity=widgets),
            ]
            Pipeline(ontology, _Replay(first), sinks=[sink]).run([Document(text="run 1")])
            stored = f"MATCH (n:`{company}`) RETURN n.key AS key, properties(n) AS props"
            before = {r["key"]: r["props"] for r in driver.execute_query(stored).records}
            assert set(before) == {acme.key, widgets.key}

            lookup = sink.lookup(tenant="t1")
            second = [
                Fact(subject=grace, predicate=employer, object_entity=incoming),
                Fact(subject=grace, predicate=employer, object_entity=widget),
            ]
            kg = Pipeline(
                ontology, _Replay(second), resolver=NativeResolver(lookup=lookup), sinks=[sink]
            ).run([Document(text="run 2")])
            assert sorted(_pairs(kg.links)) == [
                (incoming.key, acme.key, LinkKind.SAME_AS),
                (widget.key, widgets.key, LinkKind.SIMILAR),
            ]
            assert lookup.stats["transactions"] == 1 and lookup.stats["unindexed"] == []

            # Grace's employer is the node run 1 wrote; no node was made for the alias.
            employers = driver.execute_query(
                f"MATCH (:`{person}` {{key: $key}})-[:`{employer}`]->(c) RETURN c.key AS key",
                key=grace.key,
            ).records
            assert sorted(r["key"] for r in employers) == sorted([acme.key, widget.key])
            similar = driver.execute_query(
                f"MATCH (:`{company}` {{key: $a}})-[l:SIMILAR]->(:`{company}` {{key: $b}}) "
                "RETURN l.score AS score",
                a=widget.key,
                b=widgets.key,
            ).records
            assert len(similar) == 1
            # And the stored nodes hold exactly what they held.
            after = {r["key"]: r["props"] for r in driver.execute_query(stored).records}
            assert set(after) == {acme.key, widgets.key, widget.key}
            assert {key: after[key] for key in before} == before

            # Every statement reads through an index: no scan in any plan.
            for query in lookup.statements([incoming, widget]):
                with driver.session() as session:
                    plan = session.run("EXPLAIN " + query.cypher, query.params).consume().plan
                operators = _operators(plan)
                assert not {"AllNodesScan", "NodeByLabelScan"} & set(operators), operators
            # Another tenant's lookup sees none of it.
            assert sink.lookup(tenant="t2").candidates([incoming]) == {incoming.key: []}
        finally:
            driver.execute_query(f"MATCH (n) WHERE n:`{company}` OR n:`{person}` DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")
