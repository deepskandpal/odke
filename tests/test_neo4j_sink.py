"""The Neo4j sink, against a driver that records every statement it is handed.

No database runs here. What a MERGE does is Neo4j's business; what this sink
owns is which statements it sends and with which keys, and that is exactly
what a recording driver can check. `test_against_a_live_neo4j` runs the same
graph against a real server when `NEO4J_URI` is set.
"""

from __future__ import annotations

import importlib
import os
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest

from openodke import (
    Entity,
    EntityLink,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Polarity,
    Predicate,
    Resolution,
    Sink,
    SourceTier,
    Span,
)
from openodke.eval.sinks import assert_idempotent
from openodke.sinks.neo4j import EXTRA_HINT, Neo4jSink, Statement, signature_of, storable

# --------------------------------------------------------------------------- #
# A driver that records
# --------------------------------------------------------------------------- #


class _Result:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []

    def consume(self) -> None:
        return None

    def data(self) -> list[dict[str, Any]]:
        return self.rows


class _Tx:
    def __init__(self, driver: FakeDriver, mode: str) -> None:
        self.driver = driver
        self.mode = mode

    def run(self, cypher: str, parameters: dict[str, Any] | None = None, **kw: Any) -> _Result:
        params = {**(parameters or {}), **kw}
        self.driver.calls.append((self.mode, cypher, params))
        return _Result(self.driver.answer(cypher))


class _Session:
    def __init__(self, driver: FakeDriver, config: dict[str, Any]) -> None:
        self.driver = driver
        driver.sessions.append(config)

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, cypher: str, parameters: dict[str, Any] | None = None, **kw: Any) -> _Result:
        return _Tx(self.driver, "auto").run(cypher, parameters, **kw)

    def execute_write(self, fn: Callable[..., Any], *args: Any) -> Any:
        self.driver.transactions += 1
        if self.driver.fail_on == self.driver.transactions:
            raise RuntimeError("transaction failed")
        return fn(_Tx(self.driver, "write"), *args)

    def execute_read(self, fn: Callable[..., Any], *args: Any) -> Any:
        return fn(_Tx(self.driver, "read"), *args)


class FakeDriver:
    """Records (mode, cypher, parameters) for every statement, in order."""

    def __init__(self, answers: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.sessions: list[dict[str, Any]] = []
        self.transactions = 0
        self.fail_on: int | None = None
        self.closed = False
        self.answers = answers or {}

    def session(self, **config: Any) -> _Session:
        return _Session(self, config)

    def answer(self, cypher: str) -> list[dict[str, Any]]:
        return next((rows for needle, rows in self.answers.items() if needle in cypher), [])

    def close(self) -> None:
        self.closed = True

    @property
    def writes(self) -> list[tuple[str, dict[str, Any]]]:
        return [(cypher, params) for mode, cypher, params in self.calls if mode == "write"]


# --------------------------------------------------------------------------- #
# A graph with one of everything
# --------------------------------------------------------------------------- #

WHEN = datetime(2026, 9, 1, tzinfo=UTC)


def _evidence(doc: str, start: int, end: int) -> Evidence:
    return Evidence(
        doc_id=doc,
        span=Span(doc_id=doc, start=start, end=end),
        uri=f"https://example.com/{doc}",
        tier=SourceTier.AUTHORITATIVE,
        retrieved_at=WHEN,
    )


def _graph() -> KnowledgeGraph:
    """Built fresh on every call: new fact ids, new clocks — a second pipeline run."""
    ada = Entity(
        key="p:ada",
        type="Person",
        label="Ada Lovelace",
        aliases=("Ada", "Countess of Lovelace"),
        external_id="Q7259",
        resolution=Resolution(method="external_id"),
        attributes={"born": 1815, "tags": {"a": 1}},
    )
    acme = Entity(key="c:acme", type="Company", label="Acme")
    other = Entity(key="c:acme-gmbh", type="Company", label="Acme GmbH")
    facts = (
        Fact(
            subject=ada,
            predicate="employer",
            object_entity=acme,
            qualifiers={"start_time": "2019", "signature": "clash"},
            valid_from=datetime(2019, 1, 1, tzinfo=UTC),
            evidence=(_evidence("d1", 0, 24), Evidence(doc_id="d2", retrieved_at=WHEN)),
            extractor="llm",
            confidence=0.9,
            verdict=GroundingVerdict.SUPPORTED,
            support=2,
        ),
        Fact(
            subject=acme,
            predicate="uptime",
            object_value="99.9%",
            qualifiers={"percentile": "p50"},
            identity_keys=("percentile",),
            evidence=(_evidence("d3", 10, 15),),
        ),
        Fact(
            subject=acme,
            predicate="uptime",
            object_value="99.9%",
            qualifiers={"percentile": "p95"},
            identity_keys=("percentile",),
            evidence=(_evidence("d3", 20, 25),),
        ),
        Fact(subject=ada, predicate="name", object_value="Ada", evidence=(_evidence("d1", 0, 3),)),
        Fact(
            subject=acme,
            predicate="sells",
            object_value="customer_data",
            polarity=Polarity.DENIED,
            evidence=(_evidence("d4", 0, 30),),
        ),
    )
    links = (
        EntityLink(source_key="c:acme", target_key="c:acme-gmbh", kind=LinkKind.SIMILAR, score=0.9),
        EntityLink(
            source_key="c:acme",
            target_key="c:acme-gmbh",
            kind=LinkKind.DIFFERENT,
            score=0.9,
            reason="external_id mismatch: DE-114322 vs GB-889401",
        ),
        EntityLink(source_key="p:ada", target_key="p:ada", kind=LinkKind.SAME_AS, score=1.0),
    )
    return KnowledgeGraph(entities=(ada, acme, other), facts=facts, links=links)


def _written(graph: KnowledgeGraph, **kwargs: Any) -> FakeDriver:
    driver = FakeDriver()
    Neo4jSink(driver=driver, **kwargs).write(graph)
    return driver


def _rows_where(driver: FakeDriver, needle: str) -> list[dict[str, Any]]:
    return [row for cypher, params in driver.writes if needle in cypher for row in params["rows"]]


def _merge_keys(driver: FakeDriver) -> list[tuple[str, tuple[Any, ...]]]:
    """Each statement with the identity of every row it MERGEs, without the payload."""
    keys = ("key", "signature", "subject_key", "object_key", "source_key", "target_key")
    return [
        (cypher, tuple(tuple(row.get(k) for k in keys) for row in params["rows"]))
        for cypher, params in driver.writes
    ]


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


def _merged(driver: FakeDriver) -> Callable[[], dict[str, int]]:
    """What a store that honours MERGE would hold after the recorded writes: one per identity."""

    def count() -> dict[str, int]:
        identities = {(cypher, row) for cypher, rows in _merge_keys(driver) for row in rows}
        return {"identities": len(identities), "statements": len({c for c, _ in driver.writes})}

    return count


def test_writing_the_same_graph_twice_sends_identical_statements() -> None:
    """Write twice, count once — the evaluator's check — and the second write is the first."""
    graph = _graph()
    driver = FakeDriver()
    report = assert_idempotent(Neo4jSink(driver=driver), graph, _merged(driver))
    assert report.breakdown["identities"]["after_first"]
    half = len(driver.writes) // 2
    assert half and driver.writes == driver.writes[:half] * 2


def test_a_rerun_merges_on_the_same_keys_though_ids_and_clocks_change() -> None:
    """A second pipeline run mints new fact ids and timestamps; the MERGE keys must not move."""
    first, second = _graph(), _graph()
    assert first.facts[0].id != second.facts[0].id
    assert _merge_keys(_written(first)) == _merge_keys(_written(second))


def test_nothing_is_created_unconditionally_and_nothing_is_deleted() -> None:
    for cypher, _ in _written(_graph()).writes:
        assert cypher.startswith("UNWIND $rows AS row\n")
        assert "CREATE" not in cypher
        assert "DELETE" not in cypher
        # A statement either MERGEs or only SETs on nodes it matched.
        assert "MERGE" in cypher or ("MATCH" in cypher and "SET s." in cypher)


def test_edges_merge_on_the_fact_signature() -> None:
    rows = _rows_where(_written(_graph()), "[r:`employer` {signature: row.signature}]")
    (row,) = rows
    assert row["signature"] == signature_of(_graph().facts[0])
    assert (row["subject_key"], row["object_key"]) == ("p:ada", "c:acme")


def test_entities_merge_per_type_on_key_with_their_fields() -> None:
    driver = _written(_graph())
    (person,) = _rows_where(driver, "MERGE (n:`Person` {key: row.key})")
    companies = _rows_where(driver, "MERGE (n:`Company` {key: row.key})")
    assert person["aliases"] == ["Ada", "Countess of Lovelace"]
    assert (person["label"], person["external_id"], person["resolution_method"]) == (
        "Ada Lovelace",
        "Q7259",
        "external_id",
    )
    # Neo4j stores no maps: a nested attribute survives as JSON text.
    assert person["attributes"] == {"born": 1815, "tags": '{"a": 1}'}
    assert {row["key"] for row in companies} == {"c:acme", "c:acme-gmbh"}
    assert any("SET n:`Entity`" in cypher for cypher, _ in driver.writes)


def test_identity_qualifiers_and_polarity_split_what_reconcilable_qualifiers_do_not() -> None:
    graph = _graph()
    p50, p95 = graph.facts[1], graph.facts[2]
    assert signature_of(p50) != signature_of(p95)
    later = graph.facts[0].model_copy(update={"qualifiers": {"start_time": "2020"}})
    assert signature_of(later) == signature_of(graph.facts[0])
    denied = graph.facts[3].model_copy(update={"polarity": Polarity.DENIED})
    assert signature_of(denied) != signature_of(graph.facts[3])


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

PROVENANCE = {
    "fact_id",
    "signature",
    "polarity",
    "extractor",
    "verdict",
    "confidence",
    "support",
    "valid_from",
    "valid_to",
    "retrieved_at",
    "extracted_at",
    "evidence_doc_ids",
    "evidence_uris",
    "evidence_starts",
    "evidence_ends",
    "evidence_tiers",
}


def test_every_fact_relationship_carries_provenance() -> None:
    graph = _graph()
    driver = _written(graph)
    rows = _rows_where(driver, "SET r += row.props")
    assert len(rows) == len(graph.facts)
    for row in rows:
        assert set(row["props"]) >= PROVENANCE
        assert row["props"]["evidence_doc_ids"], "an edge with no document cannot be traced"
        assert len(row["props"]["evidence_starts"]) == len(row["props"]["evidence_doc_ids"])


def test_provenance_traces_an_edge_to_a_document_and_a_character_range() -> None:
    graph = _graph()
    (row,) = _rows_where(_written(graph), "[r:`employer`")
    props = row["props"]
    assert props["evidence_doc_ids"] == ["d1", "d2"]
    assert props["evidence_uris"] == ["https://example.com/d1", ""]
    assert (props["evidence_starts"], props["evidence_ends"]) == ([0, -1], [24, -1])
    assert props["evidence_tiers"] == ["authoritative", "unverified"]
    assert (props["extractor"], props["verdict"], props["confidence"], props["support"]) == (
        "llm",
        "supported",
        0.9,
        2,
    )
    assert props["valid_from"] == datetime(2019, 1, 1, tzinfo=UTC)
    assert props["valid_to"] is None
    assert props["retrieved_at"] == WHEN
    assert props["extracted_at"] == graph.created_at
    assert props["fact_id"] == graph.facts[0].id


def test_reconcilable_qualifiers_are_properties_and_never_overwrite_provenance() -> None:
    (row,) = _rows_where(_written(_graph()), "[r:`employer`")
    assert row["props"]["start_time"] == "2019"
    # A qualifier named like a receipt is kept, under a prefix.
    assert row["props"]["qualifier_signature"] == "clash"
    assert row["props"]["signature"] == row["signature"]


# --------------------------------------------------------------------------- #
# Literal facts
# --------------------------------------------------------------------------- #


def test_a_literal_fact_is_a_claim_node_carrying_provenance_on_its_relationship() -> None:
    driver = _written(_graph())
    claims = _rows_where(driver, "MERGE (c:`Claim` {signature: row.signature})")
    assert {(r["predicate"], r["value"]) for r in claims} == {
        ("uptime", "99.9%"),
        ("name", "Ada"),
        ("sells", "customer_data"),
    }
    # p50 and p95 are two claims with the same value, not one.
    assert len([r for r in claims if r["predicate"] == "uptime"]) == 2
    for row in claims:
        assert set(row["props"]) >= PROVENANCE
        assert "-[r:" in next(c for c, p in driver.writes if row in p["rows"])


def test_only_asserted_unscoped_claims_project_onto_the_node() -> None:
    """A denial is not a value, and uptime without its percentile means nothing."""
    driver = _written(_graph())
    projected = [
        (cypher.rsplit("SET ", 1)[1], row)
        for cypher, params in driver.writes
        if "SET s." in cypher
        for row in params["rows"]
    ]
    assert projected == [("s.`name` = row.value", {"subject_key": "p:ada", "value": "Ada"})]


def test_a_single_valued_projection_carries_the_best_supported_claim() -> None:
    acme = Entity(key="c:acme", type="Company")
    weak = Fact(subject=acme, predicate="hq", object_value="Berlin", support=1)
    strong = Fact(subject=acme, predicate="hq", object_value="Munich", support=3)
    driver = _written(KnowledgeGraph(facts=(weak, strong)))
    assert _rows_where(driver, "SET s.`hq`") == [{"subject_key": "c:acme", "value": "Munich"}]
    # Both claims are still held; the conflict is not resolved by dropping one.
    assert len(_rows_where(driver, "MERGE (c:`Claim`")) == 2


def test_a_multi_valued_projection_is_a_list_when_the_ontology_says_so() -> None:
    acme = Entity(key="c:acme", type="Company")
    facts = tuple(
        Fact(subject=acme, predicate="product", object_value=v, support=s)
        for v, s in (("anvils", 1), ("rockets", 2), ("anvils", 1))
    )
    ontology = Ontology(predicates={"product": Predicate(name="product", cardinality="multi")})
    driver = _written(KnowledgeGraph(facts=facts), ontology=ontology)
    assert _rows_where(driver, "SET s.`product`") == [
        {"subject_key": "c:acme", "value": ["rockets", "anvils"]}
    ]


def test_a_projected_predicate_never_overwrites_an_entity_field() -> None:
    ada = Entity(key="p:ada", type="Person", label="Ada Lovelace")
    driver = _written(
        KnowledgeGraph(facts=(Fact(subject=ada, predicate="label", object_value="x"),))
    )
    assert _rows_where(driver, "SET s.`property_label`")


def test_values_neo4j_cannot_store_are_kept_as_json_text() -> None:
    assert storable(["a", "b"]) == ["a", "b"]
    assert storable((1, 2.5)) == [1.0, 2.5]
    assert storable(WHEN) == WHEN
    assert storable({"a": [1]}) == '{"a": [1]}'
    assert storable(["a", 1]) == '["a", 1]'
    assert storable([["a"], ["b"]]) == '[["a"], ["b"]]'
    assert storable(Polarity.DENIED) == "denied"


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


def test_links_are_relationships_with_score_and_reason_and_never_merge_nodes() -> None:
    driver = _written(_graph())
    link_statements = [
        (c, p) for c, p in driver.writes if "(a:`Entity` {key: row.source_key})" in c
    ]
    kinds = {c.split("[l:", 1)[1].split("]", 1)[0] for c, _ in link_statements}
    assert kinds == {"`SAME_AS`", "`SIMILAR`", "`DIFFERENT`"}
    for cypher, _ in link_statements:
        # The ends are matched, never created or merged into one another.
        assert "MATCH (a:" in cypher and "MATCH (b:" in cypher
        assert "MERGE (a)-[l:" in cypher
    (different,) = _rows_where(driver, "[l:`DIFFERENT`]")
    assert different["props"]["score"] == 0.9
    assert different["props"]["reason"] == "external_id mismatch: DE-114322 vs GB-889401"
    assert "apoc" not in " ".join(c for c, _ in driver.writes).lower()


# --------------------------------------------------------------------------- #
# Batching, safety, surface
# --------------------------------------------------------------------------- #


def test_rows_are_written_in_batches_one_transaction_each() -> None:
    acme = Entity(key="c:acme", type="Company")
    facts = tuple(Fact(subject=acme, predicate="product", object_value=f"p{i}") for i in range(5))
    driver = _written(KnowledgeGraph(facts=facts), batch_size=2)
    claim_batches = [len(p["rows"]) for c, p in driver.writes if "MERGE (c:`Claim`" in c]
    assert claim_batches == [2, 2, 1]
    assert driver.transactions == len(driver.writes)


def test_a_failed_transaction_propagates_and_stops_the_write() -> None:
    driver = FakeDriver()
    driver.fail_on = 2
    with pytest.raises(RuntimeError, match="transaction failed"):
        Neo4jSink(driver=driver).write(_graph())
    assert len(driver.writes) == 1


def test_names_from_the_ontology_are_quoted_so_they_cannot_inject_cypher() -> None:
    evil = Entity(key="x", type="Person`) DETACH DELETE n //", label="x")
    fact = Fact(subject=evil, predicate="knows` ]->() DELETE r //", object_entity=evil)
    for cypher, _ in _written(KnowledgeGraph(facts=(fact,))).writes:
        assert "`Person``) DETACH DELETE n //`" in cypher or "`knows`` ]->() DELETE r //`" in cypher


def test_the_database_is_passed_to_every_session() -> None:
    driver = FakeDriver()
    Neo4jSink(driver=driver, database="kg").write(_graph())
    assert driver.sessions == [{"database": "kg"}]


def test_the_sink_declares_what_neo4j_does_after_the_write() -> None:
    sink = Neo4jSink(driver=FakeDriver())
    assert isinstance(sink, Sink)
    profile = sink.profile
    assert (profile.name, profile.resolves, profile.constrains, profile.prunes) == (
        "neo4j",
        False,
        True,
        False,
    )


def test_statements_are_a_dry_run_of_write() -> None:
    graph = _graph()
    sink = Neo4jSink(driver=FakeDriver())
    planned = sink.statements(graph)
    assert all(isinstance(s, Statement) for s in planned)
    driver = _written(graph)
    assert [(s.cypher, {"rows": s.rows}) for s in planned] == driver.writes


def test_the_sink_closes_its_driver() -> None:
    driver = FakeDriver()
    with Neo4jSink(driver=driver):
        pass
    assert driver.closed


def test_the_sink_needs_a_uri_or_a_driver_and_a_positive_batch() -> None:
    with pytest.raises(ValueError, match="uri"):
        Neo4jSink()
    with pytest.raises(ValueError, match="batch_size"):
        Neo4jSink(driver=FakeDriver(), batch_size=0)


def test_a_missing_driver_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module imports on the base install; only connecting needs the extra."""
    monkeypatch.setitem(sys.modules, "neo4j", None)
    monkeypatch.delitem(sys.modules, "openodke.sinks.neo4j")
    module = importlib.import_module("openodke.sinks.neo4j")
    with pytest.raises(ImportError, match=r"openodke\[neo4j\]"):
        module.Neo4jSink("bolt://localhost:7687", auth=("neo4j", "unused"))
    assert "openodke[neo4j]" in EXTRA_HINT


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_against_a_live_neo4j() -> None:
    """Write twice, count once. Types are suffixed so a shared database is left as found."""
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    graph_a, graph_b = live_graph(suffix), live_graph(suffix)
    person, company = f"Person_{suffix}", f"Company_{suffix}"
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    mine = (
        f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
        f"(n:Claim AND n.subject_type IN ['{person}', '{company}']) "
    )
    count = mine + (
        "OPTIONAL MATCH (n)-[r]-() RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
    )
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            sink.write(graph_a)
            first = driver.execute_query(count).records[0].data()
            # A second run — new fact ids and clocks, the same claims — written
            # twice, and no count may move from what the first run left.
            report = assert_idempotent(
                sink, graph_b, lambda: driver.execute_query(count).records[0].data()
            )
            assert {key: row["after_first"] for key, row in report.breakdown.items()} == first
            # Three entities and four claims; one edge, four claim edges, three links.
            assert (first["nodes"], first["rels"]) == (7, 8)
            untraced = driver.execute_query(
                mine + "MATCH (n)-[r]->() WHERE r.signature IS NOT NULL "
                "AND size(r.evidence_doc_ids) = 0 RETURN count(r) AS n"
            ).records[0]["n"]
            assert untraced == 0
        finally:
            driver.execute_query(mine + "DETACH DELETE n")


def live_graph(suffix: str) -> KnowledgeGraph:
    """`_graph()` with every entity type and key suffixed, so a shared server is left as found."""
    graph = _graph()

    def retype(entity: Entity) -> Entity:
        return entity.model_copy(
            update={"type": f"{entity.type}_{suffix}", "key": f"{entity.key}:{suffix}"}
        )

    def refact(fact: Fact) -> Fact:
        obj = retype(fact.object_entity) if fact.object_entity is not None else None
        return fact.model_copy(update={"subject": retype(fact.subject), "object_entity": obj})

    def relink(link: EntityLink) -> EntityLink:
        return link.model_copy(
            update={
                "source_key": f"{link.source_key}:{suffix}",
                "target_key": f"{link.target_key}:{suffix}",
            }
        )

    return graph.model_copy(
        update={
            "entities": tuple(retype(e) for e in graph.entities),
            "facts": tuple(refact(f) for f in graph.facts),
            "links": tuple(relink(link) for link in graph.links),
        }
    )
