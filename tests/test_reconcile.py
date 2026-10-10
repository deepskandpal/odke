"""The reconciler: retract a source on update or delete, and retire facts left with none (#116).

`retract` is a pure function on one fact, checked first. Then the issue's
done-when, against both stores: ingest A and B, which both state F; delete A,
and F remains with support 1; delete B, and F is retired; update A with
changed text, and its old facts lose support while its new ones gain it; every
step run twice gives the same store. JSONL runs here, and Neo4j in
`test_the_reconciler_against_a_live_neo4j` when `NEO4J_URI` is set. The
recording driver checks what `Neo4jSink.retract` sends: one write
transaction, facts found through the evidence indexes, never a scan.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    Ontology,
    Reconciler,
    Retractable,
    SourceTier,
    Span,
    Validator,
)
from openodke.cli.main import app
from openodke.corroborate import NEAR_DUPLICATES, SignatureCorroborator, partners
from openodke.reconcile import retract, with_parents
from openodke.sinks import JsonlSink
from openodke.sinks.neo4j import (
    Neo4jConstrainer,
    Neo4jSink,
    provenance_of,
    relationship_fact,
    signature_of,
)
from openodke.stages import PassThroughGate, PassThroughGrounder
from test_neo4j_sink import FakeDriver

WHEN = datetime(2026, 9, 1, tzinfo=UTC)
AT = datetime(2026, 10, 9, tzinfo=UTC)
ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace")
ACME = Entity(key="c:acme", type="Company", label="Acme")
ONTOLOGY = Ontology.from_dict(
    {
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "employs": {"domain": ["Company"], "range": "Person", "inverse_of": "employer"},
            "hq": {"domain": ["Company"]},
        },
    }
)


def _evidence(
    doc: str,
    *,
    uri: str | None = None,
    tier: SourceTier = SourceTier.COMMUNITY,
    at: datetime = WHEN,
) -> Evidence:
    return Evidence(
        doc_id=doc, span=Span(doc_id=doc, start=0, end=10), uri=uri, tier=tier, retrieved_at=at
    )


def _merged(*facts: Fact, **options: Any) -> Fact:
    (fact,) = SignatureCorroborator(**options).corroborate(facts)
    return fact


def _employed(*evidence: Evidence, **update: Any) -> Fact:
    fact = Fact(subject=ADA, predicate="employer", object_entity=ACME, evidence=evidence)
    return fact.model_copy(update=update) if update else fact


# --------------------------------------------------------------------------- #
# One fact
# --------------------------------------------------------------------------- #


def test_a_fact_with_another_source_keeps_it_and_loses_the_document() -> None:
    fact = _merged(_employed(_evidence("a")), _employed(_evidence("b")))
    after = retract(fact, {"a"}, AT)
    assert (after.support, [s.source for s in after.supported_by]) == (1, ["doc:b"])
    assert [e.doc_id for e in after.evidence] == ["b"]
    assert after.retired_at is None


def test_an_entry_losing_one_page_keeps_the_rest_and_reads_its_tier_and_clock_again() -> None:
    pages = [
        _evidence("p1", uri="https://news.example/1", tier=SourceTier.CURATED, at=WHEN),
        _evidence("p2", uri="https://news.example/2", at=WHEN - timedelta(days=3)),
    ]
    fact = _merged(*(_employed(e) for e in pages))
    (entry,) = retract(fact, {"p1"}, AT).supported_by
    assert (entry.doc_ids, entry.tier, entry.retrieved_at) == (
        ("p2",),
        SourceTier.COMMUNITY,
        WHEN - timedelta(days=3),
    )


def test_a_fact_left_with_no_source_is_retired_and_kept_with_its_valid_clock() -> None:
    since = datetime(2019, 1, 1, tzinfo=UTC)
    fact = _merged(_employed(_evidence("a"), valid_from=since))
    after = retract(fact, {"a"}, AT)
    assert (after.support, after.supported_by, after.evidence) == (0, (), ())
    assert (after.retired_at, after.valid_from, after.signature) == (AT, since, fact.signature)
    # Again: nothing cites the document, so nothing moves, not even the time.
    assert retract(after, {"a"}, AT + timedelta(days=1)) is after
    assert retract(fact, {"zzz"}, AT) is fact


def test_a_near_duplicate_group_loses_the_document_and_goes_when_one_is_left() -> None:
    text = " ".join(f"w{i}" for i in range(50))
    docs = [
        Document(id="orig", text=text, uri="https://a.example/"),
        Document(id="copy", text=text, uri="https://b.example/"),
    ]
    facts = [_employed(_evidence(d.id, uri=d.uri)) for d in docs]
    fact = _merged(*facts, documents=docs)
    assert fact.qualifiers[NEAR_DUPLICATES] == (("copy", "orig"),)
    after = retract(fact, {"orig"}, AT)
    assert NEAR_DUPLICATES not in after.qualifiers
    assert [(s.source, s.doc_ids) for s in after.supported_by] == [("a.example", ("copy",))]


def test_a_fact_written_before_support_lists_is_counted_afresh() -> None:
    legacy = _employed(_evidence("a"), _evidence("b"), support=2)
    after = retract(legacy, {"a"}, AT)
    assert [s.source for s in after.supported_by] == ["doc:b"] and after.support == 1


def test_a_derived_fact_is_retired_with_its_parent() -> None:
    parent = _merged(_employed(_evidence("a")))
    (partner,) = partners([parent], ONTOLOGY)
    # Even one whose list has drifted from its parent's goes with it.
    drifted = partner.model_copy(update={"evidence": (_evidence("z"),)})
    after = with_parents([retract(parent, {"a"}, AT), drifted], AT)
    assert [f.retired_at for f in after] == [AT, AT]


def test_the_reconciler_needs_a_store_that_can_retract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to reconcile"):
        Reconciler([])
    sink = JsonlSink(tmp_path)
    assert isinstance(sink, Retractable)
    report = Reconciler(sink).delete("a")
    assert report.documents == ("a",) and report.cited == 0
    assert "nothing: no fact cited them" in report.render()


# --------------------------------------------------------------------------- #
# The done-when, against a store
# --------------------------------------------------------------------------- #

TEXT_A = "Ada Lovelace works at Acme. Acme is based in Leeds."
TEXT_B = "Ada Lovelace works at Acme."
TEXT_A2 = "Ada Lovelace works at Acme. Acme moved to York."


def _claim(doc: str, predicate: str, obj: Entity | str) -> Fact:
    entity = isinstance(obj, Entity)
    return Fact(
        subject=ADA if predicate == "employer" else ACME,
        predicate=predicate,
        object_entity=obj if entity else None,
        object_value=None if entity else obj,
        evidence=(Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=10)),),
    )


VERSIONS = {
    "A": (TEXT_A, ("employer", ACME), ("hq", "Leeds")),
    "B": (TEXT_B, ("employer", ACME)),
    "A2": (TEXT_A2, ("employer", ACME), ("hq", "York")),
}


def _done_when(
    sink: Any, snapshot: Callable[[], Any], ontology: Ontology, rename: Callable[[Fact], Fact]
) -> list[Any]:
    """The issue's steps, each run twice; the store after each step."""
    validator = Validator(
        ontology, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
    )
    reconciler = Reconciler([sink])

    def ingest(name: str, doc: str, *, update: bool = False) -> None:
        text, *claims = VERSIONS[name]
        facts = [rename(_claim(doc, predicate, obj)) for predicate, obj in claims]
        validator.validate(facts, [Document(id=doc, text=text)], update=update)

    steps: list[Callable[[], Any]] = [
        lambda: (ingest("A", "doc-a"), ingest("B", "doc-b")),
        lambda: reconciler.delete("doc-a", at=AT),
        lambda: reconciler.delete("doc-b", at=AT + timedelta(hours=1)),
        lambda: ingest("A", "doc-a"),
        lambda: ingest("A2", "doc-a", update=True),
    ]
    seen = []
    for step in steps:
        step()
        once = snapshot()
        step()
        assert snapshot() == once, f"step {len(seen) + 1} is not idempotent"
        seen.append(once)
    return seen


def _check(seen: list[Any], employer: str, employs: str) -> None:
    """What each step leaves: (support, sources, retired) per claim, and the projected hq."""
    live_a, live_b = (1, ["doc:doc-a"], False), (1, ["doc:doc-b"], False)
    gone = (0, [], True)
    both = (2, ["doc:doc-a", "doc:doc-b"], False)
    assert [s["facts"] for s in seen] == [
        # Ingest A and B: F twice, with its derived partner sharing the list.
        {employer: both, employs: both, "Leeds": live_a},
        # Delete A: F keeps B; Leeds had only A.
        {employer: live_b, employs: live_b, "Leeds": gone},
        # Delete B: F is retired, and its partner with it.
        {employer: gone, employs: gone, "Leeds": gone},
        # A again: what it states regains it.
        {employer: live_a, employs: live_a, "Leeds": live_a},
        # A updated: Leeds, which it no longer states, loses it; York gains it.
        {employer: live_a, employs: live_a, "Leeds": gone, "York": live_a},
    ]
    assert [s["hq"] for s in seen] == ["Leeds", None, None, "Leeds", "York"]


def test_the_reconciler_against_jsonl(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path, merge=True)

    def snapshot() -> dict[str, Any]:
        lines = (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()
        facts = [Fact.model_validate_json(line) for line in lines]
        found = {
            (f.object_entity.key if f.object_entity else f.predicate)
            if f.predicate != "hq"
            else f.object_value: (
                f.support,
                [s.source for s in f.supported_by],
                f.retired_at is not None,
            )
            for f in facts
        }
        retired = {str(f.signature): f.retired_at for f in facts}
        live = [f.object_value for f in facts if f.predicate == "hq" and f.retired_at is None]
        return {"facts": found, "retired": retired, "hq": live[0] if live else None}

    seen = _done_when(sink, snapshot, ONTOLOGY, lambda fact: fact)
    _check(seen, ACME.key, ADA.key)
    # A retired fact keeps the time it lost its last source.
    assert set(seen[2]["retired"].values()) >= {AT, AT + timedelta(hours=1)}


def test_a_hard_delete_removes_what_it_retires_and_nothing_else(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path, merge=True)
    validator = Validator(
        ONTOLOGY, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
    )
    validator.validate([_claim("doc-a", "hq", "Leeds")], [Document(id="doc-a", text=TEXT_A)])
    validator.validate([_claim("doc-b", "hq", "York")], [Document(id="doc-b", text=TEXT_A2)])
    Reconciler(sink).delete("doc-b", at=AT)
    report = Reconciler(sink, hard_delete=True).delete("doc-a", at=AT)
    assert (report.cited, report.deleted, report.retired) == (1, 1, 0)
    lines = (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert [Fact.model_validate_json(line).object_value for line in lines] == ["York"]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["facts"] == 1


# --------------------------------------------------------------------------- #
# Neo4j, on the recording driver
# --------------------------------------------------------------------------- #


def _evidence_index(predicate: str) -> dict[str, Any]:
    return {
        "name": f"odke_evidence_{predicate}",
        "type": "FULLTEXT",
        "entityType": "RELATIONSHIP",
        "labelsOrTypes": [predicate],
        "properties": ["evidence_doc_ids"],
    }


def _row(i: int, fact: Fact) -> dict[str, Any]:
    obj = fact.object_entity
    return {
        "id": f"rel-{i}",
        "predicate": fact.predicate,
        "props": provenance_of(fact, WHEN),
        "subject_key": fact.subject.key,
        "subject_labels": [fact.subject.type, "Entity"],
        "object_key": obj.key if obj else None,
        "object_labels": [obj.type, "Entity"] if obj else ["Claim"],
        "value": None if obj else fact.object_value,
    }


class Citing(FakeDriver):
    """A store whose evidence indexes find `held`, and which has a type no index covers."""

    def __init__(self, *held: Fact) -> None:
        types = ("employer", "hq", "SIMILAR", "unindexed")
        super().__init__(
            {
                "SHOW INDEXES": [_evidence_index("employer"), _evidence_index("hq")],
                "db.relationshipTypes": [{"name": t} for t in types],
            }
        )
        self.held = held

    def answer(self, cypher: str) -> list[dict[str, Any]]:
        if "queryRelationships" in cypher:
            return [_row(i, fact) for i, fact in enumerate(self.held)]
        return super().answer(cypher)


def test_neo4j_retracts_in_one_write_transaction_through_the_evidence_indexes() -> None:
    both = _merged(_employed(_evidence("doc-a")), _employed(_evidence("doc-b")))
    leeds = _merged(Fact(subject=ACME, predicate="hq", object_value="Leeds",
                         evidence=(_evidence("doc-a"),)))  # fmt: skip
    driver = Citing(both, leeds)
    sink = Neo4jSink(driver=driver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts = sink.retract(["doc-a"], at=AT)
    assert counts == {"cited": 2, "lost": 1, "retired": 1, "deleted": 0}
    assert [str(w.message).split(",")[0] for w in caught] == [
        "the store has no index for unindexed.evidence_doc_ids"
    ]
    assert driver.transactions == 1
    writes = driver.writes
    (citing,) = [(c, p) for c, p in writes if "queryRelationships" in c]
    assert citing[1]["indexes"] == ["odke_evidence_employer", "odke_evidence_hq"]
    assert citing[1]["terms"] == '"doc-a"'
    updates = [(c, p) for c, p in writes if c.endswith("SET r += row.props")]
    assert [c.split("\n")[1] for c, _ in updates] == [
        "MATCH ()-[r:`employer` {signature: row.signature}]->()",
        "MATCH ()-[r:`hq` {signature: row.signature}]->()",
    ]
    by_key = {row["signature"]: row["props"] for _, p in updates for row in p["rows"]}
    first, second = by_key[signature_of(both)], by_key[signature_of(leeds)]
    assert (first["support"], first["support_sources"]) == (1, ["doc:doc-b"])
    assert (second["support"], second["retired_at"], second["evidence_doc_ids"]) == (0, AT, [])
    # Leeds was the projected hq; with no claim left (the driver returns none), it goes.
    (projection,) = [p for c, p in writes if c.endswith("SET s.`hq` = row.value")]
    assert projection["rows"] == [{"subject_key": "c:acme", "value": None}]
    assert not any("DELETE" in c or "SET r = " in c for c, _ in writes)

    hard = Citing(leeds)
    with pytest.warns(UserWarning, match="unindexed.evidence_doc_ids"):
        assert Neo4jSink(driver=hard).retract(["doc-a"], at=AT, hard=True)["deleted"] == 1
    (deleted,) = [p for c, p in hard.writes if "DELETE r" in c]
    assert deleted["rows"] == [{"signature": signature_of(leeds)}]


def test_a_relationship_reads_back_as_the_fact_with_its_ends() -> None:
    fact = _merged(_employed(_evidence("doc-a")))
    back = relationship_fact(_row(0, fact))
    assert back.signature == fact.signature and back.supported_by == fact.supported_by


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_odke_reconcile_retracts_from_a_jsonl_store(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "store", merge=True)
    validator = Validator(
        ONTOLOGY, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
    )
    validator.validate([_claim("doc-a", "hq", "Leeds")], [Document(id="doc-a", text=TEXT_A)])
    runner = CliRunner()
    args = ["reconcile", "--sink", str(tmp_path / "store"), "--delete", "doc-a"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "retired       1 fact left with no source, kept" in result.output
    result = runner.invoke(app, args)
    assert "cited         nothing: no fact cited them" in result.output

    result = runner.invoke(app, ["reconcile", "--sink", str(tmp_path), "--delete", "x"])
    assert result.exit_code == 2 and "holds no facts.jsonl" in result.output
    result = runner.invoke(
        app,
        ["reconcile", "--sink", "bolt://example.invalid:7687", "--delete", "x",
         "--password-env", "ODKE_UNSET_FOR_THIS_TEST"],
    )  # fmt: skip
    assert result.exit_code == 2 and "ODKE_UNSET_FOR_THIS_TEST is not set" in result.output


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


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_the_reconciler_against_a_live_neo4j() -> None:
    """The done-when in Neo4j. Names are suffixed, so a shared server is left as found."""
    pytest.importorskip("neo4j")
    from openodke.sinks.neo4j import _CITING, _retract_cypher

    suffix = uuid.uuid4().hex[:8]
    names = {
        "Person": f"Person_{suffix}",
        "Company": f"Company_{suffix}",
        "employer": f"employer_{suffix}",
        "employs": f"employs_{suffix}",
        "hq": f"hq_{suffix}",
    }
    ontology = Ontology.from_dict(
        {
            "types": {names["Person"]: {}, names["Company"]: {}},
            "predicates": {
                names["employer"]: {"domain": [names["Person"]], "range": names["Company"]},
                names["employs"]: {
                    "domain": [names["Company"]],
                    "range": names["Person"],
                    "inverse_of": names["employer"],
                },
                names["hq"]: {"domain": [names["Company"]]},
            },
        }
    )
    ada = ADA.model_copy(update={"type": names["Person"], "key": f"p:ada:{suffix}"})
    acme = ACME.model_copy(update={"type": names["Company"], "key": f"c:acme:{suffix}"})

    def rename(fact: Fact) -> Fact:
        if fact.predicate == "employer":
            return fact.model_copy(
                update={"subject": ada, "object_entity": acme, "predicate": names["employer"]}
            )
        return fact.model_copy(update={"subject": acme, "predicate": names["hq"]})

    mine = (
        f"MATCH (n) WHERE n:`{names['Person']}` OR n:`{names['Company']}` OR "
        f"(n:Claim AND n.subject_type = '{names['Company']}') "
    )
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(ontology)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth, ontology=ontology) as sink:
        driver = sink._driver

        def snapshot() -> dict[str, Any]:
            records = driver.execute_query(
                f"MATCH (s)-[r]->(o) WHERE s:`{names['Person']}` OR s:`{names['Company']}` "
                "RETURN type(r) AS predicate, coalesce(o.key, o.value) AS object, "
                "r.support AS support, r.support_sources AS sources, r.retired_at AS retired"
            ).records
            facts = {
                r["object"]: (r["support"], list(r["sources"]), r["retired"] is not None)
                for r in records
            }
            retired = {(r["predicate"], r["object"]): r["retired"] for r in records}
            (node,) = driver.execute_query(
                f"MATCH (c:`{names['Company']}` {{key: $key}}) RETURN c[$hq] AS hq",
                key=acme.key,
                hq=names["hq"],
            ).records
            return {"facts": facts, "retired": retired, "hq": node["hq"]}

        try:
            sink.bootstrap(ontology)
            driver.execute_query("CALL db.awaitIndexes(300)")
            seen = _done_when(sink, snapshot, ontology, rename)
            _check(seen, acme.key, ada.key)

            # Found through the evidence index and rewritten by signature: no scan.
            for cypher, params in (
                (_CITING, {"indexes": [f"odke_evidence_{names['hq']}"], "terms": '"doc-a"'}),
                (_retract_cypher(names["hq"]), {"rows": [{"signature": "x", "props": {}}]}),
            ):
                with driver.session() as session:
                    plan = session.run("EXPLAIN " + cypher, params).consume().plan
                operators = _operators(plan)
                assert not any("Scan" in op for op in operators), operators
        finally:
            driver.execute_query(mine + "DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")
