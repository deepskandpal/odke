"""Tenant keys, batch transactions and the write report (#159, DECISIONS #44).

A store keys each tenant apart: `<tenant>/<key>` for an entity, the signature
hashed with the tenant for a fact, and `tenant` on both. Checked here against
the recording driver from `test_neo4j_sink`, and against a real server in
`test_two_tenants_share_a_live_neo4j_and_never_merge_rerun_or_retract_across`
when `NEO4J_URI` is set: two tenants write the same facts and get no merge
across, a rerun writes nothing new, the reconciler retracts within one tenant,
and the store lookup never returns another tenant's nodes.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, KnowledgeGraph, Ontology, Span, Validator
from openodke.cli.main import app
from openodke.corroborate import MemoryLookup
from openodke.interop import read_neo4j
from openodke.reconcile import Reconciler
from openodke.sinks import JsonlSink
from openodke.sinks.neo4j import (
    Neo4jConstrainer,
    Neo4jLookup,
    Neo4jSink,
    plan,
    signature_of,
    stored_entity,
)
from openodke.sinks.report import summary
from openodke.stages import PassThroughGate, PassThroughGrounder
from openodke.tenants import scoped, tenant_name, unscoped
from openodke.types import EntityLink, LinkKind
from test_neo4j_sink import FakeDriver

ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace", attributes={"tenant": "mine"})
ACME = Entity(key="c:acme", type="Company", label="Acme")
TEXT = "Ada Lovelace works at Acme. Acme is based in Leeds."
WHEN = datetime(2026, 10, 1, tzinfo=UTC)


def _cited(doc: str) -> tuple[Evidence, ...]:
    return (Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=len(TEXT)), retrieved_at=WHEN),)


def _claims(doc: str, ada: Entity = ADA, acme: Entity = ACME) -> list[Fact]:
    return [
        Fact(subject=ada, predicate="employer", object_entity=acme, evidence=_cited(doc)),
        Fact(subject=acme, predicate="hq", object_value="Leeds", evidence=_cited(doc)),
    ]


def _graph() -> KnowledgeGraph:
    link = EntityLink(source_key="c:acme-ltd", target_key="c:acme", kind=LinkKind.SIMILAR)
    return KnowledgeGraph(entities=(ADA, ACME), facts=tuple(_claims("d1")), links=(link,))


# --------------------------------------------------------------------------- #
# Names and keys
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "a/b", "-acme", "acme corp", "acme:eu"])
def test_a_tenant_is_a_name_a_key_can_carry(bad: str) -> None:
    assert tenant_name("acme-eu.1_x") == "acme-eu.1_x" and tenant_name(None) is None
    with pytest.raises(ValueError, match="letters, digits"):
        tenant_name(bad)


def test_a_key_is_scoped_at_the_first_slash_and_back() -> None:
    assert scoped("Page:https://x/y", "acme") == "acme/Page:https://x/y"
    assert unscoped("acme/Page:https://x/y", "acme") == "Page:https://x/y"
    assert scoped("p:ada", None) == unscoped("p:ada", None) == "p:ada"
    # Another tenant's key is not this tenant's to unscope.
    assert unscoped("beta/p:ada", "acme") == "beta/p:ada"


# --------------------------------------------------------------------------- #
# The write plan
# --------------------------------------------------------------------------- #


def test_a_tenant_scopes_every_key_and_signature_and_leaves_the_cypher_alone() -> None:
    kg = _graph()
    mine, theirs, none = (plan(kg, tenant=t) for t in ("acme", "beta", None))
    assert [s.cypher for s in mine] == [s.cypher for s in none] == [s.cypher for s in theirs]
    rows = {s.kind: s.rows for s in mine}
    keys = {row["key"] for s in mine if s.kind == "entity" for row in s.rows}
    assert keys == {"acme/p:ada", "acme/c:acme"}
    assert all(row["attributes"]["tenant"] == "acme" for row in rows["entity"])
    # The sink owns `tenant`: an attribute of that name is kept under another.
    (ada,) = [row for row in rows["entity"] if row["key"] == "acme/p:ada"]
    assert ada["attributes"]["attribute_tenant"] == "mine"
    (edge,) = rows["edge"]
    assert (edge["subject_key"], edge["object_key"]) == ("acme/p:ada", "acme/c:acme")
    assert edge["props"]["tenant"] == "acme" and "tenant" not in plan(kg)[2].rows[0]["props"]
    (fact,) = [f for f in kg.facts if f.is_edge]
    signatures = {signature_of(fact, t) for t in ("acme", "beta", None)}
    assert len(signatures) == 3 and edge["signature"] == signature_of(fact, "acme")
    assert signature_of(fact) == signature_of(fact, None)
    (link,) = rows["link"]
    assert (link["source_key"], link["target_key"]) == ("acme/c:acme-ltd", "acme/c:acme")
    (projected,) = rows["projection"]
    assert projected["subject_key"] == "acme/c:acme"


def test_a_stored_node_comes_back_under_its_tenants_key_and_never_as_an_attribute() -> None:
    props = {"key": "acme/c:acme", "label": "Acme", "tenant": "acme", "attribute_tenant": "x"}
    entity = stored_entity(props, "Company", tenant="acme")
    assert entity.key == "c:acme" and entity.attributes == {"tenant": "x"}


# --------------------------------------------------------------------------- #
# The sink, the lookup, the reconciler
# --------------------------------------------------------------------------- #


def test_a_sink_scoped_to_a_tenant_shares_its_connection_and_never_writes_for_two() -> None:
    driver = FakeDriver()
    store = Neo4jSink(driver=driver, database="graph", batch_size=7)
    acme = store.scoped("acme")
    assert (acme._driver, acme.database, acme.batch_size, acme.tenant) == (
        driver,
        "graph",
        7,
        "acme",
    )
    assert acme.scoped("acme") is acme and store.scoped(None) is store
    with pytest.raises(ValueError, match="writes tenant 'acme', not 'beta'"):
        acme.scoped("beta")
    assert acme.lookup().tenant == "acme" and store.lookup().tenant is None


def test_a_rerun_writes_nothing_new_and_the_report_says_so() -> None:
    sink = Neo4jSink(driver=FakeDriver(), tenant="acme")
    sink.write(_graph())
    first = sink.writes.as_dict()
    # Two nodes, two facts and a link, in one transaction.
    assert first["entities"] == {"written": 2, "merged": 0, "skipped": 0}
    assert first["facts"] == {"written": 2, "merged": 0, "skipped": 0}
    assert first["transactions"] == 1
    sink.write(_graph())
    second = sink.writes.as_dict()
    assert second["entities"] == {"written": 2, "merged": 2, "skipped": 0}
    assert second["facts"] == {"written": 2, "merged": 2, "skipped": 0}
    assert summary(second).startswith("entities 2 written, 2 merged; facts 2 written, 2 merged")


def test_a_tenant_lookup_asks_for_its_keys_and_an_untenanted_one_for_untenanted_nodes() -> None:
    probe = Entity(key="c:acme", type="Company", label="Acme")
    lookup = Neo4jLookup(driver=FakeDriver(_indexes()), tenant="acme")
    queries = lookup.statements([probe])
    by_kind = {q.kind: q for q in queries}
    assert by_kind["key"].params["rows"] == [{"block": 0, "value": "acme/c:acme"}]
    assert all("n.`tenant` = $tenant" in q.cypher for q in queries)
    assert all(q.params["tenant"] == "acme" for q in queries)
    unscoped_queries = Neo4jLookup(driver=FakeDriver(_indexes())).statements([probe])
    assert all("n.`tenant` IS NULL" in q.cypher for q in unscoped_queries)
    assert MemoryLookup({}).scoped("acme").tenant == "acme"


def _indexes() -> dict[str, list[dict[str, Any]]]:
    return {
        "SHOW INDEXES": [
            {"name": "odke_key_Company", "type": "RANGE", "entityType": "NODE",
             "labelsOrTypes": ["Company"], "properties": ["key"]},
            {"name": "odke_names_Company", "type": "FULLTEXT", "entityType": "NODE",
             "labelsOrTypes": ["Company"], "properties": ["label", "aliases"]},
        ]
    }  # fmt: skip


def test_a_retraction_reads_its_tenants_facts_and_a_check_its_tenants_subjects() -> None:
    evidence = {"name": "odke_evidence_employer", "type": "FULLTEXT", "entityType": "RELATIONSHIP"}
    driver = FakeDriver(
        {
            "SHOW INDEXES": [
                {**evidence, "labelsOrTypes": ["employer"], "properties": ["evidence_doc_ids"]}
            ],
            "db.relationshipTypes": [{"name": "employer"}],
            "// odke:check": [
                {
                    "labels": ["Person"],
                    "subject": "acme/p:ada",
                    "objects": ["acme/c:a", "acme/c:b"],
                },
                {
                    "labels": ["Person"],
                    "subject": "beta/p:ada",
                    "objects": ["beta/c:a", "beta/c:b"],
                },
            ],
        }
    )
    sink = Neo4jSink(driver=driver, tenant="acme")
    sink.retract(["d1"], at=WHEN)
    (citing,) = [(c, p) for _, c, p in driver.calls if "queryRelationships" in c]
    assert "WHERE r.tenant = $tenant" in citing[0] and citing[1]["tenant"] == "acme"
    Neo4jSink(driver=driver).retract(["d1"], at=WHEN)
    assert (
        "WHERE r.tenant IS NULL" in [c for _, c, _ in driver.calls if "queryRelationships" in c][1]
    )

    ontology = Ontology.from_dict(
        {"types": {"Person": {}, "Company": {}},
         "predicates": {"employer": {"domain": ["Person"], "range": "Company"}}}
    )  # fmt: skip
    assert sink.check(ontology) == {
        "employer": [{"labels": ["Person"], "subject": "p:ada", "objects": ["c:a", "c:b"]}]
    }
    assert len(Neo4jSink(driver=driver).check(ontology)["employer"]) == 2


def test_read_neo4j_reads_one_tenants_facts_under_its_keys() -> None:
    row = {
        "id": "r1", "type": "employer",
        "properties": {"evidence_doc_ids": ["d1"], "tenant": "acme"},
        "subject_labels": ["Person", "Entity"], "subject_label": None, "subject_name": None,
        "subject_key": "acme/p:ada", "subject_id": "n1", "object_labels": ["Company", "Entity"],
        "object_label": None, "object_name": None, "object_key": "acme/c:acme", "object_id": "n2",
        "value": None,
    }  # fmt: skip
    driver = FakeDriver({"ORDER BY id": [row], "count(r) AS unread": [{"unread": 0}]})
    (read,), _ = read_neo4j(driver, tenant="acme")
    assert (read.subject, read.object) == ("p:ada", "c:acme")
    (query, params) = next((c, p) for _, c, p in driver.calls if "ORDER BY id" in c)
    assert "AND r.tenant = $tenant" in query and params["tenant"] == "acme"


# --------------------------------------------------------------------------- #
# The Validator, the JSONL store and the commands
# --------------------------------------------------------------------------- #


def test_the_validator_scopes_its_sinks_and_its_lookup_and_reports_each_sinks_writes(
    tmp_path: Path,
) -> None:
    driver = FakeDriver()
    store = JsonlSink(tmp_path / "out", merge=True)
    validator = Validator(
        grounder=PassThroughGrounder(),
        gate=PassThroughGate(),
        sinks=[Neo4jSink(driver=driver), store],
        lookup=MemoryLookup({}),
        tenant="acme",
    )
    kg, report = validator.validate(_claims("d1"), [Document(id="d1", text=TEXT)])
    keys = {row["key"] for cypher, p in driver.writes if "MERGE (n:" in cypher for row in p["rows"]}
    assert keys == {"acme/p:ada", "acme/c:acme"}
    assert report.tenant == "acme" and "tenant        acme" in report.render()
    assert report.writes is not None
    assert report.writes["0:Neo4jSink"]["facts"]["written"] == 2
    assert report.writes["1:JsonlSink"]["facts"] == {"written": 2, "merged": 0, "skipped": 0}
    assert kg.stats["writes"] == report.writes
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tenant"] == "acme" and manifest["stats"]["writes"] == report.writes
    assert manifest["stats"]["validation"]["writes"] == report.writes

    # A rerun merges into what both stores hold, and makes nothing new.
    _, again = validator.validate(_claims("d1"), [Document(id="d1", text=TEXT)])
    assert again.writes is not None
    assert again.writes["1:JsonlSink"]["facts"] == {"written": 0, "merged": 2, "skipped": 0}
    assert again.writes["0:Neo4jSink"]["facts"]["written"] == 0
    # One directory holds one tenant: another's merging write is refused.
    with pytest.raises(ValueError, match="holds tenant 'acme', and this sink writes 'beta'"):
        Validator(grounder=PassThroughGrounder(), sinks=[store], tenant="beta").validate(
            _claims("d1"), [Document(id="d1", text=TEXT)]
        )


def test_the_commands_take_a_tenant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    triples = Path(__file__).parent.parent / "examples" / "triples"
    args = ["validate", "--facts", str(triples / "triples.jsonl"), "--texts"]
    args += [str(triples / "texts"), "-o", "out", "--dry-run", "--tenant", "acme"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "tenant        acme" in result.output
    result = CliRunner().invoke(app, [*args[:-1], "acme/eu"])
    assert result.exit_code == 2 and "letters, digits" in result.output
    (tmp_path / "store").mkdir()
    JsonlSink(tmp_path / "store", tenant="acme").write(KnowledgeGraph(facts=tuple(_claims("d1"))))
    reconcile = ["reconcile", "--sink", "store", "--delete", "d1"]
    result = CliRunner().invoke(app, [*reconcile, "--tenant", "beta"])
    assert result.exit_code == 2 and "holds tenant 'acme'" in result.output
    result = CliRunner().invoke(app, [*reconcile, "--tenant", "acme"])
    assert result.exit_code == 0 and "retired       2 facts" in result.output


# --------------------------------------------------------------------------- #
# Done when: two tenants and a rerun in a live Neo4j
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_two_tenants_share_a_live_neo4j_and_never_merge_rerun_or_retract_across() -> None:
    """Two tenants write the same facts from a text with the same id; then everything else.

    Types and predicates are suffixed, so a shared server is left as found.
    """
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company = f"Person_{suffix}", f"Company_{suffix}"
    employer, hq = f"employer_{suffix}", f"hq_{suffix}"
    ontology = Ontology.from_dict(
        {
            "types": {person: {}, company: {}},
            "predicates": {
                employer: {"domain": [person], "range": company, "cardinality": "multi"},
                hq: {"domain": [company]},
            },
        }
    )
    ada = ADA.model_copy(update={"type": person, "attributes": {}})
    acme = ACME.model_copy(update={"type": company})
    a, b = f"acme-{suffix}", f"beta-{suffix}"

    def claims(doc: str, town: str = "Leeds") -> list[Fact]:
        return [
            Fact(subject=ada, predicate=employer, object_entity=acme, evidence=_cited(doc)),
            Fact(subject=acme, predicate=hq, object_value=town, evidence=_cited(doc)),
        ]

    mine = (
        f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
        f"(n:Claim AND n.subject_type = '{company}') "
    )
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(ontology)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth, ontology=ontology, batch_size=3) as store:
        driver = store._driver

        def run(tenant: str, doc: str, town: str = "Leeds") -> Any:
            validator = Validator(
                ontology,
                grounder=PassThroughGrounder(),
                gate=PassThroughGate(),
                sinks=[store],
                tenant=tenant,
            )
            return validator.validate(claims(doc, town), [Document(id=doc, text=TEXT)])[1]

        def held(tenant: str) -> dict[str, tuple[int, list[str], bool]]:
            records = driver.execute_query(
                f"MATCH (s)-[r]->() WHERE (s:`{person}` OR s:`{company}`) AND r.tenant = $t "
                "RETURN type(r) AS p, r.support AS support, r.support_sources AS sources, "
                "r.retired_at IS NOT NULL AS retired",
                t=tenant,
            ).records
            return {r["p"]: (r["support"], sorted(r["sources"]), r["retired"]) for r in records}

        def counted() -> dict[str, int]:
            return (
                driver.execute_query(
                    mine + "OPTIONAL MATCH (n)-[r]-() "
                    "RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
                )
                .records[0]
                .data()
            )

        try:
            # The uniqueness constraints hold for every tenant at once: one per key.
            store.bootstrap(ontology)
            driver.execute_query("CALL db.awaitIndexes(300)")
            first = run(a, "doc-1")
            run(b, "doc-1")
            # The same facts, twice over: two people, two companies, two of every fact.
            assert counted() == {"nodes": 6, "rels": 4}
            one = (1, ["doc:doc-1"], False)
            assert held(a) == held(b) == {employer: one, hq: one}
            assert first.writes is not None
            assert first.writes["0:Neo4jSink"]["facts"]["written"] == 2

            # A second source in one tenant merges with that tenant's facts alone.
            run(a, "doc-2")
            both = (2, ["doc:doc-1", "doc:doc-2"], False)
            assert held(a) == {employer: both, hq: both} and held(b) == {employer: one, hq: one}
            nodes = counted()

            # A rerun writes nothing new, in transactions of three rows.
            again = run(a, "doc-2")
            assert again.writes is not None
            wrote = again.writes["0:Neo4jSink"]
            assert wrote["entities"]["written"] == wrote["facts"]["written"] == 0
            assert wrote["facts"]["merged"] == 2 and wrote["transactions"] >= 2
            assert counted() == nodes and held(a) == {employer: both, hq: both}

            # The store lookup reads its own tenant's nodes, under its tenant's keys.
            probe = Entity(key="c:acme-plc", type=company, label="Acme")
            for tenant in (a, b):
                (found,) = store.scoped(tenant).lookup().candidates([probe])[probe.key]
                assert found.key == acme.key and "tenant" not in found.attributes
            assert store.lookup().candidates([probe]) == {probe.key: []}
            assert store.lookup(tenant=f"none-{suffix}").candidates([probe]) == {probe.key: []}

            # A check reads one tenant's subjects: a second head office is b's alone.
            run(b, "doc-3", town="York")
            assert hq in store.scoped(b).check(ontology)
            assert hq not in store.scoped(a).check(ontology)
            b_employer = (2, ["doc:doc-1", "doc:doc-3"], False)
            assert held(b)[employer] == b_employer and held(a) == {employer: both, hq: both}

            # read_neo4j reads one tenant's facts, under its keys.
            rows, _ = read_neo4j(driver, tenant=a, predicates=[employer, hq])
            assert {row.subject for row in rows} == {ada.label, acme.label}
            assert len(rows) == 4  # two facts, each citing two texts

            # The reconciler retracts within one tenant: b's doc-1 is b's.
            report = Reconciler([store.scoped(a)]).delete(["doc-1"])
            assert (report.cited, report.lost, report.retired) == (2, 2, 0)
            after = (1, ["doc:doc-2"], False)
            assert held(a) == {employer: after, hq: after}
            assert held(b)[employer] == b_employer
            # b's turn: its employer keeps doc-3, its Leeds head office is retired.
            report = Reconciler([store.scoped(b)]).delete(["doc-1"])
            assert (report.cited, report.lost, report.retired) == (2, 1, 1)
            assert held(b)[employer] == (1, ["doc:doc-3"], False)
            assert held(a) == {employer: after, hq: after}
        finally:
            driver.execute_query(mine + "DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")
