"""Support lists: which independent sources back each fact (#115, DECISIONS #33).

The corroborator names the sources it counts, one `Support` each, so `support`
is the list's length; `conftest.every_support_list_is_counted` holds that for
every fact the suite makes. Here: what the list names, that a copy and a
derived fact add nothing to it, that a 0.2.x fact still loads, and that the
list survives a write and a read-back through every sink, against a real
Neo4j in `test_the_list_survives_a_live_neo4j` when `NEO4J_URI` is set.
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    KnowledgeGraph,
    Ontology,
    SourceTier,
    Span,
    Support,
)
from openodke.corroborate import (
    NEAR_DUPLICATES,
    EvidenceScorer,
    SignatureCorroborator,
    derived_from,
    partners,
    support_of,
)
from openodke.sinks import JsonlSink
from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink
from openodke.sinks.neo4j import Neo4jSink, plan, provenance_of, support_from

WHEN = datetime(2026, 9, 1, tzinfo=UTC)
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
        doc_id=doc,
        span=Span(doc_id=doc, start=0, end=4),
        uri=uri,
        tier=tier,
        retrieved_at=at,
    )


def _employed(*evidence: Evidence) -> Fact:
    return Fact(subject=ADA, predicate="employer", object_entity=ACME, evidence=evidence)


def _two_documents() -> Fact:
    """Two documents state Ada works at Acme: one on a host, one a bare document."""
    (merged,) = SignatureCorroborator().corroborate(
        [
            _employed(_evidence("d1", uri="https://www.acme.example/about", at=WHEN)),
            _employed(_evidence("d2", tier=SourceTier.CURATED, at=WHEN + timedelta(days=1))),
        ]
    )
    return merged


# --------------------------------------------------------------------------- #
# What the list names
# --------------------------------------------------------------------------- #


def test_two_documents_stating_one_fact_give_a_list_naming_both() -> None:
    merged = _two_documents()
    assert merged.support == 2
    assert merged.supported_by == (
        Support(
            source="acme.example", doc_ids=("d1",), tier=SourceTier.COMMUNITY, retrieved_at=WHEN
        ),
        Support(
            source="doc:d2",
            doc_ids=("d2",),
            tier=SourceTier.CURATED,
            retrieved_at=WHEN + timedelta(days=1),
        ),
    )


def test_one_host_is_one_entry_naming_each_page_its_best_tier_and_newest_clock() -> None:
    later = WHEN + timedelta(hours=2)
    pages = [
        _evidence("p2", uri="https://news.example/b", tier=SourceTier.UNVERIFIED, at=later),
        _evidence("p1", uri="https://news.example/a", tier=SourceTier.AUTHORITATIVE, at=WHEN),
    ]
    (merged,) = SignatureCorroborator().corroborate(_employed(e) for e in pages)
    (entry,) = merged.supported_by
    assert (merged.support, len(merged.evidence)) == (1, 2)
    assert entry == Support(
        source="news.example",
        doc_ids=("p1", "p2"),
        tier=SourceTier.AUTHORITATIVE,
        retrieved_at=later,
    )
    # `support_of` is the same count on evidence alone, chunks of one document included.
    chunked = [_evidence("d1"), _evidence("d1").model_copy(update={"span": None})]
    assert [s.doc_ids for s in support_of(chunked)] == [("d1",)]


def test_a_near_duplicate_group_is_one_entry_under_its_least_key() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    docs = [
        Document(id="orig", text=text, uri="https://a.example/x"),
        Document(id="copy", text=text, uri="https://b.example/y"),
        Document(id="other", text="Ada Lovelace works at Acme, says the register."),
    ]
    facts = [
        _employed(_evidence("orig", uri="https://a.example/x")),
        _employed(_evidence("copy", uri="https://b.example/y")),
        _employed(_evidence("other")),
    ]
    (merged,) = SignatureCorroborator(documents=docs).corroborate(facts)
    assert merged.qualifiers[NEAR_DUPLICATES] == (("copy", "orig"),)
    assert [(s.source, s.doc_ids) for s in merged.supported_by] == [
        ("a.example", ("copy", "orig")),
        ("doc:other", ("other",)),
    ]
    assert merged.support == 2


def test_a_derived_fact_shares_its_parents_list_and_adds_nothing_to_it() -> None:
    stated = [
        _employed(_evidence("d1", uri="https://acme.example/")),
        _employed(_evidence("d2")),
    ]
    merged = SignatureCorroborator(ONTOLOGY).corroborate([*stated, *partners(stated, ONTOLOGY)])
    by_predicate = {fact.predicate: fact for fact in merged}
    parent, partner = by_predicate["employer"], by_predicate["employs"]
    assert derived_from(partner) == parent.signature
    assert partner.supported_by == parent.supported_by
    assert [s.source for s in parent.supported_by] == ["acme.example", "doc:d2"]


def test_a_claim_with_a_source_it_cannot_name_keeps_its_count_and_names_none() -> None:
    bare = Fact(subject=ADA, predicate="employer", object_entity=ACME, support=2)
    (merged,) = SignatureCorroborator().corroborate([bare, _employed(_evidence("d1"))])
    assert (merged.support, merged.supported_by) == (3, ())
    # Alone, a fact with evidence is a list of one.
    (single,) = SignatureCorroborator().corroborate([_employed(_evidence("d1"))])
    assert [s.source for s in single.supported_by] == ["doc:d1"]


def test_a_fact_serialised_by_0_2_still_loads_with_its_count() -> None:
    old = _employed(_evidence("d1")).model_dump(mode="json")
    del old["supported_by"]
    old["support"] = 3
    loaded = Fact.model_validate(old)
    assert (loaded.support, loaded.supported_by) == (3, ())


def test_the_scorer_keeps_the_count_of_a_named_list() -> None:
    merged = _two_documents()
    # A scorer counting every document as a source would say three; the list says two.
    three = merged.model_copy(update={"evidence": (*merged.evidence, _evidence("d3"))})
    scored = EvidenceScorer(source=lambda e: e.doc_id).score(three)
    assert scored.support == len(scored.supported_by) == 2
    assert scored.qualifiers["odke.score"]["support"] == 2


# --------------------------------------------------------------------------- #
# Through every sink and back
# --------------------------------------------------------------------------- #


def _graph() -> KnowledgeGraph:
    fact = _two_documents()
    value = SignatureCorroborator().corroborate(
        [
            Fact(subject=ACME, predicate="hq", object_value="Leeds", evidence=(_evidence(d),))
            for d in ("d3", "d4")
        ]
    )
    return KnowledgeGraph(entities=(ADA, ACME), facts=(fact, *value))


def _supports(kg: KnowledgeGraph) -> dict[str, tuple[Support, ...]]:
    return {fact.predicate: fact.supported_by for fact in kg.facts}


def test_the_list_survives_jsonl(tmp_path: Path) -> None:
    kg = _graph()
    JsonlSink(tmp_path).write(kg)
    lines = (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    back = [Fact.model_validate_json(line) for line in lines]
    assert {f.predicate: f.supported_by for f in back} == _supports(kg)


def test_the_list_survives_the_neo4j_plan_as_parallel_lists() -> None:
    kg = _graph()
    rows = {
        statement.names[1]: statement.rows[0]["props"]
        for statement in plan(kg)
        if statement.kind in ("edge", "claim")
    }
    employer = rows["employer"]
    assert employer["support_sources"] == ["acme.example", "doc:d2"]
    assert employer["support_doc_ids"] == ["d1", "d2"]
    assert employer["support_doc_sources"] == ["acme.example", "doc:d2"]
    assert employer["support_tiers"] == ["community", "curated"]
    assert employer["support"] == len(employer["support_sources"])
    assert {name: support_from(props) for name, props in rows.items()} == _supports(kg)
    # A relationship written before support lists existed reads back as none.
    assert support_from({"support": 2, "evidence_doc_ids": ["d1"]}) == ()


def test_the_list_survives_networkx() -> None:
    pytest.importorskip("networkx")
    from openodke.sinks.networkx import NetworkXSink

    kg = _graph()
    sink = NetworkXSink()
    sink.write(kg)
    back = {
        attrs["predicate"]: support_from(attrs)
        for _, _, attrs in sink.graph.edges(data=True)
        if attrs["kind"] == "fact"
    }
    assert back == _supports(kg)


def test_the_list_survives_the_neo4j_admin_csv(tmp_path: Path) -> None:
    kg = _graph()
    Neo4jAdminCsvSink(tmp_path).write(kg)
    back: dict[str, tuple[Support, ...]] = {}
    for path in tmp_path.glob("*edges*.csv"):
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                cells = {name.split(":", 1)[0]: value for name, value in row.items()}
                arrays = {
                    name: value.split(";") if value else []
                    for name, value in cells.items()
                    if name.startswith("support_")
                }
                back[row[":TYPE"]] = support_from(arrays)
    assert back == _supports(kg)


def test_the_list_survives_the_cypher_script(tmp_path: Path) -> None:
    kg = _graph()
    script = CypherFileSink(tmp_path / "g.cypher").script(kg)
    for fact in kg.facts:
        props = provenance_of(fact, kg.created_at)
        assert repr(props["support_sources"]).replace('"', "'") in script
    # Replayed against a server in `test_the_list_survives_a_live_neo4j`.


def test_the_list_survives_rdf(tmp_path: Path) -> None:
    pytest.importorskip("rdflib")
    from openodke.sinks.rdf import VOCAB, RdfSink

    kg = _graph()
    sink = RdfSink(tmp_path / "g.ttl")
    sink.write(kg)
    from rdflib import Graph

    graph = Graph().parse(tmp_path / "g.ttl")
    query = f"""
    PREFIX odke: <{VOCAB}>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?predicate ?support ?source ?doc ?tier ?at WHERE {{
      ?fact a odke:Fact ; rdf:predicate ?predicate ; odke:supported_by ?support .
      ?support a odke:Support ; odke:source ?source ; odke:doc_id ?doc ;
               odke:tier ?tier ; odke:retrieved_at ?at .
    }}"""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for row in graph.query(query):
        predicate = str(row.predicate).rsplit("/", 1)[-1]
        entry = found.setdefault(predicate, {}).setdefault(
            str(row.support),
            {"source": str(row.source), "doc_ids": set(), "tier": str(row.tier)},
        )
        entry["doc_ids"].add(str(row.doc))
        entry["retrieved_at"] = row.at.toPython()
    back = {
        predicate: tuple(
            sorted(
                (
                    Support(
                        source=e["source"],
                        doc_ids=tuple(sorted(e["doc_ids"])),
                        tier=SourceTier(e["tier"]),
                        retrieved_at=e["retrieved_at"],
                    )
                    for e in entries.values()
                ),
                key=lambda s: s.source,
            )
        )
        for predicate, entries in found.items()
    }
    assert back == _supports(kg)


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_the_list_survives_a_live_neo4j(tmp_path: Path) -> None:
    """Written by the driver and replayed from a script, the list reads back as it went in.

    Types are suffixed so a shared server is left as found.
    """
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company = f"Person_{suffix}", f"Company_{suffix}"

    def mine(kg: KnowledgeGraph) -> KnowledgeGraph:
        retyped = {"Person": person, "Company": company}

        def entity(e: Entity) -> Entity:
            return e.model_copy(update={"type": retyped[e.type], "key": f"{e.key}:{suffix}"})

        return kg.model_copy(
            update={
                "entities": tuple(entity(e) for e in kg.entities),
                "facts": tuple(
                    f.model_copy(
                        update={
                            "subject": entity(f.subject),
                            "object_entity": entity(f.object_entity) if f.object_entity else None,
                        }
                    )
                    for f in kg.facts
                ),
            }
        )

    kg = mine(_graph())
    read = (
        f"MATCH (s)-[r]->() WHERE s:`{person}` OR s:`{company}` "
        "RETURN type(r) AS predicate, properties(r) AS props"
    )
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            sink.write(kg)
            back = {
                r["predicate"]: support_from(r["props"]) for r in driver.execute_query(read).records
            }
            assert back == _supports(kg)

            driver.execute_query(f"MATCH (n) WHERE n:`{person}` OR n:`{company}` DETACH DELETE n")
            path = tmp_path / "g.cypher"
            CypherFileSink(path).write(kg)
            for statement in path.read_text(encoding="utf-8").split(";\n"):
                body = "\n".join(
                    line for line in statement.splitlines() if not line.startswith("//")
                ).strip()
                if body:
                    driver.execute_query(body)
            back = {
                r["predicate"]: support_from(r["props"]) for r in driver.execute_query(read).records
            }
            assert back == _supports(kg)
        finally:
            driver.execute_query(
                f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
                f"(n:Claim AND n.subject_type IN ['{person}', '{company}']) DETACH DELETE n"
            )
