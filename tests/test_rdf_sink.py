"""The RDF sink: provenance per fact that SPARQL can reach, and a denial that stays one."""

from __future__ import annotations

import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rdflib")

from rdflib import Graph, Literal, URIRef  # noqa: E402
from rdflib.compare import isomorphic  # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD  # noqa: E402

from odke import (  # noqa: E402
    Entity,
    EntityLink,
    EntityType,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Polarity,
    Predicate,
    Sink,
)
from odke.eval.sinks import assert_idempotent  # noqa: E402
from odke.sinks.neo4j import signature_of  # noqa: E402
from odke.sinks.rdf import DEFAULT_BASE, EXTRA_HINT, VOCAB, RdfSink  # noqa: E402
from test_neo4j_sink import _graph  # noqa: E402

FORMATS = ("turtle", "nt", "json-ld")
ENT = f"{DEFAULT_BASE}entity/"
ONT = f"{DEFAULT_BASE}schema/"

PREFIXES = f"""
PREFIX rdf: <{RDF}>
PREFIX odke: <{VOCAB}>
PREFIX ont: <{ONT}>
PREFIX ent: <{ENT}>
"""


def _loaded(tmp_path: Path, fmt: str, graph: KnowledgeGraph | None = None, **kw: Any) -> Graph:
    """Written to disk and parsed back: what a triple store would load."""
    path = tmp_path / f"graph.{fmt}"
    RdfSink(path, format=fmt, **kw).write(_graph() if graph is None else graph)
    return Graph().parse(path, format=fmt)


def _select(g: Graph, query: str) -> set[tuple[Any, ...]]:
    rows = g.query(PREFIXES + query)
    return {tuple(v.toPython() if v is not None else None for v in row) for row in rows}


def _ask(g: Graph, query: str) -> bool:
    return bool(g.query(PREFIXES + query).askAnswer)


# --------------------------------------------------------------------------- #
# Polarity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_denial_is_never_an_asserted_triple(tmp_path: Path, fmt: str) -> None:
    g = _loaded(tmp_path, fmt)
    assert not _ask(g, 'ASK { ent:c%3Aacme ont:sells "customer_data" }')
    assert not _ask(g, "ASK { ent:c%3Aacme ont:sells ?anything }")
    assert _select(
        g,
        """SELECT ?polarity WHERE {
             ?st rdf:subject ent:c%3Aacme ; rdf:predicate ont:sells ;
                 rdf:object "customer_data" ; odke:polarity ?polarity }""",
    ) == {("denied",)}


@pytest.mark.parametrize("polarity", [Polarity.DENIED, Polarity.PARTIAL])
def test_only_an_asserted_edge_or_value_is_written_as_a_plain_triple(
    tmp_path: Path, polarity: Polarity
) -> None:
    ada, acme = Entity(key="p:ada", type="Person"), Entity(key="c:acme", type="Company")
    edge = Fact(subject=ada, predicate="employer", object_entity=acme)
    value = Fact(subject=ada, predicate="name", object_value="Ada")
    graph = KnowledgeGraph(
        facts=(
            edge.model_copy(update={"polarity": polarity}),
            value.model_copy(update={"polarity": polarity}),
        )
    )
    g = _loaded(tmp_path, "turtle", graph)
    assert not _ask(g, "ASK { ent:p%3Aada ont:employer ent:c%3Aacme }")
    assert not _ask(g, 'ASK { ent:p%3Aada ont:name "Ada" }')
    assert _select(g, "SELECT ?p WHERE { ?st a odke:Fact ; odke:polarity ?p }") == {
        (polarity.value,)
    }

    asserted = _loaded(tmp_path, "turtle", KnowledgeGraph(facts=(edge, value)))
    assert _ask(asserted, "ASK { ent:p%3Aada ont:employer ent:c%3Aacme }")
    assert _ask(asserted, 'ASK { ent:p%3Aada ont:name "Ada" }')


def test_an_assertion_and_its_denial_are_two_statement_nodes(tmp_path: Path) -> None:
    acme = Entity(key="c:acme", type="Company")
    says = Fact(subject=acme, predicate="sells", object_value="data")
    denies = says.model_copy(update={"polarity": Polarity.DENIED})
    g = _loaded(tmp_path, "nt", KnowledgeGraph(facts=(says, denies)))
    assert _select(g, "SELECT ?st ?p WHERE { ?st odke:polarity ?p }") == {
        (f"{DEFAULT_BASE}fact/{signature_of(says)}", "asserted"),
        (f"{DEFAULT_BASE}fact/{signature_of(denies)}", "denied"),
    }


def test_a_scoped_value_is_not_a_plain_triple_and_each_scope_is_its_own_node(
    tmp_path: Path,
) -> None:
    """Uptime without its percentile says something no source said."""
    g = _loaded(tmp_path, "turtle")
    assert not _ask(g, "ASK { ent:c%3Aacme ont:uptime ?v }")
    rows = _select(
        g,
        f"""SELECT ?percentile ?key WHERE {{
             ?st rdf:predicate ont:uptime ; <{ONT}qualifier/percentile> ?percentile ;
                 odke:identity_keys ?key }}""",
    )
    assert rows == {("p50", "percentile"), ("p95", "percentile")}


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", FORMATS)
def test_provenance_is_queryable_down_to_a_document_and_a_character_range(
    tmp_path: Path, fmt: str
) -> None:
    g = _loaded(tmp_path, fmt)
    assert _select(
        g,
        """SELECT ?doc ?uri ?start ?end ?tier WHERE {
             ?st rdf:subject ent:p%3Aada ; rdf:predicate ont:employer ;
                 rdf:object ent:c%3Aacme ; odke:evidence ?ev .
             ?ev odke:doc_id ?doc ; odke:tier ?tier .
             OPTIONAL { ?ev odke:uri ?uri }
             OPTIONAL { ?ev odke:start ?start ; odke:end ?end } }""",
    ) == {
        ("d1", "https://example.com/d1", 0, 24, "authoritative"),
        ("d2", None, None, None, "unverified"),
    }
    ((confidence, support, verdict, extractor, valid_from, retrieved),) = _select(
        g,
        """SELECT ?c ?s ?v ?x ?from ?at WHERE {
             ?st rdf:predicate ont:employer ; odke:confidence ?c ; odke:support ?s ;
                 odke:verdict ?v ; odke:extractor ?x ; odke:valid_from ?from ;
                 odke:retrieved_at ?at }""",
    )
    assert (confidence, support, verdict, extractor) == (0.9, 2, "supported", "llm")
    assert valid_from == datetime(2019, 1, 1, tzinfo=UTC)
    assert retrieved == datetime(2026, 9, 1, tzinfo=UTC)


def test_forgetting_a_source_is_a_query(tmp_path: Path) -> None:
    g = _loaded(tmp_path, "turtle")
    graph = _graph()
    assert _select(g, 'SELECT ?st WHERE { ?st odke:evidence/odke:doc_id "d3" }') == {
        (f"{DEFAULT_BASE}fact/{signature_of(graph.facts[1])}",),
        (f"{DEFAULT_BASE}fact/{signature_of(graph.facts[2])}",),
    }


def test_every_fact_is_a_statement_with_its_receipts(tmp_path: Path) -> None:
    g = _loaded(tmp_path, "nt")
    graph = _graph()
    statements = set(g.subjects(RDF.type, URIRef(f"{VOCAB}Fact")))
    assert statements == {URIRef(f"{DEFAULT_BASE}fact/{signature_of(f)}") for f in graph.facts}
    for node in statements:
        assert (node, RDF.type, RDF.Statement) in g
        for name in ("fact_id", "signature", "polarity", "confidence", "extracted_at"):
            assert g.value(node, URIRef(f"{VOCAB}{name}")) is not None, name
        assert list(g.objects(node, URIRef(f"{VOCAB}evidence"))), "a fact with no document"
    employer = URIRef(f"{DEFAULT_BASE}fact/{signature_of(graph.facts[0])}")
    assert g.value(employer, URIRef(f"{ONT}qualifier/start_time")) == Literal("2019")
    # A qualifier named like a receipt does not overwrite it: different namespaces.
    assert g.value(employer, URIRef(f"{ONT}qualifier/signature")) == Literal("clash")
    assert g.value(employer, URIRef(f"{VOCAB}signature")) == Literal(signature_of(graph.facts[0]))
    assert g.value(employer, URIRef(f"{VOCAB}confidence")).datatype == XSD.double


def test_a_rerun_addresses_the_same_nodes_though_ids_and_clocks_change(tmp_path: Path) -> None:
    first = RdfSink(tmp_path / "a.ttl").graph(_graph())
    second = RdfSink(tmp_path / "b.ttl").graph(_graph())
    assert set(first.subjects()) == set(second.subjects())
    assert len(first) == len(second)


def test_two_facts_with_one_signature_are_one_node_carrying_the_later(tmp_path: Path) -> None:
    ada = Entity(key="p:ada", type="Person")
    once = Fact(id="first", subject=ada, predicate="name", object_value="Ada", confidence=0.2)
    again = once.model_copy(update={"id": "second", "confidence": 0.7})
    g = RdfSink(tmp_path / "g.ttl").graph(KnowledgeGraph(facts=(once, again)))
    node = URIRef(f"{DEFAULT_BASE}fact/{signature_of(once)}")
    assert list(g.objects(node, URIRef(f"{VOCAB}fact_id"))) == [Literal("second")]
    assert list(g.objects(node, URIRef(f"{VOCAB}confidence"))) == [Literal(0.7)]


# --------------------------------------------------------------------------- #
# Entities and links
# --------------------------------------------------------------------------- #


def test_entities_are_iris_under_the_base_typed_by_the_ontology(tmp_path: Path) -> None:
    g = _loaded(tmp_path, "turtle")
    ada = URIRef(f"{ENT}p%3Aada")
    assert (ada, RDF.type, URIRef(f"{ONT}Person")) in g
    assert (ada, RDF.type, URIRef(f"{VOCAB}Entity")) in g
    assert g.value(ada, RDFS.label) == Literal("Ada Lovelace")
    assert set(g.objects(ada, SKOS.altLabel)) == {Literal("Ada"), Literal("Countess of Lovelace")}
    assert g.value(ada, URIRef(f"{VOCAB}external_id")) == Literal("Q7259")
    assert g.value(ada, URIRef(f"{ONT}attribute/born")) == Literal(1815)
    assert g.value(ada, URIRef(f"{ONT}attribute/tags")) == Literal('{"a": 1}')
    assert (ada, URIRef(f"{ONT}employer"), URIRef(f"{ENT}c%3Aacme")) in g
    assert (ada, URIRef(f"{ONT}name"), Literal("Ada")) in g


def test_base_and_schema_are_configurable_and_odd_names_stay_valid_iris(tmp_path: Path) -> None:
    thing = Entity(key="a b/c#d", type="Odd Type")
    fact = Fact(subject=thing, predicate="has part", object_entity=thing)
    base, schema = "https://kg.example.com/", "https://schema.example.com/v1#"
    g = _loaded(tmp_path, "nt", KnowledgeGraph(facts=(fact,)), base=base, schema=schema)
    node = URIRef(f"{base}entity/a%20b%2Fc%23d")
    assert (node, URIRef(f"{schema}has%20part"), node) in g
    assert (node, RDF.type, URIRef(f"{schema}Odd%20Type")) in g


def test_links_are_owl_same_as_and_odke_properties_carrying_score_and_reason(
    tmp_path: Path,
) -> None:
    g = _loaded(tmp_path, "turtle")
    acme, gmbh, ada = (URIRef(f"{ENT}{k}") for k in ("c%3Aacme", "c%3Aacme-gmbh", "p%3Aada"))
    assert (ada, OWL.sameAs, ada) in g
    assert (acme, URIRef(f"{VOCAB}similar_to"), gmbh) in g
    assert (acme, URIRef(f"{VOCAB}different_from"), gmbh) in g
    assert _select(
        g,
        """SELECT ?kind ?score ?reason WHERE {
             ?link a odke:Link ; odke:kind ?kind ; odke:score ?score ;
                   rdf:subject ent:c%3Aacme ; rdf:object ent:c%3Aacme-gmbh .
             OPTIONAL { ?link odke:reason ?reason } }""",
    ) == {
        ("similar", 0.9, None),
        ("different", 0.9, "external_id mismatch: DE-114322 vs GB-889401"),
    }


def test_a_link_written_twice_is_one_node(tmp_path: Path) -> None:
    links = tuple(
        EntityLink(source_key="a", target_key="b", kind=LinkKind.SIMILAR, score=s)
        for s in (0.1, 0.4)
    )
    g = RdfSink(tmp_path / "g.ttl").graph(KnowledgeGraph(links=links))
    (node,) = g.subjects(RDF.type, URIRef(f"{VOCAB}Link"))
    assert list(g.objects(node, URIRef(f"{VOCAB}score"))) == [Literal(0.4)]


# --------------------------------------------------------------------------- #
# Formats, schema, idempotency, surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_loads_back_to_the_same_graph(tmp_path: Path, fmt: str) -> None:
    ontology = Ontology(
        types={"Person": EntityType(name="Person"), "Company": EntityType(name="Company")},
        predicates={
            "employer": Predicate(name="employer", domain=("Person", "Company"), range="Company")
        },
    )
    sink = RdfSink(tmp_path / "g", format=fmt, ontology=ontology)
    graph = _graph()
    assert isomorphic(sink.graph(graph), _loaded(tmp_path, fmt, graph, ontology=ontology))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("g.ttl", "turtle"),
        ("g.nt", "nt"),
        ("g.jsonld", "json-ld"),
        ("g.json", "json-ld"),
        ("g", "turtle"),
    ],
)
def test_the_format_follows_the_file_suffix(name: str, expected: str) -> None:
    assert RdfSink(name).format == expected
    assert RdfSink(name, format="N-Triples").format == "nt"


def test_an_unknown_format_is_refused() -> None:
    with pytest.raises(ValueError, match="format must be one of"):
        RdfSink("g.rdf", format="rdfxml")


def test_with_an_ontology_the_output_declares_its_schema(tmp_path: Path) -> None:
    ontology = Ontology(
        types={
            "Agent": EntityType(name="Agent", description="Anything that acts."),
            "Person": EntityType(name="Person", parents=("Agent",), aliases=("Human",)),
            "Company": EntityType(name="Company", keys=("legal_name",)),
        },
        predicates={
            "employer": Predicate(name="employer", domain=("Person",), range="Company"),
            "legal_name": Predicate(name="legal_name", domain=("Company",), label="Legal name"),
            "tag": Predicate(name="tag", domain=("Person", "Company"), cardinality="multi"),
            "founded": Predicate(name="founded", domain=("Company",), range="date"),
        },
    )
    g = _loaded(tmp_path, "turtle", KnowledgeGraph(), ontology=ontology)
    term = lambda name: URIRef(f"{ONT}{name}")  # noqa: E731
    assert (term("Person"), RDFS.subClassOf, term("Agent")) in g
    assert g.value(term("Agent"), RDFS.comment) == Literal("Anything that acts.")
    assert (term("employer"), RDF.type, OWL.ObjectProperty) in g
    assert (term("employer"), RDF.type, OWL.FunctionalProperty) in g
    assert (term("tag"), RDF.type, OWL.DatatypeProperty) in g
    assert (term("tag"), RDF.type, OWL.FunctionalProperty) not in g
    assert g.value(term("founded"), RDFS.range) == XSD.date
    assert g.value(term("employer"), RDFS.range) == term("Company")
    assert _select(
        g,
        f"""PREFIX owl: <{OWL}> PREFIX rdfs: <{RDFS}>
            SELECT ?member WHERE {{
              ont:tag rdfs:domain/owl:unionOf/rdf:rest*/rdf:first ?member }}""",
    ) == {(f"{ONT}Person",), (f"{ONT}Company",)}


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_rdf_sink_is_idempotent(tmp_path: Path, fmt: str) -> None:
    path = tmp_path / f"g.{fmt}"
    sink = RdfSink(path, format=fmt)
    assert isinstance(sink, Sink)

    def counts() -> dict[str, int]:
        g = Graph().parse(path, format=fmt)
        return {
            "triples": len(g),
            "facts": len(set(g.subjects(RDF.type, URIRef(f"{VOCAB}Fact")))),
            "links": len(set(g.subjects(RDF.type, URIRef(f"{VOCAB}Link")))),
        }

    assert_idempotent(sink, _graph(), counts)


def test_a_missing_rdflib_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module imports on the base install; only building a sink needs the extra."""
    monkeypatch.setitem(sys.modules, "rdflib", None)
    monkeypatch.delitem(sys.modules, "odke.sinks.rdf")
    module = importlib.import_module("odke.sinks.rdf")
    with pytest.raises(ImportError, match=r"odke\[rdf\]"):
        module.RdfSink("g.ttl")
    assert "odke[rdf]" in EXTRA_HINT
