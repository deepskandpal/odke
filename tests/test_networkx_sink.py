"""The NetworkX sink: the Neo4j shape in memory, and every attribute back out of it."""

from __future__ import annotations

import importlib
import sys
from typing import Any

import pytest

nx = pytest.importorskip("networkx")

from openodke import (  # noqa: E402
    Entity,
    EntityLink,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    LinkKind,
    Polarity,
    Resolution,
    Sink,
    SourceTier,
    Span,
)
from openodke.eval.sinks import assert_idempotent  # noqa: E402
from openodke.sinks.neo4j import signature_of  # noqa: E402
from openodke.sinks.networkx import EXTRA_HINT, NetworkXSink, claim_node  # noqa: E402
from test_neo4j_sink import PROVENANCE, _graph  # noqa: E402

_ENTITY_FIELDS = (
    "kind",
    "type",
    "key",
    "label",
    "aliases",
    "external_id",
    "resolution_method",
    "resolution_score",
    "resolution_linker",
)
_PROVENANCE_NAMES = PROVENANCE | {"identity_keys", "evidence_retrieved_at"}


def _written(graph: KnowledgeGraph | None = None) -> Any:
    sink = NetworkXSink()
    sink.write(_graph() if graph is None else graph)
    return sink.graph


# --------------------------------------------------------------------------- #
# Reading entities and facts back out
# --------------------------------------------------------------------------- #


def _entity_back(g: Any, key: str, projected: set[str]) -> Entity:
    attrs = g.nodes[key]
    method = attrs["resolution_method"]
    return Entity(
        key=attrs["key"],
        type=attrs["type"],
        label=attrs["label"],
        aliases=tuple(attrs["aliases"]),
        external_id=attrs["external_id"],
        resolution=Resolution(
            method=method, score=attrs["resolution_score"], linker=attrs["resolution_linker"]
        )
        if method
        else None,
        attributes={
            k: v for k, v in attrs.items() if k not in _ENTITY_FIELDS and k not in projected
        },
    )


def _fact_back(g: Any, u: str, v: str, attrs: dict[str, Any], projected: set[str]) -> Fact:
    obj = g.nodes[v]
    qualifiers = {
        k.removeprefix("qualifier_") if k.removeprefix("qualifier_") in _PROVENANCE_NAMES else k: x
        for k, x in attrs.items()
        if k not in _PROVENANCE_NAMES and k not in {"kind", "predicate"}
    }
    evidence = tuple(
        Evidence(
            doc_id=doc,
            span=Span(doc_id=doc, start=start, end=end) if start >= 0 else None,
            uri=uri or None,
            tier=SourceTier(tier),
            retrieved_at=at,
        )
        for doc, uri, start, end, tier, at in zip(
            attrs["evidence_doc_ids"],
            attrs["evidence_uris"],
            attrs["evidence_starts"],
            attrs["evidence_ends"],
            attrs["evidence_tiers"],
            attrs["evidence_retrieved_at"],
            strict=True,
        )
    )
    return Fact(
        id=attrs["fact_id"],
        subject=_entity_back(g, u, projected),
        predicate=attrs["predicate"],
        object_entity=_entity_back(g, v, projected) if obj["kind"] == "entity" else None,
        object_value=obj["value"] if obj["kind"] == "claim" else None,
        polarity=Polarity(attrs["polarity"]),
        qualifiers=qualifiers,
        identity_keys=tuple(attrs["identity_keys"]),
        valid_from=attrs["valid_from"],
        valid_to=attrs["valid_to"],
        evidence=evidence,
        extractor=attrs["extractor"],
        confidence=attrs["confidence"],
        verdict=GroundingVerdict(attrs["verdict"]),
        support=attrs["support"],
    )


def test_entity_and_fact_attributes_round_trip_exactly() -> None:
    graph = _graph()
    g = _written(graph)
    projected = {"name"}
    for entity in graph.entities:
        assert _entity_back(g, entity.key, projected) == entity
    facts = [
        _fact_back(g, u, v, attrs, projected)
        for u, v, attrs in g.edges(data=True)
        if attrs["kind"] == "fact"
    ]
    assert sorted(facts, key=lambda f: f.id) == sorted(graph.facts, key=lambda f: f.id)


# --------------------------------------------------------------------------- #
# The shape
# --------------------------------------------------------------------------- #


def test_entities_are_nodes_keyed_on_their_key_with_values_kept_as_they_are() -> None:
    g = _written()
    ada = g.nodes["p:ada"]
    assert (ada["kind"], ada["type"], ada["label"], ada["external_id"]) == (
        "entity",
        "Person",
        "Ada Lovelace",
        "Q7259",
    )
    assert ada["aliases"] == ["Ada", "Countess of Lovelace"]
    # Not flattened to JSON text as Neo4j must: nothing here needs it.
    assert ada["tags"] == {"a": 1}
    assert {n for n, a in g.nodes(data=True) if a["kind"] == "entity"} == {
        "p:ada",
        "c:acme",
        "c:acme-gmbh",
    }


def test_an_edge_fact_is_keyed_on_its_signature_and_carries_provenance() -> None:
    graph = _graph()
    g = _written(graph)
    signature = signature_of(graph.facts[0])
    edge = g.edges["p:ada", "c:acme", signature]
    assert set(edge) >= PROVENANCE
    assert (edge["kind"], edge["predicate"], edge["confidence"], edge["support"]) == (
        "fact",
        "employer",
        0.9,
        2,
    )
    assert edge["evidence_doc_ids"] == ["d1", "d2"]
    assert edge["start_time"] == "2019"
    assert edge["qualifier_signature"] == "clash"
    assert edge["signature"] == signature


def test_a_literal_fact_is_an_edge_to_a_claim_node_as_in_neo4j() -> None:
    graph = _graph()
    g = _written(graph)
    claims = {n: a for n, a in g.nodes(data=True) if a["kind"] == "claim"}
    assert sorted((a["predicate"], a["value"]) for a in claims.values()) == [
        ("name", "Ada"),
        ("sells", "customer_data"),
        ("uptime", "99.9%"),
        ("uptime", "99.9%"),
    ]
    p50 = graph.facts[1]
    node = claim_node(signature_of(p50))
    assert node in claims
    (edge,) = g.get_edge_data("c:acme", node).values()
    assert set(edge) >= PROVENANCE
    assert edge["percentile"] == "p50"


def test_a_denial_keeps_its_polarity_and_is_never_projected() -> None:
    graph = _graph()
    g = _written(graph)
    (edge,) = g.get_edge_data("c:acme", claim_node(signature_of(graph.facts[4]))).values()
    assert edge["polarity"] == "denied"
    assert "sells" not in g.nodes["c:acme"]
    assert "uptime" not in g.nodes["c:acme"]
    assert g.nodes["p:ada"]["name"] == "Ada"


def test_links_are_edges_keyed_on_kind_and_only_between_nodes_the_graph_holds() -> None:
    g = _written()
    different = g.edges["c:acme", "c:acme-gmbh", "DIFFERENT"]
    assert (different["kind"], different["score"], different["reason"]) == (
        "link",
        0.9,
        "external_id mismatch: DE-114322 vs GB-889401",
    )
    assert g.has_edge("c:acme", "c:acme-gmbh", "SIMILAR")
    assert g.has_edge("p:ada", "p:ada", "SAME_AS")
    ghost = EntityLink(source_key="p:ada", target_key="nobody", kind=LinkKind.SAME_AS)
    g = _written(_graph().model_copy(update={"links": (ghost,)}))
    assert "nobody" not in g


def test_names_the_sink_owns_are_never_overwritten() -> None:
    thing = Entity(key="k", type="T", attributes={"type": "x", "key": "y", "free": 1})
    fact = Fact(
        subject=thing,
        predicate="kind",
        object_value="v",
        qualifiers={"predicate": "q", "support": 9},
    )
    g = _written(KnowledgeGraph(facts=(fact,)))
    node = g.nodes["k"]
    assert (node["type"], node["key"], node["attribute_type"], node["attribute_key"]) == (
        "T",
        "k",
        "x",
        "y",
    )
    assert node["property_kind"] == "v" and node["kind"] == "entity"
    (edge,) = g.get_edge_data("k", claim_node(signature_of(fact))).values()
    assert (edge["predicate"], edge["qualifier_predicate"]) == ("kind", "q")
    assert (edge["support"], edge["qualifier_support"]) == (1, 9)


# --------------------------------------------------------------------------- #
# Idempotency and surface
# --------------------------------------------------------------------------- #


def test_the_networkx_sink_is_idempotent() -> None:
    sink = NetworkXSink()
    assert isinstance(sink, Sink)

    def counts() -> dict[str, int]:
        g = sink.graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "claims": sum(1 for _, a in g.nodes(data=True) if a["kind"] == "claim"),
        }

    assert_idempotent(sink, _graph(), counts)
    before = counts()
    # A second pipeline run: new fact ids and clocks, the same keys.
    sink.write(_graph())
    assert counts() == before


def test_the_sink_fills_the_graph_it_is_given_and_to_graph_matches_write() -> None:
    existing = nx.MultiDiGraph()
    existing.add_node("elsewhere", kind="entity")
    sink = NetworkXSink(existing)
    graph = _graph()
    sink.write(graph)
    assert sink.graph is existing and "elsewhere" in existing
    fresh = NetworkXSink().to_graph(graph)
    existing.remove_node("elsewhere")
    assert nx.utils.graphs_equal(fresh, existing)


def test_only_a_multidigraph_can_hold_the_facts() -> None:
    with pytest.raises(TypeError, match="MultiDiGraph"):
        NetworkXSink(nx.DiGraph())


def test_the_graph_can_be_laid_out_for_drawing() -> None:
    pytest.importorskip("numpy")
    positions = nx.spring_layout(_written(), seed=1)
    assert len(positions) == 3 + 4


def test_a_missing_networkx_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The module imports on the base install; only building a sink needs the extra."""
    monkeypatch.setitem(sys.modules, "networkx", None)
    monkeypatch.delitem(sys.modules, "openodke.sinks.networkx")
    module = importlib.import_module("openodke.sinks.networkx")
    with pytest.raises(ImportError, match=r"openodke\[networkx\]"):
        module.NetworkXSink()
    assert "openodke[networkx]" in EXTRA_HINT
