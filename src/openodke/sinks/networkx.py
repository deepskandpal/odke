"""The NetworkX sink: a `MultiDiGraph` in memory, to look at a graph without a database.

networkx is imported when a sink is built, never at the top, so this module
imports on the base install (DECISIONS #1) and a missing `[networkx]` extra
fails with the command that fixes it.
"""

from __future__ import annotations

from typing import Any

from openodke.ontology import Ontology
from openodke.sinks.neo4j import (
    entities_of,
    link_row,
    projections,
    provenance_of,
    qualifier_property,
)
from openodke.types import KnowledgeGraph

EXTRA_HINT = "networkx is not installed; run: pip install 'openodke[networkx]'"

# Node attribute names the sink writes from `Entity` fields. An attribute or a
# projected predicate with one of these names is prefixed, never merged into.
_NODE_OWNED = frozenset(
    {
        "kind",
        "type",
        "key",
        "label",
        "aliases",
        "external_id",
        "resolution_method",
        "resolution_score",
        "resolution_linker",
    }
)
# Edge attribute names beside provenance's own; `qualifier_property` covers those.
_EDGE_OWNED = frozenset({"kind", "predicate"})


def _networkx() -> Any:
    try:
        import networkx  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    return networkx


def claim_node(signature: str) -> str:
    """The node id of a literal fact's claim."""
    return f"claim:{signature}"


class NetworkXSink:
    """Fills a `networkx.MultiDiGraph` with the same shape `Neo4jSink` writes.

    Kept deliberately the same as Neo4j, so a graph looked at here and one
    queried there are one graph:

    - An entity is a node whose id is its key, with `kind="entity"`, `type`,
      `key`, `label`, `aliases`, `external_id`, `resolution_*`, and its
      attributes as attributes of their own. Values are kept as they are —
      nothing here needs a nested attribute flattened to text.
    - An edge fact is an edge `subject → object` whose key is the fact's
      signature (DECISIONS #14, #15) and whose attributes are `kind="fact"`,
      `predicate`, and the provenance a Neo4j relationship carries —
      evidence ids, uris, spans and tiers, extractor, verdict, confidence,
      support, both clocks — with the qualifiers beside them.
    - A literal fact follows the `:Claim` decision: an edge of the same shape
      to a claim node `claim:<signature>` holding `kind="claim"` and the
      `value`. An attribute cannot carry provenance, two contested values or a
      denial; an edge can, so both kinds of fact are read the same way. The
      value is also projected onto the subject node under the predicate's
      name, by the same rule as Neo4j: only asserted, unscoped claims, the
      best-supported one for a single-valued predicate.
    - An `EntityLink` is an edge keyed `SAME_AS`, `SIMILAR` or `DIFFERENT`
      with `kind="link"`, `score` and `reason`, drawn only between nodes the
      graph holds, as Neo4j's `MATCH` on both ends would. Nodes are never
      merged (DECISIONS #16).

    Writing adds and updates, never removes: `add_node`/`add_edge` on a key
    that exists updates its attributes, as `MERGE … SET +=` does. So writing
    one graph twice, or rerunning the pipeline into the same `graph`, leaves
    the counts where they were. The node id is the key alone, as in RDF; a
    key two types share is one node.
    """

    def __init__(self, graph: Any = None, *, ontology: Ontology | None = None) -> None:
        nx = _networkx()
        self.graph = graph if graph is not None else nx.MultiDiGraph()
        if not (self.graph.is_directed() and self.graph.is_multigraph()):
            raise TypeError(
                "NetworkXSink fills a MultiDiGraph: facts are directed, and one pair of "
                "entities can hold many"
            )
        # Decides whether a projected attribute is one value or a list.
        self.ontology = ontology

    def to_graph(self, kg: KnowledgeGraph) -> Any:
        """A new `MultiDiGraph` holding `kg`, leaving `self.graph` untouched."""
        graph = _networkx().MultiDiGraph()
        _fill(graph, kg, self.ontology)
        return graph

    def write(self, kg: KnowledgeGraph) -> None:
        _fill(self.graph, kg, self.ontology)


def _node_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if name in _NODE_OWNED else name


def _edge_name(key: str) -> str:
    name = qualifier_property(key)
    return f"qualifier_{key}" if name in _EDGE_OWNED else name


def _fill(graph: Any, kg: KnowledgeGraph, ontology: Ontology | None) -> None:
    # Attributes go in through `.update`, not `**kwargs`: an attribute named
    # `key` would otherwise collide with `add_edge`'s own parameter.
    for (type_name, key), entity in sorted(entities_of(kg).items()):
        resolution = entity.resolution
        attrs = {_node_name("attribute", k): v for k, v in entity.attributes.items()}
        attrs.update(
            kind="entity",
            type=type_name,
            key=key,
            label=entity.label,
            aliases=list(entity.aliases),
            external_id=entity.external_id,
            resolution_method=resolution.method if resolution else None,
            resolution_score=resolution.score if resolution else None,
            resolution_linker=resolution.linker if resolution else None,
        )
        graph.add_node(key)
        graph.nodes[key].update(attrs)

    for fact in kg.facts:
        props = provenance_of(fact, kg.created_at)
        for name in fact.qualifiers:
            props.pop(qualifier_property(name), None)
        signature = props["signature"]
        attrs = {"kind": "fact", "predicate": fact.predicate, **props}
        attrs.update({_edge_name(k): v for k, v in fact.qualifiers.items()})
        if fact.object_entity is not None:
            target = fact.object_entity.key
        else:
            target = claim_node(signature)
            graph.add_node(target)
            graph.nodes[target].update(
                kind="claim",
                signature=signature,
                predicate=fact.predicate,
                subject_key=fact.subject.key,
                subject_type=fact.subject.type,
                value=fact.object_value,
            )
        graph.add_edge(fact.subject.key, target, key=signature)
        graph.edges[fact.subject.key, target, signature].update(attrs)

    for (_, predicate), rows in sorted(projections(kg, ontology).items()):
        name = _node_name("property", predicate)
        for row in rows:
            graph.nodes[row["subject_key"]][name] = row["value"]

    for link in kg.links:
        if not (graph.has_node(link.source_key) and graph.has_node(link.target_key)):
            continue
        kind = link.kind.value.upper()
        graph.add_edge(link.source_key, link.target_key, key=kind)
        edge = graph.edges[link.source_key, link.target_key, kind]
        edge.update(kind="link", link=kind, **link_row(link)["props"])


__all__ = ["EXTRA_HINT", "NetworkXSink", "claim_node"]
