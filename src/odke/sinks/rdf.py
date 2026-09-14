"""The RDF sink: Turtle, N-Triples and JSON-LD, through rdflib.

rdflib is imported where a graph is built, never at the top, so this module
imports on the base install (DECISIONS #1) and constructing a sink without the
`[rdf]` extra fails with the command that fixes it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from odke.ontology import Ontology
from odke.sinks.neo4j import entities_of, is_scoped, provenance_of, signature_of, storable
from odke.types import Evidence, Fact, KnowledgeGraph, LinkKind, Polarity

if TYPE_CHECKING:
    from rdflib import Graph, URIRef
    from rdflib.term import Node

EXTRA_HINT = "rdflib is not installed; run: pip install 'odke[rdf]'"

# The namespace of the terms this sink owns: `odke:Fact`, `odke:polarity`,
# `odke:evidence`. A name, not yet a published document.
VOCAB = "https://deepskandpal.github.io/odke/vocab#"
# A documentation domain on purpose: data minted under it is visibly unplaced.
DEFAULT_BASE = "https://example.org/odke/"

_FORMATS = {
    "turtle": "turtle",
    "ttl": "turtle",
    "nt": "nt",
    "ntriples": "nt",
    "n-triples": "nt",
    "json-ld": "json-ld",
    "jsonld": "json-ld",
}
_SUFFIXES = {".ttl": "turtle", ".nt": "nt", ".jsonld": "json-ld", ".json": "json-ld"}

# The provenance a fact's statement node carries, named as on a Neo4j
# relationship, so one concept has one name in Cypher and in SPARQL.
_FACT_PROVENANCE = (
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
)
# An ontology literal range as the XSD datatype it is written as.
XSD_RANGES = {
    "string": "string",
    "integer": "integer",
    "number": "decimal",
    "float": "double",
    "boolean": "boolean",
    "date": "date",
    "datetime": "dateTime",
}


def _require_rdflib() -> None:
    try:
        import rdflib  # noqa: F401
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc


class RdfSink:
    """Writes a `KnowledgeGraph` as RDF: Turtle, N-Triples or JSON-LD.

    The shape, and why:

    - An entity is `<base>entity/<key>`, typed with its ontology class
      `<schema><Type>` and `odke:Entity`, with `odke:key`, `rdfs:label`,
      `skos:altLabel` per alias, `odke:external_id`, `odke:resolution_*`, and
      its attributes under `<schema>attribute/`. The IRI is the key alone,
      because an `EntityLink` knows only keys; a key two types share is one
      resource with both types.
    - Every fact is a **reified statement** `<base>fact/<signature>` — an
      `rdf:Statement` with `rdf:subject`, `rdf:predicate`, `rdf:object` — and
      its provenance hangs off that node: `odke:polarity`, `odke:confidence`,
      `odke:support`, both clocks, `odke:identity_keys`, the qualifiers under
      `<schema>qualifier/`, and one `odke:evidence` node per source with
      `odke:doc_id`, `odke:uri`, `odke:start`/`odke:end`, `odke:tier` and
      `odke:retrieved_at`. The node plays the part the relationship plays in
      Neo4j and carries the same property names. The IRI is the fact's
      signature, so a rerun with new ids and clocks addresses the same node.
    - The plain triple `<s> <schema><predicate> <o>` is written **only** for
      an asserted, unscoped fact — the same rule that decides the Neo4j
      projection. It is what a SPARQL query reaches for first, and it is a
      claim of truth: a denial written that way would say the opposite of its
      source, and "uptime 99.9%" without its percentile says something no
      source said.
    - A link is `owl:sameAs`, `odke:similar_to` or `odke:different_from`
      between the two entity IRIs, reified the same way so `odke:score`,
      `odke:reason` and `odke:created_at` are readable. Nodes are never
      merged (DECISIONS #16); a store that reasons over `owl:sameAs` will
      merge them itself, which is that platform's call (DECISIONS #21).

    **Why reification.** The three candidates for provenance per fact were
    RDF-star, named graphs and standard reification (05-context §2).

    - *RDF-star* reads cleanest, but rdflib 7 has no quoted-triple term: its
      Turtle parser rejects `<< >>` and no serializer writes it, and neither
      N-Triples 1.1 nor JSON-LD 1.1 can carry it. Worse, a quoted triple has
      no identity beyond its three terms, so uptime at p50 and at p95 with
      the same value would be one quoted triple with two provenance records
      mixed together — exactly the merge DECISIONS #15 exists to prevent.
    - *Named graphs* need TriG or N-Quads, not the three formats asked for,
      and a denial placed in its own graph is still an asserted triple in any
      store that queries the union of its graphs — polarity would not survive.
    - *Reification* works in all three formats and every store, gives each
      fact a node whose IRI is its signature, and is non-asserting by RDF's
      semantics: describing a statement does not entail it. So a denial is
      a statement node with `odke:polarity "denied"` and nothing else, and
      the only asserted triples are the ones that should be. The cost is
      bulk — six-plus triples per fact — which is the price of provenance
      that is queryable rather than merely present.

    With an `ontology`, the output also declares its schema — `owl:Class`
    with `rdfs:subClassOf`, `owl:ObjectProperty` or `owl:DatatypeProperty`
    with domain, range and `owl:FunctionalProperty` for single cardinality —
    so the file describes itself and `Ontology.from_owl` reads it back.

    Two facts with one signature are one statement node, carrying the later
    fact, as successive `SET r += props` would leave a Neo4j relationship.
    The file is rewritten on every `write`.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: str | None = None,
        base: str = DEFAULT_BASE,
        schema: str | None = None,
        ontology: Ontology | None = None,
    ) -> None:
        _require_rdflib()
        self.path = Path(path)
        chosen = (format or _SUFFIXES.get(self.path.suffix.lower(), "turtle")).lower()
        if chosen not in _FORMATS:
            raise ValueError(f"format must be one of {sorted(set(_FORMATS.values()))!r}")
        self.format = _FORMATS[chosen]
        self.base = base
        # Types and predicates: the ontology's IRIs. Under the base by default.
        self.schema = schema if schema is not None else f"{base}schema/"
        self.ontology = ontology

    # -- names --------------------------------------------------------------- #

    def entity_iri(self, key: str) -> str:
        return f"{self.base}entity/{quote(key, safe='')}"

    def fact_iri(self, fact: Fact) -> str:
        return f"{self.base}fact/{signature_of(fact)}"

    def term_iri(self, name: str) -> str:
        """A type or predicate name as an ontology IRI."""
        return f"{self.schema}{quote(name, safe='')}"

    # -- the graph ----------------------------------------------------------- #

    def graph(self, kg: KnowledgeGraph) -> Graph:
        """The RDF `write` serialises. Pure, so it is testable and queryable in memory."""
        return _Builder(self).build(kg)

    def serialize(self, kg: KnowledgeGraph) -> str:
        return self.graph(kg).serialize(format=self.format)

    def write(self, kg: KnowledgeGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.serialize(kg), encoding="utf-8")


class _Builder:
    """One graph under construction; rdflib is imported here and only here."""

    def __init__(self, sink: RdfSink) -> None:
        from rdflib import Graph, Namespace
        from rdflib.namespace import SKOS

        self.sink = sink
        self.g = Graph(bind_namespaces="core")
        self.odke = Namespace(VOCAB)
        for prefix, namespace in (
            ("odke", VOCAB),
            ("skos", str(SKOS)),
            ("ont", sink.schema),
            ("ent", f"{sink.base}entity/"),
        ):
            self.g.bind(prefix, namespace)

    def build(self, kg: KnowledgeGraph) -> Graph:
        for (type_name, _), entity in sorted(entities_of(kg).items()):
            self._entity(type_name, entity)
        latest: dict[str, Fact] = {}
        for fact in kg.facts:
            latest[signature_of(fact)] = fact
        for fact in latest.values():
            self._fact(fact, kg)
        links = {_link_id(link.kind, link.source_key, link.target_key): link for link in kg.links}
        for ident, link in links.items():
            self._link(ident, link)
        if self.sink.ontology is not None:
            self._schema(self.sink.ontology)
        return self.g

    # -- terms --------------------------------------------------------------- #

    def _iri(self, text: str) -> URIRef:
        from rdflib import URIRef

        return URIRef(text)

    def _vocab(self, name: str) -> URIRef:
        return self.odke[name]

    def _literal(self, value: Any) -> Any:
        from rdflib import Literal

        value = storable(value)
        if isinstance(value, list):
            value = json.dumps(value, default=str, ensure_ascii=False)
        return Literal(value)

    def _put(self, subject: URIRef, predicate: URIRef, value: Any) -> None:
        if value is not None:
            self.g.add((subject, predicate, self._literal(value)))

    # -- the graph's parts --------------------------------------------------- #

    def _entity(self, type_name: str, entity: Any) -> None:
        from rdflib.namespace import RDF, RDFS, SKOS

        sink = self.sink
        node = self._iri(sink.entity_iri(entity.key))
        self.g.add((node, RDF.type, self._iri(sink.term_iri(type_name))))
        self.g.add((node, RDF.type, self._vocab("Entity")))
        self._put(node, self._vocab("key"), entity.key)
        self._put(node, RDFS.label, entity.label)
        for alias in entity.aliases:
            self._put(node, SKOS.altLabel, alias)
        self._put(node, self._vocab("external_id"), entity.external_id)
        if entity.resolution is not None:
            self._put(node, self._vocab("resolution_method"), entity.resolution.method)
            self._put(node, self._vocab("resolution_score"), entity.resolution.score)
            self._put(node, self._vocab("resolution_linker"), entity.resolution.linker)
        for name, value in entity.attributes.items():
            self._put(node, self._iri(f"{sink.schema}attribute/{quote(name, safe='')}"), value)

    def _fact(self, fact: Fact, kg: KnowledgeGraph) -> None:
        from rdflib.namespace import RDF

        sink = self.sink
        subject = self._iri(sink.entity_iri(fact.subject.key))
        predicate = self._iri(sink.term_iri(fact.predicate))
        obj: Any = None
        if fact.object_entity is not None:
            obj = self._iri(sink.entity_iri(fact.object_entity.key))
        elif fact.object_value is not None:
            obj = self._literal(fact.object_value)
        if obj is not None and fact.polarity is Polarity.ASSERTED and not is_scoped(fact):
            self.g.add((subject, predicate, obj))

        node = self._iri(sink.fact_iri(fact))
        self.g.add((node, RDF.type, RDF.Statement))
        self.g.add((node, RDF.type, self._vocab("Fact")))
        self.g.add((node, RDF.subject, subject))
        self.g.add((node, RDF.predicate, predicate))
        if obj is not None:
            self.g.add((node, RDF.object, obj))
        props = provenance_of(fact, kg.created_at)
        for name in _FACT_PROVENANCE:
            self._put(node, self._vocab(name), props[name])
        for key in fact.identity_keys:
            self._put(node, self._vocab("identity_keys"), key)
        for key, value in fact.qualifiers.items():
            self._put(node, self._iri(f"{sink.schema}qualifier/{quote(key, safe='')}"), value)
        self._evidence(node, fact.evidence)

    def _evidence(self, owner: URIRef, evidence: tuple[Evidence, ...]) -> None:
        from rdflib import Literal
        from rdflib.namespace import RDF, XSD

        for i, item in enumerate(evidence):
            node = self._iri(f"{owner}/evidence/{i}")
            self.g.add((owner, self._vocab("evidence"), node))
            self.g.add((node, RDF.type, self._vocab("Evidence")))
            self._put(node, self._vocab("doc_id"), item.doc_id)
            if item.uri is not None:
                self.g.add((node, self._vocab("uri"), Literal(item.uri, datatype=XSD.anyURI)))
            if item.span is not None:
                self._put(node, self._vocab("start"), item.span.start)
                self._put(node, self._vocab("end"), item.span.end)
                self._put(node, self._vocab("quote"), item.span.quote)
            self._put(node, self._vocab("tier"), item.tier.value)
            self._put(node, self._vocab("retrieved_at"), item.retrieved_at)

    def _link(self, ident: str, link: Any) -> None:
        from rdflib.namespace import OWL, RDF

        sink = self.sink
        kinds = {
            LinkKind.SAME_AS: OWL.sameAs,
            LinkKind.SIMILAR: self._vocab("similar_to"),
            LinkKind.DIFFERENT: self._vocab("different_from"),
        }
        source = self._iri(sink.entity_iri(link.source_key))
        target = self._iri(sink.entity_iri(link.target_key))
        self.g.add((source, kinds[link.kind], target))
        node = self._iri(f"{sink.base}link/{ident}")
        self.g.add((node, RDF.type, RDF.Statement))
        self.g.add((node, RDF.type, self._vocab("Link")))
        self.g.add((node, RDF.subject, source))
        self.g.add((node, RDF.predicate, kinds[link.kind]))
        self.g.add((node, RDF.object, target))
        self._put(node, self._vocab("kind"), link.kind.value)
        self._put(node, self._vocab("score"), link.score)
        self._put(node, self._vocab("reason"), link.reason)
        self._put(node, self._vocab("created_at"), link.created_at)
        self._evidence(node, link.evidence)

    def _schema(self, ontology: Ontology) -> None:
        from rdflib import BNode
        from rdflib.collection import Collection
        from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

        sink, g = self.sink, self.g
        for name, entity_type in ontology.types.items():
            cls = self._iri(sink.term_iri(name))
            g.add((cls, RDF.type, OWL.Class))
            for parent in entity_type.parents:
                g.add((cls, RDFS.subClassOf, self._iri(sink.term_iri(parent))))
            self._put(cls, RDFS.comment, entity_type.description)
            for alias in entity_type.aliases:
                self._put(cls, SKOS.altLabel, alias)
            if entity_type.keys:
                keys = BNode()
                listed: list[Node] = [self._iri(sink.term_iri(k)) for k in entity_type.keys]
                Collection(g, keys, listed)
                g.add((cls, OWL.hasKey, keys))
        for name, predicate in ontology.predicates.items():
            prop = self._iri(sink.term_iri(name))
            edge = predicate.range in ontology.types
            g.add((prop, RDF.type, OWL.ObjectProperty if edge else OWL.DatatypeProperty))
            if predicate.cardinality == "single":
                g.add((prop, RDF.type, OWL.FunctionalProperty))
            self._put(prop, RDFS.label, predicate.label)
            self._put(prop, RDFS.comment, predicate.description)
            for alias in predicate.aliases:
                self._put(prop, SKOS.altLabel, alias)
            domains: list[Node] = [self._iri(sink.term_iri(d)) for d in predicate.domain]
            if len(domains) == 1:
                g.add((prop, RDFS.domain, domains[0]))
            elif domains:
                # Several rdfs:domain triples mean their intersection in OWL; a
                # predicate's domain is any of them, which is a union.
                union, members = BNode(), BNode()
                Collection(g, members, domains)
                g.add((union, RDF.type, OWL.Class))
                g.add((union, OWL.unionOf, members))
                g.add((prop, RDFS.domain, union))
            if edge:
                g.add((prop, RDFS.range, self._iri(sink.term_iri(predicate.range))))
            else:
                g.add((prop, RDFS.range, XSD[XSD_RANGES.get(predicate.range, "string")]))


def _link_id(kind: LinkKind, source: str, target: str) -> str:
    # One node per (kind, ends), as Neo4j MERGEs one relationship per kind and pair.
    canonical = json.dumps([kind.value, source, target], ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["DEFAULT_BASE", "EXTRA_HINT", "VOCAB", "XSD_RANGES", "RdfSink"]
