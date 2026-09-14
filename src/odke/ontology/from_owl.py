"""OWL, RDFS and SKOS to an ontology: the path for users whose schema is already RDF.

Plenty of schemas already exist as OWL, and writing them again as JSON is how
two copies drift. So the RDF vocabularies that say what an ontology here says
are read directly:

- `owl:Class`, `rdfs:Class`, `skos:Concept` — and anything used as a class by
  `rdfs:subClassOf`, `skos:broader`, a domain or an object range — become an
  `EntityType`, named by the IRI's local name;
- `rdfs:subClassOf`, `skos:broader` (and `skos:narrower`, read backwards) name
  its parents; `owl:hasKey` its keys;
- `owl:ObjectProperty` is an edge predicate to its `rdfs:range`;
  `owl:DatatypeProperty` a literal one, its XSD range mapped to a literal type;
  a bare `rdf:Property` is an edge when its range is a class and a literal
  otherwise;
- `rdfs:domain` is the domain, and an `owl:unionOf` domain is several — which
  is exactly what a predicate's domain tuple means;
- `owl:FunctionalProperty` is `cardinality="single"`; every other property is
  `multi`, because OWL's open world lets a property hold any number of values
  unless it says otherwise;
- `rdfs:label` / `skos:prefLabel` is a predicate's label; `rdfs:comment` /
  `skos:definition` is a description, and a type with no comment takes its
  label as one when the label says more than the name; `skos:altLabel` and
  `skos:hiddenLabel` are aliases. Text in `language` wins, then untagged text.

`importance` is left at its default. Nothing in an OWL file says how often a
predicate is used, and inventing a ranking from, say, declaration order would
rank snippets by an accident of authoring; `from_neo4j` is where counts exist.

Everything the model cannot hold is reported by the subject it was found on —
restrictions, property characteristics other than functional, inverse and
sub-properties, equivalence and disjointness, union ranges, unmapped
datatypes, individuals, imports. With `strict` (the default) the conversion
raises `OntologyLoadError` listing every one, as `from_pydantic` does: a
dropped axiom is a rule the extractor is silently never held to. With
`strict=False` whatever maps is loaded and the same list arrives as one
`OntologyImportWarning`, which is what loading a large public ontology needs.
Annotations outside the OWL, RDF, RDFS and SKOS vocabularies — Dublin Core,
`rdfs:seeAlso` — are documentation and are not reported.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

from odke.ontology.load import (
    OntologyLoadError,
    Source,
    _prefix,
    _read,
    load_dict,
    report_problems,
)

if TYPE_CHECKING:
    from rdflib import Graph
    from rdflib.term import Node

    from odke.ontology import Ontology

EXTRA_HINT = 'rdflib is not installed. Run: pip install "odke[rdf]"'

_XSD = "http://www.w3.org/2001/XMLSchema#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_VOCABULARIES = (_OWL, _RDF, _RDFS, _SKOS, _XSD)

_LITERAL_RANGES = {
    **dict.fromkeys(
        (
            f"{_XSD}string",
            f"{_XSD}normalizedString",
            f"{_XSD}token",
            f"{_XSD}language",
            f"{_XSD}Name",
            f"{_XSD}NCName",
            f"{_XSD}anyURI",
            f"{_RDF}langString",
            f"{_RDF}PlainLiteral",
            f"{_RDF}XMLLiteral",
            f"{_RDF}HTML",
            f"{_RDFS}Literal",
        ),
        "string",
    ),
    **dict.fromkeys(
        (
            f"{_XSD}{t}"
            for t in (
                "integer",
                "int",
                "long",
                "short",
                "byte",
                "nonNegativeInteger",
                "positiveInteger",
                "nonPositiveInteger",
                "negativeInteger",
                "unsignedLong",
                "unsignedInt",
                "unsignedShort",
                "unsignedByte",
            )
        ),
        "integer",
    ),
    **dict.fromkeys((f"{_XSD}decimal", f"{_XSD}float", f"{_XSD}double"), "number"),
    f"{_XSD}boolean": "boolean",
    f"{_XSD}date": "date",
    f"{_XSD}dateTime": "datetime",
    f"{_XSD}dateTimeStamp": "datetime",
}

# Types and predicates in the four vocabularies this reader understands or
# deliberately ignores. Anything else from those vocabularies is reported.
_HANDLED_TYPES = frozenset(
    {
        f"{_OWL}Class",
        f"{_RDFS}Class",
        f"{_SKOS}Concept",
        f"{_SKOS}ConceptScheme",
        f"{_OWL}ObjectProperty",
        f"{_OWL}DatatypeProperty",
        f"{_RDF}Property",
        f"{_OWL}FunctionalProperty",
        f"{_OWL}AnnotationProperty",
        f"{_OWL}Ontology",
        f"{_OWL}NamedIndividual",
        f"{_RDFS}Datatype",
    }
)
_HANDLED_PREDICATES = frozenset(
    {
        f"{_RDF}type",
        f"{_RDFS}subClassOf",
        f"{_RDFS}domain",
        f"{_RDFS}range",
        f"{_RDFS}label",
        f"{_RDFS}comment",
        f"{_RDFS}seeAlso",
        f"{_RDFS}isDefinedBy",
        f"{_OWL}hasKey",
        f"{_OWL}versionInfo",
        f"{_OWL}versionIRI",
        f"{_OWL}deprecated",
        f"{_OWL}priorVersion",
        f"{_OWL}backwardCompatibleWith",
        f"{_OWL}incompatibleWith",
        *(
            f"{_SKOS}{p}"
            for p in (
                "prefLabel",
                "altLabel",
                "hiddenLabel",
                "definition",
                "broader",
                "narrower",
                "inScheme",
                "topConceptOf",
                "hasTopConcept",
                "note",
                "scopeNote",
                "example",
                "editorialNote",
                "historyNote",
                "changeNote",
                "notation",
            )
        ),
    }
)


def ontology_from_owl(
    cls: type[Ontology],
    source: Source | Graph,
    *,
    format: str | None,
    name: str | None,
    version: str | None,
    language: str,
    strict: bool,
) -> Ontology:
    graph, where = _parse(source, format)
    data, problems = _Reader(graph, language).read()
    if name is not None:
        data["name"] = name
    if version is not None:
        data["version"] = version
    report_problems(problems, strict=strict, where=where, stacklevel=4)
    return load_dict(cls, data, strict=strict, where=where)


def _parse(source: Any, format: str | None) -> tuple[Graph, str | None]:
    try:
        from rdflib import Graph
        from rdflib.util import guess_format
    except ImportError as exc:
        raise ImportError(EXTRA_HINT) from exc
    if isinstance(source, Graph):
        return source, None
    text, where = _read(source)
    chosen = format or (guess_format(where) if where else None) or _sniff(text)
    try:
        return Graph().parse(data=text, format=chosen), where
    except Exception as exc:  # each rdflib parser raises its own error type
        raise OntologyLoadError(_prefix(where) + f"not valid {chosen}: {exc}") from None


def _sniff(text: str) -> str:
    head = text.lstrip()[:200]
    if head.startswith(("<?xml", "<rdf:RDF")):
        return "xml"
    if head.startswith(("{", "[")):
        return "json-ld"
    # N-Triples is a subset of Turtle, so Turtle reads both.
    return "turtle"


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("_", " ").split())


class _Reader:
    """One pass over one graph; collects the ontology's data and every problem."""

    def __init__(self, graph: Graph, language: str) -> None:
        self.g = graph
        self.language = language
        self.problems: set[str] = set()

    # -- terms --------------------------------------------------------------- #

    def iri(self, text: str) -> Any:
        from rdflib import URIRef

        return URIRef(text)

    def is_iri(self, node: Node) -> bool:
        from rdflib import URIRef

        return isinstance(node, URIRef)

    def builtin(self, node: Node) -> bool:
        return self.is_iri(node) and str(node).startswith(_VOCABULARIES)

    def show(self, node: Node) -> str:
        return str(node.n3(self.g.namespace_manager))

    def local(self, node: Node) -> str:
        text = str(node)
        for sep in ("#", "/"):
            if sep in text:
                tail = text.rsplit(sep, 1)[1]
                if tail:
                    return unquote(tail)
                break
        else:
            if ":" in text:
                return unquote(text.rsplit(":", 1)[1])
        self.report(node, "has no local name to use as a name — end the IRI in a name")
        return text

    def report(self, node: Node, message: str) -> None:
        self.problems.add(f"{self.show(node)}: {message}")

    def texts(self, node: Node, predicates: Iterable[str], *, one: bool) -> list[str]:
        """Literal values of `predicates` on `node`: in `language`, then untagged, then any."""
        from rdflib import Literal

        found = []
        for rank, predicate in enumerate(predicates):
            for value in self.g.objects(node, self.iri(predicate)):
                if not isinstance(value, Literal):
                    continue
                lang = value.language
                if lang is None:
                    fit = 1
                elif lang == self.language or lang.startswith(f"{self.language}-"):
                    fit = 0
                else:
                    fit = 2
                found.append((rank if one else 0, fit, lang or "", str(value).strip()))
        if one:
            return [min(found)[3]] if found else []
        # Aliases in another language match mentions in that language only, so
        # they are kept only when nothing is in `language` or untagged.
        near = {text for _, fit, _, text in found if fit <= 1 and text}
        return sorted(near or {text for _, _, _, text in found if text})

    def text(self, node: Node, *predicates: str) -> str | None:
        found = self.texts(node, predicates, one=True)
        return found[0] if found else None

    def items(self, head: Node) -> list[Node]:
        from rdflib.collection import Collection

        return list(Collection(self.g, head))

    def describe(self, node: Node) -> str:
        """What an anonymous class expression is, in the words a message needs."""
        g = self.g
        if (node, self.iri(f"{_RDF}type"), self.iri(f"{_OWL}Restriction")) in g:
            on = g.value(node, self.iri(f"{_OWL}onProperty"))
            return (
                f"an owl:Restriction on {self.show(on)}" if on is not None else "an owl:Restriction"
            )
        for operator in ("unionOf", "intersectionOf", "complementOf", "oneOf"):
            if g.value(node, self.iri(f"{_OWL}{operator}")) is not None:
                return f"an owl:{operator} expression"
        return "an anonymous class expression"

    # -- the read ------------------------------------------------------------ #

    def read(self) -> tuple[dict[str, Any], list[str]]:
        classes = self.classes()
        names = {c: self.local(c) for c in classes}
        self.collisions(names, "class")
        properties = self.properties(classes)
        property_names = {p: self.local(p) for p in properties}
        self.collisions(property_names, "property")

        types: dict[str, dict[str, Any]] = {}
        for node in sorted(classes, key=str):
            types.setdefault(names[node], self.entity_type(node, names, property_names))
        predicates: dict[str, dict[str, Any]] = {}
        for node in sorted(properties, key=str):
            predicate = self.predicate(node, names)
            if predicate is not None:
                predicates.setdefault(property_names[node], predicate)
        self.individuals(classes, properties)
        self.unsupported()

        data: dict[str, Any] = {"types": types, "predicates": predicates, **self.header()}
        return data, sorted(self.problems)

    def classes(self) -> set[Node]:
        g, rdf_type = self.g, self.iri(f"{_RDF}type")
        found: set[Node] = set()
        for kind in (f"{_OWL}Class", f"{_RDFS}Class", f"{_SKOS}Concept"):
            found.update(g.subjects(rdf_type, self.iri(kind)))
        for relation in (f"{_RDFS}subClassOf", f"{_SKOS}broader", f"{_SKOS}narrower"):
            for s, o in g.subject_objects(self.iri(relation)):
                found.update((s, o))
        found.update(g.objects(None, self.iri(f"{_RDFS}domain")))
        for prop, range_ in g.subject_objects(self.iri(f"{_RDFS}range")):
            if (
                not self.is_datatype(range_)
                and (prop, rdf_type, self.iri(f"{_OWL}DatatypeProperty")) not in g
            ):
                found.add(range_)
        return {c for c in found if self.is_iri(c) and not self.builtin(c)}

    def is_datatype(self, node: Node) -> bool:
        return (
            str(node) in _LITERAL_RANGES
            or str(node).startswith(_XSD)
            or (node, self.iri(f"{_RDF}type"), self.iri(f"{_RDFS}Datatype")) in self.g
        )

    def properties(self, classes: set[Node]) -> set[Node]:
        g, rdf_type = self.g, self.iri(f"{_RDF}type")
        found: set[Node] = set()
        for kind in (
            f"{_OWL}ObjectProperty",
            f"{_OWL}DatatypeProperty",
            f"{_RDF}Property",
            f"{_OWL}FunctionalProperty",
        ):
            found.update(g.subjects(rdf_type, self.iri(kind)))
        for relation in (f"{_RDFS}domain", f"{_RDFS}range"):
            found.update(g.subjects(self.iri(relation)))
        annotations = set(g.subjects(rdf_type, self.iri(f"{_OWL}AnnotationProperty")))
        out = set()
        for node in found:
            if not self.is_iri(node):
                self.report(node, "an anonymous property expression is not supported")
            elif not self.builtin(node) and node not in classes and node not in annotations:
                out.add(node)
        return out

    def collisions(self, names: dict[Node, str], kind: str) -> None:
        owners: dict[str, list[Node]] = {}
        for node, name in names.items():
            owners.setdefault(name, []).append(node)
        for name, nodes in owners.items():
            if len(nodes) > 1:
                shown = " and ".join(sorted(self.show(n) for n in nodes))
                self.problems.add(
                    f"{shown}: each {kind} is named by its local name, and these share "
                    f"{name!r} — they would be one entry"
                )

    def entity_type(
        self, node: Node, names: dict[Node, str], property_names: dict[Node, str]
    ) -> dict[str, Any]:
        from rdflib import BNode

        g, name = self.g, names[node]
        parents: set[str] = set()
        for relation in (f"{_RDFS}subClassOf", f"{_SKOS}broader"):
            for parent in g.objects(node, self.iri(relation)):
                if isinstance(parent, BNode):
                    self.report(
                        node,
                        f"subclass of {self.describe(parent)} — class expressions are not "
                        "supported; state the rule as a property's domain, range or "
                        "owl:FunctionalProperty",
                    )
                elif not self.builtin(parent):
                    parents.add(names[parent])
        parents.update(
            names[c] for c in g.subjects(self.iri(f"{_SKOS}narrower"), node) if c in names
        )

        description = self.text(node, f"{_RDFS}comment", f"{_SKOS}definition")
        label = self.text(node, f"{_RDFS}label", f"{_SKOS}prefLabel")
        if description is None and label is not None and _norm(label) != _norm(name):
            description = label
        keys: list[str] = []
        for head in g.objects(node, self.iri(f"{_OWL}hasKey")):
            for key in self.items(head):
                if key in property_names:
                    keys.append(property_names[key])
                else:
                    self.report(node, f"owl:hasKey names {self.show(key)}, which is not a property")
        return {
            "name": name,
            "description": description,
            "parents": tuple(sorted(parents - {name})),
            "keys": tuple(keys),
            "aliases": tuple(
                self.texts(node, (f"{_SKOS}altLabel", f"{_SKOS}hiddenLabel"), one=False)
            ),
        }

    def predicate(self, node: Node, names: dict[Node, str]) -> dict[str, Any] | None:
        g, rdf_type = self.g, self.iri(f"{_RDF}type")
        is_object = (node, rdf_type, self.iri(f"{_OWL}ObjectProperty")) in g
        is_datatype = (node, rdf_type, self.iri(f"{_OWL}DatatypeProperty")) in g
        if is_object and is_datatype:
            self.report(node, "is both an owl:ObjectProperty and an owl:DatatypeProperty")
            return None
        ranges = list(g.objects(node, self.iri(f"{_RDFS}range")))
        if not is_object and not is_datatype:
            is_object = any(r in names for r in ranges)

        range_ = (
            self.object_range(node, ranges, names)
            if is_object
            else self.literal_range(node, ranges)
        )
        domain = self.domain(node, names)
        if range_ is None or domain is None:
            return None
        return {
            "name": self.local(node),
            "label": self.text(node, f"{_RDFS}label", f"{_SKOS}prefLabel"),
            "description": self.text(node, f"{_RDFS}comment", f"{_SKOS}definition"),
            "domain": domain,
            "range": range_,
            "cardinality": "single"
            if (node, rdf_type, self.iri(f"{_OWL}FunctionalProperty")) in g
            else "multi",
            "aliases": tuple(
                self.texts(node, (f"{_SKOS}altLabel", f"{_SKOS}hiddenLabel"), one=False)
            ),
        }

    def object_range(self, node: Node, ranges: list[Node], names: dict[Node, str]) -> str | None:
        if not ranges:
            self.report(
                node, "an object property with no rdfs:range — an edge needs the type it points at"
            )
            return None
        if len(ranges) > 1:
            self.report(node, f"{len(ranges)} rdfs:range values — an ontology range is one type")
            return None
        (range_,) = ranges
        if range_ in names:
            return names[range_]
        shown = self.show(range_) if self.is_iri(range_) else self.describe(range_)
        self.report(node, f"rdfs:range {shown} — an edge's range must be one named class")
        return None

    def literal_range(self, node: Node, ranges: list[Node]) -> str:
        if not ranges:
            return "string"
        if len(ranges) > 1:
            self.report(node, f"{len(ranges)} rdfs:range values — an ontology range is one type")
            return "string"
        (range_,) = ranges
        found = _LITERAL_RANGES.get(str(range_))
        if found is None:
            shown = self.show(range_) if self.is_iri(range_) else "an anonymous datatype"
            self.report(
                node,
                f"rdfs:range {shown} has no ontology literal type (string, integer, number, "
                "boolean, date, datetime); read as string",
            )
            return "string"
        return found

    def domain(self, node: Node, names: dict[Node, str]) -> tuple[str, ...] | None:
        domains = list(self.g.objects(node, self.iri(f"{_RDFS}domain")))
        found: list[str] = []
        for domain in domains:
            if domain in names:
                found.append(names[domain])
            elif self.builtin(domain):
                continue  # owl:Thing: applies everywhere, which an empty domain says
            else:
                head = self.g.value(domain, self.iri(f"{_OWL}unionOf"))
                members = self.items(head) if head is not None else []
                if members and all(m in names for m in members):
                    found.extend(names[m] for m in members)
                else:
                    self.report(
                        node,
                        f"rdfs:domain {self.describe(domain)} — a domain is named classes, "
                        "or an owl:unionOf of them",
                    )
        if domains and not found and not any(self.builtin(d) for d in domains):
            # Every domain failed: an empty tuple would make it apply everywhere.
            return None
        return tuple(sorted(set(found)))

    def individuals(self, classes: set[Node], properties: set[Node]) -> None:
        g, rdf_type = self.g, self.iri(f"{_RDF}type")
        found = {
            s
            for s, o in g.subject_objects(rdf_type)
            if self.is_iri(s)
            and s not in classes
            and s not in properties
            and (o in classes or o == self.iri(f"{_OWL}NamedIndividual"))
        }
        if found:
            shown = sorted(self.show(s) for s in found)
            listed = ", ".join(shown[:5]) + (", …" if len(shown) > 5 else "")
            self.problems.add(
                f"{len(found)} individual(s) — instance data, not schema — not imported: {listed}"
            )

    def unsupported(self) -> None:
        from rdflib import BNode

        g, rdf_type = self.g, self.iri(f"{_RDF}type")
        referenced = set(g.objects())
        for s, p, o in g:
            if str(p) == f"{_OWL}imports":
                self.report(
                    s,
                    f"owl:imports {self.show(o)} is not followed — parse the imported ontology "
                    "into the same rdflib Graph and pass the Graph",
                )
            elif isinstance(s, BNode):
                if p == rdf_type and s not in referenced and self.vocabulary(o):
                    self.problems.add(f"an anonymous {self.show(o)} axiom is not supported")
            elif p == rdf_type:
                if self.vocabulary(o) and str(o) not in _HANDLED_TYPES:
                    self.report(s, f"is a {self.show(o)}, which the ontology model cannot express")
            elif self.vocabulary(p) and str(p) not in _HANDLED_PREDICATES:
                self.report(s, f"{self.show(p)} is not supported")

    def vocabulary(self, node: Node) -> bool:
        return self.is_iri(node) and str(node).startswith((_OWL, _RDFS, _RDF, _SKOS))

    def header(self) -> dict[str, str]:
        g = self.g
        ontologies = sorted(
            g.subjects(self.iri(f"{_RDF}type"), self.iri(f"{_OWL}Ontology")), key=str
        )
        if not ontologies:
            return {}
        node = ontologies[0]
        out: dict[str, str] = {}
        label = self.text(node, f"{_RDFS}label")
        if label:
            out["name"] = label
        elif self.is_iri(node):
            stem = str(node).rstrip("/#").rsplit("/", 1)[-1]
            out["name"] = (
                stem.rsplit(".", 1)[0] if stem.endswith((".owl", ".ttl", ".rdf")) else stem
            )
        version = self.text(node, f"{_OWL}versionInfo")
        version_iri = g.value(node, self.iri(f"{_OWL}versionIRI"))
        if version:
            out["version"] = version
        elif version_iri is not None:
            out["version"] = str(version_iri).rstrip("/#").rsplit("/", 1)[-1]
        return out


__all__ = ["EXTRA_HINT", "ontology_from_owl"]
