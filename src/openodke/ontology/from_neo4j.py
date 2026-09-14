"""A live Neo4j graph's schema to an ontology: extend the graph you already have.

The most common starting point is not a blank page but a database that already
holds a graph, and its schema is already in it. Three procedures read it, each
present and not deprecated in Neo4j 5:

- `db.schema.nodeTypeProperties()` — per label combination, each property key,
  the value types it holds, and whether every such node has it;
- `db.schema.relTypeProperties()` — the same per relationship type;
- `db.schema.visualization()` — which labels start and end each relationship
  type. It is computed from the count store, so it may pair a start label with
  an end label that never meet on one relationship; but every start label and
  every end label it names is real, and that is all a domain and a range need.

The two property procedures derive their answer by reading the store and are
not free on a large graph. Nothing here writes.

The mapping:

- a label is an entity type;
- a relationship type is a predicate — an edge to the label it ends at, or a
  literal predicate when it ends at `:Claim` nodes — and its properties that
  are not provenance become qualifiers;
- a node property is a literal predicate on the labels holding it, its range
  read from the value types Neo4j reports, `multi` when it holds lists, and
  `required` when every node of those labels has it.

`importance` comes from counts, which is the frequency signal the paper ranks
snippets by: relationships per type from the count store, and one pass per
label counting the nodes that hold each property. Each count is log-scaled
against the most-used predicate's, so the ranking is by use and the spread
stays readable when one predicate is used a million times and another ten.

A graph `Neo4jSink` wrote is read back as the shape it wrote: `:Entity` and
`:Claim` are the sink's labels, not types; the entity fields and provenance
properties are not predicates or qualifiers; a projected property and the
claims behind it are one predicate; the `SAME_AS`/`SIMILAR`/`DIFFERENT`
links are not predicates.

What the store cannot say is not invented. Labels have no hierarchy, so there
are no parents. Edges are `multi`: the schema does not say how many one node
holds, and finding out is a scan. Qualifiers are reconcilable: whether one
bears identity is a decision (DECISIONS #15), not something a count reveals.
What cannot be mapped — a value type with no literal range, a relationship
ending at several labels, one name used by a relationship and a property — is
reported: `strict` raises `OntologyLoadError`, `strict=False` loads the rest
and warns once with `OntologyImportWarning`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openodke.ontology.load import load_dict, report_problems

if TYPE_CHECKING:
    from openodke.ontology import Ontology

NODE_TYPE_PROPERTIES = (
    "CALL db.schema.nodeTypeProperties() "
    "YIELD nodeLabels, propertyName, propertyTypes, mandatory "
    "RETURN nodeLabels, propertyName, propertyTypes, mandatory"
)
REL_TYPE_PROPERTIES = (
    "CALL db.schema.relTypeProperties() "
    "YIELD relType, propertyName, propertyTypes, mandatory "
    "RETURN relType, propertyName, propertyTypes, mandatory"
)
SCHEMA_VISUALIZATION = (
    "CALL db.schema.visualization() YIELD nodes, relationships RETURN nodes, relationships"
)

_LINK_TYPES = frozenset({"SAME_AS", "SIMILAR", "DIFFERENT"})
_VALUE_TYPES = {
    "STRING": "string",
    "CHAR": "string",
    "LONG": "integer",
    "INTEGER": "integer",
    "INT": "integer",
    "SHORT": "integer",
    "BYTE": "integer",
    "DOUBLE": "number",
    "FLOAT": "number",
    "BOOLEAN": "boolean",
    "DATE": "date",
    "DATETIME": "datetime",
    "LOCALDATETIME": "datetime",
    "ZONED DATETIME": "datetime",
    "LOCAL DATETIME": "datetime",
}

Read = Callable[[str], list[dict[str, Any]]]


def value_type(reported: str) -> tuple[str | None, bool]:
    """A Neo4j value type as `(literal range or None, holds a list)`, in either spelling.

    The schema procedures say `String`, `Long`, `StringArray`; Cypher's own
    type names are `STRING`, `INTEGER`, `LIST<STRING NOT NULL>`. Both are read,
    so the answer does not depend on which a given server release reports.
    """
    text = " ".join(reported.upper().replace(" NOT NULL", "").split())
    multi = False
    if text.startswith("LIST<") and text.endswith(">"):
        text, multi = text[5:-1].strip(), True
    elif text.endswith("ARRAY") and text != "ARRAY":
        text, multi = text[: -len("ARRAY")], True
    return _VALUE_TYPES.get(text), multi


def rel_type_name(reported: str) -> str:
    """`:\\`WORKS_AT\\`` as `WORKS_AT`: the procedure quotes the type as Cypher would."""
    text = reported.removeprefix(":")
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        return text[1:-1].replace("``", "`")
    return text


def ontology_from_neo4j(
    cls: type[Ontology],
    source: Any,
    *,
    auth: Any,
    database: str | None,
    name: str | None,
    version: str,
    strict: bool,
) -> Ontology:
    from openodke.sinks.neo4j import _connect

    driver = _connect(source, auth) if isinstance(source, str) else source
    try:
        data, problems = _reflect(driver, database)
    finally:
        if isinstance(source, str):
            driver.close()
    data.update(name=name or database or "neo4j", version=version)
    report_problems(problems, strict=strict, where=None, stacklevel=4)
    return load_dict(cls, data, strict=strict)


def _reflect(driver: Any, database: str | None) -> tuple[dict[str, Any], list[str]]:
    from openodke.sinks.neo4j import _rows

    config = {"database": database} if database else {}
    with driver.session(**config) as session:

        def read(cypher: str) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = session.execute_read(_rows, cypher)
            return rows

        return _Reflection(read).run()


@dataclass
class _Property:
    """One node property key, across every label combination that holds it."""

    key: str
    domain: set[str] = field(default_factory=set)
    ranges: set[str] = field(default_factory=set)
    unmapped: set[str] = field(default_factory=set)
    multi: bool = False
    mandatory: bool = True


class _Reflection:
    def __init__(self, read: Read) -> None:
        from openodke.sinks.neo4j import _ENTITY_FIELDS, _PROVENANCE, CLAIM_LABEL, ENTITY_LABEL

        self.read = read
        self.claim, self.entity = CLAIM_LABEL, ENTITY_LABEL
        self.sink_labels = frozenset({CLAIM_LABEL, ENTITY_LABEL})
        self.entity_fields = _ENTITY_FIELDS
        self.provenance = _PROVENANCE
        self.problems: list[str] = []
        self.types: set[str] = set()
        self.properties: dict[str, _Property] = {}
        self.rel_properties: dict[str, set[str]] = {}
        self.starts: dict[str, set[str]] = {}
        self.ends: dict[str, set[str]] = {}
        self.uses: dict[str, int] = {}

    def run(self) -> tuple[dict[str, Any], list[str]]:
        self.nodes()
        self.relationships()
        self.visualization()
        predicates = self.edges()
        self.literals(predicates)
        top = max(self.uses.values(), default=0)
        if top:
            for name, predicate in predicates.items():
                predicate["importance"] = math.log1p(self.uses.get(name, 0)) / math.log1p(top)
        types = {t: {"name": t} for t in sorted(self.types)}
        return {"types": types, "predicates": predicates}, self.problems

    # -- reading the procedures --------------------------------------------- #

    def nodes(self) -> None:
        for row in self.read(NODE_TYPE_PROPERTIES):
            labels = set(row.get("nodeLabels") or ())
            if self.claim in labels:
                continue  # the sink's record of a literal fact; its claims are read as edges
            written = self.entity in labels
            types = labels - self.sink_labels
            if not types:
                continue
            self.types |= types
            key = row.get("propertyName")
            if key is None or (written and key in self.entity_fields):
                continue
            name = key
            if written and key.startswith("property_") and key[9:] in self.entity_fields:
                name = key[9:]  # a projection renamed so it could not overwrite a field
            prop = self.properties.setdefault(name, _Property(key=key))
            prop.domain |= types
            prop.mandatory = prop.mandatory and bool(row.get("mandatory"))
            for reported in row.get("propertyTypes") or ():
                range_, multi = value_type(str(reported))
                if range_ is None:
                    prop.unmapped.add(str(reported))
                else:
                    prop.ranges.add(range_)
                    prop.multi = prop.multi or multi

    def relationships(self) -> None:
        for row in self.read(REL_TYPE_PROPERTIES):
            rel = rel_type_name(str(row.get("relType") or ""))
            if not rel:
                continue
            props = self.rel_properties.setdefault(rel, set())
            if row.get("propertyName") is not None:
                props.add(str(row["propertyName"]))

    def visualization(self) -> None:
        for row in self.read(SCHEMA_VISUALIZATION):
            for node in row.get("nodes") or ():
                label = _name_of(node)
                if label and label not in self.sink_labels:
                    self.types.add(label)
            for relationship in row.get("relationships") or ():
                start, rel, end = _ends_of(relationship)
                if rel:
                    self.starts.setdefault(rel, set()).add(start)
                    self.ends.setdefault(rel, set()).add(end)

    def count(self, cypher: str) -> int:
        rows = self.read(cypher)
        return int(rows[0].get("n") or 0) if rows else 0

    # -- building predicates ------------------------------------------------- #

    def edges(self) -> dict[str, dict[str, Any]]:
        from openodke.sinks.neo4j import _ident

        predicates: dict[str, dict[str, Any]] = {}
        for rel in sorted({*self.rel_properties, *self.starts, *self.ends}):
            props = self.rel_properties.get(rel, set())
            if rel in _LINK_TYPES and "created_at" in props and "signature" not in props:
                continue  # an EntityLink the sink wrote: an opinion about identity, not a fact
            domain = tuple(sorted(self.starts.get(rel, set()) - self.sink_labels))
            ends = self.ends.get(rel, set()) - {self.entity}
            qualifiers: dict[str, dict[str, Any]] = {
                _qualifier_name(p, self.provenance): {}
                for p in sorted(props)
                if p not in self.provenance
            }
            where = f"relationship type {rel!r}"
            if self.claim in ends:
                if ends - {self.claim}:
                    others = ", ".join(sorted(ends - {self.claim}))
                    self.problems.append(
                        f"{where}: ends at Claim nodes and at {others} — a predicate is an "
                        "edge or a literal, not both"
                    )
                    continue
                range_ = "string"
            elif not ends:
                self.problems.append(
                    f"{where}: the schema records no label its relationships end at, so it "
                    "has no range"
                )
                continue
            elif len(ends) == 1:
                (range_,) = ends
            else:
                used = {
                    label: self.count(
                        f"MATCH ()-[r:{_ident(rel)}]->(:{_ident(label)}) RETURN count(r) AS n"
                    )
                    for label in sorted(ends)
                }
                range_ = max(sorted(used), key=used.__getitem__)
                self.problems.append(
                    f"{where}: ends at {', '.join(sorted(ends))} — an ontology range is one "
                    f"type; read as {range_!r}, the most used"
                )
            if range_ != "string" or self.claim not in ends:
                self.types.add(range_)
            self.types |= set(domain)
            predicates[rel] = {
                "name": rel,
                "domain": domain,
                "range": range_,
                "cardinality": "multi",
                "qualifiers": qualifiers,
            }
            self.uses[rel] = self.count(f"MATCH ()-[r:{_ident(rel)}]->() RETURN count(r) AS n")
        return predicates

    def literals(self, predicates: dict[str, dict[str, Any]]) -> None:
        from openodke.sinks.neo4j import _ident

        by_label: dict[str, list[tuple[str, str]]] = {}
        for name, prop in sorted(self.properties.items()):
            existing = predicates.get(name)
            where = f"property {name!r}"
            if existing is not None and existing["range"] in self.types:
                self.problems.append(
                    f"{where}: is also a relationship type between entities — a predicate is "
                    "an edge or a literal, so rename one; the relationship is kept"
                )
                continue
            range_ = self.literal_range(where, prop)
            shape = {
                "range": range_,
                "cardinality": "multi" if prop.multi else "single",
                "required": prop.mandatory,
            }
            if existing is not None:
                # A projection of claims the sink wrote: one predicate, and the
                # property knows the value type the claims were stored as.
                existing.update(shape, domain=tuple(sorted({*existing["domain"], *prop.domain})))
            else:
                predicates[name] = {"name": name, "domain": tuple(sorted(prop.domain)), **shape}
            for label in prop.domain:
                by_label.setdefault(label, []).append((prop.key, name))

        counted: dict[str, int] = {}
        for label, keys in sorted(by_label.items()):
            columns = sorted({key for key, _ in keys})
            returns = ", ".join(f"count(n.{_ident(k)}) AS {_ident(k)}" for k in columns)
            rows = self.read(f"MATCH (n:{_ident(label)}) RETURN {returns}")
            row = rows[0] if rows else {}
            for key, name in keys:
                counted[name] = counted.get(name, 0) + int(row.get(key) or 0)
        for name, n in counted.items():
            self.uses[name] = max(self.uses.get(name, 0), n)

    def literal_range(self, where: str, prop: _Property) -> str:
        if prop.unmapped:
            self.problems.append(
                f"{where}: holds {', '.join(sorted(prop.unmapped))}, which has no ontology "
                "literal type (string, integer, number, boolean, date, datetime); read as string"
            )
            return "string"
        if prop.ranges == {"integer", "number"}:
            return "number"
        if len(prop.ranges) > 1:
            self.problems.append(
                f"{where}: holds values of several types ({', '.join(sorted(prop.ranges))}); "
                "read as string"
            )
            return "string"
        return next(iter(prop.ranges), "string")


def _qualifier_name(key: str, provenance: frozenset[str]) -> str:
    stripped = key.removeprefix("qualifier_")
    return stripped if stripped != key and stripped in provenance else key


def _name_of(node: Any) -> str:
    if isinstance(node, Mapping):
        return str(node.get("name") or "")
    try:
        return str(node["name"])
    except (KeyError, TypeError):
        return ""


def _ends_of(relationship: Any) -> tuple[str, str, str]:
    """(start label, type, end label), from the driver's `.data()` tuple or a Relationship."""
    if isinstance(relationship, Sequence) and not isinstance(relationship, str):
        if len(relationship) == 3:
            start, rel, end = relationship
            return _name_of(start), str(rel), _name_of(end)
        return "", "", ""
    start, end = getattr(relationship, "start_node", None), getattr(relationship, "end_node", None)
    return _name_of(start), str(getattr(relationship, "type", "") or ""), _name_of(end)


__all__ = [
    "NODE_TYPE_PROPERTIES",
    "REL_TYPE_PROPERTIES",
    "SCHEMA_VISUALIZATION",
    "ontology_from_neo4j",
    "rel_type_name",
    "value_type",
]
