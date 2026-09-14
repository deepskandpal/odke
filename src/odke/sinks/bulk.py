"""Bulk sinks: the Neo4j write plan as a Cypher script, or as `neo4j-admin` CSV.

A driver write is the wrong tool past a certain size, and the first import of a
large corpus is that size: a round trip per batch, a transaction per batch, an
index lookup per `MERGE`. Both sinks here take the plan `Neo4jSink` would run —
`odke.sinks.neo4j.plan()`, the same rows under the same keys — and lay it out
for an offline tool instead. Nothing about the graph's shape is decided here,
so a graph written by a script, by an import or by the driver is one graph.

Neither imports the driver, so both work on the base install (DECISIONS #1).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, NamedTuple

from odke.ontology import Ontology
from odke.sinks.neo4j import (
    CLAIM_LABEL,
    ENTITY_LABEL,
    Neo4jConstrainer,
    Neo4jSink,
    Statement,
    _batches,
    _schema_name,
    plan,
    projection_property,
)
from odke.types import KnowledgeGraph

# --------------------------------------------------------------------------- #
# Cypher script
# --------------------------------------------------------------------------- #

_PARAMETER = "UNWIND $rows AS row\n"
_SIMPLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_ESCAPES = {
    **{i: f"\\u{i:04x}" for i in range(0x20)},
    ord("\\"): "\\\\",
    ord("'"): "\\'",
    ord("\n"): "\\n",
    ord("\r"): "\\r",
    ord("\t"): "\\t",
    ord("\b"): "\\b",
    ord("\f"): "\\f",
}


def cypher_literal(value: Any) -> str:
    """A value as Cypher source, typed the way the driver would have sent it.

    A naive datetime is a `localdatetime` and an aware one a `datetime`, as the
    Python driver maps them, so a replayed script stores what `Neo4jSink.write`
    stores. A string never contains a raw newline, which is what lets
    `CypherFileSink.script` separate statements safely.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "0.0 / 0.0"
        if math.isinf(value):
            return "1.0 / 0.0" if value > 0 else "-1.0 / 0.0"
        return repr(value).replace("e+", "e")
    if isinstance(value, str):
        return "'" + value.translate(_STRING_ESCAPES) + "'"
    if isinstance(value, datetime):
        fn = "datetime" if value.tzinfo is not None else "localdatetime"
        return f"{fn}('{value.isoformat()}')"
    if isinstance(value, date):
        return f"date('{value.isoformat()}')"
    if isinstance(value, time):
        fn = "time" if value.tzinfo is not None else "localtime"
        return f"{fn}('{value.isoformat()}')"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(cypher_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        entries = (f"{_map_key(str(k))}: {cypher_literal(v)}" for k, v in value.items())
        return "{" + ", ".join(entries) + "}"
    raise TypeError(f"no Cypher literal for {type(value).__name__}: {value!r}")


def _map_key(name: str) -> str:
    if _SIMPLE_NAME.fullmatch(name):
        return name
    return "`" + name.replace("`", "``") + "`"


def _inline(statement: Statement, rows: Sequence[dict[str, Any]]) -> str:
    if not statement.cypher.startswith(_PARAMETER):
        raise ValueError(f"not an UNWIND statement: {statement.cypher.splitlines()[0]}")
    listed = ",\n".join(f"  {cypher_literal(row)}" for row in rows)
    return f"UNWIND [\n{listed}\n] AS row\n" + statement.cypher.removeprefix(_PARAMETER)


_SCRIPT_HEADER = """\
// Written by odke's CypherFileSink. Replay with:
//   cypher-shell -a <uri> -u <user> -f <this file>
// Every statement is CREATE ... IF NOT EXISTS or UNWIND ... MERGE, so replaying
// the script, or running it over a graph Neo4jSink wrote, changes nothing twice."""


class CypherFileSink:
    """Writes the graph as a `.cypher` script that `cypher-shell -f` replays.

    The statements are `Neo4jSink`'s own — `plan()` — with each batch's rows
    inlined as a list literal where the driver would bind `$rows`. Inlining
    rather than `:param` keeps the file runnable by any Cypher client, not only
    by cypher-shell, and keeps it one self-contained artefact to review or
    commit. Same `MERGE` keys, so replaying the file twice, or replaying it over
    a graph the driver wrote, updates rather than duplicates.

    With an `ontology`, the script opens with the `Neo4jConstrainer` DDL — the
    uniqueness constraints that make the `MERGE`s correct and fast — and the
    ontology decides multi-valued projections, exactly as it does for
    `Neo4jSink`. Check queries are left out: they return violators, and a
    script run for its side effects would discard the answer.

    The file is rewritten on every `write`; a graph always compiles to the same
    bytes, so two writes of one graph leave one identical file.
    """

    profile = Neo4jSink.profile

    def __init__(
        self,
        path: str | Path,
        *,
        ontology: Ontology | None = None,
        batch_size: int = 500,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.path = Path(path)
        self.ontology = ontology
        self.batch_size = batch_size

    def statements(self, kg: KnowledgeGraph) -> list[str]:
        """Each statement of the script, without its terminating `;`, in run order."""
        out: list[str] = []
        if self.ontology is not None:
            out += Neo4jConstrainer().schema(self.ontology)
        for statement in plan(kg, ontology=self.ontology):
            for batch in _batches(statement.rows, self.batch_size):
                out.append(_inline(statement, batch))
        return out

    def script(self, kg: KnowledgeGraph) -> str:
        return "\n\n".join([_SCRIPT_HEADER, *(f"{s};" for s in self.statements(kg))]) + "\n"

    def write(self, kg: KnowledgeGraph) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.script(kg), encoding="utf-8")


# --------------------------------------------------------------------------- #
# neo4j-admin CSV
# --------------------------------------------------------------------------- #

_DELIMITERS = {",": ",", ";": ";", "|": "|", "\t": "TAB"}
_CLAIM_SPACE = "odke_claim_signature"
_NODE_FIELDS = (
    "label",
    "aliases",
    "external_id",
    "resolution_method",
    "resolution_score",
    "resolution_linker",
)


class Table(NamedTuple):
    """One CSV file: what `neo4j-admin` reads it as, its header and its rows.

    `columns` are header names; `rows` hold raw values, typed by the header and
    rendered only when the file is written.
    """

    file: str
    kind: Literal["nodes", "relationships"]
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


class Neo4jAdminCsvSink:
    """Writes the graph as the node and relationship CSVs `neo4j-admin database import` reads.

    The layout is `plan()` again, one file per statement group:

    - an entity type is a node file, `key:ID(<space>)` plus a `:LABEL` column
      holding the type and `Entity`, the entity fields, its attributes and the
      literal values projected onto it — every type its own ID space, because
      Neo4j keys a node on (type, key) and two types may share a key;
    - claims are a node file per (subject type, predicate) in one `Claim` ID
      space keyed on the signature, as `Neo4jSink` MERGEs them;
    - an edge group, a claim group and a link kind are relationship files with
      `:START_ID`, `:END_ID`, `:TYPE` and every provenance property the driver
      would have set — evidence lists as arrays, both clocks as datetimes.

    Rows `Neo4jSink` would MERGE into one — two facts with one signature, a
    link written twice — are one row here, their properties combined in plan
    order as successive `SET r += row.props` would combine them. A link is
    written between every node holding each key, which is what its `MATCH` on
    `(:Entity {key})` finds; a key no entity in the graph holds finds nothing,
    since an import starts from an empty database.

    Every property column is typed in its header, inferred from the values it
    holds; a column whose values disagree on a type — an attribute that is a
    number on one entity and text on another — is written as text, because a
    CSV column has one type. Newlines in values are kept, so the import needs
    `--multiline-fields=true`; `import.args` next to the files carries that and
    the delimiters, so the import is::

        cd <directory> && neo4j-admin database import full @import.args <database>

    An import builds a new database rather than updating one; to extend a live
    graph use `Neo4jSink` or `CypherFileSink`. Files are rewritten on every
    `write`, and one graph always produces the same bytes.
    """

    profile = Neo4jSink.profile

    def __init__(
        self,
        directory: str | Path,
        *,
        ontology: Ontology | None = None,
        delimiter: str = ",",
        array_delimiter: str = ";",
    ) -> None:
        if delimiter not in _DELIMITERS or array_delimiter not in _DELIMITERS:
            raise ValueError(f"delimiters must each be one of {sorted(_DELIMITERS)!r}")
        if delimiter == array_delimiter:
            raise ValueError("delimiter and array_delimiter must differ")
        self.directory = Path(directory)
        self.ontology = ontology
        self.delimiter = delimiter
        self.array_delimiter = array_delimiter

    # -- the layout ---------------------------------------------------------- #

    def tables(self, kg: KnowledgeGraph) -> list[Table]:
        """Every file `write` would produce, in import order. Pure, so it is testable."""
        statements = plan(kg, ontology=self.ontology)
        nodes: dict[str, dict[str, dict[str, Any]]] = {}
        spaces: dict[str, list[str]] = {}
        for st in (s for s in statements if s.kind == "entity"):
            (label,) = st.names
            records = nodes.setdefault(label, {})
            for row in st.rows:
                records[row["key"]] = {
                    **row["attributes"],
                    **{field: row[field] for field in _NODE_FIELDS},
                }
                spaces.setdefault(row["key"], []).append(label)
        for st in (s for s in statements if s.kind == "projection"):
            subject_type, predicate = st.names
            column = projection_property(predicate)
            for row in st.rows:
                nodes[subject_type][row["subject_key"]][column] = row["value"]

        out = [self._node_table(label, records) for label, records in nodes.items()]
        for st in statements:
            if st.kind == "claim":
                out += self._claim_tables(st)
            elif st.kind == "edge":
                out.append(self._edge_table(st))
            elif st.kind == "link":
                out += self._link_tables(st, spaces)
        return out

    def _node_table(self, label: str, records: dict[str, dict[str, Any]]) -> Table:
        props = _columns(records.values(), first=_NODE_FIELDS)
        header = (f"key:ID({_space(label)})", ":LABEL", *_typed(props, records.values()))
        rows = tuple(
            (key, [label, ENTITY_LABEL], *(record.get(p) for p in props))
            for key, record in records.items()
        )
        return Table(f"{_schema_name('nodes', label)}.csv", "nodes", header, rows)

    def _claim_tables(self, st: Statement) -> list[Table]:
        subject_type, predicate = st.names
        fields = ("predicate", "subject_key", "subject_type", "value")
        claims: dict[str, dict[str, Any]] = {}
        merged: dict[str, dict[str, Any]] = {}
        for row in st.rows:
            claims[row["signature"]] = {f: row[f] for f in fields}
            merged.setdefault(row["signature"], {}).update(row["props"])
        stem = f"{subject_type}-{predicate}"
        node_header = (f"signature:ID({_CLAIM_SPACE})", ":LABEL", *_typed(fields, claims.values()))
        node_rows = tuple(
            (sig, [CLAIM_LABEL], *(claim[f] for f in fields)) for sig, claim in claims.items()
        )
        rels = self._relationships(
            f"{_schema_name('claim_edges', stem)}.csv",
            predicate,
            (_space(subject_type), _CLAIM_SPACE),
            ((claims[sig]["subject_key"], sig, props) for sig, props in merged.items()),
        )
        return [
            Table(f"{_schema_name('claims', stem)}.csv", "nodes", node_header, node_rows),
            rels,
        ]

    def _edge_table(self, st: Statement) -> Table:
        subject_type, predicate, object_type = st.names
        merged: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for row in st.rows:
            start, end, props = merged.setdefault(
                row["signature"], (row["subject_key"], row["object_key"], {})
            )
            props.update(row["props"])
        return self._relationships(
            f"{_schema_name('edges', f'{subject_type}-{predicate}-{object_type}')}.csv",
            predicate,
            (_space(subject_type), _space(object_type)),
            merged.values(),
        )

    def _link_tables(self, st: Statement, spaces: dict[str, list[str]]) -> list[Table]:
        (kind,) = st.names
        groups: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
        for row in st.rows:
            for source_type in spaces.get(row["source_key"], ()):
                for target_type in spaces.get(row["target_key"], ()):
                    group = groups.setdefault((source_type, target_type), {})
                    ends = (row["source_key"], row["target_key"])
                    group.setdefault(ends, {}).update(row["props"])
        return [
            self._relationships(
                f"{_schema_name('links', f'{kind}-{source_type}-{target_type}')}.csv",
                kind,
                (_space(source_type), _space(target_type)),
                ((start, end, props) for (start, end), props in group.items()),
            )
            for (source_type, target_type), group in groups.items()
        ]

    def _relationships(
        self,
        file: str,
        rel_type: str,
        id_spaces: tuple[str, str],
        rows: Iterable[tuple[str, str, dict[str, Any]]],
    ) -> Table:
        listed = list(rows)
        records = [props for _, _, props in listed]
        props = _columns(records)
        header = (
            f":START_ID({id_spaces[0]})",
            f":END_ID({id_spaces[1]})",
            ":TYPE",
            *_typed(props, records),
        )
        body = tuple(
            (start, end, rel_type, *(record.get(p) for p in props))
            for (start, end, _), record in zip(listed, records, strict=True)
        )
        return Table(file, "relationships", header, body)

    # -- the files ----------------------------------------------------------- #

    def arguments(self, tables: Sequence[Table], *, root: str | Path | None = None) -> list[str]:
        """The `neo4j-admin database import full` options for these files.

        `root` prefixes each file — the directory as the import will see it, a
        container mount say. Without it the paths are bare file names, which is
        what `import.args` holds and why the import runs from the directory.
        """
        prefix = f"{root}/" if root is not None else ""
        return [
            *(f"--{t.kind}={prefix}{t.file}" for t in tables),
            f"--delimiter={_DELIMITERS[self.delimiter]}",
            f"--array-delimiter={_DELIMITERS[self.array_delimiter]}",
            "--multiline-fields=true",
            "--id-type=string",
        ]

    def render(self, table: Table) -> str:
        """One table as CSV text: the header, then a line per row, `\\n`-terminated."""
        typed = [(column, _declared_type(column)) for column in table.columns]
        lines = [self.delimiter.join(self._header_cell(c) for c in table.columns)]
        for row in table.rows:
            cells = (self._cell(v, t, c) for v, (c, t) in zip(row, typed, strict=True))
            lines.append(self.delimiter.join(cells))
        return "\n".join(lines) + "\n"

    def write(self, kg: KnowledgeGraph) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tables = self.tables(kg)
        for table in tables:
            (self.directory / table.file).write_text(self.render(table), encoding="utf-8")
        args = [
            "# Run from this directory:",
            "#   neo4j-admin database import full @import.args <database>",
            *self.arguments(tables),
        ]
        (self.directory / "import.args").write_text("\n".join(args) + "\n", encoding="utf-8")

    def _header_cell(self, column: str) -> str:
        if any(ch in column for ch in (self.delimiter, '"', "\n", "\r")):
            return _quote(column)
        return column

    def _cell(self, value: Any, column_type: str, column: str) -> str:
        if value is None:
            return ""
        if column_type.endswith("[]"):
            element = column_type[:-2]
            items = [_scalar(v, element) for v in value]
            for item in items:
                if self.array_delimiter in item:
                    raise ValueError(
                        f"{column}: {item!r} contains the array delimiter "
                        f"{self.array_delimiter!r}; pass another array_delimiter"
                    )
            return _quote(self.array_delimiter.join(items))
        text = _scalar(value, column_type)
        return _quote(text) if column_type == "string" else text


def _space(type_name: str) -> str:
    """The ID space of one entity type: a plain identifier, distinct per type name."""
    return _schema_name("key", type_name)


def _columns(records: Iterable[dict[str, Any]], *, first: Sequence[str] = ()) -> tuple[str, ...]:
    """Property names in first-seen order, `first` leading; stable for one graph."""
    seen = dict.fromkeys(first)
    for record in records:
        seen.update(dict.fromkeys(record))
    for name in seen:
        if ":" in name:
            raise ValueError(
                f"property {name!r} contains ':', which neo4j-admin reads as a header type"
            )
    return tuple(seen)


def _typed(names: Sequence[str], records: Iterable[dict[str, Any]]) -> list[str]:
    listed = list(records)
    return [f"{name}:{_column_type([r.get(name) for r in listed])}" for name in names]


def _declared_type(column: str) -> str:
    """The type a header cell declares: ids and types are text, `:LABEL` a text array."""
    name, _, declared = column.rpartition(":")
    if declared == "LABEL":
        return "string[]"
    if not name or declared.startswith("ID("):
        return "string"
    return declared


def _kind(value: Any) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "long"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "datetime" if value.tzinfo is not None else "localdatetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time" if value.tzinfo is not None else "localtime"
    return None


def _one(kinds: set[str | None]) -> str:
    if len(kinds) == 1 and None not in kinds:
        return str(next(iter(kinds)))
    if kinds == {"long", "double"}:
        return "double"
    return "string"


def _column_type(values: Sequence[Any]) -> str:
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, list) for v in present):
        elements = {_kind(item) for v in present for item in v}
        return f"{_one(elements) if elements else 'string'}[]"
    if any(isinstance(v, list | dict) for v in present):
        return "string"
    return _one({_kind(v) for v in present}) if present else "string"


def _scalar(value: Any, column_type: str) -> str:
    if column_type == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, date | time):
            return value.isoformat()
        return json.dumps(value, default=str, ensure_ascii=False)
    if column_type == "boolean":
        return "true" if value else "false"
    if column_type == "long":
        return str(int(value))
    if column_type == "double":
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return repr(number)
    return str(value.isoformat())


def _quote(text: str) -> str:
    return '"' + text.replace('"', '""') + '"'


__all__ = ["CypherFileSink", "Neo4jAdminCsvSink", "Table", "cypher_literal"]
