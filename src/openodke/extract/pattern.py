"""The extractor that needs no model: records, pipe tables and `Key: value` blocks.

Exact, free and deterministic — and the paper's hybrid design exists because
this is strictly better than a model wherever it applies. Three shapes:

- **Records.** A `structured` document from `openodke.loaders` carries its row and
  the offsets of every rendered cell. A `RecordMapping` says which paths feed
  which predicates; with none, field names are matched against the ontology.
  These are the highest-precision facts the pipeline ever sees.
- **Pipe tables.** Header cells are matched to predicate names, labels and
  aliases, case- and punctuation-blind; each body row is one entity.
- **`Key: value` blocks.** Consecutive lines — an infobox, a spec sheet — are
  one entity, named by the line holding an identity predicate or else by the
  heading above the block.

Every fact cites the exact cell or value it came from, checked with
`Span.is_faithful`. A subject the source does not name is never invented: a
row or block with no identifiable subject yields nothing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openodke._text import iter_lines, trim
from openodke.extract._common import (
    ChunkContext,
    Documents,
    PredicateNames,
    applies,
    best_type,
    index_documents,
    key_predicates,
    subject_entity,
)
from openodke.loaders.records import (
    Step,
    format_path,
    leaves,
    parse_path,
    render_value,
    resolve_path,
)
from openodke.loaders.text import markdown_headings
from openodke.ontology import Ontology, Predicate
from openodke.types import Chunk, Fact

_KEY_VALUE = re.compile(
    r"[ \t]*(?:[-*+][ \t]+)?(?:\*\*|__)?"
    r"(?P<key>[^\s:|*_][^:|\r\n]{0,60}?)"
    r"(?:\*\*|__)?[ \t]*:(?:\*\*|__)?[ \t]+"
    r"(?P<value>\S(?:.*\S)?)[ \t]*$"
)
_TABLE_RULE = re.compile(r"[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$")


class RecordMapping(BaseModel):
    """Which paths of a record feed which predicates, for one entity type.

    Data rather than code, so it lives beside the ontology as JSON and changes
    without a release:

        {"subject_type": "Person", "subject": "person.name",
         "predicates": {"full_name": "person.name", "employer": "jobs[*].company"}}

    `subject` is the path whose value names the entity; left out, the type's
    identity predicates (`EntityType.keys`) name it. `predicates` left empty
    matches the record's field names against the ontology instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_type: str
    subject: str | None = None
    predicates: dict[str, str] = Field(default_factory=dict)


class PatternExtractor:
    """Facts from structure alone, with `extractor="pattern"` and zero model calls.

    `documents` is the lookup a structured chunk needs to reach its row
    (DECISIONS #19); a chunk whose document is not registered is still read
    for tables and `Key: value` lines. `subject_type` pins the type of every
    table and block instead of inferring it from which predicates matched.

    `confidence` defaults to 1.0: the value is exactly what the source says
    under that column. It is a prior for the scorer, and a caller who trusts
    header matching less than an explicit mapping can lower it.
    """

    name = "pattern"

    def __init__(
        self,
        *,
        mappings: Sequence[RecordMapping] = (),
        documents: Documents = None,
        subject_type: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        self.mappings = tuple(mappings)
        self.documents = index_documents(documents)
        self.subject_type = subject_type
        self.confidence = confidence

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        ctx = ChunkContext(chunk, self.documents.get(chunk.doc_id))
        doc = ctx.doc
        if doc is not None and doc.modality == "structured" and _is_record(doc.metadata):
            return self._records(ctx, ontology)
        return self._text(ctx, ontology)

    # ------------------------------------------------------------------ #
    # Records
    # ------------------------------------------------------------------ #

    def _records(self, ctx: ChunkContext, ontology: Ontology) -> list[Fact]:
        assert ctx.doc is not None
        row, fields = ctx.doc.metadata["row"], ctx.doc.metadata["fields"]
        names = PredicateNames(ontology)
        facts: list[Fact] = []
        for mapping in self.mappings or (None,):
            pairs = _mapped(row, mapping, ontology) if mapping and mapping.predicates else []
            if not (mapping and mapping.predicates):
                for steps, value in leaves(row):
                    predicate = _field_predicate(names, steps)
                    if predicate is not None:
                        pairs.append((predicate, format_path(steps), value))
            type_name = (
                mapping.subject_type
                if mapping
                else best_type(ontology, [p for p, _, _ in pairs], self.subject_type)
            )
            if type_name is None:
                continue
            if not (mapping and mapping.predicates):
                pairs = [pair for pair in pairs if applies(ontology, pair[0], type_name)]
            name = _record_subject(row, mapping, pairs, ontology, type_name)
            if name is None:
                continue
            subject = subject_entity(type_name, name)
            for predicate, path, value in pairs:
                if _blank(value) or path not in fields:
                    continue
                start, end = fields[path]
                # A cell in another chunk is that chunk's fact.
                if start < ctx.chunk.start or end > ctx.chunk.end:
                    continue
                span = ctx.span(start - ctx.chunk.start, render_value(value))
                if span is not None:
                    facts.append(self._fact(ctx, ontology, subject, predicate, value, span))
        return facts

    # ------------------------------------------------------------------ #
    # Tables and key-value blocks
    # ------------------------------------------------------------------ #

    def _text(self, ctx: ChunkContext, ontology: Ontology) -> list[Fact]:
        text = ctx.chunk.text
        names = PredicateNames(ontology)
        lines = list(iter_lines(text))
        facts: list[Fact] = []
        block: list[tuple[str, int, int]] = []
        heading: str | None = None
        block_heading: str | None = None
        i = 0
        while i < len(lines):
            start, end = lines[i]
            table_end = _table_end(text, lines, i)
            kv = None if table_end else _KEY_VALUE.match(text, start, end)
            if kv is not None:
                if not block:
                    block_heading = heading
                block.append((kv.group("key"), kv.start("value"), kv.end("value")))
                i += 1
                continue
            facts += self._block(ctx, ontology, names, block, block_heading)
            block = []
            if table_end:
                facts += self._table(ctx, ontology, names, lines[i:table_end])
                i = table_end
                continue
            found = markdown_headings(text[start:end])
            if found:
                heading = found[0]["text"]
            i += 1
        facts += self._block(ctx, ontology, names, block, block_heading)
        return facts

    def _table(
        self,
        ctx: ChunkContext,
        ontology: Ontology,
        names: PredicateNames,
        rows: list[tuple[int, int]],
    ) -> list[Fact]:
        text = ctx.chunk.text
        columns = [names.match(text[s:e]) for s, e in _cells(text, *rows[0])]
        type_name = best_type(ontology, [p for p in columns if p], self.subject_type)
        if type_name is None:
            return []
        keys = key_predicates(ontology, type_name)
        subject_column = next(
            (i for key in keys for i, p in enumerate(columns) if p is not None and p.name == key),
            0,
        )
        facts: list[Fact] = []
        for row_start, row_end in rows[2:]:
            cells = _cells(text, row_start, row_end)
            if subject_column >= len(cells) or cells[subject_column][0] == cells[subject_column][1]:
                continue
            subject = subject_entity(type_name, text[slice(*cells[subject_column])])
            for (cell_start, cell_end), predicate in zip(cells, columns, strict=False):
                if predicate is None or cell_start == cell_end:
                    continue
                if not applies(ontology, predicate, type_name):
                    continue
                value = text[cell_start:cell_end]
                span = ctx.span(cell_start, value)
                if span is not None:
                    facts.append(self._fact(ctx, ontology, subject, predicate, value, span))
        return facts

    def _block(
        self,
        ctx: ChunkContext,
        ontology: Ontology,
        names: PredicateNames,
        block: list[tuple[str, int, int]],
        heading: str | None,
    ) -> list[Fact]:
        text = ctx.chunk.text
        matched = [(names.match(key), s, e) for key, s, e in block]
        found = [(p, s, e) for p, s, e in matched if p is not None]
        type_name = best_type(ontology, [p for p, _, _ in found], self.subject_type)
        if type_name is None:
            return []
        found = [m for m in found if applies(ontology, m[0], type_name)]
        keys = key_predicates(ontology, type_name)
        name = next(
            (text[s:e] for key in keys for p, s, e in found if p.name == key),
            heading,
        )
        if not name:
            return []
        subject = subject_entity(type_name, name)
        facts: list[Fact] = []
        for predicate, start, end in found:
            value = text[start:end]
            span = ctx.span(start, value)
            if span is not None:
                facts.append(self._fact(ctx, ontology, subject, predicate, value, span))
        return facts

    def _fact(
        self,
        ctx: ChunkContext,
        ontology: Ontology,
        subject: Any,
        predicate: Predicate,
        value: Any,
        span: Any,
    ) -> Fact:
        return ctx.fact(
            ontology,
            subject=subject,
            predicate=predicate,
            value=value,
            span=span,
            extractor=self.name,
            confidence=self.confidence,
        )


def _is_record(metadata: Mapping[str, Any]) -> bool:
    return isinstance(metadata.get("row"), Mapping) and isinstance(metadata.get("fields"), Mapping)


def _blank(value: Any) -> bool:
    return value is None or value == "" or isinstance(value, Mapping | list | tuple)


def _field_predicate(names: PredicateNames, steps: Sequence[Step]) -> Predicate | None:
    # `address.city` may be the predicate, or `city` may; `jobs[0]` is `jobs`.
    keys = [s for s in steps if isinstance(s, str)]
    if not keys:
        return None
    return names.match(".".join(keys)) or names.match(keys[-1])


def _mapped(
    row: Mapping[str, Any], mapping: RecordMapping, ontology: Ontology
) -> list[tuple[Predicate, str, Any]]:
    pairs: list[tuple[Predicate, str, Any]] = []
    for name, path in mapping.predicates.items():
        predicate = ontology.predicates.get(name)
        if predicate is None:
            raise ValueError(
                f"a RecordMapping maps {path!r} to {name!r}, which ontology "
                f"{ontology.name!r} does not declare"
            )
        for steps, value in resolve_path(row, parse_path(path)):
            pairs.append((predicate, format_path(steps), value))
    return pairs


def _record_subject(
    row: Mapping[str, Any],
    mapping: RecordMapping | None,
    pairs: list[tuple[Predicate, str, Any]],
    ontology: Ontology,
    type_name: str,
) -> str | None:
    if mapping is not None and mapping.subject:
        values = [v for _, v in resolve_path(row, parse_path(mapping.subject)) if not _blank(v)]
        return str(values[0]) if values else None
    for key in key_predicates(ontology, type_name):
        for predicate, _, value in pairs:
            if predicate.name == key and not _blank(value):
                return str(value)
    return None


def _table_end(text: str, lines: list[tuple[int, int]], i: int) -> int:
    """Index just past the pipe table whose header is line `i`, or 0 if there is none."""
    if i + 1 >= len(lines):
        return 0
    (head_start, head_end), (rule_start, rule_end) = lines[i], lines[i + 1]
    if "|" not in text[head_start:head_end] or "|" not in text[rule_start:rule_end]:
        return 0
    if not _TABLE_RULE.match(text, rule_start, rule_end):
        return 0
    if len(_cells(text, head_start, head_end)) != len(_cells(text, rule_start, rule_end)):
        return 0
    end = i + 2
    while end < len(lines) and "|" in text[slice(*lines[end])] and text[slice(*lines[end])].strip():
        end += 1
    return end


def _cells(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Trimmed cell offsets in one pipe-table row. An escaped `\\|` stays in its cell."""
    start, end = trim(text, start, end)
    if start < end and text[start] == "|":
        start += 1
    if end > start and text[end - 1] == "|" and (end - 2 < start or text[end - 2] != "\\"):
        end -= 1
    cells: list[tuple[int, int]] = []
    cell_start = k = start
    while k < end:
        if text[k] == "\\":
            k += 2
            continue
        if text[k] == "|":
            cells.append(trim(text, cell_start, k))
            cell_start = k + 1
        k += 1
    cells.append(trim(text, cell_start, end))
    return cells


__all__ = ["PatternExtractor", "RecordMapping"]
