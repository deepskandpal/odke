"""The deterministic proposers: what a corpus says about its own schema, for free.

The design decision behind M5 is that a model should name and rank a schema,
not discover one (`_odke-design` §6, 2026-09-04). Three routes find candidates
first, each exact about what it saw and each citing the characters it saw it in:

- **`RecordShapeProposer` — emergent schema.** A record set is a type, named
  from its file (`people.csv` → `Person`). Each column is a predicate whose
  range is read off the values: all dates, `date`; all integers, `integer`. A
  column whose values are the names another record set is keyed on is an edge
  to that type; a column of capitalised names that recur is an edge to a type
  named after the column. A column that names every row once is the type's key.
- **`HearstProposer` — lexico-syntactic patterns** (Hearst, 1992). "languages
  such as Python and Rust", "cars, trucks and other vehicles", "vehicles
  including cars", "A mathematician is a scientist", "Ada Lovelace was a
  mathematician". A proper name is an instance of the class; a common noun is a
  subclass, and its `is-a` parent is the class.
- **`CooccurrenceProposer` — co-occurrence.** Two known instances of types met
  in one clause, a few words apart, and the words between them are a candidate
  predicate: "Ada Lovelace *works at* Acme Corp" → `works_at: Person → Company`.

Patterns are regular expressions and a stopword list, standard library only.
They are conservative — a noun phrase is a head and at most one modifier, and a
capitalised word next to a pronoun is not trusted as a modifier — because a
proposal that is wrong costs a reviewer's minute, and one that is noise costs
every reviewer that minute.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from openodke.infer.candidates import PredicateCandidate, Proposals, Proposer, TypeCandidate
from openodke.infer.names import (
    NOISE_ADJECTIVES,
    PRONOUNS,
    STOPWORDS,
    fold,
    predicate_name,
    type_name,
    words,
)
from openodke.infer.sample import CorpusSample
from openodke.loaders.records import format_path, leaves
from openodke.types import Chunk, Document, Span


def deterministic_proposers() -> tuple[Proposer, ...]:
    """Records, then Hearst patterns, then co-occurrence over what those two found."""
    return (RecordShapeProposer(), HearstProposer(), CooccurrenceProposer())


def propose(sample: CorpusSample, proposers: Sequence[Proposer] | None = None) -> Proposals:
    """Every proposer in turn, each seeing what the ones before it found."""
    found = Proposals()
    for proposer in deterministic_proposers() if proposers is None else proposers:
        found = found + proposer.propose(sample, found)
    return found


# --------------------------------------------------------------------------- #
# Accumulators
# --------------------------------------------------------------------------- #


def _span(doc: Document, start: int, end: int) -> Span:
    return Span(doc_id=doc.id, start=start, end=end, quote=doc.text[start:end])


@dataclass
class _TypeAcc:
    name: str
    parents: dict[str, None] = field(default_factory=dict)
    keys: dict[str, None] = field(default_factory=dict)
    examples: dict[str, None] = field(default_factory=dict)
    evidence: dict[tuple[str, int, int], Span] = field(default_factory=dict)

    def add(
        self, *, span: Span | None = None, example: str | None = None, parent: str | None = None
    ) -> None:
        if span is not None:
            self.evidence.setdefault((span.doc_id, span.start, span.end), span)
        if example:
            self.examples.setdefault(example)
        if parent and parent != self.name:
            self.parents.setdefault(parent)

    def candidate(self, proposer: str) -> TypeCandidate:
        return TypeCandidate(
            name=self.name,
            proposer=proposer,
            parents=tuple(self.parents),
            keys=tuple(self.keys),
            examples=tuple(self.examples),
            evidence=tuple(self.evidence.values()),
        )


@dataclass
class _PredicateAcc:
    objects: dict[str, dict[str, None]] = field(default_factory=dict)
    examples: dict[str, None] = field(default_factory=dict)
    evidence: dict[tuple[str, int, int], Span] = field(default_factory=dict)

    def add(self, *, subject: str, value: str, example: str, span: Span) -> None:
        self.objects.setdefault(subject, {}).setdefault(value)
        self.examples.setdefault(example)
        self.evidence.setdefault((span.doc_id, span.start, span.end), span)

    def candidate(self, name: str, domain: str, range_: str, proposer: str) -> PredicateCandidate:
        return PredicateCandidate(
            name=name,
            proposer=proposer,
            domain=(domain,),
            range=range_,
            # Observed multiplicity: one subject seen holding two values.
            cardinality="multi" if any(len(v) > 1 for v in self.objects.values()) else "single",
            examples=tuple(self.examples)[:5],
            observations=tuple((s, v) for s, values in self.objects.items() for v in values),
            evidence=tuple(self.evidence.values()),
        )


# --------------------------------------------------------------------------- #
# Records: emergent schema
# --------------------------------------------------------------------------- #

_INTEGER = re.compile(r"[-+]?\d+")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_NAMING_WORDS = frozenset({"name", "title", "label"})


@dataclass(frozen=True, slots=True)
class _Cell:
    value: Any
    span: Span
    multi: bool


@dataclass
class _Row:
    doc: Document
    cells: dict[str, list[_Cell]]


class RecordShapeProposer:
    """Types from record sets, predicates from their columns, ranges from their values.

    `repeat_ratio` is how few distinct values a column of names may have, as a
    share of its cells, to count as recurring references. `overlap` is the share
    of a column's distinct values that must be another record set's key values
    for the column to be an edge to that type.
    """

    name = "records"

    def __init__(self, *, repeat_ratio: float = 0.5, overlap: float = 0.5) -> None:
        self.repeat_ratio = repeat_ratio
        self.overlap = overlap

    def propose(self, sample: CorpusSample, found: Proposals | None = None) -> Proposals:
        groups = _record_groups(sample)
        keys = {label: _identity_column(rows) for label, rows in groups.items()}
        named = {
            label: {
                fold(str(c.value)) for row in rows for c in row.cells.get(keys[label] or "", ())
            }
            for label, rows in groups.items()
        }
        types: dict[str, _TypeAcc] = {}
        predicates: list[PredicateCandidate] = []
        for label in sorted(groups):
            rows, key = groups[label], keys[label]
            acc = types.setdefault(label, _TypeAcc(label))
            if key:
                acc.keys.setdefault(predicate_name(key))
            for row in rows:
                cells = row.cells.get(key) if key else None
                first = cells[0] if cells else next(iter(row.cells.values()))[0]
                acc.add(span=first.span, example=str(first.value) if cells else None)
            for column in _columns(rows):
                candidate = self._column(label, column, rows, key, named, types)
                if candidate is not None:
                    predicates.append(candidate)
        return Proposals(
            types=tuple(acc.candidate(self.name) for acc in types.values()),
            predicates=tuple(predicates),
        )

    def _column(
        self,
        label: str,
        column: str,
        rows: list[_Row],
        key: str | None,
        named: dict[str, set[str]],
        types: dict[str, _TypeAcc],
    ) -> PredicateCandidate | None:
        name = predicate_name(column)
        cells = [c for row in rows for c in row.cells.get(column, ())]
        if not name or not cells:
            return None
        values = [c.value for c in cells]
        multi = any(c.multi for c in cells) or any(len(r.cells.get(column, ())) > 1 for r in rows)
        range_ = _literal_range(values)
        if range_ == "string" and column != key and all(_name_like(v) for v in values):
            target = self._referenced(values, named)
            if target is None and self._recurring(values) and type_name(column):
                target = type_name(column)
                referenced = types.setdefault(target, _TypeAcc(target))
                for cell in cells:
                    referenced.add(span=cell.span, example=str(cell.value))
            range_ = target or range_
        observations: dict[tuple[str, str], None] = {}
        for row in rows:
            subjects = row.cells.get(key) if key else None
            if subjects:
                for cell in row.cells.get(column, ()):
                    observations.setdefault((fold(str(subjects[0].value)), fold(str(cell.value))))
        return PredicateCandidate(
            name=name,
            proposer=self.name,
            domain=(label,),
            range=range_,
            cardinality="multi" if multi else "single",
            examples=tuple(dict.fromkeys(str(v) for v in values))[:5],
            observations=tuple(observations),
            evidence=tuple(c.span for c in cells),
        )

    def _referenced(self, values: list[Any], named: dict[str, set[str]]) -> str | None:
        distinct = {fold(str(v)) for v in values}
        best: tuple[int, str] | None = None
        for label in sorted(named):
            shared = len(distinct & named[label])
            if shared and shared >= self.overlap * len(distinct) and (not best or shared > best[0]):
                best = (shared, label)
        return best[1] if best else None

    def _recurring(self, values: list[Any]) -> bool:
        distinct = {fold(str(v)) for v in values}
        return (
            len(values) >= 2
            and len(distinct) <= self.repeat_ratio * len(values)
            and all(str(v)[:1].isupper() for v in values)
        )


def _record_groups(sample: CorpusSample) -> dict[str, list[_Row]]:
    chunks: dict[str, list[Chunk]] = {}
    for chunk in sample.chunks:
        chunks.setdefault(chunk.doc_id, []).append(chunk)
    groups: dict[str, list[_Row]] = {}
    for doc_id, taken in chunks.items():
        doc = sample.documents[doc_id]
        meta = doc.metadata
        if doc.modality != "structured" or not isinstance(meta.get("row"), dict):
            continue
        if not isinstance(meta.get("fields"), dict):
            continue
        row = _row(doc, taken)
        if row.cells:
            groups.setdefault(_record_label(doc), []).append(row)
    return groups


def _row(doc: Document, taken: list[Chunk]) -> _Row:
    fields = doc.metadata["fields"]
    cells: dict[str, list[_Cell]] = {}
    for steps, value in leaves(doc.metadata["row"]):
        if value is None or value == "" or isinstance(value, dict | list | tuple):
            continue
        at = fields.get(format_path(steps))
        keys = [s for s in steps if isinstance(s, str)]
        if not at or not keys:
            continue
        start, end = at
        if not any(c.start <= start and end <= c.end for c in taken):
            continue
        cell = _Cell(value, _span(doc, start, end), multi=len(keys) != len(steps))
        cells.setdefault(".".join(keys), []).append(cell)
    return _Row(doc, cells)


def _record_label(doc: Document) -> str:
    if doc.uri:
        stem = PurePosixPath(unquote(urlparse(doc.uri).path)).stem
        if type_name(stem):
            return type_name(stem)
    if doc.title and type_name(doc.title):
        return type_name(doc.title)
    return "Record"


def _columns(rows: Iterable[_Row]) -> list[str]:
    return list(dict.fromkeys(column for row in rows for column in row.cells))


def _identity_column(rows: list[_Row]) -> str | None:
    """The column that names each row exactly once, preferring one called a name."""

    def names_each_row(column: str) -> bool:
        seen: set[str] = set()
        for row in rows:
            cells = row.cells.get(column)
            if not cells or len(cells) != 1 or cells[0].multi or not _name_like(cells[0].value):
                return False
            seen.add(fold(str(cells[0].value)))
        return len(seen) == len(rows) and (len(rows) >= 2 or _names_things(column))

    unique = [c for c in _columns(rows) if names_each_row(c)]
    preferred = [c for c in unique if _names_things(c)]
    return next(iter(preferred or unique), None)


def _names_things(column: str) -> bool:
    return bool(_NAMING_WORDS & set(words(column)))


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    text = str(value).strip()
    if text.casefold() in {"true", "false"}:
        return "boolean"
    for pattern, kind in (
        (_INTEGER, "integer"),
        (_NUMBER, "number"),
        (_DATE, "date"),
        (_DATETIME, "datetime"),
    ):
        if pattern.fullmatch(text):
            return kind
    return "string"


def _literal_range(values: Sequence[Any]) -> str:
    """The narrowest literal range every value fits."""
    kinds = {_kind(v) for v in values}
    if len(kinds) == 1:
        return kinds.pop()
    if kinds and kinds <= {"integer", "number"}:
        return "number"
    if kinds and kinds <= {"date", "datetime"}:
        return "datetime"
    return "string"


def _name_like(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _kind(value) == "string"
        and any(ch.isalpha() for ch in value)
        and len(value.split()) <= 6
    )


# --------------------------------------------------------------------------- #
# Hearst patterns
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[^\W_][\w'’&.-]*[^\W_]|[^\W_]|&")
_TRIGGER = re.compile(
    r"\b(?:(?P<forward>such[ \t]+as|including)|(?P<back>(?:and|or)[ \t]+other)"
    r"|(?P<copula>(?:is|was)[ \t]+an?))\b",
    re.IGNORECASE,
)
_BOUNDARY = re.compile(r"[.;:!?()\[\]\n\r\"“”]")
_SEPARATOR = re.compile(r",|\b(?:and|or)\b")
_CONNECTORS = frozenset({"&", "of", "de", "da", "del", "van", "von", "der", "la", "le"})


@dataclass(frozen=True, slots=True)
class _Tok:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class _Phrase:
    start: int
    end: int
    text: str
    proper: bool


class HearstProposer:
    """Classes, subclasses and instances from "X such as Y" and its relatives."""

    name = "hearst"

    def propose(self, sample: CorpusSample, found: Proposals | None = None) -> Proposals:
        types: dict[str, _TypeAcc] = {}
        for chunk in sample.chunks:
            doc = sample.document(chunk)
            if doc.modality == "structured":
                continue
            for hyper, items in _hearst(chunk.text):
                self._emit(types, doc, chunk, hyper, items)
        return Proposals(types=tuple(acc.candidate(self.name) for acc in types.values()))

    def _emit(
        self,
        types: dict[str, _TypeAcc],
        doc: Document,
        chunk: Chunk,
        hyper: _Phrase,
        items: list[_Phrase],
    ) -> None:
        parent = type_name(hyper.text)
        if not _usable(parent):
            return
        low = min(hyper.start, *(i.start for i in items))
        high = max(hyper.end, *(i.end for i in items))
        span = _span(doc, chunk.start + low, chunk.start + high)
        types.setdefault(parent, _TypeAcc(parent)).add(span=span)
        for item in items:
            if item.proper:
                types[parent].add(example=item.text)
                continue
            child = type_name(item.text)
            if _usable(child) and child != parent:
                types.setdefault(child, _TypeAcc(child)).add(span=span, parent=parent)


def _hearst(text: str) -> Iterator[tuple[_Phrase, list[_Phrase]]]:
    """`(class, members)` for every pattern that matched, in text order."""
    for trigger in _TRIGGER.finditer(text):
        left = max((m.end() for m in _BOUNDARY.finditer(text, 0, trigger.start())), default=0)
        stop = _BOUNDARY.search(text, trigger.end())
        right = stop.start() if stop else len(text)
        before = _tokens(text, left, trigger.start())
        after = _tokens(text, trigger.end(), right)
        hyper: _Phrase | None
        if trigger.lastgroup == "forward":
            hyper = _common_back(text, before, head_only=False)
            items = _list_forward(text, trigger.end(), right)
        elif trigger.lastgroup == "back":
            hyper = _common_forward(text, after)[0]
            items = _list_back(text, left, trigger.start())
        else:
            hyper = _copula_class(text, after)
            subject = _copula_subject(text, before)
            items = [subject] if subject else []
        if hyper is not None and items:
            yield hyper, items


def _tokens(text: str, start: int, end: int) -> list[_Tok]:
    return [_Tok(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text, start, end)]


def _phrase(text: str, toks: Sequence[_Tok], *, proper: bool) -> _Phrase:
    return _Phrase(toks[0].start, toks[-1].end, text[toks[0].start : toks[-1].end], proper)


def _word(tok: _Tok) -> bool:
    """A lower-case word that is not a stopword: the stuff of a common noun phrase."""
    lowered = tok.text.casefold()
    return (
        tok.text[0].islower()
        and tok.text.replace("-", "").replace("'", "").isalpha()
        and lowered not in STOPWORDS
    )


def _upper(tok: _Tok) -> bool:
    return tok.text[0].isupper()


def _usable(name: str) -> bool:
    return len(name) >= 3 and name[0].isalpha()


def _common_back(text: str, toks: Sequence[_Tok], *, head_only: bool) -> _Phrase | None:
    """The common noun phrase ending `toks`: a head, and one modifier if it is safe."""
    if not toks or not _word(toks[-1]) or toks[-1].text.casefold() in NOISE_ADJECTIVES:
        return None
    chosen = [toks[-1]]
    if not head_only and len(toks) >= 2:
        modifier = toks[-2]
        lowered = modifier.text.casefold()
        # A modifier is trusted only after a determiner, a preposition or nothing:
        # "about programming languages", not "they saw cars".
        before = toks[-3].text.casefold() if len(toks) >= 3 else None
        introduced = before is None or (before in STOPWORDS and before not in PRONOUNS)
        if (
            _word(modifier)
            and introduced
            and lowered not in NOISE_ADJECTIVES
            and not lowered.endswith(("ed", "ly", "s"))
        ):
            chosen.insert(0, modifier)
    return _phrase(text, chosen, proper=False)


def _common_forward(text: str, toks: Sequence[_Tok]) -> tuple[_Phrase | None, int]:
    chosen: list[_Tok] = []
    for tok in toks[:2]:
        if not _word(tok) or tok.text.casefold() in NOISE_ADJECTIVES:
            break
        chosen.append(tok)
    if not chosen:
        return None, 0
    # "programming languages exist" keeps its plural head; "cars exist" does not
    # take the verb.
    if len(chosen) == 2:
        first, second = (t.text.casefold() for t in chosen)
        if first.endswith("s") and not second.endswith("s"):
            chosen = chosen[:1]
    return _phrase(text, chosen, proper=False), len(chosen)


def _proper_back(text: str, toks: Sequence[_Tok]) -> _Phrase | None:
    if not toks or not _upper(toks[-1]):
        return None
    start, k = len(toks) - 1, len(toks) - 2
    while k >= 0:
        if _upper(toks[k]):
            start, k = k, k - 1
        elif toks[k].text.casefold() in _CONNECTORS and k >= 1 and _upper(toks[k - 1]):
            start, k = k - 1, k - 2
        else:
            break
    run = list(toks[start:])
    while run and run[0].text.casefold() in STOPWORDS:
        run.pop(0)
    return _phrase(text, run, proper=True) if run else None


def _proper_forward(text: str, toks: Sequence[_Tok]) -> tuple[_Phrase | None, int]:
    if not toks or not _upper(toks[0]):
        return None, 0
    end, k = 0, 1
    while k < len(toks):
        if _upper(toks[k]):
            end, k = k, k + 1
        elif toks[k].text.casefold() in _CONNECTORS and k + 1 < len(toks) and _upper(toks[k + 1]):
            end, k = k + 1, k + 2
        else:
            break
    run = list(toks[: end + 1])
    while run and run[0].text.casefold() in STOPWORDS:
        run.pop(0)
    return (_phrase(text, run, proper=True) if run else None), end + 1


def _segments(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cursor = start
    for sep in _SEPARATOR.finditer(text, start, end):
        out.append((cursor, sep.start()))
        cursor = sep.end()
    out.append((cursor, end))
    return out


def _list_forward(text: str, start: int, end: int) -> list[_Phrase]:
    """ "Python, Rust and Go are…" → Python, Rust, Go: items until one runs on."""
    items: list[_Phrase] = []
    for s, e in _segments(text, start, end):
        toks = _tokens(text, s, e)
        if not toks:
            continue
        phrase, used = _proper_forward(text, toks)
        if phrase is None and used == 0:
            phrase, used = _common_forward(text, toks)
        if phrase is None:
            break
        items.append(phrase)
        if used < len(toks):
            break
    return items


def _list_back(text: str, start: int, end: int) -> list[_Phrase]:
    """ "They saw cars, trucks" (and other…) → cars, trucks: items until one runs on."""
    items: list[_Phrase] = []
    for s, e in reversed(_segments(text, start, end)):
        toks = _tokens(text, s, e)
        if not toks:
            continue
        phrase = _proper_back(text, toks) or _common_back(text, toks, head_only=len(toks) > 2)
        if phrase is None:
            break
        items.append(phrase)
        if toks[0].start < phrase.start:
            break
    return items[::-1]


def _copula_subject(text: str, toks: Sequence[_Tok]) -> _Phrase | None:
    """ "Ada Lovelace (was a…)" is an instance; "A mathematician (is a…)" is a class."""
    proper = _proper_back(text, toks)
    if proper is not None:
        return proper
    if 2 <= len(toks) <= 3 and toks[0].text.casefold() in {"a", "an"}:
        rest = toks[1:]
        if all(_word(t) for t in rest):
            return _phrase(text, rest, proper=False)
    return None


def _copula_class(text: str, toks: Sequence[_Tok]) -> _Phrase | None:
    """ "(is an) English mathematician and writer" → mathematician."""
    k = 0
    while k < min(2, len(toks)) and (
        _upper(toks[k]) or toks[k].text.casefold() in NOISE_ADJECTIVES
    ):
        k += 1
    chosen: list[_Tok] = []
    for tok in toks[k : k + 2]:
        if not _word(tok) or tok.text.casefold() in NOISE_ADJECTIVES:
            break
        chosen.append(tok)
    if len(chosen) == 2 and chosen[1].text.casefold().endswith(("ed", "ing", "ly")):
        chosen = chosen[:1]
    return _phrase(text, chosen, proper=False) if chosen else None


# --------------------------------------------------------------------------- #
# Co-occurrence
# --------------------------------------------------------------------------- #

_GAP_BREAK = re.compile(r"[.;:!?,()\[\]\n\r\"“”]")
_AUXILIARIES = frozenset({"is", "are", "was", "were", "be", "been", "has", "have", "had"})
_DETERMINERS = frozenset({"a", "an", "the"})


class CooccurrenceProposer:
    """Predicates from the words between two known instances in one clause.

    Instances are the examples earlier proposers found — a record set's key
    values, the proper names a Hearst pattern classed. Only adjacent mentions
    are paired, at most `max_gap_words` apart, with nothing but lower-case
    words between them: a comma, a full stop or another capitalised name means
    the two were not in one relation.
    """

    name = "cooccurrence"

    def __init__(self, *, max_gap_words: int = 4) -> None:
        self.max_gap_words = max_gap_words

    def propose(self, sample: CorpusSample, found: Proposals | None = None) -> Proposals:
        lexicon = _lexicon((found or Proposals()).types)
        if not lexicon:
            return Proposals()
        names = sorted(lexicon, key=lambda n: (-len(n), n))
        mention = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, names)) + r")(?!\w)")
        seen: dict[tuple[str, str, str], _PredicateAcc] = {}
        for chunk in sample.chunks:
            doc = sample.document(chunk)
            if doc.modality == "structured":
                continue
            for a, b in pairwise(mention.finditer(chunk.text)):
                if fold(a.group()) == fold(b.group()):
                    continue
                phrase = self._phrase(chunk.text[a.end() : b.start()])
                if phrase is None:
                    continue
                key = (predicate_name(phrase), lexicon[a.group()], lexicon[b.group()])
                seen.setdefault(key, _PredicateAcc()).add(
                    subject=fold(a.group()),
                    value=fold(b.group()),
                    example=b.group(),
                    span=_span(doc, chunk.start + a.start(), chunk.start + b.end()),
                )
        return Proposals(
            predicates=tuple(
                acc.candidate(name, domain, range_, self.name)
                for (name, domain, range_), acc in seen.items()
            )
        )

    def _phrase(self, between: str) -> str | None:
        if _GAP_BREAK.search(between):
            return None
        parts = between.split()
        if not 1 <= len(parts) <= self.max_gap_words:
            return None
        if not all(p.replace("-", "").isalpha() and p[0].islower() for p in parts):
            return None
        lowered = [p.casefold() for p in parts]
        while len(lowered) > 1 and lowered[0] in _AUXILIARIES | _DETERMINERS:
            lowered.pop(0)
        while lowered and lowered[-1] in _DETERMINERS:
            lowered.pop()
        if not lowered or all(w in STOPWORDS for w in lowered):
            return None
        return " ".join(lowered)


def _lexicon(types: Iterable[TypeCandidate]) -> dict[str, str]:
    """Instance name → the best-supported type it was seen as."""
    out: dict[str, str] = {}
    for candidate in sorted(types, key=lambda t: (-t.support, t.name)):
        for example in candidate.examples:
            if len(example) >= 2 and example[0].isupper() and example.casefold() not in STOPWORDS:
                out.setdefault(example, candidate.name)
    return out


__all__ = [
    "CooccurrenceProposer",
    "HearstProposer",
    "RecordShapeProposer",
    "deterministic_proposers",
    "propose",
]
