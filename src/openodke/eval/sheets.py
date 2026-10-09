"""Label sheets: markdown a person ticks, read back into the eval label formats.

`odke eval` scores against labels, and someone has to make them. A sheet is
plain markdown (headings, a quoted passage, three checkboxes), so it can be
labelled wherever a markdown file opens, a tablet included, one sheet a
sitting. Everything an item needs to become a label row is kept in a sidecar,
`<sheet>.items.jsonl`, beside the sheet; the markdown carries only what a
person adds to it, which is one tick per item and an optional note.

Two kinds: `grounding` sheets read back as `GroundingLabel` rows, and `pair`
sheets as `PairLabel` rows, with every pair answered `unsure` kept out of them.

The sheet shows a grounding claim exactly as `render_claim` puts it to the
grounder, so the person judges what the model judged. Passage text is escaped
so that markdown cannot hide or restyle it: a `$` stays a dollar sign rather
than opening a formula.

Reading back is strict where a slip would corrupt the labels and lenient where
it would not. An item with no tick is unlabelled, counted and reported. An item
with two ticks, or a heading the sidecar does not know, is an error naming the
file and the line of the item's heading, and nothing is written until every
sheet reads cleanly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openodke.eval.formats import GroundingLabel, PairLabel
from openodke.ground.llm import render_claim
from openodke.ground.span import located
from openodke.types import Document, Fact, Frozen

GLOB = "sheet-*.md"
NOTE = "note:"


class GroundingItem(Frozen):
    """One row of `odke label make grounding`: a `GroundingLabel` before its verdict.

    The fact, the text its evidence span indexes into, and optionally the
    `doc_id` the span points at, exactly as on a `GroundingLabel`. A line of a
    grounding labels file with its `verdict` removed is one. Give the fact an
    `id`: predictions are joined on it.
    """

    text: str
    fact: Fact
    doc_id: str | None = None

    def document(self) -> Document:
        """The document the grounder would be handed, as `GroundingLabel.as_document`."""
        cited = next((e.doc_id for e in self.fact.evidence), None)
        return Document(id=self.doc_id or cited or self.fact.id, text=self.text)


@dataclass(frozen=True)
class Kind:
    """What one kind of sheet asks, and how it shows an item."""

    name: str
    prefix: str
    title: str
    item: type[BaseModel]
    question: str
    explain: tuple[str, str]
    boxes: dict[str, str]  # the box as written → the answer it records
    render: Callable[[Any], list[str]]
    output: type[BaseModel]
    label: Callable[[dict[str, Any], str], dict[str, Any]]
    # The answer that is never a label: its rows go to a file of their own.
    apart: str | None = None


# Markdown, and Obsidian on top of it, would hide or restyle text containing
# these: `$` opens a formula, `%%` a comment, `==` a highlight, `<` HTML.
_SPECIAL = re.compile(r"([\\`*_\[\]<>#$%=~^|])")


# Every line break but `\n`. A reader may break a line on any of them, which
# would move a passage out of its quote; each becomes a space, one for one, so
# span offsets still hold.
_BREAKS = re.compile("[\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")


def plain(text: str) -> str:
    """`text` on one line, with every character markdown would act on escaped."""
    return _SPECIAL.sub(r"\\\1", _BREAKS.sub(" ", text.replace("\n", " ")))


def quote(text: str, start: int | None = None, end: int | None = None) -> list[str]:
    """`text` as blockquote lines, with the characters in `[start, end)` in bold.

    Bold is applied per line, so a span that crosses a line break stays bold on
    both sides of it, and surrounding whitespace is kept outside the markers,
    where markdown still reads them as bold.
    """
    out = []
    at = 0
    for line in text.split("\n"):
        lo, hi = at, at + len(line)
        at = hi + 1
        rendered = plain(line)
        if start is not None and end is not None and start < hi and end > lo:
            s, e = max(start, lo) - lo, min(end, hi) - lo
            core = line[s:e].strip()
            if core:
                s += len(line[s:e]) - len(line[s:e].lstrip())
                e = s + len(core)
                rendered = f"{plain(line[:s])}**{plain(core)}**{plain(line[e:])}"
        out.append(f"> {rendered}".rstrip())
    return out


def _grounding(item: GroundingItem) -> list[str]:
    claim = render_claim(item.fact)
    evidence = located(item.fact, item.document())
    span = evidence.span if evidence is not None else None
    passage = quote(item.text, span.start, span.end) if span else quote(item.text)
    return [plain(claim), "", *passage]


GROUNDING = Kind(
    name="grounding",
    prefix="G",
    title="Grounding sheet",
    item=GroundingItem,
    question="does the passage alone support the claim?",
    explain=(
        "supported: it says so, or plainly implies it.",
        "contradicted: it says otherwise. not found: it does not say.",
    ),
    boxes={"supported": "supported", "contradicted": "contradicted", "not found": "not_found"},
    render=_grounding,
    output=GroundingLabel,
    label=lambda row, answer: {**row, "verdict": answer},
)


class Mention(Frozen):
    """One side of a pair: an entity key and its type, and how and where it was written.

    `label` is the name as the text has it (the key is shown when it is
    absent); `context` is the passage it appeared in.
    """

    key: str
    type: str
    label: str | None = None
    context: str | None = None


class PairItem(Frozen):
    """One row of `odke label make pair`: two mentions, one entity or two?

    `same` and `different` read back as a `PairLabel` on the two keys.
    `unsure` never does: those rows go to a file of their own, in this shape,
    ready to be made into a sheet again.
    """

    a: Mention
    b: Mention


def _side(name: str, mention: Mention) -> list[str]:
    shown = mention.label or mention.key
    lines = [plain(f"{name}: {shown} ({mention.type})")]
    if mention.context:
        found = re.search(re.escape(shown), mention.context, re.IGNORECASE)
        span = (found.start(), found.end()) if found else (None, None)
        lines += ["", *quote(mention.context, *span)]
    return lines


def _pair(item: PairItem) -> list[str]:
    return [*_side("A", item.a), "", *_side("B", item.b)]


PAIR = Kind(
    name="pair",
    prefix="P",
    title="Pair sheet",
    item=PairItem,
    question="do A and B name the same thing?",
    explain=(
        "same: one thing in the world. different: two things.",
        "unsure: cannot tell. Kept apart, never a label.",
    ),
    boxes={"same": "same", "different": "different", "unsure": "unsure"},
    render=_pair,
    output=PairLabel,
    label=lambda row, answer: {
        "a": row["a"]["key"],
        "b": row["b"]["key"],
        "same": answer == "same",
    },
    apart="unsure",
)
KINDS = {kind.name: kind for kind in (GROUNDING, PAIR)}


def _kind(name: str) -> Kind:
    if name not in KINDS:
        raise ValueError(f"unknown sheet kind {name!r}; one of {', '.join(KINDS)}")
    return KINDS[name]


def _rows(path: Path, kind: Kind) -> list[tuple[dict[str, Any], BaseModel]]:
    """Every non-blank line of `path`, as parsed and as `kind.item`, or an error naming it."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append((raw, kind.item.model_validate(raw)))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a {kind.name} item: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no items")
    return rows


def render_sheet(kind: Kind, title: str, items: list[tuple[str, BaseModel]]) -> str:
    """One sheet's markdown: a short header, then each item and its boxes."""
    lines = [
        f"# {title}",
        "",
        f"{items[0][0]} to {items[-1][0]}: {kind.question}",
        "",
        *kind.explain,
        "",
        "Tick exactly one box per item. No tick skips it.",
    ]
    for item_id, item in items:
        boxes = [f"- [ ] {box}" for box in kind.boxes]
        lines += ["", f"### {item_id}", "", *kind.render(item), "", *boxes, "", NOTE]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Made:
    """What `make_sheets` wrote, and what is worth knowing before labelling it."""

    sheets: list[Path]
    items: int
    first: str
    last: str
    warnings: list[str]


def make_sheets(kind: str, items: Path, out: Path, *, per_sheet: int = 50) -> Made:
    """Write `items` (JSONL) into `out` as numbered sheets of `per_sheet` items.

    Item ids run across the sheets (`G-0001`, `G-0002`, …) and each sheet gets
    its sidecar. The same input writes byte-identical files. A directory that
    already holds sheets is refused rather than overwritten: those may be
    hours of ticks.
    """
    spec = _kind(kind)
    if per_sheet < 1:
        raise ValueError(f"per_sheet must be at least 1, got {per_sheet}")
    rows = _rows(Path(items), spec)
    out = Path(out)
    if out.is_dir() and any(out.glob("sheet-*")):
        raise ValueError(f"{out} already has sheets; write to an empty directory")
    width = max(4, len(str(len(rows))))
    ids = [f"{spec.prefix}-{n:0{width}d}" for n in range(1, len(rows) + 1)]
    chunks = [range(i, min(i + per_sheet, len(rows))) for i in range(0, len(rows), per_sheet)]
    digits = max(2, len(str(len(chunks))))
    out.mkdir(parents=True, exist_ok=True)
    sheets = []
    for number, chunk in enumerate(chunks, start=1):
        name = f"sheet-{number:0{digits}d}"
        markdown = render_sheet(
            spec, f"{spec.title} {number:0{digits}d}", [(ids[i], rows[i][1]) for i in chunk]
        )
        sidecar = "".join(
            json.dumps({"id": ids[i], "kind": spec.name, "row": rows[i][0]}, ensure_ascii=False)
            + "\n"
            for i in chunk
        )
        (out / f"{name}.md").write_text(markdown, encoding="utf-8", newline="\n")
        (out / f"{name}.items.jsonl").write_text(sidecar, encoding="utf-8", newline="\n")
        sheets.append(out / f"{name}.md")
    warnings = []
    if spec is GROUNDING:
        missing = sum("id" not in raw.get("fact", {}) for raw, _ in rows)
        if missing:
            warnings.append(
                f"{missing} of {len(rows)} facts have no id; predictions are joined on "
                "Fact.id, so these labels can only score a grounder run over them in-process"
            )
    return Made(sheets=sheets, items=len(rows), first=ids[0], last=ids[-1], warnings=warnings)


# --------------------------------------------------------------------------- #
# Reading back
# --------------------------------------------------------------------------- #


class SheetError(ValueError):
    """Every problem found in the sheets, one `path:line: …` per line of the message."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


_HEADING = re.compile(r"^###\s+(\S+)\s*$")
_BOX = re.compile(r"^\s*[-*+]\s+\[(.)\]\s+(.*?)\s*$")


@dataclass
class _Marked:
    """What a person left on one item: the heading's line, the ticked boxes, the note."""

    line: int
    ticked: list[str]
    note: list[str] | None = None


def _sidecar(path: Path) -> list[tuple[str, Kind, dict[str, Any]]]:
    if not path.is_file():
        raise SheetError([f"{path}: missing; it holds the items its sheet was made from"])
    entries = []
    # Split on "\n" alone: a row may hold a raw U+2028, which splitlines() breaks on.
    lines = path.read_text(encoding="utf-8").split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            entries.append((str(entry["id"]), _kind(entry["kind"]), dict(entry["row"])))
        except (ValueError, KeyError, TypeError) as exc:
            raise SheetError([f"{path}:{number}: not a sidecar row: {exc}"]) from exc
    return entries


def _marks(sheet: Path, known: set[str], kind: Kind, problems: list[str]) -> dict[str, _Marked]:
    """Each item heading in `sheet`, and the boxes ticked and the note written under it."""
    found: dict[str, _Marked] = {}
    current: _Marked | None = None
    # Lines as an editor numbers them: "\n" ends one, nothing else does.
    lines = [line.removesuffix("\r") for line in sheet.read_text(encoding="utf-8").split("\n")]
    for number, line in enumerate(lines, start=1):
        heading = _HEADING.match(line)
        if heading:
            item_id, current = heading.group(1), None
            if item_id not in known:
                sidecar = sheet.with_suffix(".items.jsonl").name
                problems.append(f"{sheet}:{number}: {item_id} is not in {sidecar}")
            elif item_id in found:
                problems.append(f"{sheet}:{number}: {item_id} appears twice")
            else:
                current = found[item_id] = _Marked(number, [])
            continue
        if current is None:
            continue
        box = _BOX.match(line)
        if box:
            mark, text = box.groups()
            name = next((b for b in kind.boxes if text == b or text.startswith(f"{b} ")), None)
            if name is None:
                problems.append(f"{sheet}:{number}: no box is called {text!r}")
            elif mark in "xX":
                current.ticked.append(name)
            elif mark != " ":
                problems.append(f"{sheet}:{number}: [{mark}] is neither [x] nor [ ]")
        elif current.note is not None:
            current.note.append(line)
        elif line.startswith(NOTE):
            current.note = [line[len(NOTE) :]]
    for item_id, marked in found.items():
        if len(marked.ticked) > 1:
            problems.append(
                f"{sheet}:{marked.line}: {item_id} has {len(marked.ticked)} boxes ticked "
                f"({', '.join(marked.ticked)}); tick exactly one"
            )
    return found


@dataclass(frozen=True)
class Reading:
    """What a set of sheets says, ready to write as JSONL."""

    kind: Kind
    sheets: list[Path]
    items: int
    labels: list[dict[str, Any]]
    apart: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    counts: dict[str, int]  # per box, in the order the sheet shows them

    @property
    def unlabelled(self) -> int:
        return self.items - sum(self.counts.values())

    def summary(self) -> str:
        sheets = f"{len(self.sheets)} sheet{'' if len(self.sheets) == 1 else 's'}"
        labelled = sum(self.counts.values())
        per_box = ", ".join(f"{box} {n}" for box, n in self.counts.items())
        return (
            f"{sheets}, {self.items} items: {labelled} labelled, {self.unlabelled} unlabelled\n"
            f"{per_box}"
        )

    def write(self, out: Path) -> list[tuple[Path, int]]:
        """The labels to `out`; notes, and any rows kept apart, to files beside it."""
        out = Path(out)
        files = [(out, self.labels)]
        if self.kind.apart is not None:
            files.append((out.with_suffix(f".{self.kind.apart}.jsonl"), self.apart))
        files.append((out.with_suffix(".notes.jsonl"), self.notes))
        out.parent.mkdir(parents=True, exist_ok=True)
        for path, rows in files:
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            path.write_text(text, encoding="utf-8", newline="\n")
        return [(path, len(rows)) for path, rows in files]


def read_sheets(path: Path) -> Reading:
    """Every sheet at `path` (one sheet, or a directory of them) as labels.

    Exactly one tick is a label; no tick leaves the item unlabelled; anything
    else raises `SheetError` listing every problem in every sheet, each with
    the file and line of the item's heading.
    """
    path = Path(path)
    if path.is_dir():
        sheets = sorted(path.glob(GLOB))
        if not sheets:
            raise ValueError(f"{path}: no {GLOB} in this directory")
    elif path.is_file():
        sheets = [path]
    else:
        raise ValueError(f"{path}: no such sheet or directory")

    problems: list[str] = []
    read = []
    kinds: dict[str, Kind] = {}
    for sheet in sheets:
        entries = _sidecar(sheet.with_suffix(".items.jsonl"))
        if not entries:
            continue
        kinds.update((kind.name, kind) for _, kind, _ in entries)
        marks = _marks(sheet, {item_id for item_id, _, _ in entries}, entries[0][1], problems)
        read.append((sheet, entries, marks))
    if len(kinds) > 1:
        problems.append(f"{path}: mixes {' and '.join(sorted(kinds))} sheets; read them apart")
    if problems:
        raise SheetError(problems)
    if not kinds:
        raise ValueError(f"{path}: the sheets hold no items")
    (kind,) = kinds.values()

    labels: list[dict[str, Any]] = []
    apart: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    counts = dict.fromkeys(kind.boxes, 0)
    items = 0
    for sheet, entries, marks in read:
        for item_id, _, row in entries:
            items += 1
            marked = marks.get(item_id)
            answer = None
            if marked is not None and len(marked.ticked) == 1:
                box = marked.ticked[0]
                counts[box] += 1
                answer = kind.boxes[box]
                if answer == kind.apart:
                    apart.append(row)
                else:
                    label = kind.label(row, answer)
                    try:
                        kind.output.model_validate(label)
                    except ValueError as exc:
                        problems.append(f"{sheet}: {item_id}: not a {kind.output.__name__}: {exc}")
                    labels.append(label)
            note = "\n".join(marked.note).strip() if marked and marked.note else ""
            if note:
                notes.append({"id": item_id, "sheet": sheet.name, "answer": answer, "note": note})
    if problems:
        raise SheetError(problems)
    return Reading(kind, sheets, items, labels, apart, notes, counts)


__all__ = [
    "GROUNDING",
    "KINDS",
    "PAIR",
    "GroundingItem",
    "Kind",
    "Made",
    "Mention",
    "PairItem",
    "Reading",
    "SheetError",
    "make_sheets",
    "plain",
    "quote",
    "read_sheets",
]
