"""Label sheets: markdown a person ticks, read back into the eval label formats.

`odke eval` scores against labels, and someone has to make them. A sheet is
plain markdown (headings, a quoted passage, three checkboxes), so it can be
labelled wherever a markdown file opens, a tablet included, one sheet a
sitting. Everything an item needs to become a label row is kept in a sidecar,
`<sheet>.items.jsonl`, beside the sheet; the markdown carries only what a
person adds to it, which is one tick per item and an optional note.

The sheet shows a grounding claim exactly as `render_claim` puts it to the
grounder, so the person judges what the model judged. Passage text is escaped
so that markdown cannot hide or restyle it: a `$` stays a dollar sign rather
than opening a formula.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
)
KINDS = {kind.name: kind for kind in (GROUNDING,)}


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


__all__ = ["GROUNDING", "KINDS", "GroundingItem", "Kind", "Made", "make_sheets", "plain", "quote"]
