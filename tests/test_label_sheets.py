"""`odke label`: sheets a person ticks, read back into the eval label formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke.eval.sheets import make_sheets, quote

ADA = "Ada Lovelace was born in London in 1815."


def _grounding(n: int, text: str = ADA, *, span: tuple[int, int] | None = (0, 22)) -> dict:
    fact: dict[str, Any] = {
        "id": f"f{n}",
        "subject": {"key": "ada", "type": "Person", "label": "Ada Lovelace"},
        "predicate": "born_in",
        "object_value": 1815 + n,
    }
    if span is not None:
        start, end = span
        fact["evidence"] = [{"doc_id": "d1", "span": {"doc_id": "d1", "start": start, "end": end}}]
    return {"text": text, "fact": fact}


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _item(sheet: Path, item_id: str) -> str:
    """The block of `sheet` from `### item_id` up to the next item."""
    text = sheet.read_text(encoding="utf-8")
    return text.split(f"### {item_id}\n", 1)[1].split("\n### ", 1)[0]


# --------------------------------------------------------------------------- #
# make: grounding
# --------------------------------------------------------------------------- #


def test_a_grounding_item_shows_the_claim_as_the_grounder_reads_it(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(0)])
    made = make_sheets("grounding", items, tmp_path / "sheets")

    assert [p.name for p in made.sheets] == ["sheet-01.md"]
    assert _item(made.sheets[0], "G-0001") == (
        "\n"
        "Ada Lovelace (Person) — born in — 1815.\n"
        "\n"
        "> **Ada Lovelace was born** in London in 1815.\n"
        "\n"
        "- [ ] supported\n"
        "- [ ] contradicted\n"
        "- [ ] not found\n"
        "\n"
        "note:\n"
    )


def test_the_header_explains_the_answers_and_asks_for_one_tick(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(3)])
    sheet = make_sheets("grounding", items, tmp_path / "sheets").sheets[0]
    header = sheet.read_text(encoding="utf-8").split("\n### ", 1)[0]

    assert header.startswith("# Grounding sheet 01\n\nG-0001 to G-0003: ")
    assert "supported: " in header and "contradicted: " in header and "not found: " in header
    assert "Tick exactly one box per item." in header
    # Mobile-friendly: short lines of our own, and nothing but headings, text and boxes.
    assert max(len(line) for line in header.splitlines()) <= 64


def test_items_run_across_numbered_sheets_with_sidecars(tmp_path: Path) -> None:
    rows = [_grounding(n) for n in range(5)]
    items = _jsonl(tmp_path / "in.jsonl", rows)
    made = make_sheets("grounding", items, tmp_path / "s", per_sheet=2)

    assert [p.name for p in made.sheets] == ["sheet-01.md", "sheet-02.md", "sheet-03.md"]
    assert (made.items, made.first, made.last) == (5, "G-0001", "G-0005")
    assert "### G-0003" in made.sheets[1].read_text(encoding="utf-8")
    sidecars = [
        json.loads(line)
        for n in (1, 2, 3)
        for line in (tmp_path / "s" / f"sheet-0{n}.items.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in sidecars] == [f"G-000{n}" for n in range(1, 6)]
    assert [row["row"] for row in sidecars] == rows


def test_the_same_input_writes_byte_identical_sheets(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(7)])
    make_sheets("grounding", items, tmp_path / "a", per_sheet=3)
    make_sheets("grounding", items, tmp_path / "b", per_sheet=3)

    a = {p.name: p.read_bytes() for p in sorted((tmp_path / "a").iterdir())}
    b = {p.name: p.read_bytes() for p in sorted((tmp_path / "b").iterdir())}
    assert len(a) == 6 and a == b


def test_passage_markdown_is_escaped_so_nothing_is_hidden() -> None:
    text = "It costs $5 and $10; uptime 99.9% %%hidden%% <b>x</b> ==hi== [[link]] #tag"
    (line,) = quote(text)
    assert "\\$5" in line and "\\%\\%hidden\\%\\%" in line and "\\<b\\>" in line
    assert "\\=\\=hi\\=\\=" in line and "\\[\\[link\\]\\]" in line and "\\#tag" in line


def test_a_span_across_lines_is_bold_on_each_and_whitespace_stays_outside() -> None:
    text = "First line here.\n\nSecond line."
    # From the space before "line" to the end of "Second".
    assert quote(text, 5, 24) == ["> First **line here.**", ">", "> **Second** line."]


def test_a_fact_with_no_span_shows_the_passage_unmarked(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(0, span=None)])
    sheet = make_sheets("grounding", items, tmp_path / "s").sheets[0]
    assert f"> {ADA}\n" in _item(sheet, "G-0001")


def test_a_bad_row_names_its_line(tmp_path: Path) -> None:
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(_grounding(0)) + "\n\n" + '{"text": "no fact"}\n', "utf-8")
    with pytest.raises(ValueError, match=r"items\.jsonl:3: not a grounding item"):
        make_sheets("grounding", items, tmp_path / "s")
    assert not (tmp_path / "s").exists()


def test_a_row_already_labelled_is_refused(tmp_path: Path) -> None:
    row = {**_grounding(0), "verdict": "supported"}
    with pytest.raises(ValueError, match="verdict"):
        make_sheets("grounding", _jsonl(tmp_path / "i.jsonl", [row]), tmp_path / "s")


def test_sheets_are_never_overwritten(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(0)])
    make_sheets("grounding", items, tmp_path / "s")
    with pytest.raises(ValueError, match="already has sheets"):
        make_sheets("grounding", items, tmp_path / "s")


def test_facts_without_an_id_are_warned_about(tmp_path: Path) -> None:
    rows = [_grounding(0), _grounding(1)]
    del rows[1]["fact"]["id"]
    made = make_sheets("grounding", _jsonl(tmp_path / "i.jsonl", rows), tmp_path / "s")
    assert made.warnings and made.warnings[0].startswith("1 of 2 facts have no id")
