"""`odke label`: sheets a person ticks, read back into the eval label formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke.eval import GroundingLabel, load_jsonl
from openodke.eval.sheets import SheetError, make_sheets, quote, read_sheets

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


def _tick(sheet: Path, item_id: str, box: str, mark: str = "x") -> None:
    """Tick `box` under `item_id`, as a person would in Obsidian."""
    head, tail = sheet.read_text(encoding="utf-8").split(f"### {item_id}\n", 1)
    tail = tail.replace(f"- [ ] {box}\n", f"- [{mark}] {box}\n", 1)
    sheet.write_text(f"{head}### {item_id}\n{tail}", encoding="utf-8")


def _note(sheet: Path, item_id: str, note: str) -> None:
    head, tail = sheet.read_text(encoding="utf-8").split(f"### {item_id}\n", 1)
    sheet.write_text(f"{head}### {item_id}\n{tail.replace('note:', note, 1)}", encoding="utf-8")


def _line(sheet: Path, text: str) -> int:
    return sheet.read_text(encoding="utf-8").splitlines().index(text) + 1


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


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #


def test_a_round_trip_gives_the_labels_that_were_ticked(tmp_path: Path) -> None:
    rows = [_grounding(n) for n in range(5)]
    items = _jsonl(tmp_path / "items.jsonl", rows)
    sheets = make_sheets("grounding", items, tmp_path / "s", per_sheet=3).sheets
    ticked = {"G-0001": "supported", "G-0002": "not found", "G-0004": "contradicted"}
    _tick(sheets[0], "G-0001", "supported")
    _tick(sheets[0], "G-0002", "not found", mark="X")  # Obsidian writes either
    _tick(sheets[1], "G-0004", "contradicted")

    reading = read_sheets(tmp_path / "s")
    reading.write(tmp_path / "labels.jsonl")

    verdict = {"supported": "supported", "contradicted": "contradicted", "not found": "not_found"}
    expected = [
        {**rows[int(item_id[2:]) - 1], "verdict": verdict[box]} for item_id, box in ticked.items()
    ]
    written = (tmp_path / "labels.jsonl").read_text(encoding="utf-8")
    assert written == "".join(json.dumps(row) + "\n" for row in expected)
    labels = load_jsonl(tmp_path / "labels.jsonl", GroundingLabel)
    assert [label.verdict for label in labels] == ["supported", "not_found", "contradicted"]


def test_odd_line_breaks_neither_leave_the_quote_nor_shift_line_numbers(tmp_path: Path) -> None:
    # A page break from a PDF, and a line separator, which JSON keeps raw.
    row = _grounding(0, text="Ada Lovelace\x0cwas born\u2028in London in 1815.")
    items = _jsonl(tmp_path / "items.jsonl", [row, _grounding(1)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    assert "> **Ada Lovelace was born** in London in 1815.\n" in _item(sheet, "G-0001")

    _tick(sheet, "G-0001", "supported")
    _tick(sheet, "G-0002", "supported")
    _tick(sheet, "G-0002", "contradicted")
    with pytest.raises(SheetError, match=f"sheet-01.md:{_line(sheet, '### G-0002')}: "):
        read_sheets(sheet)
    text = sheet.read_text(encoding="utf-8")
    sheet.write_text(text.replace("- [x] contradicted", "- [ ] contradicted"), encoding="utf-8")
    labels = read_sheets(sheet).labels
    assert labels == [{**row, "verdict": "supported"}, {**_grounding(1), "verdict": "supported"}]


def test_unlabelled_items_are_counted_not_refused(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(4)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    _tick(sheet, "G-0003", "supported")

    reading = read_sheets(sheet)
    assert (reading.items, len(reading.labels), reading.unlabelled) == (4, 1, 3)
    assert reading.counts == {"supported": 1, "contradicted": 0, "not found": 0}
    assert reading.summary() == (
        "1 sheet, 4 items: 1 labelled, 3 unlabelled\nsupported 1, contradicted 0, not found 0"
    )


def test_two_ticks_fail_on_the_line_of_the_item_heading(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(3)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    _tick(sheet, "G-0002", "supported")
    _tick(sheet, "G-0002", "not found")

    with pytest.raises(SheetError) as caught:
        read_sheets(tmp_path / "s")
    line = _line(sheet, "### G-0002")
    assert caught.value.problems == [
        f"{sheet}:{line}: G-0002 has 2 boxes ticked (supported, not found); tick exactly one"
    ]


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(4)])
    sheets = make_sheets("grounding", items, tmp_path / "s", per_sheet=2).sheets
    for box in ("supported", "contradicted", "not found"):
        _tick(sheets[0], "G-0001", box)
    text = sheets[1].read_text(encoding="utf-8").replace("### G-0004", "### G-0009")
    sheets[1].write_text(text, encoding="utf-8")

    with pytest.raises(SheetError) as caught:
        read_sheets(tmp_path / "s")
    assert [problem.split(": ", 1)[0] for problem in caught.value.problems] == [
        f"{sheets[0]}:{_line(sheets[0], '### G-0001')}",
        f"{sheets[1]}:{_line(sheets[1], '### G-0009')}",
    ]
    assert "G-0009 is not in sheet-02.items.jsonl" in caught.value.problems[1]


def test_a_mark_that_is_not_a_tick_is_refused(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(0)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    _tick(sheet, "G-0001", "supported", mark="/")
    with pytest.raises(SheetError, match=r"\[/\] is neither \[x\] nor \[ \]"):
        read_sheets(sheet)


def test_notes_are_kept_beside_the_labels(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(n) for n in range(3)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    _tick(sheet, "G-0001", "contradicted")
    _note(sheet, "G-0001", "note: the span stops short\nof the year")
    _note(sheet, "G-0002", "note: come back to this")

    written = dict(read_sheets(sheet).write(tmp_path / "labels.jsonl"))
    notes = [
        json.loads(line) for line in (tmp_path / "labels.notes.jsonl").read_text().splitlines()
    ]
    assert written == {tmp_path / "labels.jsonl": 1, tmp_path / "labels.notes.jsonl": 2}
    assert notes == [
        {
            "id": "G-0001",
            "sheet": "sheet-01.md",
            "answer": "contradicted",
            "note": "the span stops short\nof the year",
        },
        {"id": "G-0002", "sheet": "sheet-01.md", "answer": None, "note": "come back to this"},
    ]


def test_a_sheet_without_its_sidecar_is_refused(tmp_path: Path) -> None:
    items = _jsonl(tmp_path / "items.jsonl", [_grounding(0)])
    (sheet,) = make_sheets("grounding", items, tmp_path / "s").sheets
    sheet.with_suffix(".items.jsonl").unlink()
    with pytest.raises(SheetError, match="sheet-01.items.jsonl: missing"):
        read_sheets(sheet)
