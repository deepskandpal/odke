"""The track record (#141): each run's predicted fixes, and what the next run measured.

A run is named by its per-document outcomes, so `compare`, which reads only the
`--items` files, finds the run's predictions wherever the files went. The file
only grows: a measurement is appended beside the prediction, never written over
it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke.cli.main import app
from openodke.eval.compare import ItemRow, compare_items, write_items
from openodke.eval.eval_report import read_report
from openodke.eval.fixes import Fix, Gain
from openodke.eval.formats import load_jsonl
from openodke.eval.track import (
    MeasuredLine,
    RunLine,
    measured,
    read,
    record_comparison,
    record_run,
    run_id,
    summary,
)

TRIPLES = Path(__file__).resolve().parents[1] / "examples" / "triples"
runner = CliRunner()

A = [ItemRow(id=f"d{i}", tp=2, fp=1, fn=2) for i in range(12)]
B = [ItemRow(id=f"d{i}", tp=3 + i % 2, fp=1, fn=1 - i % 2) for i in range(12)]


def fix(id_: str, expected: float = 0.1, ceiling: float = 0.3) -> Fix:
    return Fix(
        id=id_,
        rank=1,
        row="pipeline",
        buckets=("never_offered",),
        misses=4,
        action="x",
        knob="y",
        recall=Gain(expected=expected, ceiling=ceiling, basis="arithmetic"),
    )


def compared(**options: Any) -> Any:
    return compare_items(A, B, resamples=300, seed=1, **options)


def test_a_run_is_named_by_its_outcomes_wherever_they_were_written(tmp_path: Path) -> None:
    write_items(tmp_path / "a.items.jsonl", list(reversed(A)))
    assert run_id(load_jsonl(tmp_path / "a.items.jsonl", ItemRow)) == run_id(A)
    assert run_id(A) != run_id(B) and len(run_id(A)) == 16


def test_a_run_s_predictions_are_appended_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    line = record_run(path, run_id(A), [fix("offer-relations")], report="a.json", config={"x": 1})
    record_run(path, run_id(B), [])
    assert read(path) == [line, read(path)[1]]
    assert line.predictions[0].fix == "offer-relations" and line.config == {"x": 1}
    assert read(tmp_path / "none.jsonl") == []
    path.write_text(path.read_text() + '{"kind": "guess"}\n')
    with pytest.raises(ValueError, match=r"record.jsonl:3: not a track record line"):
        read(path)


def test_compare_appends_what_b_measured_of_a_fix_a_predicted(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    record_run(path, run_id(A), [fix("offer-relations"), fix("reextract")])
    comparison = compared()
    lines, notes = record_comparison(comparison, A, B, [path], applied=["offer-relations"])
    assert notes == []
    (line,) = lines
    recall = comparison.guardrails["recall"]
    assert (line.fix, line.applied, line.run, line.against) == (
        "offer-relations",
        "--applied",
        run_id(A),
        run_id(B),
    )
    assert (line.measured, line.low, line.high) == (recall.difference, *recall.interval)
    assert (line.expected, line.ceiling, line.verdict) == (0.1, 0.3, recall.verdict)
    # Appended beside the prediction, never over it.
    assert [type(x) for x in read(path)] == [RunLine, MeasuredLine]
    (shown,) = measured(read(path), "offer-relations")
    assert shown.measured == recall.difference and measured(read(path), "reextract") == ()
    text = summary(read(path), path)
    assert text[0].endswith(
        "2 prediction(s) from 1 run(s), 1 measured, 0 inside its expected-to-ceiling range"
    )
    assert "offer-relations: expected +10.0 (ceiling +30.0), measured +" in text[1]


def test_the_config_diff_names_the_fix_when_one_knob_changed(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    before = {"stages": {"extractor": {"use": "llm"}, "chunker": {"use": "sentence"}}}
    after = {"stages": {"extractor": {"use": "llm", "snippet_limit": 96}, **_chunker()}}
    record_run(path, run_id(A), [fix("offer-relations"), fix("reextract")], config=before)
    record_run(path, run_id(B), [], config=after)
    (line,), notes = record_comparison(compared(), A, B, [path])
    assert notes == []
    assert line.applied == "config: stages.extractor.snippet_limit (unset) → 96"

    # A change at two predicted fixes' knobs is attributed to neither.
    record_run(path, run_id(A), [fix("wider-context"), fix("smaller-chunks")], config=before)
    record_run(path, run_id(B), [], config={**before, "stages": {"chunker": {"use": "none"}}})
    lines, notes = record_comparison(compared(), A, B, [path])
    assert lines == []
    assert notes == [
        "track record: the config change touches smaller-chunks, wider-context; "
        "name one with --applied"
    ]


def _chunker() -> dict[str, Any]:
    return {"chunker": {"use": "sentence"}}


def test_what_cannot_be_recorded_says_why(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"
    assert record_comparison(compared(), A, B, [path], applied=["inverses"]) == (
        [],
        [f"track record: run A ({run_id(A)}) predicted nothing in {path}"],
    )
    record_run(path, run_id(A), [fix("offer-relations")])
    lines, notes = record_comparison(compared(), A, B, [path], applied=["inverses"])
    assert lines == [] and notes == ["track record: B applied inverses, which A did not predict"]
    lines, notes = record_comparison(compared(), A, B, [path])
    assert notes == ["track record: no config for both runs to diff; name the fix with --applied"]
    with pytest.raises(ValueError, match="--applied 'nonsense': no such fix"):
        record_comparison(compared(), A, B, [path], applied=["nonsense"])
    right = [ItemRow(id=f"i{n}", correct=n % 2 == 0) for n in range(10)]
    wrong = [ItemRow(id=f"i{n}", correct=False) for n in range(10)]
    record_run(path, run_id(right), [fix("offer-relations")])
    comparison = compare_items(right, wrong, resamples=300)
    lines, notes = record_comparison(comparison, right, wrong, [path], applied=["offer-relations"])
    assert notes == ["track record: the comparison carries no recall to measure a fix by"]


# --------------------------------------------------------------------------- #
# From the shell: report, compare, report again
# --------------------------------------------------------------------------- #


def _pipeline(predictions: Path, out: Path) -> list[str]:
    return [
        "eval", "pipeline",
        "--labels", str(TRIPLES / "gold.jsonl"),
        "--documents", str(TRIPLES / "texts"),
        "--ontology", str(TRIPLES / "ontology.json"),
        "--predictions", str(predictions),
        "--report", str(out / "report.json"),
        "--items", str(out / "items.jsonl"),
    ]  # fmt: skip


def test_the_record_grows_with_each_report_and_each_comparison(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    fixed = tmp_path / "fixed.jsonl"
    rows = [json.loads(line) for line in (TRIPLES / "triples.jsonl").read_text().splitlines()]
    rows[-1] = {**rows[-1], "object": 2014, "quote": "founded in Leeds in 2014"}
    fixed.write_text("".join(json.dumps(row) + "\n" for row in rows))

    first = runner.invoke(app, _pipeline(TRIPLES / "triples.jsonl", a))
    assert first.exit_code == 0, first.output
    report = read_report(a / "report.json")
    assert [f.id for f in report.fixes] == ["reextract"]
    assert "where it loses facts  (pipeline: 1 miss)" in first.output
    assert f"track record  ({a / 'track-record.jsonl'}): 1 prediction(s) from 1 run(s)" in (
        first.output
    )
    assert runner.invoke(app, _pipeline(fixed, b)).exit_code == 0

    shared = tmp_path / "record.jsonl"
    compare = [
        "eval", "compare", str(a / "items.jsonl"), str(b / "items.jsonl"),
        "--applied", "reextract",
    ]  # fmt: skip
    # Each report went beside itself; compare finds A's prediction beside A's items.
    done = runner.invoke(app, compare)
    assert done.exit_code == 0, done.output
    assert "track record: reextract expected +18.8 (ceiling +25.0), measured +25.0" in done.output
    (line,) = [x for x in read(a / "track-record.jsonl") if isinstance(x, MeasuredLine)]
    assert line.applied == "--applied"

    # A named record holds nothing of A: nothing is appended, and it says so.
    alone = runner.invoke(app, [*compare, "--track-record", str(shared)])
    assert alone.exit_code == 0 and "predicted nothing in" in alone.output
    assert not shared.exists()

    again = runner.invoke(app, _pipeline(TRIPLES / "triples.jsonl", a))
    assert again.exit_code == 0, again.output
    (prediction,) = read_report(a / "report.json").fixes
    (record,) = prediction.record
    assert (record.expected, record.measured) == (pytest.approx(0.1875), pytest.approx(0.25))
    assert "     - track record: expected +18.8 (ceiling +25.0), measured +25.0" in again.output
    assert (
        "2 prediction(s) from 2 run(s), 1 measured, 1 inside its expected-to-ceiling range"
        in again.output
    )


def test_extract_records_its_fixes_beside_its_report(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "eval"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"stats": {"coverage": {"not_offered": ["employer"]}}}))
    args = ["eval", "extract", "--labels", str(fixtures / "extract.labels.jsonl")]
    args += ["--predictions", str(fixtures / "extract.predictions.jsonl")]
    result = runner.invoke(
        app, [*args, "--trace", str(manifest), "--report", str(tmp_path / "r.json")]
    )
    assert result.exit_code == 0, result.output
    report = read_report(tmp_path / "r.json")
    never = next(b for b in report.diagnosis if b.bucket == "never_offered")
    assert never.count == 2 and never.note == "from manifest.json"
    (line,) = read(tmp_path / "track-record.jsonl")
    assert isinstance(line, RunLine) and line.report == str(tmp_path / "r.json")
    assert [p.fix for p in line.predictions] == [f.id for f in report.fixes]
    assert line.predictions[0].fix == "offer-relations"


def test_the_track_record_flags_belong_to_their_stages() -> None:
    labels = str(TRIPLES / "gold.jsonl")
    for args, message in (
        (["eval", "ground", "--labels", labels, "--trace", "t.json"], "--trace: for extract"),
        (["eval", "ground", "--labels", labels, "--track-record", "t.jsonl"], "--track-record"),
        (["eval", "extract", "--labels", labels, "--applied", "inverses"], "--applied: for"),
        (["eval", "extract", "--labels", labels, "--reference", "r.jsonl"], "for pipeline"),
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2 and message in result.output, (args, result.output)
