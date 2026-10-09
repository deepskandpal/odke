"""`odke eval compare`: two runs' item files in, a verdict with its detection limit out.

The extract fixture, per document (tests/fixtures/eval/extract.*.jsonl):

    d1  tp 2  fp 1  fn 1    born and birthplace right, one wrong value
    d2  tp 0  fp 2  fn 2    a wrong entity and a spurious fact against two gold facts

which sums to the report's tp 2, fp 3, fn 3: precision = recall = F1 = 0.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke.cli.main import app
from openodke.eval.compare import (
    Comparison,
    ItemRow,
    compare_items,
    gate,
    item_rows,
    write_items,
)
from openodke.eval.runner import load_inputs, score

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
runner = CliRunner()


def _flat(text: str) -> str:
    return " ".join(text.split())


def _pass_fail(both: int, gained: int, lost: int, neither: int) -> tuple[list[ItemRow], ...]:
    a = [True] * both + [False] * gained + [True] * lost + [False] * neither
    b = [True] * both + [True] * gained + [False] * lost + [False] * neither
    return tuple([ItemRow(id=f"i{n}", correct=c) for n, c in enumerate(run)] for run in (a, b))


def _files(tmp_path: Path, a: list[ItemRow], b: list[ItemRow]) -> list[str]:
    write_items(tmp_path / "a.jsonl", a)
    write_items(tmp_path / "b.jsonl", b)
    return [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]


def _items_args(stage: str, out: Path) -> list[str]:
    args = ["eval", stage, "--labels", str(FIXTURES / f"{stage}.labels.jsonl")]
    predictions = FIXTURES / f"{stage}.predictions.jsonl"
    if predictions.exists():
        args += ["--predictions", str(predictions)]
    return [*args, "--items", str(out)]


# --------------------------------------------------------------------------- #
# --items: one run's outcomes, per item
# --------------------------------------------------------------------------- #


def test_extract_items_are_per_document_and_sum_to_the_report(tmp_path: Path) -> None:
    out = tmp_path / "items.jsonl"
    result = runner.invoke(app, _items_args("extract", out))
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("extract  (n=5)")
    assert f"wrote {out}: 2 documents" in result.stderr
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [
        {"id": "d1", "tp": 2, "fp": 1, "fn": 1},
        {"id": "d2", "tp": 0, "fp": 2, "fn": 2},
    ]


@pytest.mark.parametrize("stage", ["extract", "ground", "validate", "route"])
def test_items_sum_to_what_the_report_scored(stage: str) -> None:
    labels = FIXTURES / f"{stage}.labels.jsonl"
    predictions = FIXTURES / f"{stage}.predictions.jsonl"
    rows, predicted = load_inputs(stage, labels, predictions if predictions.exists() else None)
    report = score(stage, rows, predicted)
    items, _ = item_rows(stage, rows, predicted)
    if stage == "extract":
        totals = {k: sum(getattr(r, k) for r in items) for k in ("tp", "fp", "fn")}
        assert totals == {k: report.metrics[k] for k in ("tp", "fp", "fn")}
        return
    accuracy = {"ground": "accuracy", "validate": "agreement", "route": "action_accuracy"}
    assert len(items) == report.n
    assert sum(bool(r.correct) for r in items) / len(items) == report.metrics[accuracy[stage]]


def test_items_are_not_written_for_a_stage_with_no_per_item_outcome(tmp_path: Path) -> None:
    result = runner.invoke(app, _items_args("resolve", tmp_path / "items.jsonl"))
    assert result.exit_code == 2
    assert "--items is written for extract, ground, validate, route" in _flat(result.output)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def test_a_run_against_itself_is_inconclusive_and_says_why(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    runner.invoke(app, _items_args("extract", out))
    result = runner.invoke(app, ["eval", "compare", str(out), str(out)])
    assert result.exit_code == 0, result.output
    text = _flat(result.stdout)
    assert "compare f1 over 2 documents" in text
    assert "inconclusive: no document changed between the runs" in text
    assert "guardrails: reported, not deciding the verdict" in text


def test_a_paired_drop_is_worse_with_its_interval_flips_and_mcnemar(tmp_path: Path) -> None:
    a, b = _pass_fail(both=240, gained=3, lost=27, neither=30)
    result = runner.invoke(app, ["eval", "compare", *_files(tmp_path, a, b)])
    assert result.exit_code == 1
    text = _flat(result.stdout)
    assert "verdict worse: B is below A, and the whole 95% interval is below zero" in text
    assert "accuracy A 0.890 B 0.810 difference -8.0 points" in text
    assert "limit 5.1 points: the smallest change these 300 items detect" in text
    assert "30 of 300 items changed (10.0%): 3 gained, 27 lost, McNemar exact p" in text


def test_items_are_paired_by_id_not_by_line(tmp_path: Path) -> None:
    a, b = _pass_fail(both=240, gained=3, lost=27, neither=30)
    in_order = compare_items(a, b)
    shuffled = compare_items(a, list(reversed(b)))
    assert in_order.primary == shuffled.primary


def test_different_items_are_refused_naming_the_difference(tmp_path: Path) -> None:
    a, b = _pass_fail(both=20, gained=2, lost=2, neither=6)
    b = [*b[:-2], ItemRow(id="new-1", correct=True)]
    result = runner.invoke(app, ["eval", "compare", *_files(tmp_path, a, b)])
    assert result.exit_code == 2
    text = _flat(result.output)
    assert "the runs score different items: 2 only in" in text
    assert "(i28, i29); 1 only in" in text and "(new-1)" in text


def test_an_item_listed_twice_is_refused(tmp_path: Path) -> None:
    a, b = _pass_fail(both=5, gained=0, lost=0, neither=5)
    with pytest.raises(ValueError, match="'i0' is listed twice"):
        compare_items([*a, a[0]], b)


def test_json_is_one_comparison_block(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    runner.invoke(app, _items_args("extract", out))
    result = runner.invoke(app, ["eval", "compare", str(out), str(out), "--json"])
    comparison = Comparison.model_validate_json(result.stdout)
    assert (comparison.metric, comparison.unit, comparison.verdict) == (
        "f1",
        "document",
        "inconclusive",
    )
    assert set(comparison.guardrails) == {"precision", "recall"}
    assert comparison.primary.resamples == 2000 and comparison.primary.seed == 0


def test_the_primary_metric_is_chosen_and_the_rest_are_guardrails(tmp_path: Path) -> None:
    out = tmp_path / "a.jsonl"
    runner.invoke(app, _items_args("extract", out))
    args = ["eval", "compare", str(out), str(out), "--metric", "recall", "--json"]
    comparison = Comparison.model_validate_json(runner.invoke(app, args).stdout)
    assert comparison.metric == "recall"
    assert set(comparison.guardrails) == {"f1", "precision"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--metric", "accuracy"], "accuracy needs correct on every row of both runs"),
        (["--metric", "kappa"], "unknown metric 'kappa'"),
    ],
)
def test_a_metric_the_rows_cannot_give_is_refused(
    tmp_path: Path, args: list[str], message: str
) -> None:
    out = tmp_path / "a.jsonl"
    runner.invoke(app, _items_args("extract", out))
    result = runner.invoke(app, ["eval", "compare", str(out), str(out), *args])
    assert result.exit_code == 2
    assert message in _flat(result.output)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "compare", "a.jsonl"], "compare takes two runs' --items files, A then B; got 1"),
        (
            ["eval", "compare", "a.jsonl", "b.jsonl", "--labels", "x.jsonl"],
            "takes no other inputs",
        ),
        (["eval", "extract", "a.jsonl", "--labels", "x.jsonl"], "only compare takes runs"),
        (
            ["eval", "extract", "--labels", "x.jsonl", "--metric", "f1"],
            "--metric: for compare only",
        ),
        (["eval", "extract", "--labels", "x.jsonl", "--seed", "3"], "--seed: for compare only"),
    ],
)
def test_compare_mistakes_exit_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in _flat(result.output)


def test_describe_says_what_an_item_row_is() -> None:
    result = runner.invoke(app, ["eval", "compare", "--describe"])
    assert result.exit_code == 0
    assert '{"id": "d1", "tp": 3, "fp": 1, "fn": 0}' in result.stdout
    assert "guardrails" in result.stdout


def test_an_item_row_carries_an_outcome() -> None:
    with pytest.raises(ValueError, match="carries all three"):
        ItemRow(id="d1", tp=1, fp=0)
    with pytest.raises(ValueError, match="tp, fp and fn, or correct"):
        ItemRow(id="d1")


# --------------------------------------------------------------------------- #
# The CI gate
# --------------------------------------------------------------------------- #


def test_worse_fails_the_gate_with_the_reason(tmp_path: Path) -> None:
    a, b = _pass_fail(both=240, gained=3, lost=27, neither=30)
    result = runner.invoke(app, ["eval", "compare", *_files(tmp_path, a, b)])
    assert result.exit_code == 1
    assert "gate: fail: accuracy is worse: -8.0 points, 95% interval" in _flat(result.stderr)


def test_inconclusive_passes_unless_asked_to_fail(tmp_path: Path) -> None:
    """A change too small for 300 items to see is not a regression they saw."""
    files = _files(tmp_path, *_pass_fail(both=240, gained=14, lost=16, neither=30))
    passed = runner.invoke(app, ["eval", "compare", *files])
    assert passed.exit_code == 0, passed.output
    assert "smallest change it can detect is 5.1 points" in _flat(passed.stdout)
    strict = runner.invoke(app, ["eval", "compare", *files, "--fail-on-inconclusive"])
    assert strict.exit_code == 1
    assert "gate: fail: --fail-on-inconclusive: inconclusive" in _flat(strict.stderr)


def test_fail_under_is_a_floor_for_b_with_its_uncertainty(tmp_path: Path) -> None:
    """B scores 0.890, better than A; its 95% range reaches down to about 0.85."""
    a, b = _pass_fail(both=240, gained=27, lost=3, neither=30)
    files = _files(tmp_path, a, b)
    comparison = compare_items(a, b)
    low = comparison.primary.b_interval[0]
    assert comparison.verdict == "better" and 0.84 < low < 0.87
    cleared = runner.invoke(app, ["eval", "compare", *files, "--fail-under", "0.80"])
    assert cleared.exit_code == 0, cleared.output
    # Above B's low end, though below its point estimate of 0.890: B fails the floor.
    missed = runner.invoke(app, ["eval", "compare", *files, "--fail-under", "0.88"])
    assert missed.exit_code == 1
    expected = f"B's accuracy could be as low as {low:.3f} (95% range), under --fail-under 0.88"
    assert expected in _flat(missed.stderr)


def test_the_gate_reads_the_comparison() -> None:
    better = compare_items(*_pass_fail(both=240, gained=27, lost=3, neither=30))
    assert gate(better) == [] and gate(better, fail_under=0.5, fail_on_inconclusive=True) == []
    same = compare_items(*_pass_fail(both=240, gained=0, lost=0, neither=60))
    assert gate(same) == []
    assert gate(same, fail_on_inconclusive=True) == [
        "--fail-on-inconclusive: inconclusive: no item changed between the runs"
    ]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "compare", "a.jsonl", "b.jsonl", "--fail-under", "80"], "--fail-under"),
        (["eval", "extract", "--labels", "x", "--fail-under", "0.8"], "for compare only"),
        (["eval", "extract", "--labels", "x", "--fail-on-inconclusive"], "for compare only"),
    ],
)
def test_gate_mistakes_exit_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in _flat(result.output)
