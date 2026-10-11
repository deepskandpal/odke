"""`bench/adjudication.py`: how often adjudication is right on label set G, with no model.

Label set G's labels are not in the repository yet, so the script must skip
with a message; with labels and verdicts it must count what the docstring says.
The verdicts and labels here are made up, to check the counting.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from openodke import Document, Fact, GroundingVerdict

BENCH = Path(__file__).parent.parent / "bench"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "odke_bench_adjudication", BENCH / "adjudication.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adjudication = _load()


def test_it_skips_until_label_set_g_has_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert adjudication.main(["--labels", str(tmp_path / "labels.jsonl")]) == 0
    assert capsys.readouterr().out.startswith(f"skipped: {tmp_path / 'labels.jsonl'} is not there")


def test_the_not_in_gold_items_are_g_s_200_real_and_100_planted() -> None:
    rows = adjudication.not_in_gold()
    real = [r for r in rows if not r["planted"]]
    assert (len(real), len(rows) - len(real)) == (200, 100)
    assert sum(r["dataset"] == "redocred" for r in real) == 100
    assert all(r["item"]["fact"]["id"] == r["fact_id"] for r in rows)


def test_it_measures_the_list_against_the_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = adjudication.not_in_gold()
    labels, verdicts = {}, {}
    for n, row in enumerate(rows):
        # Every other real item is supported; every third is listed (two runs say so).
        labels[row["fact_id"]] = "supported" if n % 2 == 0 else "not_found"
        verdicts[row["fact_id"]] = (
            ["supported", "not_found", "supported"] if n % 3 == 0 else ["supported"] + ["x"] * 2
        )
    real = [(n, r) for n, r in enumerate(rows) if not r["planted"]]
    found = adjudication.measure(rows, labels, verdicts)
    everything = found["groups"]["all"]
    listed = [n for n, _ in real if n % 3 == 0]
    assert everything["labelled"] == 200
    assert everything["listed"] == len(listed)
    assert everything["listed_supported"] == sum(n % 6 == 0 for n, _ in real)
    precision = everything["precision"]
    assert precision["value"] == everything["listed_supported"] / everything["listed"]
    low, high = precision["interval"]
    assert low < precision["value"] < high
    planted = [n for n, r in enumerate(rows) if r["planted"]]
    assert found["planted"] == {"items": 100, "listed": sum(n % 3 == 0 for n in planted)}
    by_dataset = (
        found["groups"]["text2kgbench"]["labelled"] + found["groups"]["redocred"]["labelled"]
    )
    assert by_dataset == 200

    label_file, verdict_file = tmp_path / "labels.jsonl", tmp_path / "v.jsonl"
    label_file.write_text(
        "".join(json.dumps({"fact": {"id": k}, "verdict": v}) + "\n" for k, v in labels.items())
    )
    verdict_file.write_text(
        "".join(json.dumps({"fact_id": k, "verdicts": v}) + "\n" for k, v in verdicts.items())
    )
    assert adjudication.main(["--labels", str(label_file), "--verdicts", str(verdict_file)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("gold adjudication on label set G: listed when supported in 2 of 3 runs")
    assert "planted false facts listed:" in out
    with pytest.raises(SystemExit):
        adjudication.main(["--labels", str(label_file)])


def test_a_live_run_is_saved_and_never_paid_for_twice(tmp_path: Path) -> None:
    class Counting:
        def __init__(self) -> None:
            self.calls = 0

        def ground(self, fact: Fact, doc: Document) -> Fact:
            self.calls += 1
            return fact.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})

    rows = adjudication.not_in_gold()[:3]
    judge = Counting()
    saved = adjudication.collect(rows, judge, tmp_path)
    assert judge.calls == 9
    assert saved == {row["fact_id"]: ["supported"] * 3 for row in rows}
    again = adjudication.collect(rows, judge, tmp_path)
    assert judge.calls == 9 and again == saved
    assert len((tmp_path / "G.verdicts.jsonl").read_text().splitlines()) == 3


def test_wilson() -> None:
    assert adjudication.wilson(0, 0) is None
    low, high = adjudication.wilson(8, 10)
    assert (round(low, 3), round(high, 3)) == (0.49, 0.943)
