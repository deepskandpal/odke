"""`bench/calibrate.py` and `bench/spend.py`: the contest, the cards, and the spend guard.

Loaded by path, as the other bench scripts are. No model is called: the runner
is driven on recorded answers (`--replay`), and the guard refuses before any
client is built.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke.calibration import JudgedFact, load_card
from openodke.llm.cache import replay_directory

BENCH = Path(__file__).parent.parent / "bench"
HAIKU = "anthropic/claude-haiku-4-5-20251001"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calibrate = _load("calibrate")
spend = calibrate.spend

REPLAY = [
    {"match": "It is NOT the case", "response": {"verdict": "contradicted"}},
    {"match": "It is not the case", "response": "no"},
    {"match": "## Statement", "response": '{"verdict": "supported"}'},
    {"match": "Mention A:", "response": {"because": "one person", "decision": "same"}},
    {"match": "Fact 1:", "response": {"decision": "different"}},
    {"match": "Claim", "response": {"verdict": "supported"}},
]


# --------------------------------------------------------------------------- #
# The spend guard
# --------------------------------------------------------------------------- #


def _ledger(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "spend.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_a_run_past_its_provider_s_cap_is_refused_before_anything_is_called(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, {"who": "x", "model": HAIKU, "calls": 1, "usd": 19.5})
    assert spend.check(HAIKU, 0.4, ledger=ledger).endswith("$0.10 left of $20.00")
    with pytest.raises(spend.SpendRefused, match=r"\$19\.50 spent \+ \$0\.60 estimated"):
        spend.check(HAIKU, 0.6, ledger=ledger)


def test_each_provider_has_its_own_cap_and_one_with_none_is_refused(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, {"who": "x", "model": HAIKU, "calls": 1, "usd": 19.99})
    gpt = "openai/gpt-4.1-mini-2025-04-14"
    with pytest.raises(spend.SpendRefused, match="no cap for openai"):
        spend.check(gpt, 0.1, ledger=ledger)
    assert "openai: $0.00 spent" in spend.check(gpt, 0.1, cap=5.0, ledger=ledger)


def test_a_run_s_spend_is_appended_and_a_raw_log_seeds_once(tmp_path: Path) -> None:
    ledger = tmp_path / "spend.jsonl"
    row = spend.record(
        "t", HAIKU, calls=2, input_tokens=10, output_tokens=1, usd=0.0001, ledger=ledger
    )
    assert set(row) == {"who", "model", "calls", "input_tokens", "output_tokens", "usd", "date"}
    (tmp_path / "raw").mkdir()
    raw_row = {"who": "a", "model": HAIKU, "calls": 1, "input_tokens": None, "output_tokens": None}
    raw = _ledger(tmp_path / "raw", {**raw_row, "usd": 0.31})
    assert spend.seed(raw, ledger) == 1 and spend.seed(raw, ledger) == 0
    assert spend.spent(ledger)["anthropic"] == pytest.approx(0.3101)
    assert spend.cost(HAIKU, 1_000_000, 1_000_000) == 6.0


def test_the_ledger_holds_rows_of_one_shape_and_no_paths() -> None:
    rows = spend.rows()
    assert rows, "bench/spend.jsonl is seeded with the runs before calibration"
    for row in rows:
        assert {"who", "model", "calls", "input_tokens", "output_tokens", "usd"} <= set(row)
        assert "/home" not in json.dumps(row) and "/root" not in json.dumps(row)


def test_the_runner_refuses_past_the_cap_and_calls_nothing(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, {"who": "x", "model": HAIKU, "calls": 1, "usd": 19.99})
    out = tmp_path / "out"
    code = calibrate.main(["--judges", "fact", "-o", str(out), "--ledger", str(ledger)])
    assert code == 3 and not out.exists()
    code = calibrate.main(["--model", "openai/gpt-4.1-mini-2025-04-14", "--judges", "fact",
                           "-o", str(out), "--ledger", str(ledger)])  # fmt: skip
    assert code == 3 and not out.exists()


# --------------------------------------------------------------------------- #
# The contest, adjudication and self-preference
# --------------------------------------------------------------------------- #


def _items(n: int, wrong: int, label: str = "supported") -> list[JudgedFact]:
    """n items the person labelled `label`; the first `wrong` the judge gets wrong."""
    right = "supported" if label == "supported" else "not_found"
    other = "not_found" if label == "supported" else "supported"
    return [
        JudgedFact(id=f"g{k}", label=label, runs=(other if k < wrong else right,) * 3)
        for k in range(n)
    ]


def test_a_challenger_replaces_p1_only_by_more_than_the_detection_limit() -> None:
    p1 = _items(100, 40)
    found = calibrate.contest(
        {"span-inline": p1, "onyx": _items(100, 5), "statement": _items(100, 37)}
    )
    assert found["onyx"]["replaces_p1"] is True
    assert found["onyx"]["decision"].startswith("onyx replaces span-inline: better by 35.0 points")
    assert found["statement"]["replaces_p1"] is False
    assert found["statement"]["decision"].startswith("span-inline stays: inconclusive")


def test_no_answer_is_a_failed_item() -> None:
    assert not calibrate.passes(JudgedFact(id="a", label="supported", runs=("unchecked",)))
    assert calibrate.passes(JudgedFact(id="a", label="contradicted", runs=("not_found",)))


def test_adjudication_reads_two_of_three_runs_against_the_person() -> None:
    items = {
        "a": calibrate.GItem("a", None, "gate", False, "not_in_gold"),  # type: ignore[arg-type]
        "b": calibrate.GItem("b", None, "gate", False, "not_in_gold"),  # type: ignore[arg-type]
        "c": calibrate.GItem("c", None, "dev", False, "match"),  # type: ignore[arg-type]
    }
    p1 = [
        JudgedFact(id="a", label="supported", runs=("supported", "not_found", "supported")),
        JudgedFact(id="b", label="not_found", runs=("supported", "not_found", "supported")),
        JudgedFact(id="c", label="supported", runs=("supported",) * 3),
    ]
    found = calibrate.adjudication(p1, items)
    assert set(found) == {"gate", "all"}
    assert found["gate"]["agreement"]["k"] == 1 and found["gate"]["precision"]["k"] == 1


def test_self_preference_compares_fail_recall_on_the_same_facts() -> None:
    claude = _items(20, 15, label="not_found")  # refuses 5 of 20
    other = _items(20, 5, label="not_found")  # refuses 15 of 20
    found = calibrate.self_preference(claude, other, ("claude", "other"))
    assert found["n"] == 20
    assert (found["fail_recall"]["claude"]["k"], found["fail_recall"]["other"]["k"]) == (5, 15)
    assert found["mcnemar"]["p_value"] < 0.05


# --------------------------------------------------------------------------- #
# A whole run, on recorded answers
# --------------------------------------------------------------------------- #


def test_a_run_writes_cards_items_and_the_table(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(REPLAY), encoding="utf-8")
    ledger = tmp_path / "spend.jsonl"
    out = tmp_path / "out"
    args = ["--replay", str(replay), "--variants", "span-inline,statement", "--cache",
            str(tmp_path / "cache"), "-o", str(out), "--ledger", str(ledger)]  # fmt: skip
    assert calibrate.main(args) == 0
    cards = sorted(p.name for p in (out / "cards").iterdir())
    assert cards == [
        f"fact_equiv@1__{HAIKU.replace('/', '--')}__-.json",
        f"ground.span@1__{HAIKU.replace('/', '--')}__inline.json",
        f"ground.statement@1__{HAIKU.replace('/', '--')}__sentence.json",
        f"pair@1__{HAIKU.replace('/', '--')}__-.json",
    ]
    span = load_card(out / "cards" / cards[1])
    assert span.labels.split == "gate" and span.labels.n == 104 and span.grounding is not None
    assert span.grounding.runs == 3 and span.grounding.planted is not None
    assert span.grounding.planted.n == 100
    pair = load_card(out / "cards" / cards[3])
    # 64 gate pairs were audited and 3 of those left unsure: 61 labels are the person's.
    assert pair.pairs is not None and pair.labels.n == 400 and pair.pairs.audited == 61
    assert pair.pairs.label_noise is not None and pair.pairs.label_noise.n == 96
    fact = load_card(out / "cards" / cards[0])
    assert fact.labels.n == 62 and fact.equivalence is not None
    table = (out / "results.md").read_text()
    assert "### The contest" in table and "span-inline stays" in table
    assert "Gold adjudication" in table
    items = (out / "items" / HAIKU.replace("/", "--") / "ground-span-inline.jsonl").read_text()
    assert len(items.splitlines()) == 263
    # Recorded answers spend nothing, and are cached apart from the live model's (R4), in the
    # folder odke itself caches that recording in.
    assert not ledger.exists()
    assert [p.name for p in (tmp_path / "cache" / "replay").iterdir()] == [
        replay_directory(tmp_path / "cache", replay).name
    ]
    estimated = calibrate.estimate(
        argparse.Namespace(
            cache=tmp_path / "cache",
            model=[HAIKU],
            judges={"ground"},
            variants=["span-inline"],
            runs=3,
        ),
        calibrate.load_g(),
        [],
        [],
    )
    (job,) = estimated[HAIKU]
    assert job["calls"] == 789 and job["cached"] == 0 and job["usd"] > 0
    # Asked through the cache, a recorded answer is never served as the live model's.
    assert not list((tmp_path / "cache").glob("[0-9a-f][0-9a-f]"))
