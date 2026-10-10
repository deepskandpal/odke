"""The track record: what each run's fixes were expected to gain, and what they gained (#141).

An expected gain is a prediction, and a prediction nobody checks is a guess
with a decimal point. So every run whose report is written appends one line to
a JSON Lines file, `track-record.jsonl` beside the report unless named: the
run, the fixes it predicted with their expected gains, and the run config that
produced it when that is known. When `odke eval compare A B` finds that B
applied a fix A predicted, it appends the measured gain beside the prediction.

- **A run is its outcomes.** Its id is a hash of the per-document counts that
  `--items` writes, so `compare`, which reads only those files, finds the run
  wherever they were moved, and two runs with the same outcomes are one run.
- **A fix is applied** when `--applied <fix-id>` says so, or when the two runs'
  configs differ at the knobs of exactly one fix A predicted. A change that
  touches the knobs of two is attributed to neither: the flag says which.
- **Measured** is the change in recall from the comparison, with its 95%
  interval and verdict: the primary metric when that is recall, otherwise the
  recall guardrail.

The file only grows. Nothing in it is rewritten, so a prediction cannot be
edited after its measurement arrives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import TypeAdapter

from openodke.eval.compare import Comparison, ItemRow
from openodke.eval.fixes import REMEDIES, Fix, Measured
from openodke.eval.stats import Paired
from openodke.types import Frozen

TRACK_RECORD = "track-record.jsonl"


class Prediction(Frozen):
    """One fix a run predicted, and its expected recall gain."""

    fix: str
    buckets: tuple[str, ...]
    expected: float
    ceiling: float
    exact: bool = False
    reference: float | None = None


class RunLine(Frozen):
    """One run: its id, where its report went, its config, and the fixes it predicted."""

    kind: Literal["run"] = "run"
    run: str
    at: str
    report: str | None = None
    row: str = ""
    config: dict[str, Any] | None = None
    predictions: tuple[Prediction, ...] = ()


class MeasuredLine(Frozen):
    """A fix run A predicted, applied in run B, with the gain the comparison measured."""

    kind: Literal["measured"] = "measured"
    run: str
    against: str
    at: str
    fix: str
    applied: str
    expected: float
    ceiling: float
    reference: float | None = None
    measured: float
    low: float
    high: float
    verdict: str


Line = RunLine | MeasuredLine
_LINE: TypeAdapter[Line] = TypeAdapter(Line)


def run_id(rows: Sequence[ItemRow]) -> str:
    """A run's id: a hash of its per-document outcomes, in id order, whatever file held them."""
    text = "\n".join(row.model_dump_json(exclude_none=True) for row in sorted(rows, key=_id))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read(path: str | Path) -> list[Line]:
    """Every line of a track record; none when there is no file yet."""
    source = Path(path)
    if not source.is_file():
        return []
    lines = []
    for n, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            lines.append(_LINE.validate_json(text))
        except ValueError as exc:
            raise ValueError(f"{source}:{n}: not a track record line: {exc}") from None
    return lines


def record_run(
    path: str | Path,
    run: str,
    fixes: Sequence[Fix],
    *,
    report: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> RunLine:
    """Append one run's predictions to the record at `path`, and return the line written."""
    line = RunLine(
        run=run,
        at=_now(),
        report=None if report is None else str(report),
        row=fixes[0].row if fixes else "",
        config=None if config is None else dict(config),
        predictions=tuple(
            Prediction(
                fix=fix.id,
                buckets=fix.buckets,
                expected=fix.recall.expected,
                ceiling=fix.recall.ceiling,
                exact=fix.recall.exact,
                reference=fix.recall.reference,
            )
            for fix in fixes
        ),
    )
    _append(path, [line])
    return line


def record_comparison(
    comparison: Comparison,
    a: Sequence[ItemRow],
    b: Sequence[ItemRow],
    records: Sequence[str | Path],
    *,
    applied: Sequence[str] = (),
    into: str | Path | None = None,
) -> tuple[list[MeasuredLine], list[str]]:
    """Measurements of the fixes B applied that A predicted, appended; and notes on what was not.

    `records` are the track records to look in: A's prediction is appended to
    in the first that holds it, unless `into` names another file.
    """
    a_id, b_id = run_id(a), run_id(b)
    found = [(Path(p), line) for p in records for line in read(p)]
    mine = [(p, ln) for p, ln in found if isinstance(ln, RunLine) and ln.run == a_id]
    if not mine:
        names = ", ".join(str(p) for p in records) or "no track record"
        return [], [f"track record: run A ({a_id}) predicted nothing in {names}"]
    home, predicted = mine[-1]
    theirs = [ln for _, ln in found if isinstance(ln, RunLine) and ln.run == b_id]
    notes: list[str] = []
    unknown = [fix for fix in applied if fix not in REMEDIES]
    if unknown:
        raise ValueError(f"--applied {unknown[0]!r}: no such fix; one of {', '.join(REMEDIES)}")
    if applied:
        how = dict.fromkeys(applied, "--applied")
    else:
        how, said = _from_configs(predicted, theirs[-1] if theirs else None)
        notes += said
    recall = _recall(comparison)
    if recall is None:
        return [], [*notes, "track record: the comparison carries no recall to measure a fix by"]
    by_fix = {p.fix: p for p in predicted.predictions}
    lines = []
    for fix, why in how.items():
        prediction = by_fix.get(fix)
        if prediction is None:
            notes.append(f"track record: B applied {fix}, which A did not predict")
            continue
        lines.append(
            MeasuredLine(
                run=a_id,
                against=b_id,
                at=_now(),
                fix=fix,
                applied=why,
                expected=prediction.expected,
                ceiling=prediction.ceiling,
                reference=prediction.reference,
                measured=recall.difference,
                low=recall.interval[0],
                high=recall.interval[1],
                verdict=recall.verdict,
            )
        )
    _append(into if into is not None else home, lines)
    return lines, notes


def measured(lines: Iterable[Line], fix: str) -> tuple[Measured, ...]:
    """Every measurement of `fix` in a record, as a report's fix carries them."""
    return tuple(
        Measured(
            run=ln.run,
            against=ln.against,
            applied=ln.applied,
            expected=ln.expected,
            ceiling=ln.ceiling,
            measured=ln.measured,
            low=ln.low,
            high=ln.high,
            verdict=ln.verdict,
        )
        for ln in lines
        if isinstance(ln, MeasuredLine) and ln.fix == fix
    )


def summary(lines: Sequence[Line], path: str | Path) -> list[str]:
    """What a report prints of its track record: predicted against measured, fix by fix."""
    runs = [ln for ln in lines if isinstance(ln, RunLine)]
    done = [ln for ln in lines if isinstance(ln, MeasuredLine)]
    predicted = sum(len(ln.predictions) for ln in runs)
    inside = sum(1 for ln in done if ln.expected <= ln.measured <= ln.ceiling)
    head = (
        f"track record  ({path}): {predicted} prediction(s) from {len(runs)} run(s), "
        f"{len(done)} measured"
    )
    if done:
        head += f", {inside} inside its expected-to-ceiling range"
    out = [head]
    for ln in done:
        reference = "" if ln.reference is None else f", reference {_points(ln.reference)}"
        out.append(
            f"  {ln.fix}: expected {_points(ln.expected)} (ceiling {_points(ln.ceiling)}"
            f"{reference}), measured {_points(ln.measured)} [{_points(ln.low)}, "
            f"{_points(ln.high)}] {ln.verdict}; {ln.run} → {ln.against}, {ln.applied}"
        )
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _from_configs(a: RunLine, b: RunLine | None) -> tuple[dict[str, str], list[str]]:
    """The fix A predicted whose knobs alone the configs differ at, if exactly one."""
    if b is None or a.config is None or b.config is None:
        return {}, ["track record: no config for both runs to diff; name the fix with --applied"]
    before, after = _flat(a.config), _flat(b.config)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    touched: dict[str, list[str]] = {}
    for prediction in a.predictions:
        knobs = REMEDIES[prediction.fix].knobs if prediction.fix in REMEDIES else ()
        hits = [k for k in changed if any(k == n or k.startswith(f"{n}.") for n in knobs)]
        if hits:
            touched[prediction.fix] = hits
    if len(touched) > 1:
        names = ", ".join(sorted(touched))
        return {}, [f"track record: the config change touches {names}; name one with --applied"]
    return {
        fix: "config: "
        + "; ".join(f"{k} {before.get(k, '(unset)')} → {after.get(k, '(unset)')}" for k in keys)
        for fix, keys in touched.items()
    }, []


def _flat(data: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flat(value, f"{path}."))
        else:
            out[path] = json.dumps(value, sort_keys=True)
    return out


def _recall(comparison: Comparison) -> Paired | None:
    if comparison.metric == "recall":
        return comparison.primary
    return comparison.guardrails.get("recall")


def _append(path: str | Path, lines: Sequence[Line]) -> None:
    if not lines:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line.model_dump_json(exclude_none=True) + "\n")


def _id(row: ItemRow) -> str:
    return row.id


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _points(share: float) -> str:
    return f"{share * 100:+.1f}"


__all__ = [
    "TRACK_RECORD",
    "Line",
    "MeasuredLine",
    "Prediction",
    "RunLine",
    "measured",
    "read",
    "record_comparison",
    "record_run",
    "run_id",
    "summary",
]
