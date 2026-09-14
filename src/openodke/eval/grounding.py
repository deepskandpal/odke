"""Scoring a Grounder: verdict accuracy, its confusion, and the ablation.

The grounder is the precision stage, so its two errors are not symmetric.
Stamping `supported` on a fact the passage does not support lets a false fact
through to the graph (`false_support`); stamping anything else on a fact it
does support throws a true one away (`lost_support`). Both are counted, apart
from the accuracy that would blur them.

The ablation — extraction alone against extraction plus grounding (#38) — is
what the release's sentence rests on, so its scorer lives here and the runner
does not. `grounding_ablation` takes one labelled slice and the same facts
after a grounder has stamped them, and reports the precision of what survives
with grounding off (everything) and on (what the verdicts keep). Against
extraction gold instead, the same comparison is
`evaluate_extraction(gold, facts)` beside
`evaluate_extraction(gold, [f for f in grounded if kept(f)])`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

from openodke.eval.formats import GroundingLabel
from openodke.eval.report import Metric, StageReport, accuracy, macro_f1, per_class, prf, ratio
from openodke.stages import Grounder
from openodke.types import Fact, GroundingVerdict

VERDICTS = ("supported", "contradicted", "not_found")
# What the pipeline's gate would drop. `unchecked` is kept: a grounder that did
# not decide is the pass-through, and the pass-through refuses nothing.
DROPPED: tuple[GroundingVerdict, ...] = (GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND)


def run_ground(grounder: Grounder, labels: Iterable[GroundingLabel]) -> list[Fact]:
    """Ground every labelled fact against its own passage, in-process."""
    return [grounder.ground(row.fact, row.as_document()) for row in labels]


def kept(fact: Fact, drop: Collection[GroundingVerdict] = DROPPED) -> bool:
    """Whether a fact survives grounding under a gate that drops `drop`."""
    return fact.verdict not in drop


def evaluate_grounding(
    labels: Sequence[GroundingLabel], predictions: Iterable[Fact] | None = None
) -> StageReport:
    """Accuracy per verdict, the confusion, and the two errors that cost differently."""
    pairs, notes = _join(labels, predictions)
    by_verdict, confusion = per_class(
        ((row.verdict, f.verdict.value) for row, f in pairs), VERDICTS
    )
    unchecked = sum(1 for _, f in pairs if f.verdict is GroundingVerdict.UNCHECKED)
    if unchecked:
        notes.append(f"{unchecked} fact(s) came back unchecked; they are scored wrong, not skipped")
    false_support = sum(
        1
        for row, f in pairs
        if row.verdict != "supported" and f.verdict is GroundingVerdict.SUPPORTED
    )
    lost_support = sum(
        1
        for row, f in pairs
        if row.verdict == "supported" and f.verdict is not GroundingVerdict.SUPPORTED
    )
    metrics: dict[str, Metric] = {
        "accuracy": accuracy(confusion),
        # Over the three real verdicts: `unchecked` is a failure to answer, not a class.
        "macro_f1": macro_f1({v: by_verdict[v] for v in VERDICTS}),
        "false_support": false_support,
        "lost_support": lost_support,
        "unchecked": unchecked,
    }
    return StageReport(
        stage="ground",
        n=len(pairs),
        metrics=metrics,
        breakdown=by_verdict,
        confusion=confusion,
        notes=tuple(notes),
    )


def grounding_ablation(
    labels: Sequence[GroundingLabel],
    grounded: Iterable[Fact],
    *,
    drop: Collection[GroundingVerdict] = DROPPED,
) -> StageReport:
    """Precision and recall of the kept facts with grounding off and on.

    A fact is true when its label is `supported`. Off, every labelled fact is
    kept, whatever verdict its row happens to carry. On, a fact is kept when
    its grounded verdict is not in `drop`. Recall is the share of true facts
    still kept, which is the price of the precision grounding buys.
    """
    off = [(row, True) for row in labels]
    pairs, notes = _join(labels, grounded)
    on = [(row, kept(f, drop)) for row, f in pairs]
    rows = {"off": _kept(off), "on": _kept(on)}
    p_off, p_on = rows["off"]["precision"], rows["on"]["precision"]
    gain = p_on - p_off if p_on is not None and p_off is not None else None
    if gain is not None:
        notes.insert(
            0,
            f"grounding moved precision from {p_off:.3f} to {p_on:.3f}, keeping "
            f"{rows['on']['tp']} of {rows['off']['tp']} true facts",
        )
    metrics: dict[str, Metric] = {
        "precision_off": p_off,
        "precision_on": p_on,
        "precision_gain": gain,
        "recall_on": rows["on"]["recall"],
        "kept_off": rows["off"]["kept"],
        "kept_on": rows["on"]["kept"],
    }
    return StageReport(
        stage="ground:ablation", n=len(pairs), metrics=metrics, breakdown=rows, notes=tuple(notes)
    )


def _join(
    labels: Sequence[GroundingLabel], predictions: Iterable[Fact] | None
) -> tuple[list[tuple[GroundingLabel, Fact]], list[str]]:
    if predictions is None:
        return [(row, row.fact) for row in labels], []
    by_id = {f.id: f for f in predictions}
    pairs = [(row, by_id[row.fact.id]) for row in labels if row.fact.id in by_id]
    notes = []
    if len(pairs) < len(labels):
        notes.append(
            f"{len(labels) - len(pairs)} labelled fact(s) had no prediction and were not scored"
        )
    known = {row.fact.id for row in labels}
    if stray := sum(1 for key in by_id if key not in known):
        notes.append(f"{stray} prediction(s) matched no labelled fact and were ignored")
    return pairs, notes


def _kept(rows: Sequence[tuple[GroundingLabel, bool]]) -> dict[str, Metric]:
    true_kept = sum(1 for row, keep in rows if keep and row.verdict == "supported")
    false_kept = sum(1 for row, keep in rows if keep and row.verdict != "supported")
    true_dropped = sum(1 for row, keep in rows if not keep and row.verdict == "supported")
    counts = prf(true_kept, false_kept, true_dropped)
    return {
        **counts,
        "kept": true_kept + false_kept,
        "dropped": len(rows) - true_kept - false_kept,
        "true_rate": ratio(true_kept, true_kept + false_kept),
    }


__all__ = [
    "DROPPED",
    "VERDICTS",
    "evaluate_grounding",
    "grounding_ablation",
    "kept",
    "run_ground",
]
