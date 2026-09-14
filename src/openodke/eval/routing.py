"""Scoring a Router: P/R/F1 per label, and the failure that costs facts.

A router is wrong in two ways that do not cost the same. Sending marketing
prose to the extractor wastes a model call and risks over-extraction, which
the grounder and the validator can still catch. Skipping a chunk that states
facts loses them silently: no later stage sees the chunk, so no later stage
can recover it. The report therefore leads with the skip/extract confusion
and puts the skipped-facts count on its own line, instead of letting it
average away into an accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from openodke.eval.formats import RouteLabel, RoutePrediction
from openodke.eval.report import Metric, StageReport, accuracy, macro_f1, per_class, ratio
from openodke.stages import Router

ACTIONS = ("extract", "skip", "defer")
# What a prediction with no label is scored as, against a row that has one.
NO_LABEL = "<none>"


def run_route(router: Router, labels: Iterable[RouteLabel]) -> list[RoutePrediction]:
    """Route every labelled chunk in-process.

    Each row is routed on its own, so a `scope="document"` verdict skips only
    that row: a label row is a chunk, and the document it came from is not
    part of the format.
    """
    predictions = []
    for row in labels:
        verdict = router.route(row.as_chunk())
        predictions.append(RoutePrediction(id=row.id, action=verdict.action, label=verdict.label))
    return predictions


def evaluate_routing(
    labels: Sequence[RouteLabel], predictions: Iterable[RoutePrediction]
) -> StageReport:
    """Per-label P/R/F1, the action confusion, and how many fact chunks were skipped."""
    by_id = {p.id: p for p in predictions}
    joined = [(row, by_id[row.id]) for row in labels if row.id in by_id]
    notes = _join_notes(labels, by_id)

    actions, confusion = per_class(((r.action, p.action) for r, p in joined), ACTIONS)
    to_extract = confusion["extract"]
    skipped, deferred = to_extract["skip"], to_extract["defer"]
    wasted = confusion["skip"]["extract"] + confusion["defer"]["extract"]
    if skipped:
        notes.insert(
            0,
            f"{skipped} of {sum(to_extract.values())} chunks labelled extract were skipped: "
            "their facts never reached the extractor, and nothing downstream can recover them",
        )

    labelled = [(r.label, p.label or NO_LABEL) for r, p in joined if r.label is not None]
    by_label, label_confusion = per_class(labelled)
    if joined and not labelled:
        notes.append("no row carries a label, so only the action metrics are reported")

    metrics: dict[str, Metric] = {
        "action_accuracy": accuracy(confusion),
        "extract_precision": actions["extract"]["precision"],
        "extract_recall": actions["extract"]["recall"],
        "extract_f1": actions["extract"]["f1"],
        "skip_precision": actions["skip"]["precision"],
        "skip_recall": actions["skip"]["recall"],
        "facts_skipped": skipped,
        "facts_skipped_rate": ratio(skipped, sum(to_extract.values())),
        "facts_deferred": deferred,
        "wasted_extractions": wasted,
        "label_accuracy": accuracy(label_confusion),
        "label_macro_f1": macro_f1(by_label),
    }
    return StageReport(
        stage="route",
        n=len(joined),
        metrics=metrics,
        breakdown=by_label,
        confusion=confusion,
        notes=tuple(notes),
    )


def _join_notes(labels: Sequence[RouteLabel], by_id: dict[str, RoutePrediction]) -> list[str]:
    notes = []
    unpredicted = sum(1 for row in labels if row.id not in by_id)
    if unpredicted:
        notes.append(f"{unpredicted} labelled chunk(s) had no prediction and were not scored")
    known = {row.id for row in labels}
    unlabelled = sum(1 for key in by_id if key not in known)
    if unlabelled:
        notes.append(f"{unlabelled} prediction(s) matched no labelled chunk and were ignored")
    return notes


__all__ = ["ACTIONS", "NO_LABEL", "evaluate_routing", "run_route"]
