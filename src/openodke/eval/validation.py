"""Scoring a Validator: agreement with the verdicts a person would give.

The validator is the gate — what it refuses is not written — so its two
costly disagreements point in opposite directions. Refusing a fact a person
would accept is silent data loss (`wrongly_refused`). Accepting one a person
would refuse puts it in the graph (`wrongly_written`). And a conflict
accepted without its flag (`conflicts_missed`) is a contradiction nobody will
be asked to look at. Each is counted on its own line.

Agreement is reported as accuracy and as Cohen's kappa, because a validator
that accepts everything scores a high accuracy on a slice that is mostly
acceptable, and a kappa of zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from openodke.eval.formats import ValidationLabel, ValidationPrediction
from openodke.eval.report import Metric, StageReport, accuracy, cohen_kappa, macro_f1, per_class
from openodke.ontology import Ontology
from openodke.stages import Validator

ACTIONS = ("accept", "refuse", "conflict")


def run_validate(
    validator: Validator, labels: Iterable[ValidationLabel], ontology: Ontology
) -> list[ValidationPrediction]:
    """Validate every labelled fact against `ontology`, in-process."""
    predictions = []
    for row in labels:
        verdict = validator.validate(row.fact, ontology)
        predictions.append(
            ValidationPrediction(id=row.fact.id, action=verdict.action, reason=verdict.reason)
        )
    return predictions


def evaluate_validation(
    labels: Sequence[ValidationLabel], predictions: Iterable[ValidationPrediction]
) -> StageReport:
    """Agreement, kappa, per-action P/R/F1, and the three disagreements that cost."""
    by_id = {p.id: p for p in predictions}
    pairs = [(row.action, by_id[row.fact.id].action) for row in labels if row.fact.id in by_id]
    by_action, confusion = per_class(pairs, ACTIONS)

    notes = []
    if len(pairs) < len(labels):
        notes.append(
            f"{len(labels) - len(pairs)} labelled fact(s) had no prediction and were not scored"
        )
    known = {row.fact.id for row in labels}
    if stray := sum(1 for key in by_id if key not in known):
        notes.append(f"{stray} prediction(s) matched no labelled fact and were ignored")

    wrongly_refused = confusion["accept"]["refuse"] + confusion["conflict"]["refuse"]
    wrongly_written = confusion["refuse"]["accept"] + confusion["refuse"]["conflict"]
    conflicts_missed = confusion["conflict"]["accept"]
    if wrongly_refused:
        notes.append(f"{wrongly_refused} fact(s) a person would write were refused")
    metrics: dict[str, Metric] = {
        "agreement": accuracy(confusion),
        "kappa": cohen_kappa(confusion),
        "macro_f1": macro_f1(by_action),
        "wrongly_refused": wrongly_refused,
        "wrongly_written": wrongly_written,
        "conflicts_missed": conflicts_missed,
    }
    return StageReport(
        stage="validate",
        n=len(pairs),
        metrics=metrics,
        breakdown=by_action,
        confusion=confusion,
        notes=tuple(notes),
    )


__all__ = ["ACTIONS", "evaluate_validation", "run_validate"]
