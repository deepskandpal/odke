"""Scoring a gate: agreement with the verdicts a person would give.

What the gate refuses is not written, so its two costly disagreements point
in opposite directions. Refusing a fact a person would accept is silent data
loss (`wrongly_refused`). Accepting one a person would refuse puts it in the
graph (`wrongly_written`). And a conflict accepted without its flag
(`conflicts_missed`) is a contradiction nobody will be asked to look at. Each
is counted on its own line.

Agreement is reported as accuracy and as Cohen's kappa, because a gate that
accepts everything scores a high accuracy on a slice that is mostly
acceptable, and a kappa of zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from openodke._renamed import Renamed, module_getattr
from openodke.eval.formats import GateLabel, ValidationPrediction
from openodke.eval.report import (
    Metric,
    StageReport,
    accuracy,
    cohen_kappa,
    join_by_id,
    macro_f1,
    per_class,
)
from openodke.ontology import Ontology
from openodke.stages import Gate

ACTIONS = ("accept", "refuse", "conflict")


def run_validate(
    validator: Gate, labels: Iterable[GateLabel], ontology: Ontology
) -> list[ValidationPrediction]:
    """Validate every labelled fact against `ontology`, in-process."""
    predictions = []
    for row in labels:
        verdict = validator.validate(row.fact, ontology)
        predictions.append(
            ValidationPrediction(id=row.fact.id, action=verdict.action, reason=verdict.reason)
        )
    return predictions


def evaluate_gate(
    labels: Sequence[GateLabel], predictions: Iterable[ValidationPrediction]
) -> StageReport:
    """Agreement, kappa, per-action P/R/F1, and the three disagreements that cost."""
    joined, notes = join_by_id(labels, predictions, lambda row: row.fact.id)
    pairs = [(row.action, p.action) for row, p in joined]
    by_action, confusion = per_class(pairs, ACTIONS)

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


if TYPE_CHECKING:
    # What a type checker sees; at run time the 0.2 name comes from `__getattr__`.
    evaluate_validation = evaluate_gate

# The gate's evaluator, under its 0.2 name, until 2.0.0 (release audit call C).
__getattr__ = module_getattr(
    __name__,
    {"evaluate_validation": Renamed(evaluate_gate, "openodke.eval.validation.evaluate_gate")},
)

__all__ = ["ACTIONS", "evaluate_gate", "run_validate"]
