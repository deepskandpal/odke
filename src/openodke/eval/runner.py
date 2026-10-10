"""What `odke eval <stage>` does, as a function the CLI is a few lines over.

Labels always come from a file. Predictions come from a second file — the
cheap path, and the one a sink's output already fits — or, with `run`, from
importing a stage (`package.module:Name`) and running it over the labels
in-process. Running is offered where the labels carry everything the stage
needs: route, ground and score from the rows alone; extract with a documents
file and an ontology; validate with an ontology. Resolution is not runnable
from labels — a resolver takes facts, not pairs — so its links are always a
file, which is also how a platform's merges arrive.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED
from openodke.eval.calibration import evaluate_calibration, run_score
from openodke.eval.diagnosis import EXAMPLES, Offered, diagnose, gold_view
from openodke.eval.eval_report import Dataset, EvalReport, Run, extraction_rows, from_stage
from openodke.eval.extraction import evaluate_extraction, run_extract
from openodke.eval.formats import (
    LABEL_FORMATS,
    CalibrationLabel,
    GoldFact,
    GroundingLabel,
    LinkRow,
    PairLabel,
    RouteLabel,
    RoutePrediction,
    ValidationLabel,
    ValidationPrediction,
    load_jsonl,
)
from openodke.eval.grounding import evaluate_grounding, run_ground
from openodke.eval.report import StageReport
from openodke.eval.resolution import evaluate_resolution
from openodke.eval.routing import evaluate_routing, run_route
from openodke.eval.validation import evaluate_validation, run_validate
from openodke.ontology import Ontology
from openodke.types import Document, Fact

STAGES = tuple(LABEL_FORMATS)


def load_stage(spec: str) -> Any:
    """`package.module:Name` imported; a class is instantiated with no arguments."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"--run takes package.module:Name, got {spec!r}")
    try:
        found = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load {spec!r}: {exc}") from exc
    if not isinstance(found, type):
        return found
    try:
        return found()
    except TypeError as exc:
        raise ValueError(
            f"cannot instantiate {spec!r} with no arguments ({exc}); "
            "point --run at a module-level instance instead"
        ) from exc


def evaluate_files(
    stage: str,
    labels: str | Path,
    predictions: str | Path | None = None,
    *,
    run: str | None = None,
    ontology: str | Path | None = None,
    documents: str | Path | None = None,
) -> StageReport:
    """Load the labels for `stage`, get predictions from a file or a run, and score them."""
    rows, predicted = load_inputs(
        stage, labels, predictions, run=run, ontology=ontology, documents=documents
    )
    return score(stage, rows, predicted)


def score(stage: str, rows: Sequence[Any], predicted: Any) -> StageReport:
    """Score loaded labels against loaded (or produced) predictions."""
    return SCORERS[stage](rows, predicted)


def load_inputs(
    stage: str,
    labels: str | Path,
    predictions: str | Path | None = None,
    *,
    run: str | None = None,
    ontology: str | Path | None = None,
    documents: str | Path | None = None,
) -> tuple[list[Any], Any]:
    """The labelled rows for `stage`, and its predictions from a file or a run.

    Split from scoring so one set of predictions can be both scored and written
    out per item (`odke eval --items`) without running a stage twice.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(STAGES)}")
    if run is not None and predictions is not None:
        raise ValueError("pass --predictions or --run, not both")
    component = load_stage(run) if run is not None else None

    if stage == "route":
        route_rows = load_jsonl(labels, RouteLabel)
        if component is not None:
            return route_rows, run_route(component, route_rows)
        return route_rows, load_jsonl(_required(stage, predictions), RoutePrediction)

    if stage == "extract":
        gold = load_jsonl(labels, GoldFact)
        if component is not None:
            if documents is None or ontology is None:
                raise ValueError("--run with extract needs --documents and --ontology")
            docs = load_jsonl(documents, Document)
            return gold, run_extract(component, docs, _ontology(ontology))
        return gold, load_jsonl(_required(stage, predictions), Fact)

    if stage == "ground":
        ground_rows = load_jsonl(labels, GroundingLabel)
        if component is not None:
            return ground_rows, run_ground(component, ground_rows)
        return ground_rows, _optional(predictions)

    if stage == "resolve":
        if component is not None:
            raise ValueError(
                "resolve cannot run from labels: a resolver takes facts, not pairs. Pass the "
                "links it emitted — a sink's links.jsonl, or a platform's merges — as --predictions"
            )
        pairs = load_jsonl(labels, PairLabel)
        return pairs, load_jsonl(_required(stage, predictions), LinkRow)

    if stage == "score":
        score_rows = load_jsonl(labels, CalibrationLabel)
        if component is not None:
            return score_rows, run_score(component, score_rows)
        return score_rows, _optional(predictions)

    validate_rows = load_jsonl(labels, ValidationLabel)
    if component is not None:
        if ontology is None:
            raise ValueError("--run with validate needs --ontology")
        return validate_rows, run_validate(component, validate_rows, _ontology(ontology))
    return validate_rows, load_jsonl(_required(stage, predictions), ValidationPrediction)


SCORERS: dict[str, Callable[[Any, Any], StageReport]] = {
    "route": evaluate_routing,
    "extract": evaluate_extraction,
    "ground": evaluate_grounding,
    "resolve": evaluate_resolution,
    "score": evaluate_calibration,
    "validate": evaluate_validation,
}


def report_files(
    stage: str,
    labels: str | Path,
    predictions: str | Path | None = None,
    *,
    run: str | None = None,
    ontology: str | Path | None = None,
    documents: str | Path | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> EvalReport:
    """`evaluate_files` as an eval report."""
    rows, predicted = load_inputs(
        stage, labels, predictions, run=run, ontology=ontology, documents=documents
    )
    return report_inputs(
        stage,
        rows,
        predicted,
        labels=labels,
        ontology=ontology,
        resamples=resamples,
        seed=seed,
        level=level,
    )


def report_inputs(
    stage: str,
    rows: Sequence[Any],
    predicted: Any,
    *,
    labels: str | Path | None = None,
    ontology: str | Path | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
    offered: Offered | None = None,
    texts: Mapping[str, str] | None = None,
    examples: int = EXAMPLES,
) -> EvalReport:
    """Loaded labels and predictions scored as an eval report (`load_inputs`, then this).

    For `extract` the report has a row: precision, recall and F1 with their
    ranges, the counts, and conformance when an ontology is given, whether or
    not anything ran; and the diagnosis of its misses (#140), with `offered`
    the trace of what the extractor was shown and `texts` the documents.
    Every other stage's report carries its `StageReport` and no row.
    `labels` names the dataset.
    """
    name = Path(labels).name if labels is not None else f"{stage} labels"
    dataset = Dataset(name=name, path=None if labels is None else str(labels))
    found = score(stage, rows, predicted)
    if stage != "extract":
        return from_stage(found, run=Run(dataset=dataset))
    schema = _ontology(ontology) if ontology is not None else None
    found_rows, how = extraction_rows(
        [("extract", list(predicted), None)],
        rows,
        ontology=schema,
        resamples=resamples,
        seed=seed,
        level=level,
    )
    dataset = dataset.model_copy(
        update={"documents": len({g.doc_id for g in rows}), "labels": len(rows)}
    )
    view = gold_view(rows, list(predicted), texts=texts, row="extract")
    inverses = schema.inverses if schema is not None else {}
    diagnosed = diagnose(view, offered=offered, inverses=inverses, examples=examples)
    return from_stage(
        found,
        rows=found_rows,
        bootstrap=how,
        run=Run(dataset=dataset),
        diagnosis=diagnosed.buckets,
    )


def _required(stage: str, predictions: str | Path | None) -> str | Path:
    if predictions is None:
        raise ValueError(f"{stage} needs --predictions, or --run to produce them")
    return predictions


def _optional(predictions: str | Path | None) -> list[Fact] | None:
    return None if predictions is None else load_jsonl(predictions, Fact)


def _ontology(path: str | Path) -> Ontology:
    return Ontology.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "SCORERS",
    "STAGES",
    "evaluate_files",
    "load_inputs",
    "load_stage",
    "report_files",
    "report_inputs",
    "score",
]
