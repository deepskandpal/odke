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
from pathlib import Path
from typing import Any

from odke.eval.calibration import evaluate_calibration, run_score
from odke.eval.extraction import evaluate_extraction, run_extract
from odke.eval.formats import (
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
from odke.eval.grounding import evaluate_grounding, run_ground
from odke.eval.report import StageReport
from odke.eval.resolution import evaluate_resolution
from odke.eval.routing import evaluate_routing, run_route
from odke.eval.validation import evaluate_validation, run_validate
from odke.ontology import Ontology
from odke.types import Document, Fact

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
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of: {', '.join(STAGES)}")
    if run is not None and predictions is not None:
        raise ValueError("pass --predictions or --run, not both")
    component = load_stage(run) if run is not None else None

    if stage == "route":
        route_rows = load_jsonl(labels, RouteLabel)
        if component is not None:
            return evaluate_routing(route_rows, run_route(component, route_rows))
        return evaluate_routing(
            route_rows, load_jsonl(_required(stage, predictions), RoutePrediction)
        )

    if stage == "extract":
        gold = load_jsonl(labels, GoldFact)
        if component is not None:
            if documents is None or ontology is None:
                raise ValueError("--run with extract needs --documents and --ontology")
            docs = load_jsonl(documents, Document)
            return evaluate_extraction(gold, run_extract(component, docs, _ontology(ontology)))
        return evaluate_extraction(gold, load_jsonl(_required(stage, predictions), Fact))

    if stage == "ground":
        ground_rows = load_jsonl(labels, GroundingLabel)
        if component is not None:
            return evaluate_grounding(ground_rows, run_ground(component, ground_rows))
        return evaluate_grounding(ground_rows, _optional(predictions))

    if stage == "resolve":
        if component is not None:
            raise ValueError(
                "resolve cannot run from labels: a resolver takes facts, not pairs. Pass the "
                "links it emitted — a sink's links.jsonl, or a platform's merges — as --predictions"
            )
        pairs = load_jsonl(labels, PairLabel)
        return evaluate_resolution(pairs, load_jsonl(_required(stage, predictions), LinkRow))

    if stage == "score":
        score_rows = load_jsonl(labels, CalibrationLabel)
        if component is not None:
            return evaluate_calibration(score_rows, run_score(component, score_rows))
        return evaluate_calibration(score_rows, _optional(predictions))

    validate_rows = load_jsonl(labels, ValidationLabel)
    if component is not None:
        if ontology is None:
            raise ValueError("--run with validate needs --ontology")
        return evaluate_validation(
            validate_rows, run_validate(component, validate_rows, _ontology(ontology))
        )
    return evaluate_validation(
        validate_rows, load_jsonl(_required(stage, predictions), ValidationPrediction)
    )


def _required(stage: str, predictions: str | Path | None) -> str | Path:
    if predictions is None:
        raise ValueError(f"{stage} needs --predictions, or --run to produce them")
    return predictions


def _optional(predictions: str | Path | None) -> list[Fact] | None:
    return None if predictions is None else load_jsonl(predictions, Fact)


def _ontology(path: str | Path) -> Ontology:
    return Ontology.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["STAGES", "evaluate_files", "load_stage"]
