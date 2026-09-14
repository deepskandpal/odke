"""Evaluation, BYOLD: bring your own labelled dataset.

Precision and recall claims need a harness or they are marketing — but a
number computed against the pipeline's own output measures nothing, and this
package cannot label a corpus for you. So `openodke.eval` ships, for every stage,
three things and never a fourth:

- a **dataset format** — a JSONL row model in `openodke.eval.formats` whose
  docstring says exactly what one row is and what "correct" means;
- an **evaluator** — labels and predictions in, a `StageReport` out, and a
  `run_*` helper that scores a stage in-process where that is cheap;
- a **fixture** — a handful of rows under `tests/fixtures/eval/` so this
  package's own tests run.

**The fixtures are not a benchmark.** They exist to check the arithmetic;
every expected number in the tests is computed by hand, and no result on them
says anything about any extractor, grounder or resolver. There is no corpus,
no gold slice and no leaderboard. Brier, B-cubed and P/R/F1 are arithmetic,
so the subpackage adds no dependency to the base install (DECISIONS #1).

Cost needs no labels: `CostMeter` wraps the `LLMClient` a stage was built
with, so a run is measured without any stage Protocol changing.
"""

from openodke.eval.ablation import per_document, run_ablation
from openodke.eval.calibration import evaluate_calibration, run_score
from openodke.eval.cost import (
    CallRecord,
    CostMeter,
    CostReport,
    MeteredClient,
    StageCost,
    compare_costs,
)
from openodke.eval.extraction import evaluate_extraction, match_extraction, run_extract
from openodke.eval.formats import (
    LABEL_FORMATS,
    PREDICTION_FORMATS,
    CalibrationLabel,
    GoldFact,
    GroundingLabel,
    LinkRow,
    PairLabel,
    RouteLabel,
    RoutePrediction,
    ValidationLabel,
    ValidationPrediction,
    describe,
    dump_jsonl,
    load_jsonl,
)
from openodke.eval.grounding import evaluate_grounding, grounding_ablation, kept, run_ground
from openodke.eval.report import StageReport
from openodke.eval.resolution import as_triples, evaluate_resolution, links_from_clusters
from openodke.eval.routing import evaluate_routing, run_route
from openodke.eval.sinks import assert_idempotent, check_idempotency, jsonl_counts
from openodke.eval.validation import evaluate_validation, run_validate

__all__ = [
    "LABEL_FORMATS",
    "PREDICTION_FORMATS",
    "CalibrationLabel",
    "CallRecord",
    "CostMeter",
    "CostReport",
    "GoldFact",
    "GroundingLabel",
    "LinkRow",
    "MeteredClient",
    "PairLabel",
    "RouteLabel",
    "RoutePrediction",
    "StageCost",
    "StageReport",
    "ValidationLabel",
    "ValidationPrediction",
    "as_triples",
    "assert_idempotent",
    "check_idempotency",
    "compare_costs",
    "describe",
    "dump_jsonl",
    "evaluate_calibration",
    "evaluate_extraction",
    "evaluate_grounding",
    "evaluate_resolution",
    "evaluate_routing",
    "evaluate_validation",
    "grounding_ablation",
    "jsonl_counts",
    "kept",
    "links_from_clusters",
    "load_jsonl",
    "match_extraction",
    "per_document",
    "run_ablation",
    "run_extract",
    "run_ground",
    "run_route",
    "run_score",
    "run_validate",
]
