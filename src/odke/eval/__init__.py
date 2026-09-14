"""Evaluation, BYOLD: bring your own labelled dataset.

Precision and recall claims need a harness or they are marketing — but a
number computed against the pipeline's own output measures nothing, and this
package cannot label a corpus for you. So `odke.eval` ships, for every stage,
three things and never a fourth:

- a **dataset format** — a JSONL row model in `odke.eval.formats` whose
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
"""

from odke.eval.formats import (
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
from odke.eval.report import StageReport
from odke.eval.routing import evaluate_routing, run_route

__all__ = [
    "LABEL_FORMATS",
    "PREDICTION_FORMATS",
    "CalibrationLabel",
    "GoldFact",
    "GroundingLabel",
    "LinkRow",
    "PairLabel",
    "RouteLabel",
    "RoutePrediction",
    "StageReport",
    "ValidationLabel",
    "ValidationPrediction",
    "describe",
    "dump_jsonl",
    "evaluate_routing",
    "load_jsonl",
    "run_route",
]
