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

Neither does span width, and it is the one evaluator that scores a real error
without them: `evaluate_spans` splits cited span width by the grounder's
verdict. Where the narrow spans are the `not_found` ones, the citations are too
narrow to carry their claims, and the `not_found` rate measures that with no
gold set at all.

Whether a change between two runs was real is `openodke.eval.stats`: a paired
bootstrap over the items both runs scored, three verdicts (better, worse,
inconclusive) and the detection limit printed beside every one.
`compare_items` runs it over two runs' per-item outcomes (`item_rows`).
"""

from openodke.eval.ablation import per_document, run_ablation
from openodke.eval.calibration import evaluate_calibration, run_score
from openodke.eval.compare import Comparison, ItemRow, compare_files, compare_items, item_rows
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
from openodke.eval.spans import evaluate_spans, load_facts, span_width
from openodke.eval.stats import (
    McNemar,
    Paired,
    bootstrap_interval,
    detection_limit,
    mcnemar,
    paired_bootstrap,
)
from openodke.eval.validation import evaluate_validation, run_validate

__all__ = [
    "LABEL_FORMATS",
    "PREDICTION_FORMATS",
    "CalibrationLabel",
    "CallRecord",
    "Comparison",
    "CostMeter",
    "CostReport",
    "GoldFact",
    "GroundingLabel",
    "ItemRow",
    "LinkRow",
    "McNemar",
    "MeteredClient",
    "PairLabel",
    "Paired",
    "RouteLabel",
    "RoutePrediction",
    "StageCost",
    "StageReport",
    "ValidationLabel",
    "ValidationPrediction",
    "as_triples",
    "assert_idempotent",
    "bootstrap_interval",
    "check_idempotency",
    "compare_files",
    "compare_items",
    "compare_costs",
    "describe",
    "detection_limit",
    "dump_jsonl",
    "evaluate_calibration",
    "evaluate_extraction",
    "evaluate_grounding",
    "evaluate_resolution",
    "evaluate_routing",
    "evaluate_spans",
    "evaluate_validation",
    "grounding_ablation",
    "item_rows",
    "jsonl_counts",
    "kept",
    "links_from_clusters",
    "load_facts",
    "load_jsonl",
    "match_extraction",
    "mcnemar",
    "paired_bootstrap",
    "per_document",
    "run_ablation",
    "run_extract",
    "run_ground",
    "run_route",
    "run_score",
    "run_validate",
    "span_width",
]
