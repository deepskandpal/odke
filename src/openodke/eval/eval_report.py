"""The eval report: one versioned JSON document for every run the Evaluator scores.

A `StageReport` is what one evaluator concluded about one stage, and its
shape is whatever that stage partitions on. That suits a person reading one
table. It does not suit a CI job, a dashboard, or a second run to compare
against, because each of those needs to know where precision is without
knowing which stage made the report. So an Evaluator run also writes an
`EvalReport`, whose shape is fixed and versioned (`schema_version`, "1.1"):

- **rows**: one per configuration scored against gold facts (a single
  extraction, or the ablation's three), each with precision, recall and F1
  and a 95% range around each (`openodke.eval.bootstrap`, over documents);
  hits, over-extraction and under-extraction; conformance to the ontology;
  hallucination, where the dataset defines it; cost and latency, where the
  run was metered;
- **run**: the package version, the models, the registered prompt keys the
  stages sent (DECISIONS #27), and the dataset;
- **stages**: the `StageReport`s the rows were computed beside, unchanged, so
  nothing a 0.x reader used is lost (DECISIONS #24);
- **diagnosis**: where the final row lost facts, every miss in one cause
  bucket, and its false positives split by what the gate did
  (`openodke.eval.diagnosis`, #140);
- **fixes**: what should change, ranked by the recall gain the arithmetic
  expects (`openodke.eval.fixes`, #141);
- **comparison**: this run against a baseline (#142); **calibration**: empty,
  and typed, until #135 fills it.

1.1 adds the sections that score without gold, or with gold that is
incomplete, each `null` when not asked for: **judged_precision**, a judge's
precision corrected by a labelled sample (#144, `openodke.eval.ppi`), and
**adjudication**, the predictions the gold lacks that the grounder supports
(#145, `openodke.eval.adjudication`), and **pooled_recall**, two or more
pipelines' recall relative to the pool of what they found (#146,
`openodke.eval.pooling`).

`eval_report.schema.json`, beside this file, is the contract. `check_report`
holds a report to it in plain Python, with no dependency; every report the
package writes is checked before it is written. A reader checks
`schema_version`: a minor version adds fields, a major one changes them.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED, Range, bootstrap
from openodke.eval.compare import Comparison
from openodke.eval.cost import CallRecord, StageCost
from openodke.eval.diagnosis import Bucket
from openodke.eval.diagnosis import render as render_buckets
from openodke.eval.extraction import document_counts
from openodke.eval.fixes import Fix
from openodke.eval.fixes import render as render_fixes
from openodke.eval.formats import GoldFact
from openodke.eval.report import Metric, StageReport, _fmt, _table, prf
from openodke.ground.checks import CHECKS, schema_problem
from openodke.ontology import Ontology
from openodke.types import Fact, Frozen

SCHEMA_VERSION = "1.1"
SCHEMA_PATH = Path(__file__).with_name("eval_report.schema.json")


Average = Literal["micro", "macro"]


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


class Estimate(Frozen):
    """A number and the range around it; all three `None` where it is undefined."""

    value: float | None
    low: float | None = None
    high: float | None = None

    @classmethod
    def of(cls, value: Metric, found: Range) -> Estimate:
        """`value` with its bootstrap range, widened, if it must be, to include it.

        On very few documents a percentile range can miss the number measured on
        all of them. A range printed beside a number that it excludes reads as
        an error, so the range is stretched to reach it.
        """
        if value is None:
            return cls(value=None)
        low, high = found
        if low is None or high is None:
            return cls(value=float(value))
        return cls(value=float(value), low=min(low, value), high=max(high, value))


class Performance(Frozen):
    """Precision, recall and F1, each with its range.

    `micro` pools every fact before dividing; `macro` is the mean of each
    document's own number, which is how Text2KGBench averages.
    """

    precision: Estimate
    recall: Estimate
    f1: Estimate
    average: Average = "micro"


class Counts(Frozen):
    """Hits, and the two ways to miss: a fact written that the gold lacks, and the reverse.

    A wrong value or a wrong entity is one of each: a wrong fact was written and
    the right one was not. `predicted` and `gold` are what was scored, so
    `predicted == hits + over_extraction` and `gold == hits + under_extraction`.
    `uncited` counts the spurious predictions among `over_extraction` that cite
    no document, and so are in no document's counts; `unscored`, predictions
    left out of every number, and the stage's notes say why.
    """

    hits: int = Field(ge=0)
    over_extraction: int = Field(ge=0)
    under_extraction: int = Field(ge=0)
    predicted: int = Field(ge=0)
    gold: int = Field(ge=0)
    uncited: int = Field(default=0, ge=0)
    unscored: int = Field(default=0, ge=0)
    documents: int = Field(ge=0)


class Conformance(Frozen):
    """The share of predicted facts the ontology has room for.

    `checks` says what was checked: the relation alone (a benchmark's own
    measure), or the relation and both ends' types (`conforms`).
    """

    rate: float | None
    conformant: int = Field(ge=0)
    facts: int = Field(ge=0)
    checks: tuple[str, ...]
    average: Average = "micro"


class Hallucination(Frozen):
    """Predicted facts the dataset calls invented, by its own `definition`.

    `rate` is `hallucinated / facts`. `subject`, `relation` and `object` are
    the dataset's per-part rates where it reports them.
    """

    definition: str
    hallucinated: int = Field(ge=0)
    facts: int = Field(ge=0)
    rate: float | None
    subject: float | None = None
    relation: float | None = None
    object: float | None = None


class Cost(Frozen):
    """What a configuration's model calls cost. `usd` is `None` when any call was unpriced."""

    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    usd: float | None
    priced_usd: float
    unpriced_calls: int = Field(ge=0)


class Latency(Frozen):
    """Wall-clock time inside model calls, summed: a concurrent run beats it."""

    calls: int = Field(ge=0)
    seconds: float
    per_call: float | None


class Row(Frozen):
    """One configuration scored against gold facts."""

    name: str
    performance: Performance
    counts: Counts
    conformance: Conformance | None = None
    hallucination: Hallucination | None = None
    cost: Cost | None = None
    latency: Latency | None = None


class Bootstrap(Frozen):
    """How the ranges were drawn, so a reader can draw them again."""

    unit: Literal["document"] = "document"
    units: int = Field(ge=0)
    resamples: int = Field(ge=0)
    seed: int
    level: float
    method: Literal["percentile"] = "percentile"


class Dataset(Frozen):
    """What was scored: a benchmark's name, or the labels file's."""

    name: str
    path: str | None = None
    documents: int | None = None
    # Label rows: gold facts for extraction, labelled rows for a stage.
    labels: int | None = None
    # The dataset's own coordinates: source, ontology id, split.
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def _version() -> str:
    return importlib.metadata.version("openodke")


class Run(Frozen):
    """What produced the report: the package, the models, the prompts, the data."""

    openodke: str = Field(default_factory=_version)
    # Role (extract, ground) -> provider-qualified model, as the calls named it.
    models: dict[str, str] = Field(default_factory=dict)
    # Registered prompt keys the stages sent, `id@version` (DECISIONS #27).
    prompts: tuple[str, ...] = ()
    dataset: Dataset | None = None


# --------------------------------------------------------------------------- #
# Without gold (#144)
# --------------------------------------------------------------------------- #


class CoverageTotals(Frozen):
    """The coverage report's totals (`openodke.coverage`): where facts could be and none are.

    With no gold, nothing says what a pipeline missed. These counts are the
    nearest thing, and they need no model: sentences naming two known entities
    that no fact covers, and known entities no fact of their document names.
    """

    documents: int = Field(ge=0)
    sentences: int = Field(ge=0)
    uncovered: int = Field(ge=0)
    missed_entities: int = Field(ge=0)
    not_offered: tuple[str, ...] | None
    unused: tuple[str, ...]


class JudgedPrecision(Frozen):
    """Precision with no gold (#144): a judge graded every fact, a person a random sample.

    `judge_only` is the share of the `facts` the judge called supported. It has
    no range, because a range would only say how precisely the judge is biased.
    `corrected` is the prediction-powered estimate (`method`): the judge's
    share plus the mean gap between the labels and the judge on the `labels`
    sample, with its interval at `level`. `labels_only` is the labels' own
    share and classical interval. Under `minimum` labels the corrected number
    is an uncalibrated estimate, and `calibrated` is false. `false_support`
    and `lost_support` are the judge's two errors on the sample.
    """

    facts: int = Field(ge=0)
    supported: int = Field(ge=0)
    labels: int = Field(ge=0)
    labelled_supported: int = Field(ge=0)
    false_support: int = Field(ge=0)
    lost_support: int = Field(ge=0)
    judge_only: Estimate
    corrected: Estimate
    labels_only: Estimate
    level: float
    method: Literal["ppi-closed-form"] = "ppi-closed-form"
    minimum: int = Field(ge=0)
    calibrated: bool
    # The coverage report over the judged facts; None without their documents.
    coverage: CoverageTotals | None = None


# --------------------------------------------------------------------------- #
# Incomplete gold (#145)
# --------------------------------------------------------------------------- #


class AdjudicatedRow(Frozen):
    """One row's precision, strict and with the predictions the grounder vouches for as hits.

    `not_in_gold` is the row's predictions the gold lacks (its over-extraction);
    `asked`, those of them that cite a document to be grounded against;
    `possibly_missing`, those the grounder supported in enough runs. `strict`
    is the row's own precision and range, unchanged; `adjudicated` counts the
    possibly missing as hits, resampled on the same draws.
    """

    name: str
    not_in_gold: int = Field(ge=0)
    asked: int = Field(ge=0)
    possibly_missing: int = Field(ge=0)
    strict: Estimate
    adjudicated: Estimate


class Adjudication(Frozen):
    """Gold adjudication (#145): each prediction the gold lacks, grounded `runs` times.

    One supported in `needed` of the `runs` is possibly missing from gold.
    `questions` is how many distinct predictions were asked, each once for
    every row it is in. `audit` is where the list was written.
    """

    runs: int = Field(ge=1)
    needed: int = Field(ge=1)
    questions: int = Field(ge=0)
    rows: tuple[AdjudicatedRow, ...]
    audit: str | None = None


# --------------------------------------------------------------------------- #
# Recall relative to a pool (#146)
# --------------------------------------------------------------------------- #


class PoolMember(Frozen):
    """One run in a pool: its share of what every run found, and its own coverage.

    `supported` is the run's distinct supported facts, per document, after
    normalisation; `unique`, those no other run found. `relative_recall` is
    `supported` over the pool's, with its range.
    """

    name: str
    facts: int = Field(ge=0)
    supported: int = Field(ge=0)
    unique: int = Field(ge=0)
    relative_recall: Estimate
    # The coverage report over the run's facts; None without their documents.
    coverage: CoverageTotals | None = None


class PooledRecall(Frozen):
    """Recall relative to a pool (#146): two or more runs on the same documents, no gold.

    `pool` is every supported fact any run wrote, per document, once each after
    normalisation; `documents`, the documents it cites, which are the
    bootstrap's units. `caveat` says why the numbers overstate true recall.
    """

    pool: int = Field(ge=0)
    documents: int = Field(ge=0)
    runs: tuple[PoolMember, ...]
    bootstrap: Bootstrap
    caveat: str


class EvalReport(Frozen):
    """One run, scored: the versioned document `odke eval` and `odke bench run` write."""

    schema_version: str = SCHEMA_VERSION
    title: str
    # As a StageReport's: the rows scored — gold facts, or labelled rows.
    n: int = Field(ge=0)
    run: Run = Field(default_factory=Run)
    bootstrap: Bootstrap | None = None
    rows: tuple[Row, ...] = ()
    stages: tuple[StageReport, ...] = ()
    notes: tuple[str, ...] = ()
    # Every miss of the final row in one cause bucket, then its false positives (#140),
    diagnosis: tuple[Bucket, ...] = ()
    # ranked fixes with their expected gain (#141),
    fixes: tuple[Fix, ...] = ()
    # this run against a baseline: `odke eval compare`'s block, as `--json` prints it (#142),
    comparison: Comparison | None = None
    # and, typed and empty until filled, the grounder's calibration cards (#135).
    calibration: tuple[dict[str, Any], ...] = ()
    # Added in 1.1, so a 1.0 report has none of them: precision with no gold,
    # the judge corrected by a labelled sample (#144).
    judged_precision: JudgedPrecision | None = None
    # gold that is incomplete, adjudicated by the grounder (#145).
    adjudication: Adjudication | None = None
    # and recall relative to a pool of pipelines, with no gold (#146).
    pooled_recall: PooledRecall | None = None

    def as_json(self) -> str:
        return self.model_dump_json(indent=2)

    def write(self, path: str | Path) -> Path:
        """The report as JSON at `path`, checked against the schema first."""
        data = self.model_dump(mode="json")
        problems = check_report(data)
        if problems:
            raise ValueError("the report does not match its schema: " + "; ".join(problems[:5]))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return target

    def render(self) -> str:
        """The plain-text form: the rows with their ranges, then each stage's own table.

        Read from the same fields the JSON holds, at three decimals. A stage
        whose breakdown is keyed by the rows is the rows table again, so only
        its notes are printed.
        """
        head = f"{self.title}  (n={self.n})"
        lines = [head]
        names = {row.name for row in self.rows}
        if self.rows:
            lines += ["", *_rows_table(self.rows)]
            if self.bootstrap is not None:
                lines.append(f"  {_ranges(self.bootstrap)}")
        lines += _without_gold(self)
        for stage in self.stages:
            if stage.breakdown and set(stage.breakdown) == names:
                body = [f"  - {note}" for note in stage.notes]
            else:
                text = stage.render().splitlines()
                body = text[1:] if text and text[0] == head else text
            while body and not body[0].strip():
                body = body[1:]
            if body:
                lines += ["", *body]
        if self.notes:
            lines += ["", *(f"  - {note}" for note in self.notes)]
        if self.diagnosis:
            lines += ["", *render_buckets(self.diagnosis)]
        if self.fixes:
            lines += ["", *render_fixes(self.fixes)]
        if self.comparison is not None:
            lines += ["", *self.comparison.render().splitlines()]
        lines += ["", *_provenance(self)]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Building rows
# --------------------------------------------------------------------------- #


def micro(units: Sequence[tuple[int, int, int]]) -> dict[str, Metric]:
    """Precision, recall and F1 over `(tp, fp, fn)` per document, pooled."""
    tp = sum(u[0] for u in units)
    fp = sum(u[1] for u in units)
    fn = sum(u[2] for u in units)
    scores = prf(tp, fp, fn)
    return {k: scores[k] for k in ("precision", "recall", "f1")}


def performance(
    point: Mapping[str, Metric],
    ranges: Mapping[str, Range],
    average: Average = "micro",
) -> Performance:
    """Point values beside their ranges, as one `Performance`."""
    return Performance(
        **{
            k: Estimate.of(point.get(k), ranges.get(k, (None, None)))
            for k in ("precision", "recall", "f1")
        },
        average=average,
    )


Configuration = tuple[str, Sequence[Fact], Sequence[CallRecord] | None]


def extraction_rows(
    configurations: Sequence[Configuration],
    gold: Sequence[GoldFact],
    *,
    ontology: Ontology | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[list[Row], Bootstrap]:
    """Each `(name, facts, calls)` against gold facts, with ranges over the documents.

    The numbers are `evaluate_extraction`'s: the same matching, so the same
    precision, recall and F1 to the last digit. Each labelled document is one
    unit of the bootstrap, with the counts `odke eval --items` writes for it
    (`document_counts`). A spurious prediction that cites no document is in no
    document: it is in the numbers, and in every draw as it is in the run,
    and `counts.uncited` says how many. `calls`, when the run was metered, give
    a row its cost and latency; `ontology` gives every row its conformance.
    """
    rows = []
    for name, facts, calls in configurations:
        found = document_counts(gold, facts)
        units = list(found.by_doc.values())
        fixed = (0, found.uncited, 0)

        def statistic(
            draw: Sequence[tuple[int, int, int]], fixed: tuple[int, int, int] = fixed
        ) -> dict[str, Metric]:
            return micro([*draw, fixed])

        tp, fp, fn = (sum(u[i] for u in units) + fixed[i] for i in range(3))
        ranges = bootstrap(units, statistic, resamples=resamples, seed=seed, level=level)
        counts = Counts(
            hits=tp,
            over_extraction=fp,
            under_extraction=fn,
            predicted=tp + fp,
            gold=tp + fn,
            uncited=found.uncited,
            unscored=found.unscored,
            documents=len(units),
        )
        rows.append(
            Row(
                name=name,
                performance=performance(statistic(units), ranges),
                counts=counts,
                conformance=conformance(facts, ontology),
                **spend(name, calls),
            )
        )
    documents = len(dict.fromkeys(g.doc_id for g in gold))
    return rows, Bootstrap(units=documents, resamples=resamples, seed=seed, level=level)


def conforms(fact: Fact, ontology: Ontology) -> bool:
    """Whether `ontology` has room for `fact`: its predicate, its domain, its range.

    The grounder's free checks (`openodke.ground.checks.schema_problem`), so a
    fact the report counts as conforming is one the free checks let through.
    """
    return schema_problem(fact, ontology) is None


def conformance(facts: Sequence[Fact], ontology: Ontology | None) -> Conformance | None:
    """`conforms` over every fact. `None` with no ontology, or one that declares no predicates."""
    if ontology is None or not ontology.predicates:
        return None
    fitting = sum(1 for fact in facts if conforms(fact, ontology))
    return Conformance(
        rate=fitting / len(facts) if facts else None,
        conformant=fitting,
        facts=len(facts),
        checks=CHECKS,
    )


def spend(name: str, calls: Sequence[CallRecord] | None) -> dict[str, Any]:
    """`cost` and `latency` for a row, from the calls a meter recorded; empty when unmetered."""
    if calls is None:
        return {}
    total = StageCost.of(name, list(calls))
    return {
        "cost": Cost(
            calls=total.calls,
            prompt_tokens=total.prompt_tokens,
            completion_tokens=total.completion_tokens,
            usd=total.cost_usd,
            priced_usd=total.priced_usd,
            unpriced_calls=total.unpriced_calls,
        ),
        "latency": Latency(
            calls=total.calls,
            seconds=total.latency_s,
            per_call=total.latency_s / total.calls if total.calls else None,
        ),
    }


def models_called(
    calls: Iterable[CallRecord], configured: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Role -> model, as the calls named it, then any configured role that made none."""
    found: dict[str, str] = {}
    for call in calls:
        found.setdefault(call.stage, call.model)
    for role, model in (configured or {}).items():
        found.setdefault(role, model)
    return found


def from_stage(
    report: StageReport,
    *,
    rows: Sequence[Row] = (),
    run: Run | None = None,
    bootstrap: Bootstrap | None = None,
    notes: Sequence[str] = (),
    diagnosis: Sequence[Bucket] = (),
    fixes: Sequence[Fix] = (),
) -> EvalReport:
    """A stage's report as an eval report: titled and counted as the stage is."""
    return EvalReport(
        title=report.stage,
        n=report.n,
        run=run if run is not None else Run(),
        bootstrap=bootstrap if rows else None,
        rows=tuple(rows),
        stages=(report,),
        notes=tuple(notes),
        diagnosis=tuple(diagnosis),
        fixes=tuple(fixes),
    )


# --------------------------------------------------------------------------- #
# Reading and checking
# --------------------------------------------------------------------------- #


def schema() -> dict[str, Any]:
    """The JSON Schema every report is held to."""
    loaded: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return loaded


def check_report(data: Any) -> list[str]:
    """Every way `data` departs from the schema, as `path: problem` lines; none when it fits.

    Plain Python over the subset of JSON Schema the schema file uses: `type`,
    `properties`, `required`, `additionalProperties`, `items`, `minItems`,
    `maxItems`, `enum`, `const`, `minimum`, `pattern`, `anyOf` and local
    `$ref`s. No dependency, so a report can be checked wherever the package is
    installed.
    """
    root = schema()
    return list(_problems(data, root, root, "$"))


def read_report(path: str | Path) -> EvalReport:
    """A report written by `EvalReport.write`, checked against the schema.

    A report from another major version is refused by name, not by whichever
    field moved first.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version") if isinstance(data, dict) else None
    if not isinstance(version, str) or version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"{path}: schema_version {version!r}; this package reads {SCHEMA_VERSION}")
    problems = check_report(data)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems[:5]))
    return EvalReport.model_validate(data)


_TYPES: dict[str, Any] = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list | tuple),
    "object": lambda v: isinstance(v, dict),
}


def _problems(
    value: Any, node: Mapping[str, Any], root: Mapping[str, Any], at: str
) -> Iterator[str]:
    if "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"only local $refs are supported, got {ref!r}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        yield from _problems(value, target, root, at)
        return
    if "anyOf" in node:
        found = [list(_problems(value, option, root, at)) for option in node["anyOf"]]
        if all(found):
            # One option of the right type is the shape meant: say what is wrong inside it.
            near = [f for f in found if not f[0].startswith(f"{at}: expected ")]
            yield from near[0] if len(near) == 1 else [f"{at}: matches none of the allowed shapes"]
        return
    expected = node.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_TYPES[name](value) for name in names):
            yield f"{at}: expected {' or '.join(names)}, got {type(value).__name__}"
            return
    if "const" in node and value != node["const"]:
        yield f"{at}: must be {node['const']!r}"
    if "enum" in node and value not in node["enum"]:
        yield f"{at}: must be one of {node['enum']!r}"
    if "minimum" in node and _TYPES["number"](value) and value < node["minimum"]:
        yield f"{at}: below the minimum {node['minimum']}"
    if "pattern" in node and isinstance(value, str) and not re.search(node["pattern"], value):
        yield f"{at}: does not match {node['pattern']!r}"
    if isinstance(value, dict):
        properties = node.get("properties", {})
        for name in node.get("required", ()):
            if name not in value:
                yield f"{at}: missing {name!r}"
        extra = node.get("additionalProperties", True)
        for name, item in value.items():
            where = f"{at}.{name}"
            if name in properties:
                yield from _problems(item, properties[name], root, where)
            elif extra is False:
                yield f"{at}: unexpected {name!r}"
            elif isinstance(extra, Mapping):
                yield from _problems(item, extra, root, where)
    if isinstance(value, list | tuple):
        if "minItems" in node and len(value) < node["minItems"]:
            yield f"{at}: fewer than {node['minItems']} items"
        if "maxItems" in node and len(value) > node["maxItems"]:
            yield f"{at}: more than {node['maxItems']} items"
        for index, item in enumerate(value if "items" in node else ()):
            yield from _problems(item, node["items"], root, f"{at}[{index}]")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _estimate(estimate: Estimate) -> str:
    if estimate.value is None:
        return "—"
    if estimate.low is None or estimate.high is None:
        return f"{estimate.value:.3f}"
    return f"{estimate.value:.3f} [{estimate.low:.3f}, {estimate.high:.3f}]"


def _rows_table(rows: Sequence[Row]) -> list[str]:
    headers = ["", "precision", "recall", "f1", "hits", "over", "under"]
    conformance = any(r.conformance is not None for r in rows)
    hallucination = any(r.hallucination is not None for r in rows)
    cost = any(r.cost is not None for r in rows)
    if conformance:
        headers.append("conformance")
    if hallucination:
        headers.append("hallucinated")
    if cost:
        headers += ["calls", "usd", "seconds"]
    body = []
    for row in rows:
        p, c = row.performance, row.counts
        cells = [
            row.name,
            _estimate(p.precision),
            _estimate(p.recall),
            _estimate(p.f1),
            str(c.hits),
            str(c.over_extraction),
            str(c.under_extraction),
        ]
        if conformance:
            cells.append(_fmt(row.conformance.rate) if row.conformance else "—")
        if hallucination:
            cells.append(str(row.hallucination.hallucinated) if row.hallucination else "—")
        if cost:
            spent, timed = row.cost, row.latency
            cells += [
                str(spent.calls) if spent else "—",
                f"{spent.usd:.4f}" if spent and spent.usd is not None else "—",
                f"{timed.seconds:.1f}" if timed else "—",
            ]
        body.append(cells)
    return _table(headers, body)


def _ranges(how: Bootstrap) -> str:
    share = (
        f"{how.level:.0%}"
        if math.isclose(how.level * 100, round(how.level * 100))
        else f"{how.level:.1%}"
    )
    return (
        f"{share} ranges: {_count(how.units, 'document')} resampled {how.resamples} times "
        f"({how.method} bootstrap, seed {how.seed})"
    )


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _without_gold(report: EvalReport) -> list[str]:
    """The sections that score without gold, or with gold that is incomplete, as text."""
    lines: list[str] = []
    if report.judged_precision is not None:
        lines += ["", *_judged(report.judged_precision)]
    if report.adjudication is not None:
        lines += ["", *_adjudicated(report.adjudication)]
    if report.pooled_recall is not None:
        lines += ["", *_pooled(report.pooled_recall)]
    return lines


def _pooled(section: PooledRecall) -> list[str]:
    lines = [
        f"recall relative to a pool  ({_count(len(section.runs), 'run')}, "
        f"{_count(section.pool, 'supported fact')} pooled over "
        f"{_count(section.documents, 'document')})"
    ]
    body = [
        [run.name, _estimate(run.relative_recall), str(run.supported), str(run.unique)]
        for run in section.runs
    ]
    lines += _table(["", "relative recall", "supported", "found by no other"], body)
    lines += [f"  {_ranges(section.bootstrap)}", f"  {section.caveat}"]
    covered = [run for run in section.runs if run.coverage is not None]
    if covered:
        lines.append("  coverage, which needs no pool:")
        width = max(len(run.name) for run in covered)
        lines += [
            f"    {run.name:<{width}}  {_coverage(run.coverage)}"
            for run in covered
            if run.coverage is not None
        ]
    return lines


def _adjudicated(section: Adjudication) -> list[str]:
    lines = [
        f"gold adjudication  (each prediction the gold lacks grounded {section.runs} times; "
        f"supported in {section.needed} is possibly missing from gold)"
    ]
    body = [
        [
            row.name,
            _estimate(row.strict),
            _estimate(row.adjudicated),
            str(row.not_in_gold),
            str(row.asked),
            str(row.possibly_missing),
        ]
        for row in section.rows
    ]
    lines += _table(["", "strict", "adjudicated", "not in gold", "asked", "possibly missing"], body)
    where = f": {section.audit}" if section.audit else ""
    lines.append(f"  the strict precision never changes; the list, with every verdict{where}")
    return lines


def _level(level: float) -> str:
    return f"{level:.0%}" if math.isclose(level * 100, round(level * 100)) else f"{level:.1%}"


def _judged(section: JudgedPrecision) -> list[str]:
    labelled = _count(section.labels, "label") if section.labels else "no labels"
    level = _level(section.level)
    body = [
        ["judge only", _estimate(section.judge_only), "the judge's verdicts, uncorrected"],
        ["corrected", _estimate(section.corrected), f"prediction-powered (PPI), {level}"],
        ["labels only", _estimate(section.labels_only), f"the labels alone, {level}"],
    ]
    if not section.labels:
        body[1][2] = "label a random sample of the facts to correct the judge"
        body[2][2] = ""
    lines = [f"precision without gold  ({_count(section.facts, 'fact')} judged, {labelled})"]
    lines += _table(["", "", ""], body)[1:]
    if section.labels:
        lines.append(
            f"  the judge on the sample: {section.false_support} false support, "
            f"{section.lost_support} lost support"
        )
    if section.labels and not section.calibrated:
        lines.append(
            f"  uncalibrated estimate: {_count(section.labels, 'label')}, under the "
            f"{section.minimum} a calibrated one needs"
        )
    if section.coverage is not None:
        lines.append(f"  coverage, in place of recall: {_coverage(section.coverage)}")
    return [line.rstrip() for line in lines]


def _coverage(totals: CoverageTotals) -> str:
    from openodke.coverage import summary

    return summary(
        {
            "sentences": totals.sentences,
            "uncovered": totals.uncovered,
            "missed_entities": totals.missed_entities,
            "not_offered": totals.not_offered,
            "unused": totals.unused,
        }
    )


def _provenance(report: EvalReport) -> list[str]:
    run = report.run
    lines = [f"  openodke {run.openodke} · eval report {report.schema_version}"]
    if run.dataset is not None:
        data = run.dataset
        sizes = [
            _count(data.labels, "label") if data.labels is not None else "",
            _count(data.documents, "document") if data.documents is not None else "",
        ]
        said = ", ".join(s for s in sizes if s)
        lines.append(f"  dataset  {data.name}" + (f" ({said})" if said else ""))
    if run.models:
        lines.append("  models   " + ", ".join(f"{role} {m}" for role, m in run.models.items()))
    if run.prompts:
        lines.append("  prompts  " + ", ".join(run.prompts))
    return lines


__all__ = [
    "CHECKS",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "AdjudicatedRow",
    "Adjudication",
    "Bootstrap",
    "Conformance",
    "Cost",
    "Counts",
    "CoverageTotals",
    "Dataset",
    "Estimate",
    "EvalReport",
    "Hallucination",
    "JudgedPrecision",
    "PoolMember",
    "PooledRecall",
    "Latency",
    "Performance",
    "Row",
    "Run",
    "check_report",
    "conformance",
    "conforms",
    "extraction_rows",
    "from_stage",
    "micro",
    "models_called",
    "performance",
    "read_report",
    "schema",
    "spend",
]
