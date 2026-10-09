"""The eval report: one versioned JSON document for every run the Evaluator scores.

A `StageReport` is what one evaluator concluded about one stage, and its
shape is whatever that stage partitions on. That suits a person reading one
table. It does not suit a CI job, a dashboard, or a second run to compare
against, because each of those needs to know where precision is without
knowing which stage made the report. So an Evaluator run also writes an
`EvalReport`, whose shape is fixed and versioned (`schema_version`, "1.0"):

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
- **diagnosis**, **fixes**, **comparison** and **calibration**: empty, and
  typed, until the issues that fill them land (#140, #141, #142, #135).

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
from openodke.eval.cost import CallRecord, StageCost
from openodke.eval.extraction import match_extraction, per_document
from openodke.eval.formats import GoldFact
from openodke.eval.report import Metric, StageReport, _fmt, _table, prf
from openodke.ground.checks import CHECKS, schema_problem
from openodke.ontology import Ontology
from openodke.types import Fact, Frozen

SCHEMA_VERSION = "1.0"
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
    `unscored` counts predictions left out of every number, and the stage's
    notes say why.
    """

    hits: int = Field(ge=0)
    over_extraction: int = Field(ge=0)
    under_extraction: int = Field(ge=0)
    predicted: int = Field(ge=0)
    gold: int = Field(ge=0)
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
    # Reserved, typed and empty until filled: every miss in one cause bucket (#140),
    diagnosis: tuple[dict[str, Any], ...] = ()
    # ranked fixes with their expected gain (#141),
    fixes: tuple[dict[str, Any], ...] = ()
    # this run against a baseline (#142),
    comparison: dict[str, Any] | None = None
    # and the grounder's calibration cards (#135).
    calibration: tuple[dict[str, Any], ...] = ()

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
    unit of the bootstrap, and predictions that cite no document and match
    nothing form one more. `calls`, when the run was metered, give a row its
    cost and latency; `ontology` gives every row its conformance.
    """
    rows, most = [], 0
    for name, facts, calls in configurations:
        units, unscored = extraction_units(gold, facts)
        most = max(most, len(units))
        tp, fp, fn = (sum(u[i] for u in units) for i in range(3))
        ranges = bootstrap(units, micro, resamples=resamples, seed=seed, level=level)
        counts = Counts(
            hits=tp,
            over_extraction=fp,
            under_extraction=fn,
            predicted=tp + fp,
            gold=tp + fn,
            unscored=unscored,
            documents=len({g.doc_id for g in gold}),
        )
        rows.append(
            Row(
                name=name,
                performance=performance(micro(units), ranges),
                counts=counts,
                conformance=conformance(facts, ontology),
                **spend(name, calls),
            )
        )
    return rows, Bootstrap(units=most, resamples=resamples, seed=seed, level=level)


def extraction_units(
    gold: Sequence[GoldFact], predictions: Iterable[Fact]
) -> tuple[list[tuple[int, int, int]], int]:
    """`(tp, fp, fn)` per labelled document in label order, and the unscored count.

    A wrong value or entity is a false positive and a false negative in the
    document it was matched in. Predictions that cite no document and match no
    gold fact are one more unit, after the documents.
    """
    split = per_document(predictions)
    labelled = {g.doc_id for g in gold}
    unscored = sum(
        1 for p in split if p.evidence and not any(e.doc_id in labelled for e in p.evidence)
    )
    by_doc: dict[str | None, list[int]] = {
        d: [0, 0, 0] for d in dict.fromkeys(g.doc_id for g in gold)
    }
    for outcome in match_extraction(gold, split):
        unit = by_doc.setdefault(outcome.doc_id, [0, 0, 0])
        if outcome.kind == "correct":
            unit[0] += 1
        elif outcome.kind == "spurious":
            unit[1] += 1
        elif outcome.kind == "missing":
            unit[2] += 1
        else:  # a wrong value or entity: one fact written wrong, one not written
            unit[1] += 1
            unit[2] += 1
    return [(tp, fp, fn) for tp, fp, fn in by_doc.values()], unscored


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
    `properties`, `required`, `additionalProperties`, `items`, `enum`, `const`,
    `minimum`, `pattern`, `anyOf` and local `$ref`s. No dependency, so a
    report can be checked wherever the package is installed.
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
        if not any(not list(_problems(value, option, root, at)) for option in node["anyOf"]):
            yield f"{at}: matches none of the allowed shapes"
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
    if isinstance(value, list | tuple) and "items" in node:
        for index, item in enumerate(value):
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
    "Bootstrap",
    "Conformance",
    "Cost",
    "Counts",
    "Dataset",
    "Estimate",
    "EvalReport",
    "Hallucination",
    "Latency",
    "Performance",
    "Row",
    "Run",
    "check_report",
    "conformance",
    "conforms",
    "extraction_rows",
    "extraction_units",
    "from_stage",
    "micro",
    "models_called",
    "performance",
    "read_report",
    "schema",
    "spend",
]
