"""What every dataset adapter shares: downloading, names, the run config, the report."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openodke.coverage import summary as coverage_summary
from openodke.eval.ablation import AblationRun
from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED, Range, bootstrap
from openodke.eval.cost import CallRecord, StageCost
from openodke.eval.diagnosis import View
from openodke.eval.eval_report import (
    Bootstrap,
    Configuration,
    Dataset,
    EvalReport,
    Row,
    Run,
    from_stage,
    models_called,
)
from openodke.eval.report import Metric, StageReport
from openodke.interop.triples import _file_name
from openodke.types import Document, Fact

Triple = tuple[str, str, str]
Opener = Callable[[str], Any]
# What a bench run writes beside `predictions/`: what the extractor refused (#109).
REJECTIONS_FILE = "rejections.jsonl"

# Literal ranges a dataset may name, in openodke's vocabulary (ontology value types).
LITERALS = {
    "": "string",
    "string": "string",
    "langstring": "string",
    "text": "string",
    "date": "date",
    "time": "date",
    "datetime": "date",
    "year": "integer",
    "number": "number",
    "integer": "integer",
    "int": "integer",
    "float": "number",
    "double": "number",
    "quantity": "number",
    "boolean": "boolean",
}


def download(url: str, dest: Path, *, opener: Opener | None = None) -> Path:
    """`url` saved at `dest`, unless it is already there. Written atomically."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener if opener is not None else urllib.request.urlopen
    with (
        open_url(url) as response,
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
    Path(tmp.name).replace(dest)
    return dest


def pascal(label: str) -> str:
    """`film production company` -> `FilmProductionCompany`; `University` stays."""
    words = re.split(r"[^0-9A-Za-z]+", label.strip())
    return "".join(w[:1].upper() + w[1:] for w in words if w) or "Thing"


def snake(label: str) -> str:
    """`cast member` -> `cast_member`; `academicStaffSize` -> `academic_staff_size`."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label.strip())
    return "_".join(w.lower() for w in re.split(r"[^0-9A-Za-z]+", spaced) if w) or "related_to"


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return path


def check_out(out: Path) -> None:
    """Refuse to prepare into a directory that something other than `prepare` owns.

    `prepare` overwrites `ontology.json` and `odke.json` and replaces the text
    files under `docs/`, so `--out .` at a repository's root would rewrite its
    files. A new or empty directory is fine, and so is one an earlier prepare
    wrote, which its `dataset.json` marks.
    """
    if not out.exists() or (out / "dataset.json").is_file():
        return
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError(
            f"{out} is not empty and has no dataset.json from an earlier prepare; "
            "prepare into a new or empty directory"
        )


def write_documents(out: Path, docs: Iterable[tuple[str, str]]) -> int:
    """One `<id>.txt` per document under `out/docs`: the id survives as the file's stem.

    Only the text files an earlier prepare wrote are removed; anything else in
    `docs/` stays.
    """
    folder = out / "docs"
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()
    count = 0
    for doc_id, text in docs:
        (folder / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        count += 1
    return count


def run_config(
    *,
    extract_model: str | None,
    ground_model: str | None,
    paper: bool,
    relations: int,
    max_tokens: int = 16000,
) -> dict[str, Any]:
    """The run config a prepared dataset ships: text in, nothing written out.

    No chunker, so each document reaches the extractor whole — the paper's unit
    is a page — and no value normalizer, so values stay in the words the gold
    uses. `paper=True` is ODKE+'s own grounder and gate: the whole context,
    True or False, and only affirmed facts kept. `relations` is the ontology's
    size: the extractor is shown every relation a type can take, as a
    competitor given the same schema is, not the default 25-predicate snippet.
    `max_tokens` is the extractor's output room: 16,000 leaves a reasoning
    model space to think; lower it for a model with a smaller output cap.
    """
    grounder: dict[str, Any] = {"use": "llm"}
    gate: dict[str, Any] = {"use": "verdict"}
    if paper:
        grounder |= {"context": "document", "verdicts": "binary"}
        gate |= {"refuse_not_found": True}
    config: dict[str, Any] = {
        "ontology": "ontology.json",
        "inputs": [{"path": "docs", "loader": "directory"}],
        "stages": {
            # No response schema, as the paper prompts (App. A); also the way
            # round Anthropic's grammar limits on a typed multi-type schema.
            "extractor": {"use": "llm", "structured": False, "snippet_limit": relations},
            "grounder": grounder,
            # No value normalizer: both datasets' gold is in the source's own words,
            # and "13 March 1963" rewritten to "1963-03-13" scores as wrong. The
            # corroborator still merges, scores and gates. (The paper's corroborator
            # does normalise; on these datasets that would only measure the scorer.)
            "corroborator": "signature",
            "scorer": "evidence",
            "gate": gate,
        },
    }
    models: dict[str, Any] = {}
    if extract_model:
        # Room for a reasoning model's thinking, which counts against max_tokens;
        # only the tokens used are billed.
        models["extract"] = {"model": extract_model, "max_tokens": max_tokens}
    if ground_model:
        models["ground"] = ground_model
    if models:
        config["models"] = models
    return config


def doc_names(documents: Sequence[Document]) -> dict[str, str]:
    """Document id -> the dataset's own id, read back from the file's stem."""
    out: dict[str, str] = {}
    for doc in documents:
        if (name := _file_name(doc)) is not None:
            out[doc.id] = name
    return out


def triples_by_doc(
    facts: Iterable[Fact], names: Mapping[str, str], relation: Callable[[str], str]
) -> dict[str, list[Triple]]:
    """Each fact as `(subject, relation, object)` strings, under every document it cites.

    Labels, not keys: a benchmark's gold is written in surface strings. `relation`
    maps an openodke predicate name back to the dataset's own relation label.
    """
    out: dict[str, list[Triple]] = {}
    for fact in facts:
        triple = (
            fact.subject.label or fact.subject.key,
            relation(fact.predicate),
            (fact.object_entity.label or fact.object_entity.key)
            if fact.object_entity is not None
            else str(fact.object_value),
        )
        for doc_id in dict.fromkeys(e.doc_id for e in fact.evidence):
            name = names.get(doc_id)
            if name is not None and triple not in out.setdefault(name, []):
                out[name].append(triple)
    return out


def report(
    stage: str,
    n: int,
    rows: Sequence[tuple[str, Mapping[str, Metric], Sequence[Any] | None, int]],
    notes: Sequence[str],
    *,
    labels: Sequence[str] = ("extraction", "grounding", "corroboration"),
) -> StageReport:
    """The ablation table: one row per configuration, its metrics, calls and cost.

    `labels` name the rows in the summary metrics (`precision_extraction`, ...),
    one per row. A row whose calls were not metered has no call or cost column.
    """
    breakdown: dict[str, dict[str, Metric]] = {}
    for name, row, calls, facts in rows:
        cost = None if calls is None else StageCost.of(name, list(calls))
        breakdown[name] = {
            **row,
            "facts": facts,
            "model_calls": None if cost is None else cost.calls,
            "prompt_tokens": None if cost is None else cost.prompt_tokens,
            "completion_tokens": None if cost is None else cost.completion_tokens,
            "cost_usd": None if cost is None else cost.cost_usd,
        }
    summary: dict[str, Metric] = {}
    for label, (name, _, _, _) in zip(labels, rows, strict=True):
        for key in ("precision", "recall", "f1"):
            summary[f"{key}_{label}"] = breakdown[name].get(key)
    return StageReport(stage=stage, n=n, metrics=summary, breakdown=breakdown, notes=tuple(notes))


# A dataset's scoring, split so the eval report can resample it: one configuration's
# triples by document -> one unit per document; units -> the dataset's metrics;
# and a configuration's name, units, metrics, ranges and calls -> its report row.
Units = Callable[[Mapping[str, Sequence[Triple]]], list[Any]]
Aggregate = Callable[[Sequence[Any]], dict[str, Metric]]
ToRow = Callable[
    [str, Sequence[Any], Mapping[str, Metric], Mapping[str, Range], Sequence[CallRecord] | None],
    Row,
]
# Triples written, and those a gate refused (None: no record) -> the diagnosis's view.
ToView = Callable[[Mapping[str, Sequence[Triple]], Mapping[str, Sequence[Triple]] | None], View]


@dataclass(frozen=True)
class Scoring:
    """How one prepared dataset scores facts: what each dataset module's `scoring` returns.

    `units` takes one configuration's triples by document and scores each
    document; `aggregate` turns those into the dataset's numbers, and is what
    the bootstrap recomputes on each draw of documents; `row` makes the eval
    report's row. `stage` names the set, and `dataset` describes it. `view`,
    where the dataset has one, puts its gold and a configuration's triples in
    the form the diagnosis reads (#140), matched as `units` matches them.
    """

    stage: str
    meta: Mapping[str, Any]
    units: Units
    aggregate: Aggregate
    row: ToRow
    dataset: Dataset
    view: ToView | None = None


@dataclass(frozen=True)
class Scored:
    """Configurations scored: the stage table's rows, the eval report's, and the triples."""

    table: list[tuple[str, dict[str, Metric], Sequence[CallRecord] | None, int]]
    rows: list[Row]
    predictions: dict[str, dict[str, list[Triple]]]


def score_configurations(
    configurations: Sequence[Configuration],
    documents: Sequence[Document],
    scoring: Scoring,
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> Scored:
    """Each `(name, facts, calls)` scored with a dataset's own metrics, ranges included.

    Facts are read back as the dataset's triples: labels rather than keys, and
    the dataset's own relation names, under each document they cite.
    """
    labels: Mapping[str, str] = scoring.meta["relation_labels"]
    names = doc_names(documents)
    scored = Scored(table=[], rows=[], predictions={})

    def statistic(draw: Sequence[Any]) -> dict[str, Metric]:
        found = scoring.aggregate(draw)
        return {k: found[k] for k in ("precision", "recall", "f1")}

    for name, facts, calls in configurations:
        predicted = triples_by_doc(facts, names, lambda p: labels.get(p, p))
        scored.predictions[name] = predicted
        units = scoring.units(predicted)
        metrics = scoring.aggregate(units)
        ranges = bootstrap(units, statistic, resamples=resamples, seed=seed, level=level)
        scored.table.append((name, metrics, calls, len(facts)))
        scored.rows.append(scoring.row(name, units, metrics, ranges, calls))
    return scored


def report_run(
    ablation: AblationRun,
    gold: Sequence[Mapping[str, Any]],
    scoring: Scoring,
    *,
    notes: Callable[[Mapping[str, Metric], Mapping[str, Metric], Mapping[str, Metric]], list[str]],
    save_to: Path | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> EvalReport:
    """Each configuration of an `AblationRun` scored with a dataset's own metrics.

    `notes` takes the three rows' metrics — extracted, after the gate, after
    corroboration — and writes the dataset's lines beside the paper's numbers;
    the grounder's verdicts and the run's own notes follow them.
    """
    scored = score_configurations(
        ablation.configurations(),
        ablation.documents,
        scoring,
        resamples=resamples,
        seed=seed,
        level=level,
    )
    names = doc_names(ablation.documents)
    if save_to is not None:
        save_predictions(save_to, scored.predictions)
        save_rejections(save_to, ablation.rejections or [], names)
    raw, gated, full = (metrics for _, metrics, _, _ in scored.table)
    lines = [
        *notes(raw, gated, full),
        f"grounder verdicts on the candidates: {verdicts(ablation.grounded)}",
        rejected(ablation.rejections, saved=save_to is not None),
        *ablation.notes,
    ]
    if ablation.coverage is not None:
        lines.append(f"coverage of the candidates: {coverage_summary(ablation.coverage)}")
    stage = report(scoring.stage, len(gold), scored.table, lines)
    if ablation.rejections is not None:
        counts = Counter(str(r.reason) for r in ablation.rejections)
        by_reason = {f"rejected: {why}": n for why, n in sorted(counts.items())}
        stage = stage.model_copy(
            update={"metrics": {**stage.metrics, "rejected": len(ablation.rejections), **by_reason}}
        )
    return from_stage(
        stage,
        rows=scored.rows,
        bootstrap=Bootstrap(units=len(gold), resamples=resamples, seed=seed, level=level),
        run=Run(
            models=models_called(ablation.all_calls, ablation.models),
            prompts=ablation.prompts,
            dataset=scoring.dataset,
        ),
    )


def verdicts(facts: Iterable[Fact]) -> str:
    """How the grounder judged the candidates: `supported 60, not_found 4, unchecked 0`.

    The gate lets an unchecked fact through, so a row that drops nothing could
    be a grounder that confirmed everything or one whose calls all failed. This
    says which.
    """
    counts = Counter(f.verdict.value for f in facts)
    order = ("supported", "not_found", "contradicted", "unchecked")
    return ", ".join(f"{v} {counts.get(v, 0)}" for v in order)


def rejected(rejections: Sequence[Any] | None, *, saved: bool = False) -> str:
    """What the extractor refused before grounding, by reason: the report's line for it."""
    if rejections is None:
        return "extractor rejections: none recorded; this extractor keeps no record of them"
    counts = Counter(str(r.reason) for r in rejections)
    by_reason = ", ".join(
        f"{why} {n}" for why, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    where = f"; each in {REJECTIONS_FILE}" if saved and rejections else ""
    return (
        f"extractor rejections: {len(rejections)}"
        + (f" ({by_reason})" if by_reason else "")
        + where
    )


def save_rejections(
    folder: Path, rejections: Iterable[Any], names: Mapping[str, str] | None = None
) -> Path:
    """`folder/rejections.jsonl`, beside `predictions/`: each candidate the extractor refused.

    One row a rejection: the document, by the dataset's own id where `names`
    has it, the chunk, the reason, and the candidate as the model wrote it
    (`subject`, `predicate`, `value`, `quote`), as far as it got. Written on
    every run, empty when nothing was refused, so a missing fact can be told
    from one the extractor threw away.
    """
    rows = [
        {
            "doc": (names or {}).get(r.doc_id, r.doc_id),
            "chunk": r.chunk_index,
            "reason": r.reason,
            "candidate": {
                "subject": getattr(r, "subject", None),
                "predicate": r.predicate,
                "value": getattr(r, "value", None),
                "quote": r.quote,
            },
        }
        for r in rejections
    ]
    return write_jsonl(folder / REJECTIONS_FILE, rows)


def save_predictions(folder: Path, predicted: Mapping[str, Mapping[str, Sequence[Triple]]]) -> None:
    """`folder/predictions/<row>.jsonl`: each document's triples, per configuration.

    What an audit reads: a precision is only an error rate once someone has looked
    at the "false positives" the gold missed.
    """
    for row, by_doc in predicted.items():
        slug = re.sub(r"[^a-z0-9]+", "-", row.lower()).strip("-")
        rows = [
            {"id": doc, "triples": [list(t) for t in triples]} for doc, triples in by_doc.items()
        ]
        write_jsonl(folder / "predictions" / f"{slug}.jsonl", rows)


def change(before: float, after: float) -> str:
    """`-35%`, `+4%`, or `n/a` when there was nothing to change."""
    if before == 0:
        return "n/a"
    return f"{(after - before) / before:+.0%}"
