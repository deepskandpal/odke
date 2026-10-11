"""The Evaluator's harness: point it at your pipeline, and get the eval report (#138).

Your pipeline is whatever turns documents into triples. The harness runs it
one of three ways, reads what it returned, and scores that with the same
arithmetic as every other evaluator here:

- **command**: a shell command template that reads a folder of texts and writes
  triples, `python my_extract.py {in} {out}`. The template is split the way a
  shell would split it, `{in}` and `{out}` are put into the words, and the
  words are run with no shell in between, so a path with a space or a `;` in it
  is one argument and nothing else. It has a timeout, and a command that exits
  without writing `{out}` is an error that says so.
- **callable**: `module:function`, or `path/to/file.py:function`, called with
  the `Document`s and returning triples rows, `TripleRow`s or `Fact`s.
- **files**: output already written, `--predictions`.

Output is the triples format (`openodke.interop.triples`), or any adapter's
format by name (`ADAPTERS`): a LangChain `GraphDocument` dump, LangExtract's
JSON Lines, a neo4j-graphrag graph. A file of `Fact` rows, such as a sink's
`facts.jsonl`, is read as facts.

What it is scored against is either your labelled sample (`GoldFact` rows and
the documents they name) or a set `odke bench prepare` wrote, scored with that
benchmark's own metrics. With `validator` it also runs `openodke.Validator`
over the same output (#129), and the report has both rows: the pipeline, and
the pipeline with the Validator: grounded, normalised, resolved, corroborated
and gated, with what that cost.

With `adjudicate`, each prediction your gold lacks is grounded three times,
and one supported in two of them is listed as possibly missing from gold
(`openodke.eval.adjudication`, #145): an adjudicated precision is printed beside
each row's strict one, and the list is written for audit.

The last row is diagnosed: every miss in one cause bucket, with the gate's
refusals when the Validator ran and what the extractor was offered when a
`trace` says (`openodke.eval.diagnosis`, #140), and the fixes ranked by the
recall the arithmetic expects of them (`openodke.eval.fixes`, #141).

With `lenient`, the fact-equivalence judge reads each pair the pre-filter makes
from a row's misses and its unmatched predictions (`openodke.eval.equivalence`,
#143): a lenient score, counting a surface form judged the gold fact as a hit,
is printed beside each row's strict one, and every pair is written for audit.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError

from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED
from openodke.eval.compare import ItemRow
from openodke.eval.cost import CallRecord
from openodke.eval.diagnosis import (
    EXAMPLES,
    Bucket,
    EquivalenceJudge,
    Offered,
    View,
    diagnose,
    gold_view,
    read_trace,
)
from openodke.eval.equivalence import Decided, Pairing
from openodke.eval.eval_report import (
    Bootstrap,
    Configuration,
    Dataset,
    EvalReport,
    Lenient,
    Run,
    extraction_rows,
    models_called,
)
from openodke.eval.extraction import document_counts, evaluate_extraction
from openodke.eval.fixes import Fix
from openodke.eval.fixes import fixes as rank_fixes
from openodke.eval.formats import GoldFact, load_jsonl

# From the modules, not the package: `openodke.interop` imports `openodke.eval`.
from openodke.interop.graphrag import from_graphrag
from openodke.interop.langchain import from_graph_documents
from openodke.interop.langextract import from_langextract
from openodke.interop.triples import TripleRow, _file_name, read_triples, to_fact
from openodke.ontology import Ontology
from openodke.types import Document, Fact

TITLE = "pipeline"
PIPELINE, VALIDATOR = "pipeline", "+ validator"
# Seconds a command may run before it is stopped.
TIMEOUT = 600.0

# Other libraries' output, read into triples rows and the texts they cite.
Adapter = Callable[[Path], tuple[list[TripleRow], list[Document]]]
ADAPTERS: dict[str, Adapter] = {
    "langchain": from_graph_documents,
    "langextract": from_langextract,
    "graphrag": from_graphrag,
}
FORMATS = ("triples", *ADAPTERS)

DESCRIPTION = """\
odke eval pipeline (--cmd TEMPLATE | --run MODULE:FUNCTION | --predictions FILE)
                   (--labels GOLD_FACTS --documents DOCS | --bench PREPARED)
                   [--adapter NAME] [--ontology FILE] [--validator] [--config RUN_CONFIG]
                   [--adjudicate LIST] [--lenient PAIRS]

Runs your pipeline over the documents and scores what it returns.

--cmd       a command template: {in} is a folder with one <name>.txt per
            document, {out} the file it must write. Run without a shell, with a
            --timeout (default 600 s).
--run       module:function or path/file.py:function, called with the
            Documents; returns triples rows, TripleRows or Facts.
--predictions  what it already wrote.

Output is triples JSONL, one row per triple: doc (a document's id, or its
file name without the suffix), subject, predicate, object, and optionally
subject_type, object_type, start, end, quote. --adapter langchain,
langextract or graphrag reads that library's output instead. A file of Fact
rows (a sink's facts.jsonl) is read as facts.

--labels and --documents: GoldFact rows, and the documents they name (a
folder, each file named by its path inside it, or Document JSONL).
--bench: a directory `odke bench prepare` wrote, scored with its metrics.

--validator runs the Validator over the same output and reports both rows:
with the stages and models of --config (a bench set's own odke.json by
default), or the Validator's defaults.

--adjudicate LIST, with --labels: gold is incomplete, so each prediction the
gold lacks is grounded three times against its document (--config's grounder,
or the default one). One supported in two of the three runs is possibly
missing from gold. Each row's adjudicated precision, counting those as hits,
is printed beside its strict one, which never changes, and every prediction
the gold lacks goes to LIST (JSONL) with its three verdicts.

--lenient PAIRS, with --labels: a missed gold fact and an unmatched prediction
with the relation and one end in common are asked about, in both orders, by the
fact-equivalence judge (--config's ground model, or the default one). A pair
judged the same fact in both orders counts as a hit in each row's lenient
precision, recall and F1, printed beside the strict ones, which never change;
every pair goes to PAIRS (JSONL) with both answers."""


class PipelineError(RuntimeError):
    """The pipeline ran and failed: it exited non-zero, timed out, or wrote nothing."""


# --------------------------------------------------------------------------- #
# What is scored
# --------------------------------------------------------------------------- #


@dataclass
class Corpus:
    """The documents a pipeline reads, and what its output is scored against.

    Exactly one of `gold` (a labelled sample) and `scoring` (a prepared set's
    own metrics) is set. `files` is the name each document's text gets under
    `{in}`.
    """

    documents: list[Document]
    ontology: Ontology | None
    dataset: Dataset
    gold: list[GoldFact] | None = None
    scoring: Any = None
    files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.files:
            self.files = _file_names(self.documents)


def labelled(
    labels: str | Path, documents: str | Path, ontology: str | Path | Ontology | None = None
) -> Corpus:
    """Your labelled sample: `GoldFact` rows, and the documents they name.

    `documents` is a folder, read as `odke run` reads one, each document named
    by its path inside the folder (`halden.txt`, `2024/report.txt`); or a
    JSON Lines file of `Document` rows, named by their ids.
    """
    gold = load_jsonl(labels, GoldFact)
    docs = load_documents(documents)
    known = {doc.id for doc in docs}
    if missing := sorted({g.doc_id for g in gold} - known):
        raise ValueError(
            f"{len(missing)} labelled document(s) are not among the documents: "
            f"{', '.join(missing[:5])}; a gold fact names a document by its id"
        )
    return Corpus(
        documents=docs,
        ontology=_ontology(ontology),
        gold=gold,
        dataset=Dataset(
            name=Path(labels).name,
            path=str(labels),
            documents=len({g.doc_id for g in gold}),
            labels=len(gold),
        ),
    )


def prepared(folder: str | Path) -> Corpus:
    """A set `odke bench prepare` wrote: its documents, ontology and gold."""
    root = Path(folder)
    if not (root / "dataset.json").is_file():
        raise ValueError(
            f"{root} has no dataset.json: point --bench at what `odke bench prepare` wrote"
        )
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rows = (root / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    gold = [json.loads(line) for line in rows if line.strip()]
    scoring = _bench_module(meta).scoring(gold, meta, path=root)
    docs = [
        Document(id=path.stem, text=path.read_text(encoding="utf-8"), uri=path.resolve().as_uri())
        for path in sorted((root / "docs").glob("*.txt"))
    ]
    return Corpus(
        documents=docs,
        ontology=_ontology(root / "ontology.json"),
        scoring=scoring,
        dataset=scoring.dataset,
    )


def load_documents(source: str | Path) -> list[Document]:
    """A folder of texts, each named by its path inside it, or `Document` JSONL."""
    path = Path(source)
    if path.is_dir():
        from openodke.loaders import DirectoryLoader
        from openodke.run.build import with_path_ids

        docs = with_path_ids(list(DirectoryLoader().load(path)), path.resolve())
        if not docs:
            raise ValueError(f"{path}: no file in it has a loader")
        return docs
    if not path.is_file():
        raise ValueError(f"{path}: no such folder or file")
    return load_jsonl(path, Document)


# --------------------------------------------------------------------------- #
# Running the pipeline
# --------------------------------------------------------------------------- #


@dataclass
class Output:
    """What a pipeline returned, before it is read as facts."""

    items: list[Any]
    # The texts an adapter says its rows came from; matched to the corpus's documents.
    texts: list[Document] = field(default_factory=list)
    seconds: float | None = None
    how: str = ""


def run_command(
    template: str, corpus: Corpus, *, adapter: str = "triples", timeout: float = TIMEOUT
) -> Output:
    """Run `template` with `{in}` a folder of the corpus's texts and `{out}` a file to write.

    No shell: the template is split into words first, and the paths go into
    the words, so they are never parsed. Exit non-zero, a timeout, or no file
    at `{out}` is a `PipelineError` naming which.
    """
    if "{in}" not in template or "{out}" not in template:
        raise ValueError(
            "--cmd needs {in} and {out}: the folder of texts it reads and the file it writes, "
            'e.g. --cmd "python my_extract.py {in} {out}"'
        )
    words = shlex.split(template)
    suffix = ".json" if adapter in ("langchain", "graphrag") else ".jsonl"
    with tempfile.TemporaryDirectory(prefix="odke-pipeline-") as scratch:
        inbox, out = Path(scratch) / "in", Path(scratch) / "out" / f"predictions{suffix}"
        inbox.mkdir()
        out.parent.mkdir()
        for doc in corpus.documents:
            (inbox / corpus.files[doc.id]).write_text(doc.text, encoding="utf-8")
        argv = [w.replace("{in}", str(inbox)).replace("{out}", str(out)) for w in words]
        start = time.perf_counter()
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError:
            raise PipelineError(f"cannot run {argv[0]!r}: no such program") from None
        except subprocess.TimeoutExpired:
            raise PipelineError(
                f"the command ran past {timeout:g} s and was stopped; raise --timeout"
            ) from None
        seconds = time.perf_counter() - start
        if done.returncode != 0:
            tail = "\n".join(done.stderr.strip().splitlines()[-5:])
            raise PipelineError(
                f"the command exited {done.returncode}" + (f":\n{tail}" if tail else "")
            )
        if not out.is_file():
            raise PipelineError(
                f"the command exited 0 but wrote nothing at {{out}}; it must write its "
                f"{adapter} output to the path given as {{out}}"
            )
        read = read_output(out, adapter=adapter)
    read.seconds, read.how = seconds, f"command {template!r}"
    return read


def run_callable(spec: str | Callable[..., Any], corpus: Corpus) -> Output:
    """Call `spec` with the corpus's `Document`s; it returns rows, `TripleRow`s or `Fact`s."""
    function = load_callable(spec) if isinstance(spec, str) else spec
    start = time.perf_counter()
    returned = function(list(corpus.documents))
    seconds = time.perf_counter() - start
    if returned is None or isinstance(returned, str | bytes | Mapping):
        raise PipelineError(
            f"{_name(spec)} returned {type(returned).__name__}; it must return rows or facts"
        )
    return Output(items=list(returned), seconds=seconds, how=f"callable {_name(spec)}")


def read_output(path: str | Path, *, adapter: str = "triples") -> Output:
    """What a pipeline wrote: triples JSONL, `Fact` JSONL, or an adapter's format."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{source}: no such file")
    if adapter in ADAPTERS:
        rows, texts = ADAPTERS[adapter](source)
        return Output(items=list(rows), texts=list(texts), how=f"{adapter} file {source.name}")
    if adapter != "triples":
        raise ValueError(f"unknown adapter {adapter!r}; one of {', '.join(FORMATS)}")
    first = next((ln for ln in source.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
    if _is_fact(first):
        return Output(items=load_jsonl(source, Fact), how=f"facts file {source.name}")
    return Output(items=list(read_triples(source)), how=f"triples file {source.name}")


def load_callable(spec: str) -> Callable[..., Any]:
    """`package.module:function`, or `path/to/file.py:function`; a class is instantiated."""
    where, _, name = spec.rpartition(":")
    if not where or not name:
        raise ValueError(f"--run takes module:function or path/file.py:function, got {spec!r}")
    try:
        if where.endswith(".py"):
            path = Path(where)
            if not path.is_file():
                raise ValueError(f"{path}: no such file")
            loader = importlib.util.spec_from_file_location(f"odke_pipeline_{path.stem}", path)
            if loader is None or loader.loader is None:
                raise ValueError(f"cannot import {path}")
            module = importlib.util.module_from_spec(loader)
            sys.modules[loader.name] = module
            loader.loader.exec_module(module)
        else:
            module = importlib.import_module(where)
        found = getattr(module, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load {spec!r}: {exc}") from exc
    if isinstance(found, type):
        found = found()
    if not callable(found):
        raise ValueError(f"{spec!r} is not callable")
    return found


# --------------------------------------------------------------------------- #
# Reading the output as facts
# --------------------------------------------------------------------------- #


def as_facts(output: Output, corpus: Corpus) -> tuple[list[Fact], list[str]]:
    """The output as facts about the corpus's documents, and notes on what did not fit.

    A row names its document by id, by the name its text had under `{in}`
    without the suffix, or by its source file's name without the suffix. An
    adapter's own texts are matched to the documents by id, name, or exact
    text. A row that names no document is left out and counted.
    """
    names = _names(corpus)
    for text in output.texts:
        match = names.get(text.id) or _by_uri(names, text) or _by_text(corpus, text)
        if match is not None:
            names.setdefault(text.id, match)
    facts: list[Fact] = []
    unmatched: Counter[str] = Counter()
    for item in output.items:
        if isinstance(item, Fact):
            facts.append(_recited(item, names))
            continue
        try:
            row = item if isinstance(item, TripleRow) else TripleRow.model_validate(item)
        except ValidationError as exc:
            error = exc.errors()[0]
            where = ".".join(str(part) for part in error["loc"])
            raise ValueError(
                f"not a triples row: {where + ': ' if where else ''}{error['msg']}"
            ) from None
        doc = names.get(row.doc)
        if doc is None:
            unmatched[row.doc] += 1
            continue
        facts.append(to_fact(row, doc, corpus.ontology, extractor=TITLE))
    notes = []
    if unmatched:
        named = ", ".join(sorted(unmatched)[:5])
        notes.append(
            f"{sum(unmatched.values())} row(s) named no document the pipeline was given and were "
            f"left out: {named}"
        )
    return facts, notes


# --------------------------------------------------------------------------- #
# The check: the Validator over the same output
# --------------------------------------------------------------------------- #


@dataclass
class Checked:
    """The pipeline's facts after the Validator, what it cost, and what its gate refused."""

    facts: list[Fact]
    calls: list[CallRecord]
    prompts: tuple[str, ...]
    models: dict[str, str]
    notes: list[str]
    # Each fact the gate refused, and its reason: the diagnosis's refused bucket.
    refused: list[tuple[Fact, str]] = field(default_factory=list)


def check(facts: Sequence[Fact], corpus: Corpus, config: str | Path | None = None) -> Checked:
    """`openodke.Validator` over the pipeline's facts and the documents they cite (#129).

    With `config`, the Validator gets that run config's stages, models and
    recorded responses, as `odke validate --config` builds it; a stage the
    config leaves out is the Validator's default, not a pass-through. Without
    one, the Validator's own defaults on the default models. Metered either
    way, so the row has its calls, tokens, cost and latency.
    """
    from openodke.eval.cost import CostMeter
    from openodke.gate import Kept, VerdictGate
    from openodke.llm.roles import ModelRoles
    from openodke.validator import Validator

    if config is not None:
        from openodke.run.build import build
        from openodke.run.config import load_config

        loaded = load_config(config)
        built = build(
            loaded.model_copy(update={"models": loaded.models.model_copy(update={"meter": True})})
        )
        meter = built.context.meter
        assert meter is not None
        stage = built.stages
        kept = Kept(stage["gate"] if stage.get("gate") is not None else VerdictGate(schema=True))
        validator = Validator(
            corpus.ontology or built.ontology,
            grounder=stage["grounder"],
            roles=built.context.roles,
            client=built.context.client("ground"),
            normalizer=stage["normalizer"],
            resolver=stage["resolver"],
            corroborator=stage["corroborator"],
            scorer=stage["scorer"],
            gate=kept,
            inverses=loaded.inverses,
            coverage=loaded.coverage,
        )
        configured = {"ground": built.context.roles.ground.model}
        source = Path(config).name
    else:
        roles = ModelRoles()
        meter = CostMeter()
        client = meter.client(roles.client_for("ground"), "ground")
        kept = Kept(VerdictGate(schema=True))
        validator = Validator(corpus.ontology, roles=roles, client=client, gate=kept)
        configured = {"ground": roles.ground.model}
        source = "its defaults"
    graph, done = validator.validate(list(facts), corpus.documents)
    verdicts = ", ".join(f"{k} {v}" for k, v in done.verdicts.items())
    refused = ", ".join(f"{v} {k}" for k, v in done.refused_by.items() if v)
    notes = [
        f"{VALIDATOR}: openodke.Validator over the same facts, with {source}: grounded "
        f"{verdicts}; {done.refused} refused" + (f" ({refused})" if refused else "") + ", "
        f"{done.merged} merged, {done.linked} linked, {done.derived} derived"
    ]
    if done.unmatched:
        notes.append(f"{VALIDATOR}: {done.unmatched} fact(s) cite no text it was given")
    return Checked(
        facts=list(graph.facts),
        calls=list(meter.records),
        prompts=done.prompts,
        models=models_called(meter.records, configured),
        notes=notes,
        refused=kept.refused,
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class Evaluation(NamedTuple):
    """A pipeline scored: the report, its last row's outcome per document, and its run config.

    `items` are what `--items` writes and the track record names the run by;
    empty for a benchmark whose scorer counts no hits per document. `config` is
    the run config that produced the predictions (a `trace` that is one), else
    the Validator's, as data: what the track record diffs between runs.
    """

    report: EvalReport
    items: list[ItemRow]
    config: dict[str, Any] | None = None


def score(
    corpus: Corpus,
    facts: Sequence[Fact],
    *,
    checked: Checked | None = None,
    notes: Sequence[str] = (),
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
    offered: Offered | None = None,
    examples: int = EXAMPLES,
    reference: Sequence[Fact] | None = None,
) -> EvalReport:
    """The pipeline's facts, and the checked ones when there are, as the eval report.

    The last row is diagnosed, with `offered` the trace of what the extractor
    was shown, and its fixes ranked; `reference` is another system's facts on
    the same documents, for each fix's gain at that system's rate.
    """
    return _score(
        corpus,
        facts,
        checked=checked,
        notes=notes,
        resamples=resamples,
        seed=seed,
        level=level,
        offered=offered,
        examples=examples,
        reference=reference,
    ).report


def _score(
    corpus: Corpus,
    facts: Sequence[Fact],
    *,
    checked: Checked | None,
    notes: Sequence[str],
    resamples: int,
    seed: int,
    level: float,
    offered: Offered | None,
    examples: int,
    reference: Sequence[Fact] | None = None,
    judge: EquivalenceJudge | None = None,
) -> Evaluation:
    configurations: list[Configuration] = [(PIPELINE, facts, None)]
    if checked is not None:
        configurations.append((VALIDATOR, checked.facts, checked.calls))
    run = Run(
        models=checked.models if checked is not None else {},
        prompts=checked.prompts if checked is not None else (),
        dataset=corpus.dataset,
    )
    lines = [*notes, *(checked.notes if checked is not None else ())]
    last, final, _ = configurations[-1]
    refused = None if checked is None else checked.refused
    view: View | None = None
    other: View | None = None
    if corpus.gold is not None:
        rows, how = extraction_rows(
            configurations,
            corpus.gold,
            ontology=corpus.ontology,
            resamples=resamples,
            seed=seed,
            level=level,
        )
        stages = tuple(
            evaluate_extraction(corpus.gold, found).model_copy(update={"stage": name})
            for name, found, _ in configurations
        )
        n = len(corpus.gold)
        texts = {doc.id: doc.text for doc in corpus.documents}
        view = gold_view(corpus.gold, final, refused=refused, texts=texts, row=last)
        if reference is not None:
            other = gold_view(corpus.gold, reference)
        counts = document_counts(corpus.gold, final).by_doc.items()
        items = [ItemRow(id=doc, tp=tp, fp=fp, fn=fn) for doc, (tp, fp, fn) in counts]
    else:
        from openodke.eval.datasets._common import (
            doc_names,
            report,
            score_configurations,
            triples_by_doc,
        )

        scoring = corpus.scoring
        scored = score_configurations(
            configurations, corpus.documents, scoring, resamples=resamples, seed=seed, level=level
        )
        n = int(scoring.dataset.documents or 0)
        labels = ("pipeline", "validator")[: len(configurations)]
        stages = (report(scoring.stage, n, scored.table, (), labels=labels),)
        rows = scored.rows
        how = Bootstrap(units=n, resamples=resamples, seed=seed, level=level)
        predicted = scored.predictions[last]
        items = [r for unit in scoring.units(predicted) if (r := _unit_row(unit)) is not None]
        if scoring.view is None:
            lines.append(f"no diagnosis: {scoring.stage}'s scorer has no view for one yet")
        else:
            relation = scoring.meta["relation_labels"]
            names = doc_names(corpus.documents)

            def triples(found: Sequence[Fact]) -> dict[str, list[tuple[str, str, str]]]:
                return triples_by_doc(found, names, lambda p: relation.get(p, p))

            gated = None if refused is None else triples([fact for fact, _ in refused])
            view = scoring.view(predicted, gated)
            view.row = last
            if reference is not None:
                other = scoring.view(triples(reference), None)
    diagnosed: tuple[Bucket, ...] = ()
    ranked: tuple[Fix, ...] = ()
    if view is not None:
        inverses = corpus.ontology.inverses if corpus.ontology is not None else {}
        found = diagnose(view, offered=offered, inverses=inverses, examples=examples, judge=judge)
        caught = None if other is None else {g.id for g in other.gold if g.found}
        diagnosed, ranked = found.buckets, tuple(rank_fixes(found, reference=caught))
    report_ = EvalReport(
        title=TITLE,
        n=n,
        run=run,
        bootstrap=how,
        rows=tuple(rows),
        stages=stages,
        notes=tuple(lines),
        diagnosis=diagnosed,
        fixes=ranked,
    )
    return Evaluation(report_, items)


def _unit_row(unit: Mapping[str, Any]) -> ItemRow | None:
    """A benchmark's per-document unit as an `--items` row, when it counts hits."""
    if {"tp", "predicted", "gold"} <= unit.keys():
        tp = int(unit["tp"])
        return ItemRow(
            id=str(unit["id"]), tp=tp, fp=int(unit["predicted"]) - tp, fn=int(unit["gold"]) - tp
        )
    if {"hits", "over", "under"} <= unit.keys():
        return ItemRow(
            id=str(unit["id"]), tp=int(unit["hits"]), fp=int(unit["over"]), fn=int(unit["under"])
        )
    return None


def evaluate_pipeline(
    *,
    command: str | None = None,
    function: str | Callable[..., Any] | None = None,
    predictions: str | Path | None = None,
    adapter: str = "triples",
    labels: str | Path | None = None,
    documents: str | Path | None = None,
    bench: str | Path | None = None,
    ontology: str | Path | None = None,
    validator: bool = False,
    config: str | Path | None = None,
    timeout: float = TIMEOUT,
    adjudicate: str | Path | None = None,
    trace: str | Path | None = None,
    examples: int = EXAMPLES,
    reference: str | Path | None = None,
    lenient: str | Path | None = None,
) -> EvalReport:
    """Run a pipeline one way, read its output, validate it if asked, and score it.

    One of `command`, `function` and `predictions`; one of `labels` (with
    `documents`) and `bench`. `validator` runs `openodke.Validator` over the
    same output and adds its row; `config` is the run config it takes its
    stages and models from, a bench set's own `odke.json` by default.
    `adjudicate` names the file the adjudication list goes to: each prediction
    the gold lacks, grounded three times by `config`'s grounder or the default
    one (`openodke.eval.adjudication`). It needs `labels`. `trace` says what
    the extractor was offered, for the diagnosis: a run's `manifest.json`, or
    the run config that produced the predictions. `reference` is another
    system's output on the same documents, triples or facts, for each fix's
    gain at that system's rate. `lenient` names the file the fact-equivalence
    judge's pairs go to, and adds the lenient score (`openodke.eval.equivalence`);
    it needs `labels` too.
    """
    return evaluate(
        command=command,
        function=function,
        predictions=predictions,
        adapter=adapter,
        labels=labels,
        documents=documents,
        bench=bench,
        ontology=ontology,
        validator=validator,
        config=config,
        timeout=timeout,
        adjudicate=adjudicate,
        trace=trace,
        examples=examples,
        reference=reference,
        lenient=lenient,
    ).report


def evaluate(
    *,
    command: str | None = None,
    function: str | Callable[..., Any] | None = None,
    predictions: str | Path | None = None,
    adapter: str = "triples",
    labels: str | Path | None = None,
    documents: str | Path | None = None,
    bench: str | Path | None = None,
    ontology: str | Path | None = None,
    validator: bool = False,
    config: str | Path | None = None,
    timeout: float = TIMEOUT,
    adjudicate: str | Path | None = None,
    trace: str | Path | None = None,
    examples: int = EXAMPLES,
    reference: str | Path | None = None,
    lenient: str | Path | None = None,
) -> Evaluation:
    """`evaluate_pipeline`, with the last row's items and the run config beside the report."""
    modes = [m for m in (command, function, predictions) if m is not None]
    if len(modes) != 1:
        raise ValueError("pipeline takes exactly one of --cmd, --run and --predictions")
    if (labels is None) == (bench is None):
        raise ValueError(
            "pipeline scores against --labels (with --documents) or --bench, one of them"
        )
    if adapter not in FORMATS:
        raise ValueError(f"unknown adapter {adapter!r}; one of {', '.join(FORMATS)}")
    if function is not None and adapter != "triples":
        raise ValueError("--adapter reads files; a callable returns rows or facts itself")
    if config is not None and not (validator or adjudicate is not None or lenient is not None):
        raise ValueError(
            "--config is for --validator, --adjudicate and --lenient: the Validator's stages "
            "and models, the grounder's and the judge's"
        )
    if adjudicate is not None and labels is None:
        raise ValueError(
            "--adjudicate needs --labels: it lists the predictions your gold facts lack, by "
            "openodke's matching, and a --bench set is scored by its benchmark's own"
        )
    if lenient is not None and labels is None:
        raise ValueError(
            "--lenient needs --labels: it pairs your gold facts' misses with the predictions "
            "openodke's matching left, and a --bench set is scored by its benchmark's own"
        )
    offered, produced = read_trace(trace) if trace is not None else (None, None)
    if labels is not None:
        if documents is None:
            raise ValueError("--labels needs --documents: the texts the pipeline reads")
        corpus = labelled(labels, documents, ontology)
    else:
        assert bench is not None
        corpus = prepared(bench)
        if ontology is not None:
            corpus.ontology = _ontology(ontology)
        if validator and config is None:
            config = Path(bench) / "odke.json"
    if command is not None:
        output = run_command(command, corpus, adapter=adapter, timeout=timeout)
    elif function is not None:
        output = run_callable(function, corpus)
    else:
        assert predictions is not None
        output = read_output(predictions, adapter=adapter)
    facts, notes = as_facts(output, corpus)
    ran = f" in {output.seconds:.1f} s" if output.seconds is not None else ""
    notes.insert(0, f"{output.how}: {len(output.items)} row(s){ran}, {len(facts)} fact(s)")
    checked = check(facts, corpus, config) if validator else None
    others = None if reference is None else as_facts(read_output(reference), corpus)[0]
    judged = None
    if lenient is not None:
        rows = [(PIPELINE, facts)] + ([(VALIDATOR, checked.facts)] if checked else [])
        judged = judged_leniently(corpus, rows, Path(lenient), config)
    found = _score(
        corpus,
        facts,
        checked=checked,
        notes=notes,
        resamples=RESAMPLES,
        seed=SEED,
        level=LEVEL,
        offered=offered,
        examples=examples,
        reference=others,
        judge=Decided(judged.pairs) if judged is not None else None,
    )
    report = found.report
    if adjudicate is not None:
        rows = [(PIPELINE, facts)] + ([(VALIDATOR, checked.facts)] if checked else [])
        report = adjudicated(report, corpus, rows, Path(adjudicate), config)
    if judged is not None:
        report = judged.attach(report)
    if produced is None and validator and config is not None:
        produced = _data(config)
    return Evaluation(report, found.items, produced)


def adjudicated(
    report: EvalReport,
    corpus: Corpus,
    rows: Sequence[tuple[str, Sequence[Fact]]],
    audit: Path,
    config: str | Path | None = None,
) -> EvalReport:
    """`report` with its adjudication section, after writing the list to `audit`.

    The grounder is `config`'s, as `odke validate --config` builds it, or the
    default `LLMGrounder` on the default models; metered either way, so the
    report names the model it called.
    """
    from openodke.eval.adjudication import NEEDED, RUNS, adjudicate, write_audit
    from openodke.eval.cost import CostMeter

    assert corpus.gold is not None
    if config is not None:
        from openodke.run.build import build
        from openodke.run.config import load_config

        loaded = load_config(config)
        built = build(
            loaded.model_copy(update={"models": loaded.models.model_copy(update={"meter": True})})
        )
        grounder = built.stages["grounder"]
        meter = built.context.meter
        assert meter is not None
        configured = {"ground": built.context.roles.ground.model}
    else:
        from openodke.ground import LLMGrounder
        from openodke.llm.roles import ModelRoles

        roles = ModelRoles()
        meter = CostMeter()
        grounder = LLMGrounder(roles, client=meter.client(roles.client_for("ground"), "ground"))
        configured = {"ground": roles.ground.model}
    section, entries = adjudicate(rows, corpus.gold, corpus.documents, grounder)
    write_audit(audit, entries)
    listed = sum(entry.listed for entry in entries)
    note = (
        f"adjudication: {len(entries)} prediction(s) the gold lacks, grounded {RUNS} times "
        f"each; {listed} supported in {NEEDED} or more, possibly missing from gold "
        f"(every one, with its verdicts, in {audit})"
    )
    stats = getattr(grounder, "stats", None)
    prompts = stats.get("prompts", []) if isinstance(stats, Mapping) else []
    run = report.run.model_copy(
        update={
            "models": {**models_called(meter.records, configured), **report.run.models},
            "prompts": tuple(dict.fromkeys([*report.run.prompts, *prompts])),
        }
    )
    return report.model_copy(
        update={
            "adjudication": section.model_copy(update={"audit": str(audit)}),
            "notes": (*report.notes, note),
            "run": run,
        }
    )


@dataclass(frozen=True)
class Judged:
    """What `--lenient` found: the section, every pair asked, and what the report says of it."""

    section: Lenient
    pairs: list[Pairing]
    note: str
    models: dict[str, str]
    prompts: tuple[str, ...]

    def attach(self, report: EvalReport) -> EvalReport:
        """`report` with the lenient section, its note, and the judge's model and prompts."""
        run = report.run.model_copy(
            update={
                "models": {**self.models, **report.run.models},
                "prompts": tuple(dict.fromkeys([*report.run.prompts, *self.prompts])),
            }
        )
        return report.model_copy(
            update={"lenient": self.section, "notes": (*report.notes, self.note), "run": run}
        )


def judged_leniently(
    corpus: Corpus,
    rows: Sequence[tuple[str, Sequence[Fact]]],
    pairs: Path,
    config: str | Path | None = None,
) -> Judged:
    """The lenient section over `rows`, after writing every pair asked to `pairs`.

    The judge is `FactJudge` on `config`'s ground model, recorded responses
    and all, or the default one; metered under the stage `judge` either way,
    so the report names the model it called. The ontology describes each
    relation to it. The diagnosis reads the same decisions (`Decided`).
    """
    from openodke.eval.cost import CostMeter
    from openodke.eval.equivalence import FactJudge, lenient, write_pairs
    from openodke.llm.roles import ModelRoles

    assert corpus.gold is not None
    ontology = corpus.ontology
    if config is not None:
        from openodke.run.build import build
        from openodke.run.config import load_config

        loaded = load_config(config)
        built = build(
            loaded.model_copy(update={"models": loaded.models.model_copy(update={"meter": True})})
        )
        roles = built.context.roles
        client = built.context.client("ground", stage="judge")
        meter = built.context.meter
        assert meter is not None
        ontology = ontology or built.ontology
    else:
        roles = ModelRoles()
        meter = CostMeter()
        client = meter.client(roles.client_for("ground"), "judge")
    judge = FactJudge(roles, client=client, ontology=ontology)
    section, entries = lenient(rows, corpus.gold, corpus.documents, judge)
    write_pairs(pairs, entries)
    same = sum(1 for e in entries if e.decision is not None and e.decision.decision == "same")
    note = (
        f"lenient: {len(entries)} pair(s) the pre-filter let through, asked in both orders; "
        f"{same} judged the same fact (every one, with both answers, in {pairs})"
    )
    return Judged(
        section=section,
        pairs=entries,
        note=note,
        models=models_called(meter.records, {"judge": roles.ground.model}),
        prompts=tuple(judge.stats["prompts"]),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _file_names(documents: Sequence[Document]) -> dict[str, str]:
    """A distinct, safe `<name>.txt` for each document under `{in}`."""
    out: dict[str, str] = {}
    taken: set[str] = set()
    for doc in documents:
        stem = re.sub(r"[^\w.-]+", "_", doc.id.removesuffix(".txt")).strip("._") or "document"
        name, n = stem, 1
        while name.casefold() in taken:
            n += 1
            name = f"{stem}~{n}"
        taken.add(name.casefold())
        out[doc.id] = f"{name}.txt"
    return out


def _names(corpus: Corpus) -> dict[str, Document]:
    """Every name a row may give a document: its id, its name under `{in}`, its file's stem."""
    names: dict[str, Document] = {doc.id: doc for doc in corpus.documents}
    for doc in corpus.documents:
        names.setdefault(Path(corpus.files[doc.id]).stem, doc)
    stems = Counter(stem for doc in corpus.documents if (stem := _stem(doc)) is not None)
    for doc in corpus.documents:
        stem = _stem(doc)
        if stem is not None and stems[stem] == 1:
            names.setdefault(stem, doc)
    return names


def _stem(doc: Document) -> str | None:
    return _file_name(doc)


def _by_uri(names: Mapping[str, Document], text: Document) -> Document | None:
    stem = _stem(text)
    return names.get(stem) if stem is not None else None


def _by_text(corpus: Corpus, text: Document) -> Document | None:
    same = [doc for doc in corpus.documents if doc.text == text.text]
    return same[0] if len(same) == 1 else None


def _recited(fact: Fact, names: Mapping[str, Document]) -> Fact:
    """A fact whose evidence names a document by another of its names, renamed to its id."""
    evidence = []
    for e in fact.evidence:
        doc = names.get(e.doc_id)
        if doc is None or doc.id == e.doc_id:
            evidence.append(e)
            continue
        span = e.span.model_copy(update={"doc_id": doc.id}) if e.span is not None else None
        evidence.append(e.model_copy(update={"doc_id": doc.id, "span": span}))
    return fact.model_copy(update={"evidence": tuple(evidence)})


def _is_fact(line: str) -> bool:
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(row, dict) and isinstance(row.get("subject"), dict)


def _ontology(source: str | Path | Ontology | None) -> Ontology | None:
    if source is None or isinstance(source, Ontology):
        return source
    path = Path(source)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return Ontology.from_yaml(path)
    return Ontology.model_validate_json(path.read_text(encoding="utf-8"))


def _data(path: str | Path) -> dict[str, Any]:
    """A run config as plain data, JSON or YAML: what the track record keeps of it."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    return dict(loaded) if isinstance(loaded, dict) else {}


def _bench_module(meta: Mapping[str, Any]) -> Any:
    from openodke.eval.datasets import DATASETS

    name = meta.get("dataset")
    if name not in DATASETS:
        raise ValueError(f"dataset.json names {name!r}; one of {', '.join(DATASETS)}")
    return DATASETS[name]


def _name(spec: Any) -> str:
    return spec if isinstance(spec, str) else getattr(spec, "__qualname__", repr(spec))


__all__ = [
    "ADAPTERS",
    "DESCRIPTION",
    "FORMATS",
    "TIMEOUT",
    "Checked",
    "Corpus",
    "Evaluation",
    "Output",
    "PipelineError",
    "as_facts",
    "check",
    "evaluate",
    "evaluate_pipeline",
    "labelled",
    "load_callable",
    "load_documents",
    "prepared",
    "read_output",
    "run_callable",
    "run_command",
    "score",
]
