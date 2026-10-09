"""The `odke` command.

Thin on purpose: every command is a few lines over the library, so that anything
reachable from the CLI is reachable from Python and neither grows a behaviour the
other lacks.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import typer

from openodke import __version__
from openodke.ontology import Ontology, OntologyLoadError

if TYPE_CHECKING:
    from openodke.run import RunConfig

app = typer.Typer(
    name="odke",
    help="Ontology-guided knowledge extraction: text in, a grounded knowledge graph out.",
    no_args_is_help=True,
    add_completion=False,
)
ontology_app = typer.Typer(help="Inspect and compile ontologies.", no_args_is_help=True)
app.add_typer(ontology_app, name="ontology")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"openodke {__version__}")
        raise typer.Exit()


def _load(path: Path) -> Ontology:
    # Never strict: these commands exist to show what is wrong with a schema,
    # and refusing to load it would hide exactly that.
    if path.suffix.lower() in {".yaml", ".yml"}:
        return Ontology.from_yaml(path, strict=False)
    return Ontology.from_json(path, strict=False)


def _load_or_exit(path: Path) -> Ontology:
    try:
        return _load(path)
    except (OntologyLoadError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None


def _load_run_config(path: Path) -> RunConfig:
    """`load_config`, with what it warned about printed as `warning:` lines.

    Python hides a `DeprecationWarning` raised inside a library, and the reader
    of a config file is a person at a terminal: a key that still works under an
    old name (`validator:`, now `gate:`) has to be said where they will see it.
    """
    from openodke.run import load_config

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        config = load_config(path)
    for warning in caught:
        typer.echo(f"warning: {warning.message}", err=True)
    return config


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _qualified(model: str | None, provider: str | None) -> str | None:
    """`--model` and `--model-provider` as one provider-qualified string, or None.

    Exits 2 rather than raising: choosing a model wrongly is a usage error, and
    it is worth saying so before a config is read or a document is loaded.
    """
    if model is None:
        if provider is None:
            return None
        typer.echo("error: --model-provider needs --model", err=True)
        raise typer.Exit(2)
    from openodke.llm.providers import qualify

    try:
        return qualify(model, provider)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """openodke — ontology-guided knowledge extraction."""


@ontology_app.command("snippet")
def ontology_snippet(
    path: Path = typer.Argument(..., help="Ontology JSON or YAML file."),
    entity_type: str = typer.Argument(..., help="Entity type to render a snippet for."),
    limit: int = typer.Option(25, help="Maximum predicates in the snippet."),
    as_json_schema: bool = typer.Option(False, "--json-schema", help="Emit JSON Schema instead."),
) -> None:
    """Print the exact schema fragment the extractor would be prompted with.

    Being able to see this without spending a token is most of what makes an
    ontology debuggable: a bad extraction is usually a bad snippet.
    """
    snippet = _load_or_exit(path).snippet(entity_type, limit=limit)
    typer.echo(json.dumps(snippet.json_schema(), indent=2) if as_json_schema else snippet.render())


@ontology_app.command("types")
def ontology_types(
    path: Path = typer.Argument(..., help="Ontology JSON or YAML file."),
) -> None:
    """List entity types and how many predicates each one can carry."""
    ontology = _load_or_exit(path)
    for name in sorted(ontology.types):
        typer.echo(f"{name}\t{len(ontology.predicates_for(name))} predicates")


@ontology_app.command("validate")
def ontology_validate(
    paths: list[Path] = typer.Argument(..., help="Ontology JSON or YAML files."),
) -> None:
    """Print every diagnostic, and exit 1 if any file has an error.

    Warnings print and pass, so this can gate a pre-commit hook or a CI step
    without teaching anyone to ignore a red build. It takes several files
    because that is how pre-commit calls it.
    """
    errors = warned = 0
    for path in paths:
        try:
            diagnostics = _load(path).validate()
        except (OntologyLoadError, ImportError) as exc:
            typer.echo(f"error: {exc}")
            errors += 1
            continue
        for diagnostic in diagnostics:
            typer.echo(f"{path}: {diagnostic}")
        found = sum(d.severity == "error" for d in diagnostics)
        errors += found
        warned += len(diagnostics) - found
    typer.echo(f"{_count(errors, 'error')}, {_count(warned, 'warning')}")
    if errors:
        raise typer.Exit(1)


@ontology_app.command("diff")
def ontology_diff(
    old: Path = typer.Argument(..., help="The ontology as it was."),
    new: Path = typer.Argument(..., help="The ontology as it is now."),
    fail_on_breaking: bool = typer.Option(
        False, "--fail-on-breaking", help="Exit 1 if any change is breaking."
    ),
) -> None:
    """Added, removed and changed types, predicates, qualifiers and cardinality.

    Breaking changes print first, because "does this break the live graph?" is
    the question being asked; `--fail-on-breaking` makes the answer an exit code.
    """
    changes = _load_or_exit(old).diff(_load_or_exit(new))
    for change in sorted(changes, key=lambda c: not c.breaking):
        typer.echo(str(change))
    breaking = sum(c.breaking for c in changes)
    typer.echo(f"{_count(len(changes), 'change')}, {breaking} breaking")
    if breaking and fail_on_breaking:
        raise typer.Exit(1)


MODEL_HELP = "Provider-qualified model for every role, e.g. openai/gpt-5.5. Overrides `models`."
MODEL_PROVIDER_HELP = "Provider for --model when the string does not name one, e.g. openai."


@app.command("models")
def models_command() -> None:
    """List every provider openodke can address, and whether its key is set.

    Which provider serves a model string, which environment variable that
    provider reads, whether it is set here, and which client makes the call —
    so "use OpenAI" is something to check rather than to infer. No value is
    printed and no key is written anywhere: openodke reads the environment and
    nothing else.
    """
    # Imported here so `odke --version` stays light. Nothing on this path imports
    # a provider: the [llm] extra is looked for, never loaded.
    from openodke.llm.providers import describe
    from openodke.llm.registry import registered_providers

    typer.echo(describe(os.environ, registered=registered_providers()))


@app.command("run")
def run_command(
    config: Path = typer.Argument(..., help="Run config, YAML or JSON. See examples/run.yaml."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load, extract and ground, then print what would be written instead of writing it.",
    ),
    model: str | None = typer.Option(None, "--model", help=MODEL_HELP),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    widen: bool = typer.Option(
        False,
        "--widen",
        help="Give a not_found whose cited span is narrower than its sentence one more "
        "grounding call, against the sentence. Needs stages.grounder: llm.",
    ),
) -> None:
    """Run the whole pipeline from a config file.

    The config names the inputs, the ontology, the models and which
    implementation fills each of the thirteen stages. A dry run still calls
    the models; it is the store it spares. Exit 2 is a config that cannot run,
    exit 1 a run that failed.

    `--model` puts every role on one model and prints which, so a run states
    what it called instead of leaving it to be read out of a config; per-role
    models stay a config decision. `odke models` lists what can be named.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.llm.base import ProviderError
    from openodke.run import ConfigError, execute

    chosen = _qualified(model, model_provider)
    try:
        loaded = _load_run_config(config)
        if chosen is not None:
            loaded = loaded.with_model(chosen)
            typer.echo(f"models: every role on {chosen}")
        if widen:
            loaded = loaded.with_widen()
        result = execute(loaded, dry_run=dry_run)
    except (ConfigError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except (ProviderError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(result.render())


@ontology_app.command("infer")
def ontology_infer(
    paths: list[Path] = typer.Argument(..., help="Files or directories to infer a schema from."),
    out: Path = typer.Option(
        ..., "--out", "-o", help="Where to write the draft: .yaml, .yml or .json."
    ),
    max_types: int = typer.Option(10, "--max-types", help="Keep at most this many entity types."),
    max_predicates: int = typer.Option(
        25, "--max-predicates", help="Keep at most this many predicates."
    ),
    sample_words: int = typer.Option(
        8_000, "--sample-words", help="How many words of the corpus to sample."
    ),
    seed: int = typer.Option(
        0, "--seed", help="Sampling seed: the same files and seed give the same draft."
    ),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Deterministic proposers only: no model, no key, no cost."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Model for the infer role, e.g. ollama/llama3.1."
    ),
    model_provider: str | None = typer.Option(None, "--model-provider", help=MODEL_PROVIDER_HELP),
    name: str = typer.Option("inferred", "--name", help="Name of the drafted ontology."),
) -> None:
    """Draft an ontology for a corpus that has none: for review, never to use as is.

    Record shapes, Hearst patterns and co-occurrence propose types and predicates,
    each with the spans that produced it; unless --no-llm, a model (the infer role)
    then names and merges what they found. Writes the draft marked inferred, with
    its evidence as comments, and the full evidence beside it as
    <out>.evidence.json, and prints what backs each proposal. Then validate, edit,
    and `odke ontology freeze` it.
    """
    # Imported here so `odke --version` and the other commands stay light.
    from openodke.infer.build import infer_ontology
    from openodke.infer.review import (
        evidence_path,
        format_for,
        inferred_header,
        notes,
        render,
        summary,
    )
    from openodke.llm import ModelSpec, ProviderError
    from openodke.loaders import DirectoryLoader

    chosen = _qualified(model, model_provider)
    try:
        fmt = format_for(out)
        loader = DirectoryLoader()
        docs = [doc for path in paths for doc in loader.load(path)]
    except (OSError, ValueError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    if not docs:
        # From the loader's own table, so the hint cannot fall behind what it reads.
        suffixes = ", ".join(loader.loaders)
        typer.echo(
            f"error: no documents: no file under those paths has a loader ({suffixes})",
            err=True,
        )
        raise typer.Exit(2)
    try:
        inference = infer_ontology(
            docs,
            llm=not no_llm,
            spec=ModelSpec(model=chosen) if chosen else None,
            name=name,
            seed=seed,
            sample_words=sample_words,
            max_types=max_types,
            max_predicates=max_predicates,
        )
    except ProviderError as exc:
        typer.echo(f"error: {exc}\n--no-llm runs the deterministic proposers alone.", err=True)
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None

    evidence = evidence_path(out)
    try:
        text = render(
            inference.ontology,
            fmt=fmt,
            header=inferred_header(inference, evidence_file=evidence.name),
            notes=notes(inference),
        )
    except ImportError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    evidence.write_text(
        inference.model_dump_json(indent=2, exclude={"ontology"}) + "\n", encoding="utf-8"
    )
    typer.echo(summary(inference))
    typer.echo(f"wrote {out} — inferred: review it, then `odke ontology freeze {out}`")
    typer.echo(f"wrote {evidence}")


@ontology_app.command("freeze")
def ontology_freeze(
    path: Path = typer.Argument(..., help="The reviewed ontology, JSON or YAML."),
    by: str | None = typer.Option(
        None, "--by", help="Who reviewed it. Defaults to the current user."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write the frozen ontology here instead of in place."
    ),
) -> None:
    """Mark a reviewed ontology frozen: inferred cleared, reviewer and time recorded.

    Refuses while `validate` finds an error, printing each one and exiting 1, and
    leaves the file untouched. The frozen file is written without the review
    comments; the evidence file beside a draft is left where it is.
    """
    from openodke.infer.review import format_for, frozen_header, render
    from openodke.ontology import OntologyFreezeError

    ontology = _load_or_exit(path)
    try:
        frozen = ontology.freeze(by=by if by is not None else _current_user())
    except OntologyFreezeError as exc:
        for diagnostic in exc.diagnostics:
            typer.echo(f"{path}: {diagnostic}")
        typer.echo(f"not frozen: {_count(len(exc.diagnostics), 'error')}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    target = out if out is not None else path
    try:
        text = render(frozen, fmt=format_for(target), header=frozen_header(frozen))
    except (ValueError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    target.write_text(text, encoding="utf-8")
    when = frozen.frozen_at.isoformat() if frozen.frozen_at else "now"
    typer.echo(f"frozen: {target} (by {frozen.frozen_by} at {when})")


def _current_user() -> str:
    import getpass

    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):
        return "unknown"


@app.command("eval")
def eval_stage(
    stage: str = typer.Argument(
        ...,
        help="route, extract, ground, resolve, score, validate, ablation (with --config), "
        "or spans (with --facts, and no labels at all).",
    ),
    labels: Path | None = typer.Option(None, "--labels", help="Your labelled rows, as JSONL."),
    predictions: Path | None = typer.Option(
        None, "--predictions", help="What the stage produced, as JSONL."
    ),
    run: str | None = typer.Option(
        None, "--run", help="package.module:Name — run that stage over the labels instead."
    ),
    ontology: Path | None = typer.Option(
        None, "--ontology", help="Ontology JSON, for --run with extract or validate."
    ),
    documents: Path | None = typer.Option(
        None, "--documents", help="Document JSONL, for --run with extract."
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Run config, for ablation: the pipeline to run three ways."
    ),
    facts: Path | None = typer.Option(
        None, "--facts", help="For spans: a run's facts.jsonl, or the directory a sink wrote."
    ),
    describe: bool = typer.Option(
        False, "--describe", help="Print what a label row and a prediction row are, and exit."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Score one stage against your own labelled data.

    Bring your own labelled dataset: openodke ships the formats and the arithmetic,
    never a corpus. The fixtures in its test suite are examples of the formats
    and are not a benchmark. Run with --describe to see what to label.

    `ablation` runs a whole config three ways — extraction alone, + grounding,
    + corroboration — over your labelled extraction set.

    `spans` is the exception: it needs no labelled data. It reads a run's facts
    and reports span width split by the grounder's verdict, which scores the
    too-narrow-citation error with no gold set at all.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.eval.formats import describe as describe_formats
    from openodke.eval.runner import evaluate_files
    from openodke.llm.base import ProviderError

    try:
        if stage == "spans":
            from openodke.eval.spans import DESCRIPTION, evaluate_spans, load_facts

            if describe:
                typer.echo(DESCRIPTION)
                return
            if labels is not None:
                raise ValueError(
                    "spans needs no labelled data; pass the run's facts as --facts instead"
                )
            # A facts file is what --predictions already means for the fact
            # stages, so it is taken as --facts rather than refused.
            source = facts if facts is not None else predictions
            if any(v is not None for v in (run, ontology, documents, config)):
                raise ValueError("spans reads a run's facts; it takes --facts and nothing else")
            if source is None:
                raise ValueError(
                    "spans needs --facts: a run's facts.jsonl, or the directory a sink wrote"
                )
            report = evaluate_spans(load_facts(source))
        elif stage == "ablation":
            from openodke.eval.ablation import DESCRIPTION, run_ablation
            from openodke.eval.formats import GoldFact, load_jsonl

            if describe:
                typer.echo(
                    f"{DESCRIPTION}\n\n{describe_formats('extract').split('--predictions')[0]}"
                )
                return
            if any(v is not None for v in (predictions, run, ontology, documents, facts)):
                raise ValueError("ablation runs the config itself; it takes --config and --labels")
            if config is None or labels is None:
                raise ValueError("ablation needs --config and --labels (or --describe)")
            report = run_ablation(_load_run_config(config), load_jsonl(labels, GoldFact))
        else:
            if describe:
                typer.echo(describe_formats(stage))
                return
            if labels is None:
                raise ValueError("--labels is required (or --describe to see the format)")
            if config is not None:
                raise ValueError("--config is for ablation")
            if facts is not None:
                raise ValueError("--facts is for spans")
            report = evaluate_files(
                stage, labels, predictions, run=run, ontology=ontology, documents=documents
            )
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ProviderError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(report.model_dump_json(indent=2) if as_json else report.render())


# --------------------------------------------------------------------------- #
# odke label — labelling by hand
# --------------------------------------------------------------------------- #

label_app = typer.Typer(
    help="Write sheets to label by hand, and read the ticks back as labels.",
    no_args_is_help=True,
)
app.add_typer(label_app, name="label")


def _data_error(exc: Exception) -> NoReturn:
    # One `error:` line per problem: a sheet can have several, each with its line.
    for line in str(exc).splitlines():
        typer.echo(f"error: {line}", err=True)
    raise typer.Exit(2) from exc


@label_app.command("make")
def label_make(
    kind: str = typer.Argument(..., help="grounding or pair."),
    items: Path = typer.Argument(..., help="The rows to label, as JSONL."),
    out: Path = typer.Option(..., "--out", "-o", help="A directory with no sheets in it yet."),
    per_sheet: int = typer.Option(
        50, "--per-sheet", min=1, help="Items per sheet: one sheet is one sitting."
    ),
) -> None:
    """Write numbered markdown sheets to tick by hand, with the rows kept beside them.

    A grounding row is a GroundingLabel without its verdict: the fact and the
    text it cites. A pair row is two mentions, `a` and `b`, each a key and a
    type with an optional label and context. Each item gets a heading, what to
    judge and one box per answer; sheet-NN.items.jsonl keeps its rows. The same
    rows write the same sheets, byte for byte, and a directory that already
    holds sheets is refused. Exit 2 is a row or a directory it cannot use.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.eval.sheets import make_sheets

    try:
        made = make_sheets(kind, items, out, per_sheet=per_sheet)
    except (ValueError, OSError) as exc:
        _data_error(exc)
    for warning in made.warnings:
        typer.echo(f"warning: {warning}", err=True)
    typer.echo(f"wrote {_count(len(made.sheets), 'sheet')} to {out}: {made.first} to {made.last}")


@label_app.command("read")
def label_read(
    path: Path = typer.Argument(..., help="A sheet, or the directory `odke label make` wrote."),
    out: Path = typer.Option(..., "--out", "-o", help="Where to write the labels, as JSONL."),
) -> None:
    """Read ticked sheets back as GroundingLabel or PairLabel rows.

    One tick is a label. No tick leaves the item unlabelled, and it is counted.
    Two ticks, or a heading the sidecar does not know, exits 2 with the file and
    the line of the item's heading, and nothing is written. Notes go beside the
    labels as <out>.notes.jsonl; pairs answered unsure as <out>.unsure.jsonl.
    """
    from openodke.eval.sheets import read_sheets

    try:
        reading = read_sheets(path)
        written = reading.write(out)
    except (ValueError, OSError) as exc:
        _data_error(exc)
    typer.echo(reading.summary())
    for target, n in written:
        typer.echo(f"wrote {target}: {_count(n, 'row')}")


# --------------------------------------------------------------------------- #
# odke bench — public benchmarks
# --------------------------------------------------------------------------- #

bench_app = typer.Typer(
    help="Public benchmarks: fetch a dataset, prepare a run, run it three ways and score it.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


def _dataset(name: str) -> Any:
    from openodke.eval.datasets import DATASETS

    if name not in DATASETS:
        typer.echo(f"error: unknown dataset {name!r}; one of {', '.join(DATASETS)}", err=True)
        raise typer.Exit(2)
    return DATASETS[name]


@bench_app.command("fetch")
def bench_fetch(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    dest: Path = typer.Argument(..., help="Directory to download into."),
    source: str = typer.Option(
        "wikidata_tekgen", help="text2kgbench only: wikidata_tekgen or dbpedia_webnlg."
    ),
    ontology: list[str] | None = typer.Option(
        None, "--ontology", help="text2kgbench only: an ontology id, repeatable (default: all)."
    ),
    split: list[str] | None = typer.Option(
        None, "--split", help="redocred only: dev, test or train, repeatable (default: dev, test)."
    ),
) -> None:
    """Download a dataset's official files. Nothing is redistributed by openodke."""
    module = _dataset(dataset)
    try:
        if dataset == "text2kgbench":
            module.fetch(dest, source=source, ontologies=ontology or None)
        else:
            module.fetch(dest, splits=tuple(split or ("dev", "test")))
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"fetched {dataset} into {dest}")


@bench_app.command("prepare")
def bench_prepare(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    root: Path = typer.Argument(..., help="Where `odke bench fetch` downloaded it."),
    out: Path = typer.Option(..., "--out", help="Directory to write the runnable set into."),
    ontology: str | None = typer.Option(
        None, "--ontology", help="text2kgbench only: the ontology id, e.g. ont_1_movie."
    ),
    source: str = typer.Option(
        "wikidata_tekgen", help="text2kgbench only: wikidata_tekgen or dbpedia_webnlg."
    ),
    split: str = typer.Option("test", help="redocred only: dev, test or train."),
    limit: int | None = typer.Option(None, help="Keep the first N documents (a cheap pilot)."),
    extract_model: str | None = typer.Option(
        None,
        "--extract-model",
        help="Any LiteLLM model string: openai/gpt-5, anthropic/..., ollama/llama3.1.",
    ),
    ground_model: str | None = typer.Option(
        None, "--ground-model", help="Any LiteLLM model string; a smaller one than the extractor."
    ),
    max_tokens: int = typer.Option(
        16000, "--max-tokens", help="The extractor's output room; lower it for a small model."
    ),
    paper: bool = typer.Option(
        False,
        "--paper",
        help="ODKE+'s own grounder and gate: the whole context, True/False, affirmed facts only.",
    ),
) -> None:
    """Write documents, ontology, gold and an `odke.json` run config. Calls no model."""
    module = _dataset(dataset)
    models = {
        "extract_model": extract_model,
        "ground_model": ground_model,
        "paper": paper,
        "max_tokens": max_tokens,
    }
    try:
        if dataset == "text2kgbench":
            if ontology is None:
                raise ValueError("text2kgbench needs --ontology, e.g. ont_1_movie")
            module.prepare(root, ontology, out, source=source, limit=limit, **models)
        else:
            module.prepare(root, out, split=split, limit=limit, **models)
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"prepared {out}; run it with: odke bench run {dataset} {out}")


@bench_app.command("run")
def bench_run(
    dataset: str = typer.Argument(..., help="text2kgbench or redocred."),
    prepared: Path = typer.Argument(..., help="A directory `odke bench prepare` wrote."),
    as_json: bool = typer.Option(False, "--json", help="The report as JSON."),
) -> None:
    """Run the prepared config three ways and score each row. Calls the models it names.

    Extraction and grounding are each called once; + grounding and + corroboration
    replay them, so the bill is one run's, not three.
    """
    from openodke.llm.base import ProviderError

    module = _dataset(dataset)
    try:
        report = module.run(prepared)
    except (ValueError, OSError, ImportError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except ProviderError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(report.model_dump_json(indent=2) if as_json else report.render())


if __name__ == "__main__":  # pragma: no cover
    app()
