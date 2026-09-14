"""The `odke` command.

Thin on purpose: every command is a few lines over the library, so that anything
reachable from the CLI is reachable from Python and neither grows a behaviour the
other lacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from openodke import __version__
from openodke.ontology import Ontology, OntologyLoadError

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


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


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


@app.command("run")
def run_command(
    config: Path = typer.Argument(..., help="Run config, YAML or JSON. See examples/run.yaml."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Load, extract and ground, then print what would be written instead of writing it.",
    ),
) -> None:
    """Run the whole pipeline from a config file.

    The config names the inputs, the ontology, the models and which
    implementation fills each of the thirteen stages. A dry run still calls
    the models; it is the store it spares. Exit 2 is a config that cannot run,
    exit 1 a run that failed.
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.llm.base import ProviderError
    from openodke.run import ConfigError, execute, load_config

    try:
        result = execute(load_config(config), dry_run=dry_run)
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
            spec=ModelSpec(model=model) if model else None,
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
        ..., help="route, extract, ground, resolve, score, validate, or ablation (with --config)."
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
    """
    # Imported here so `odke --version` and the ontology commands stay light.
    from openodke.eval.formats import describe as describe_formats
    from openodke.eval.runner import evaluate_files
    from openodke.llm.base import ProviderError

    try:
        if stage == "ablation":
            from openodke.eval.ablation import DESCRIPTION, run_ablation
            from openodke.eval.formats import GoldFact, load_jsonl
            from openodke.run import load_config

            if describe:
                typer.echo(
                    f"{DESCRIPTION}\n\n{describe_formats('extract').split('--predictions')[0]}"
                )
                return
            if any(v is not None for v in (predictions, run, ontology, documents)):
                raise ValueError("ablation runs the config itself; it takes --config and --labels")
            if config is None or labels is None:
                raise ValueError("ablation needs --config and --labels (or --describe)")
            report = run_ablation(load_config(config), load_jsonl(labels, GoldFact))
        else:
            if describe:
                typer.echo(describe_formats(stage))
                return
            if labels is None:
                raise ValueError("--labels is required (or --describe to see the format)")
            if config is not None:
                raise ValueError("--config is for ablation")
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


if __name__ == "__main__":  # pragma: no cover
    app()
