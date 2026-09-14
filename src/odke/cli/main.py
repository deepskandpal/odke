"""The `odke` command.

Thin on purpose: every command is a few lines over the library, so that anything
reachable from the CLI is reachable from Python and neither grows a behaviour the
other lacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from odke import __version__
from odke.ontology import Ontology, OntologyLoadError

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
        typer.echo(f"odke {__version__}")
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
    """odke — ontology-guided knowledge extraction."""


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


if __name__ == "__main__":  # pragma: no cover
    app()
