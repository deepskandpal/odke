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
from odke.ontology import Ontology

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


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """odke — ontology-guided knowledge extraction."""


@ontology_app.command("snippet")
def ontology_snippet(
    path: Path = typer.Argument(..., help="Ontology JSON file."),
    entity_type: str = typer.Argument(..., help="Entity type to render a snippet for."),
    limit: int = typer.Option(25, help="Maximum predicates in the snippet."),
    as_json_schema: bool = typer.Option(False, "--json-schema", help="Emit JSON Schema instead."),
) -> None:
    """Print the exact schema fragment the extractor would be prompted with.

    Being able to see this without spending a token is most of what makes an
    ontology debuggable: a bad extraction is usually a bad snippet.
    """
    ontology = Ontology.model_validate_json(path.read_text(encoding="utf-8"))
    snippet = ontology.snippet(entity_type, limit=limit)
    typer.echo(json.dumps(snippet.json_schema(), indent=2) if as_json_schema else snippet.render())


@ontology_app.command("types")
def ontology_types(
    path: Path = typer.Argument(..., help="Ontology JSON file."),
) -> None:
    """List entity types and how many predicates each one can carry."""
    ontology = Ontology.model_validate_json(path.read_text(encoding="utf-8"))
    for name in sorted(ontology.types):
        typer.echo(f"{name}\t{len(ontology.predicates_for(name))} predicates")


if __name__ == "__main__":  # pragma: no cover
    app()
