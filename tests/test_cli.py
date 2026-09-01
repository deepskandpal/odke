"""The CLI is a thin shell over the library; these guard that it stays wired."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from odke import __version__
from odke.cli.main import app

runner = CliRunner()

ONTOLOGY = {
    "name": "demo",
    "types": {"Person": {"name": "Person", "description": "A human being."}},
    "predicates": {
        "birth_date": {"name": "birth_date", "domain": ["Person"], "range": "date"},
        "name": {"name": "name", "domain": ["Person"], "range": "string", "importance": 1.0},
    },
}


def _ontology_file(tmp_path):
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(ONTOLOGY), encoding="utf-8")
    return path


def test_version_matches_the_package() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_snippet_renders_the_prompt_fragment(tmp_path) -> None:
    result = runner.invoke(app, ["ontology", "snippet", str(_ontology_file(tmp_path)), "Person"])
    assert result.exit_code == 0
    assert "Entity type: Person" in result.stdout
    assert "birth_date" in result.stdout


def test_snippet_can_emit_json_schema(tmp_path) -> None:
    result = runner.invoke(
        app, ["ontology", "snippet", str(_ontology_file(tmp_path)), "Person", "--json-schema"]
    )
    assert result.exit_code == 0
    assert set(json.loads(result.stdout)["properties"]) == {"birth_date", "name"}


def test_types_lists_predicate_counts(tmp_path) -> None:
    result = runner.invoke(app, ["ontology", "types", str(_ontology_file(tmp_path))])
    assert result.exit_code == 0
    assert "Person\t2 predicates" in result.stdout
