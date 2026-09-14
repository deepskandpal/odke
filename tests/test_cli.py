"""The CLI is a thin shell over the library; these guard that it stays wired."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from openodke import __version__
from openodke.cli.main import app

runner = CliRunner()

EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"

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


def _write(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    path = tmp_path / name
    text = json.dumps(data) if name.endswith(".json") else yaml.safe_dump(data)
    path.write_text(text, encoding="utf-8")
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


def test_snippet_reads_yaml_too(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["ontology", "snippet", str(_write(tmp_path, "o.yml", ONTOLOGY)), "Person"]
    )
    assert result.exit_code == 0
    assert "birth_date" in result.stdout


def test_types_lists_predicate_counts(tmp_path) -> None:
    result = runner.invoke(app, ["ontology", "types", str(_ontology_file(tmp_path))])
    assert result.exit_code == 0
    assert "Person\t2 predicates" in result.stdout


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #

BROKEN = {
    "types": {"Person": {}},
    "predicates": {
        "employer": {"domain": ["Person"], "range": "Compnay"},
        "orphan": {"domain": ["Robot"]},
    },
}


def test_validate_passes_a_clean_ontology() -> None:
    result = runner.invoke(app, ["ontology", "validate", str(EXAMPLE)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0 errors, 0 warnings"


def test_validate_prints_every_diagnostic_and_exits_1_on_errors(tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.json", BROKEN)
    result = runner.invoke(app, ["ontology", "validate", str(path)])
    assert result.exit_code == 1
    lines = result.stdout.splitlines()
    assert lines[0].startswith(f"{path}: error: predicates.employer.range: 'Compnay'")
    assert lines[0].endswith("[unknown-range]")
    assert lines[1].endswith("[unreachable-predicate]")
    assert lines[-1] == "2 errors, 0 warnings"


def test_validate_passes_on_warnings_alone_so_it_can_gate_ci(tmp_path: Path) -> None:
    path = _write(tmp_path, "cycle.yaml", {"types": {"A": {"parents": ["A"]}}})
    result = runner.invoke(app, ["ontology", "validate", str(path)])
    assert result.exit_code == 0
    assert "warning: types.A.parents" in result.stdout
    assert result.stdout.splitlines()[-1] == "0 errors, 1 warning"


def test_validate_takes_several_files_and_explains_a_malformed_one(tmp_path: Path) -> None:
    """How pre-commit calls it; a schema that does not parse is an error, not a traceback."""
    good = _write(tmp_path, "good.json", ONTOLOGY)
    bad = tmp_path / "bad.json"
    bad.write_text('{"types": {"Person": {"parnets": ["Agent"]}}}', encoding="utf-8")
    result = runner.invoke(app, ["ontology", "validate", str(good), str(bad)])
    assert result.exit_code == 1
    assert f"error: {bad}: types.Person.parnets: unknown key — did you mean 'parents'?" in (
        result.stdout
    )
    assert "Traceback" not in result.output
    assert result.stdout.splitlines()[-1] == "1 error, 0 warnings"


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #


def test_diff_lists_breaking_changes_first_and_can_fail_on_them(tmp_path: Path) -> None:
    old = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    new = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    new["version"] = "2"
    new["types"]["Robot"] = {"name": "Robot"}
    new["predicates"]["employer"]["cardinality"] = "single"
    del new["predicates"]["headquarters"]
    before, after = _write(tmp_path, "old.json", old), _write(tmp_path, "new.yaml", new)

    result = runner.invoke(app, ["ontology", "diff", str(before), str(after)])
    assert result.exit_code == 0
    lines = result.stdout.splitlines()
    assert [line.split()[:3] for line in lines[:4]] == [
        ["breaking", "changed", "predicates.employer.cardinality:"],
        ["breaking", "removed", "predicates.headquarters:"],
        ["compatible", "changed", "version:"],
        ["compatible", "added", "types.Robot"],
    ]
    assert lines[-1] == "4 changes, 2 breaking"
    failing = ["ontology", "diff", str(before), str(after), "--fail-on-breaking"]
    assert runner.invoke(app, failing).exit_code == 1


def test_diff_of_a_file_with_itself_is_empty_and_passes() -> None:
    result = runner.invoke(
        app, ["ontology", "diff", str(EXAMPLE), str(EXAMPLE), "--fail-on-breaking"]
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "0 changes, 0 breaking"
