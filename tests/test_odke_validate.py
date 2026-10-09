"""`odke validate` (#129): the Validator from the command line, on recorded answers.

The facts come as `odke ground` takes them, or from a run config. A Neo4j sink
writes through the recording driver from `test_neo4j_sink`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from openodke.cli.main import app
from openodke.sinks import neo4j as neo4j_module
from test_neo4j_sink import FakeDriver

runner = CliRunner()
REPO = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "interop"
RECORDED = [
    {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
    {"match": "Claim:", "response": {"verdict": "supported"}},
]


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding examples/triples, recorded answers and a models file."""
    shutil.copytree(
        REPO / "examples" / "triples",
        tmp_path / "triples",
        ignore=shutil.ignore_patterns("out"),
    )
    (tmp_path / "recorded.json").write_text(json.dumps(RECORDED), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(
        "models:\n  replay:\n    ground: recorded.json\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _validate(*args: str) -> Any:
    return runner.invoke(app, ["validate", *args])


def _manifest(directory: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return manifest


def test_the_triples_example_config_runs_end_to_end_into_jsonl(here: Path) -> None:
    result = _validate("--config", "triples/odke.yaml")
    assert result.exit_code == 0, result.output
    for line in (
        "odke validate",
        "in            5 facts from 1 text",
        "grounded      supported 3, contradicted 0, not_found 2, unchecked 0",
        "free checks   1 refused: 1 not in the text",
        "cost          4 model calls",
    ):
        assert line in result.output
    # The config's own sink, where the config puts it.
    assert "wrote         jsonl → " in result.output
    manifest = _manifest(here / "triples" / "out")
    assert manifest["facts"] == 5
    report = manifest["stats"]["validation"]
    assert (report["facts_in"], report["refused"], report["merged"]) == (5, 0, 0)
    assert report["prompts"] == ["ground.span@1"]
    assert manifest["stats"]["stages"]["extractor"]["rows"] == 5


def test_any_adapters_facts_validate_into_the_directory_given(here: Path) -> None:
    result = _validate(
        "--adapter", "langchain", "--facts", str(FIXTURES / "langchain.graph_documents.jsonl"),
        "--config", "models.yaml", "-o", "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "wrote         jsonl → out: entities.jsonl" in result.output
    report = _manifest(here / "out")["stats"]["validation"]
    assert report["facts_in"] == 6 and report["verdicts"]["not_found"] == 1
    assert report["calls"] == 6


def test_a_dry_run_calls_no_model_and_writes_nothing(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run built a model client or opened a store")

    monkeypatch.setattr("openodke.llm.registry.resolve", refuse)
    monkeypatch.setattr(neo4j_module, "_connect", refuse)
    args = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "-o", "out"]
    result = _validate(*args, "--dry-run", "--locate")
    assert result.exit_code == 0, result.output
    assert "dry run, no model called" in result.output
    assert "would write   jsonl → out" in result.output
    assert not (here / "out").exists()

    # A config's dry run neither connects to its store nor writes its JSONL.
    config = yaml.safe_load((here / "triples" / "odke.yaml").read_text(encoding="utf-8"))
    config["stages"]["sink"] = [config["stages"]["sink"], _neo4j_sink()]
    (here / "triples" / "neo4j.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = _validate("--config", "triples/neo4j.yaml", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would write   neo4j → bolt://example.invalid:7687:" in result.output
    assert not (here / "triples" / "out").exists()


def _neo4j_sink() -> dict[str, Any]:
    return {"use": "neo4j", "uri": "bolt://example.invalid:7687", "password_env": "ODKE_SECRET"}


def test_a_configs_neo4j_sink_writes_through_its_driver(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(neo4j_module, "_connect", lambda uri, auth: driver)
    monkeypatch.setenv("ODKE_SECRET", "not-a-real-password")
    config = yaml.safe_load((here / "triples" / "odke.yaml").read_text(encoding="utf-8"))
    config["stages"]["sink"] = _neo4j_sink()
    config["bootstrap"] = True
    (here / "triples" / "neo4j.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = _validate("--config", "triples/neo4j.yaml")
    assert result.exit_code == 0, result.output
    # The constraints first, then the graph.
    modes = [mode for mode, _, _ in driver.calls]
    first_write = modes.index("write")
    assert modes[:first_write] and set(modes[:first_write]) == {"auto"}
    assert "bootstrap     applied" in result.output
    assert any("MERGE (n:`Company` {key: row.key})" in cypher for cypher, _ in driver.writes)
    assert driver.closed
    assert "wrote         neo4j → bolt://example.invalid:7687:" in result.output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "give the facts with --facts, or a run config with --config"),
        (
            ["--config", "triples/odke.yaml", "--texts", "triples/texts"],
            "a run config names its own inputs",
        ),
        (["--config", "triples/odke.yaml", "--locate"], "the locator is the grounder's"),
        (
            ["--config", "triples/llm.yaml"],
            "odke validate checks the triples another extractor wrote",
        ),
        (["--facts", "triples/triples.jsonl"], "give them as --texts"),
    ],
)
def test_input_that_cannot_be_read_exits_2(here: Path, args: list[str], message: str) -> None:
    config = yaml.safe_load((here / "triples" / "odke.yaml").read_text(encoding="utf-8"))
    config["stages"]["extractor"] = "llm"
    (here / "triples" / "llm.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = _validate(*args)
    assert result.exit_code == 2, result.output
    assert message in result.output


def test_the_documented_example_prints_what_the_page_shows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/validator.md's `odke validate` example, run from a clone, as the page shows it."""
    page = (REPO / "docs" / "validator.md").read_text(encoding="utf-8")
    command = page.split("```bash\n", 1)[1].split("```", 1)[0]
    shown = page.split("```text\n", 1)[1].split("```", 1)[0]
    args = command.replace("\\\n", " ").split()
    assert args[:2] == ["odke", "validate"]
    shutil.copytree(
        REPO / "examples" / "triples",
        tmp_path / "examples" / "triples",
        ignore=shutil.ignore_patterns("out"),
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, args[1:])
    assert result.exit_code == 0, result.output
    assert [line for line in shown.splitlines() if line] == result.output.splitlines()
