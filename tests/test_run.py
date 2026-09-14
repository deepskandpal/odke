"""`odke run`: a config file to a written graph, on recorded model responses.

Every model call here is answered from a file the test writes — a cassette for
extraction, match entries for grounding — through the config's own
`models.replay`, which is the path a user takes to try a config without a key.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from odke import Chunk, Ontology, RouteVerdict
from odke.cli.main import app
from odke.llm import ModelRoles, ModelSpec
from odke.run import ConfigError, build, execute, load_config, parse_config
from odke.sinks import neo4j as neo4j_module
from test_neo4j_sink import FakeDriver

runner = CliRunner()
SUPPORT = "odke_run_support"

NOTES = "# Grace Hopper\n\nGrace Hopper was born in 1906. She worked for the US Navy.\n"
PEOPLE = "name,born\nAda Lovelace,1815-12-10\n"


def _reply(entities: list[dict[str, Any]]) -> str:
    return json.dumps({"entities": entities})


EXTRACT = {
    "description": "One prose note: three true facts, one wrong birth year, one invented quote.",
    "interactions": [
        {
            "match": {"contains": ["She worked for the US Navy."]},
            "response": {
                "text": _reply(
                    [
                        {
                            "type": "Person",
                            "name": "Grace Hopper",
                            "facts": [
                                {
                                    "predicate": "full_name",
                                    "value": "Grace Hopper",
                                    "quote": "Grace Hopper",
                                },
                                {
                                    "predicate": "birth_date",
                                    "value": "1906",
                                    "quote": "born in 1906",
                                },
                                {
                                    "predicate": "employer",
                                    "value": "US Navy",
                                    "quote": "She worked for the US Navy",
                                },
                                {
                                    "predicate": "birth_date",
                                    "value": "1907",
                                    "quote": "born in 1906",
                                },
                                {
                                    "predicate": "birth_date",
                                    "value": "1908",
                                    "quote": "born in 1908",
                                },
                            ],
                        }
                    ]
                ),
                "prompt_tokens": 500,
                "completion_tokens": 120,
                "cost_usd": 0.002,
            },
        }
    ],
}


def _verdict(claim: str, verdict: str) -> dict[str, Any]:
    return {"match": f"Claim: {claim}", "response": {"verdict": verdict}}


GROUND = [
    _verdict('Grace Hopper (Person) — full name — "Grace Hopper".', "supported"),
    _verdict('Grace Hopper (Person) — birth date — "1906".', "supported"),
    _verdict('Grace Hopper (Person) — birth date — "1907".', "contradicted"),
    _verdict("Grace Hopper (Person) — employer — US Navy (Company).", "supported"),
    _verdict('Ada Lovelace (Person) — full name — "Ada Lovelace".', "supported"),
    _verdict('Ada Lovelace (Person) — birth date — "1815-12-10".', "not_found"),
]


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "ontology": "ontology.json",
        "inputs": [{"path": "corpus", "loader": {"use": "directory", "tier": "authoritative"}}],
        "models": {
            "extract": "anthropic/claude-sonnet-5",
            "replay": {"extract": "extract.json", "ground": "ground.json"},
            "meter": True,
        },
        "stages": {
            "chunker": {"use": "sentence", "max_words": 120},
            "extractor": "hybrid",
            "grounder": {"use": "llm", "max_workers": 4},
            "normalizer": {"use": "value", "person_types": ["Person"]},
            "corroborator": "signature",
            "scorer": "evidence",
            "validator": "verdict",
            "sink": {"use": "jsonl", "directory": "out"},
        },
    }
    for key, value in overrides.items():
        section, _, name = key.partition("__")
        if name:
            config[section][name] = value
        else:
            config[section] = value
    return config


@pytest.fixture
def project(tmp_path: Path, people: Ontology) -> Path:
    """A corpus, an ontology and recorded responses, laid out as a user's project would be."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text(NOTES, encoding="utf-8")
    (corpus / "people.csv").write_text(PEOPLE, encoding="utf-8")
    (tmp_path / "ontology.json").write_text(people.model_dump_json(), encoding="utf-8")
    (tmp_path / "extract.json").write_text(json.dumps(EXTRACT), encoding="utf-8")
    (tmp_path / "ground.json").write_text(json.dumps(GROUND), encoding="utf-8")
    return tmp_path


def _write(project: Path, config: dict[str, Any], name: str = "odke.json") -> Path:
    path = project / name
    text = yaml.safe_dump(config) if name.endswith((".yaml", ".yml")) else json.dumps(config)
    path.write_text(text, encoding="utf-8")
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def test_a_config_runs_end_to_end_into_jsonl_with_every_stage_counted(project: Path) -> None:
    result = runner.invoke(app, ["run", str(_write(project, _config()))])
    assert result.exit_code == 0, result.output

    facts = _lines(project / "out" / "facts.jsonl")
    # Six candidates: four from the note, two from the CSV row. The 1908 quote is
    # not in the passage and never became a fact; the 1907 one was contradicted
    # by its own span and refused at the gate.
    assert len(facts) == 5
    assert {f["verdict"] for f in facts} == {"supported", "not_found"}
    assert "1907" not in {f["object_value"] for f in facts}

    stats = json.loads((project / "out" / "manifest.json").read_text())["stats"]
    assert (stats["documents"], stats["chunks"], stats["refused"]) == (2, 2, 1)
    stages = stats["stages"]
    assert stages["extractor"]["rejections"] == {"quote not in the passage": 1}
    paths = stages["extractor"]["paths"]
    assert (paths["pattern_facts"], paths["llm_facts"], paths["model_calls"]) == (2, 4, 1)
    grounder = stages["grounder"]
    assert (grounder["supported"], grounder["contradicted"], grounder["not_found"]) == (4, 1, 1)
    assert (grounder["calls"], grounder["failed"]) == (6, 0)
    assert stages["validator"] == {"accepted": 5, "refused": {"contradicted": 1}}
    # 1906 and 1907 contested a single-valued birth date and tied; the loser's
    # twin was refused, so one stamped fact reached the graph.
    assert stages["corroborator"]["conflicts"] == {"tied": 1}
    assert stats["graph"] == {"facts": 5, "edges": 1, "properties": 4, "entities": 3, "links": {}}
    cost = stats["cost"]["metrics"]
    assert cost["calls"] == 7
    # The grounding replies carry no price, so the total is unknown, not a partial sum.
    assert cost["cost_usd"] is None and cost["priced_usd"] == pytest.approx(0.002)

    assert "wrote         jsonl →" in result.output
    assert "refused       1" in result.output


def test_document_ids_are_source_paths_so_labels_and_queries_can_name_them(project: Path) -> None:
    runner.invoke(app, ["run", str(_write(project, _config()))])
    cited = {e["doc_id"] for f in _lines(project / "out" / "facts.jsonl") for e in f["evidence"]}
    assert cited == {"corpus/notes.md", "corpus/people.csv#L2"}


def test_a_dry_run_extracts_and_grounds_but_writes_nothing(project: Path) -> None:
    result = runner.invoke(app, ["run", str(_write(project, _config())), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (project / "out").exists()
    assert result.output.startswith("odke run — dry run, nothing written")
    assert "would write   jsonl →" in result.output
    assert "facts.jsonl 5" in result.output
    assert "facts that would be written:" in result.output
    assert "Grace Hopper —employer→ US Navy  [supported" in result.output


def test_yaml_and_json_configs_are_the_same_run(project: Path) -> None:
    from_json = execute(load_config(_write(project, _config())), dry_run=True)
    from_yaml = execute(load_config(_write(project, _config(), "odke.yaml")), dry_run=True)
    assert from_json.stats["graph"] == from_yaml.stats["graph"]
    assert from_json.stats["stages"]["grounder"] == from_yaml.stats["stages"]["grounder"]


def test_paths_resolve_against_the_config_not_the_working_directory(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    result = runner.invoke(app, ["run", str(_write(project, _config()))])
    assert result.exit_code == 0, result.output
    assert (project / "out" / "manifest.json").exists()


def test_omitted_stages_are_the_pass_through(project: Path) -> None:
    """Only an extractor: candidates, unchecked, all written, no stage counts invented."""
    config = _config(stages={"extractor": "hybrid"}, models={"replay": {"extract": "extract.json"}})
    result = execute(parse_config(config, base_dir=project))
    assert result.written == []
    assert len(result.graph.facts) == 6
    assert {f.verdict.value for f in result.graph.facts} == {"unchecked"}
    assert set(result.stats["stages"]) == {"extractor"}
    assert "cost" not in result.stats


def test_models_accept_a_bare_model_string() -> None:
    config = parse_config(
        _config(models={"extract": "ollama/llama3.1", "ground": {"model": "ollama/qwen2.5:3b"}})
    )
    assert config.models.roles() == ModelRoles(
        extract=ModelSpec(model="ollama/llama3.1"), ground=ModelSpec(model="ollama/qwen2.5:3b")
    )


# --------------------------------------------------------------------------- #
# Neo4j, through a recording driver
# --------------------------------------------------------------------------- #


def _neo4j_sink(**options: Any) -> dict[str, Any]:
    return {
        "use": "neo4j",
        "uri": "bolt://example.invalid:7687",
        "password_env": "ODKE_TEST_SECRET",
        **options,
    }


def test_bootstrap_applies_the_constraints_before_the_first_write(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = FakeDriver()
    monkeypatch.setattr(neo4j_module, "_connect", lambda uri, auth: driver)
    monkeypatch.setenv("ODKE_TEST_SECRET", "not-a-real-password")
    config = _config(bootstrap=True)
    config["stages"].update(sink=_neo4j_sink(), constrainer="neo4j")
    result = runner.invoke(app, ["run", str(_write(project, config))])
    assert result.exit_code == 0, result.output
    assert "warning" not in result.output  # a Neo4j constrainer is the sink's other half

    modes = [mode for mode, _, _ in driver.calls]
    first_write = modes.index("write")
    assert modes[:first_write] and set(modes[:first_write]) == {"auto"}
    assert all("IF NOT EXISTS" in cypher for mode, cypher, _ in driver.calls if mode == "auto")
    assert "applied" in result.output and "CREATE CONSTRAINT odke_key_Person" in result.output
    assert driver.closed


def test_a_dry_run_prints_the_ddl_and_the_statements_without_connecting(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(uri: str, auth: Any) -> Any:
        raise AssertionError("a dry run must not connect")

    monkeypatch.setattr(neo4j_module, "_connect", refuse)
    monkeypatch.delenv("ODKE_TEST_SECRET", raising=False)
    config = _config(bootstrap=True)
    config["stages"]["sink"] = _neo4j_sink()
    result = runner.invoke(app, ["run", str(_write(project, config)), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "would apply" in result.output
    assert "CREATE CONSTRAINT odke_key_Person IF NOT EXISTS" in result.output
    assert "neo4j → bolt://example.invalid:7687:" in result.output
    assert "MERGE (n:`Person` {key: row.key})" in result.output


def test_a_missing_password_variable_is_named(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ODKE_TEST_SECRET", raising=False)
    config = _config()
    config["stages"]["sink"] = _neo4j_sink()
    result = runner.invoke(app, ["run", str(_write(project, config))])
    assert result.exit_code == 2
    assert "ODKE_TEST_SECRET is not set" in result.output


# --------------------------------------------------------------------------- #
# Your own stages
# --------------------------------------------------------------------------- #


class _SkipPrefix:
    """Skips any chunk that starts with `prefix`."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def route(self, chunk: Chunk) -> RouteVerdict:
        if chunk.text.startswith(self.prefix):
            return RouteVerdict(action="skip", label="heading")
        return RouteVerdict(action="extract")


@pytest.fixture
def support(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType(SUPPORT)
    module.SkipPrefix = _SkipPrefix  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, SUPPORT, module)
    return module


def test_a_stage_of_your_own_is_named_by_import_path_with_options(
    project: Path, support: types.ModuleType
) -> None:
    config = _config()
    config["stages"]["router"] = {"use": f"{SUPPORT}:SkipPrefix", "prefix": "# Grace"}
    result = execute(parse_config(config, base_dir=project), dry_run=True)
    assert result.stats["skipped"] == 1
    assert {f.subject.label for f in result.graph.facts} == {"Ada Lovelace"}


def test_pythonpath_puts_a_project_module_within_reach(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "project_stages", raising=False)
    (project / "stages").mkdir()
    (project / "stages" / "project_stages.py").write_text(
        "from odke import RouteVerdict\n\n"
        "class Everything:\n"
        "    def route(self, chunk):\n"
        "        return RouteVerdict(action='defer')\n",
        encoding="utf-8",
    )
    config = _config(pythonpath=["stages"])
    config["stages"]["router"] = "project_stages:Everything"
    try:
        result = execute(parse_config(config, base_dir=project), dry_run=True)
    finally:
        sys.modules.pop("project_stages", None)
    assert (result.stats["deferred"], len(result.graph.facts)) == (2, 0)


# --------------------------------------------------------------------------- #
# Configs that cannot run
# --------------------------------------------------------------------------- #


def _stages(**changes: Any) -> dict[str, Any]:
    config = _config()
    config["stages"].update(changes)
    return config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_stages(grounder="lm"), "stages.grounder: unknown grounder 'lm' — did you mean 'llm'?"),
        ({**_config(), "stagse": {}}, "stagse: unknown key — did you mean 'stages'?"),
        (
            {**_config(), "stages": {"grounder": "llm"}},
            "stages.extractor: Field required",
        ),
        (_stages(chunker={"use": "sentence", "max_word": 50}), "it takes max_words, overlap"),
        (_stages(chunker={"use": "sentence", "max_words": 0}), "max_words must be at least 1"),
        (_stages(normalizer={"use": "value", "ontology": "x"}), "normalizer.ontology: set by"),
        (_stages(inferrer="llm"), "odke run never infers an ontology"),
        (_stages(router="odke.stages:PassThroughGrounder"), "is not a Router; it needs a method"),
        (_stages(router="nowhere.at_all:Router"), "cannot load 'nowhere.at_all:Router'"),
        (_stages(sink={"use": "jsonl"}), "stages.sink.directory: the directory to write into"),
        (
            _stages(sink={"use": "neo4j", "uri": "bolt://x", "password": "hunter2"}),
            "never goes in a config file",
        ),
        (_stages(sink="parquet"), "unknown sink 'parquet'"),
        ({**_config(), "bootstrap": True}, "bootstrap: true needs a sink that applies"),
        (_config(inputs=["nowhere"]), "inputs[0].path:"),
        (
            _config(inputs=[{"path": "corpus", "loader": {"use": "csv", "tier": "gold"}}]),
            "is not one of curated",
        ),
        (_config(models={"replay": {"extract": "missing.json"}}), "models.replay.extract:"),
        (_config(ontology="missing.json"), "ontology:"),
    ],
)
def test_a_config_that_cannot_run_exits_2_and_names_the_key(
    project: Path, config: dict[str, Any], message: str
) -> None:
    result = runner.invoke(app, ["run", str(_write(project, config))])
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Traceback" not in result.output


def test_a_malformed_file_names_its_line(project: Path) -> None:
    path = project / "broken.json"
    path.write_text('{"ontology": "o.json",\n  "inputs": [}', encoding="utf-8")
    result = runner.invoke(app, ["run", str(path)])
    assert result.exit_code == 2
    assert "line 2 column" in result.output


def test_building_constructs_every_stage_and_calls_nothing(project: Path) -> None:
    """A config is checked whole before anything is loaded or any model is asked."""
    built = build(parse_config(_config(), base_dir=project))
    assert built.stages["router"] is None
    assert type(built.stages["grounder"]).__name__ == "LLMGrounder"
    assert [plan.name for plan in built.sinks] == ["jsonl"]
    with pytest.raises(ConfigError, match="unknown grounder"):
        build(parse_config(_stages(grounder="lm"), base_dir=project))


def test_the_commented_reference_config_is_a_valid_config() -> None:
    """examples/run.yaml documents every key; if it stops parsing, it documents nothing."""
    reference = Path(__file__).parent.parent / "examples" / "run.yaml"
    config = load_config(reference)
    assert config.stages.extractor.use == "hybrid"
    assert config.base_dir == reference.parent.resolve()
