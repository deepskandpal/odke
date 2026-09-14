"""`odke run`: a config file to a written graph, on recorded model responses.

Every model call here is answered from a file the test writes — a cassette for
extraction, match entries for grounding — through the config's own
`models.replay`, which is the path a user takes to try a config without a key.
"""

from __future__ import annotations

import json
import sys
import types
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from openodke import Chunk, Ontology, RouteVerdict, Sink
from openodke.cli.main import app
from openodke.llm import ModelRoles, ModelSpec
from openodke.loaders import DocxLoader, HtmlLoader, PdfLoader
from openodke.run import ConfigError, StageSpec, build, execute, load_config, parse_config
from openodke.run.build import NodeLinkFile, build_stage
from openodke.sinks import neo4j as neo4j_module
from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink
from openodke.sinks.networkx import NetworkXSink
from openodke.sinks.rdf import RdfSink
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
        "from openodke import RouteVerdict\n\n"
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
        (
            _stages(router="openodke.stages:PassThroughGrounder"),
            "is not a Router; it needs a method",
        ),
        (_stages(router="nowhere.at_all:Router"), "cannot load 'nowhere.at_all:Router'"),
        (_stages(sink={"use": "jsonl"}), "stages.sink.directory: the directory to write into"),
        (
            _stages(sink={"use": "neo4j", "uri": "bolt://x", "password": "hunter2"}),
            "never goes in a config file",
        ),
        (_stages(sink="parquet"), "unknown sink 'parquet'"),
        (_stages(sink="rfd"), "unknown sink 'rfd' — did you mean 'rdf'?"),
        (_stages(sink={"use": "cypher_file"}), "stages.sink.path: the file to write into"),
        (
            _stages(sink={"use": "neo4j_admin_csv", "path": "out"}),
            "stages.sink.directory: the directory to write into",
        ),
        (
            _stages(sink={"use": "rdf", "path": "graph.ttl", "formt": "nt"}),
            "it takes path, format, base, schema",
        ),
        (_stages(sink={"use": "rdf", "path": "graph.ttl", "format": "rdfxml"}), "format must be"),
        (
            _stages(sink={"use": "networkx", "pth": "graph.json"}),
            "stages.sink.pth: unknown option; networkx takes path",
        ),
        (_stages(sink={"use": "rdf", "path": "g.ttl", "ontology": "x"}), "set by odke run"),
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


# --------------------------------------------------------------------------- #
# Short names for the loaders and sinks that landed after the builder
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("use", "options", "cls", "attribute", "value"),
    [
        ("html", {"strip_boilerplate": True}, HtmlLoader, "strip_boilerplate", True),
        ("pdf", {"per_page": True, "tier": "curated"}, PdfLoader, "per_page", True),
        ("docx", {"modality": "structured"}, DocxLoader, "modality", "structured"),
    ],
)
def test_each_document_loader_has_a_short_name_and_takes_its_options(
    project: Path, use: str, options: dict[str, Any], cls: type, attribute: str, value: Any
) -> None:
    built = build(parse_config(_config(), base_dir=project))
    spec = StageSpec.model_validate({"use": use, **options})
    loader = build_stage("loader", spec, built.context, "inputs[0].loader")
    assert isinstance(loader, cls)
    assert getattr(loader, attribute) == value


def test_an_html_input_read_by_its_short_name_is_extracted_and_named_by_its_path(
    project: Path,
) -> None:
    (project / "page.html").write_text(
        "<title>Ada</title><table><tr><th>name</th><td>Ada Lovelace</td></tr>"
        "<tr><th>born</th><td>1815-12-10</td></tr></table>",
        encoding="utf-8",
    )
    config = _config(
        inputs=[{"path": "page.html", "loader": {"use": "html", "tier": "authoritative"}}],
        models={},
        stages={"extractor": "pattern"},
    )
    result = execute(parse_config(config, base_dir=project), dry_run=True)
    facts = result.graph.facts
    assert {(f.predicate, f.object_value) for f in facts} == {
        ("full_name", "Ada Lovelace"),
        ("birth_date", "1815-12-10"),
    }
    assert {(e.doc_id, e.tier.value) for f in facts for e in f.evidence} == {
        ("page.html", "authoritative")
    }


def _inner(sink: Any) -> Any:
    return sink.sink if isinstance(sink, NodeLinkFile) else sink


@pytest.mark.parametrize(
    ("sink", "cls", "check"),
    [
        (
            {"use": "cypher_file", "path": "out/graph.cypher", "batch_size": 50},
            CypherFileSink,
            lambda s: s.batch_size == 50 and s.path.name == "graph.cypher",
        ),
        (
            {"use": "neo4j_admin_csv", "directory": "out/import", "delimiter": "|"},
            Neo4jAdminCsvSink,
            lambda s: s.delimiter == "|" and s.directory.name == "import",
        ),
        (
            {"use": "rdf", "path": "out/graph.nt"},
            RdfSink,
            lambda s: s.format == "nt",
        ),
        (
            {"use": "rdf", "path": "out/graph.ttl", "format": "json-ld", "base": "urn:x:"},
            RdfSink,
            lambda s: (s.format, s.base) == ("json-ld", "urn:x:"),
        ),
        ({"use": "networkx"}, NetworkXSink, lambda s: s.graph.number_of_nodes() == 0),
        (
            {"use": "networkx", "path": "out/graph.json"},
            NodeLinkFile,
            lambda s: s.path.name == "graph.json",
        ),
    ],
)
def test_each_v02_sink_has_a_short_name_and_a_dry_run_writes_none_of_them(
    project: Path, sink: dict[str, Any], cls: type, check: Callable[[Any], bool]
) -> None:
    config = _config()
    config["stages"]["sink"] = sink
    built = build(parse_config(config, base_dir=project))
    (plan,) = built.sinks
    opened = plan.open()
    assert isinstance(opened, cls) and check(opened)
    assert isinstance(opened, Sink)
    # Paths resolve against the config, and odke run supplies the ontology.
    assert _inner(opened).ontology == built.ontology
    target = getattr(opened, "path", None) or getattr(opened, "directory", None)
    assert target is None or target.is_relative_to(project)

    result = execute(parse_config(config, base_dir=project), dry_run=True)
    assert result.written[0].startswith(f"{sink['use']} → ")
    assert not (project / "out").exists()


@pytest.mark.parametrize(
    ("module", "change", "message"),
    [
        (
            "pypdf",
            {"inputs": [{"path": "corpus", "loader": {"use": "pdf"}}]},
            "inputs[0].loader: PdfLoader needs pypdf, which is not installed. "
            'Run: pip install "openodke[pdf]"',
        ),
        (
            "docx",
            {"inputs": [{"path": "corpus", "loader": {"use": "docx"}}]},
            "inputs[0].loader: DocxLoader needs python-docx, which is not installed. "
            'Run: pip install "openodke[docx]"',
        ),
        (
            "pyarrow.parquet",
            {"inputs": ["corpus"], "stages__loader": "parquet"},
            "stages.loader: ParquetLoader needs pyarrow, which is not installed. "
            'Run: pip install "openodke[parquet]"',
        ),
        (
            "rdflib",
            {"stages__sink": {"use": "rdf", "path": "out/graph.ttl"}},
            "stages.sink: RdfSink needs rdflib, which is not installed. "
            'Run: pip install "openodke[rdf]"',
        ),
        (
            "networkx",
            {"stages__sink": {"use": "networkx", "path": "out/graph.json"}},
            "stages.sink: NetworkXSink needs networkx, which is not installed. "
            'Run: pip install "openodke[networkx]"',
        ),
    ],
)
def test_a_short_name_whose_extra_is_missing_is_a_config_error_naming_the_extra(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    change: dict[str, Any],
    message: str,
) -> None:
    # A None entry makes the lazy import fail exactly as an uninstalled extra does.
    monkeypatch.setitem(sys.modules, module, None)
    result = runner.invoke(app, ["run", str(_write(project, _config(**change)))])
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert "Traceback" not in result.output
    assert not (project / "out").exists()


def _example_into(example: Path, sink: dict[str, Any]) -> Path:
    """The end-to-end example's own config, with its sink replaced."""
    config = yaml.safe_load((example / "odke.yaml").read_text(encoding="utf-8"))
    config["stages"]["sink"] = sink
    path = example / "odke.sink.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


GMBH, LTD = "Company:halden robotics gmbh", "Company:halden robotics ltd"


def test_the_example_runs_end_to_end_into_rdf(example: Path) -> None:
    rdflib = pytest.importorskip("rdflib")
    from rdflib.namespace import OWL, RDF

    from openodke.sinks.rdf import VOCAB

    config = _example_into(example, {"use": "rdf", "path": "out/graph.ttl"})
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 0, result.output
    assert "wrote         rdf → " in result.output
    assert "graph         25 facts" in result.output

    graph = rdflib.Graph().parse(example / "out" / "graph.ttl", format="turtle")
    sink = RdfSink(example / "out" / "graph.ttl")
    odke = rdflib.Namespace(VOCAB)
    ont = rdflib.Namespace(f"{sink.base}schema/")
    gmbh, ltd = rdflib.URIRef(sink.entity_iri(GMBH)), rdflib.URIRef(sink.entity_iri(LTD))

    # One reified statement per fact the gate let through, each with its evidence.
    statements = set(graph.subjects(RDF.type, odke.Fact))
    assert len(statements) == 25
    assert all(graph.value(s, odke.evidence) is not None for s in statements)
    # The head office the model invented for the GmbH was refused, so no triple says it.
    assert (gmbh, ont.headquarters, rdflib.Literal("Leeds")) not in graph
    # Two companies with one name are two resources, and the link says why.
    assert (gmbh, odke.different_from, ltd) in graph or (ltd, odke.different_from, gmbh) in graph
    (link,) = graph.subjects(odke.kind, rdflib.Literal("different"))
    assert "external_id mismatch" in str(graph.value(link, odke.reason))
    # odke run supplied the ontology, so the file declares its schema.
    assert (ont.headquarters, RDF.type, OWL.FunctionalProperty) in graph


def test_the_example_runs_end_to_end_into_networkx(example: Path) -> None:
    nx = pytest.importorskip("networkx")
    config = _example_into(example, {"use": "networkx", "path": "out/graph.json"})
    result = runner.invoke(app, ["run", str(config)])
    assert result.exit_code == 0, result.output
    assert "wrote         networkx → " in result.output

    data = json.loads((example / "out" / "graph.json").read_text(encoding="utf-8"))
    graph = nx.node_link_graph(data, edges="edges")
    assert graph.is_directed() and graph.is_multigraph()
    nodes = Counter(attrs["kind"] for _, attrs in graph.nodes(data=True))
    assert nodes["entity"] == 7
    facts = [attrs for _, _, attrs in graph.edges(data=True) if attrs["kind"] == "fact"]
    assert len(facts) == 25
    assert all(isinstance(f["extracted_at"], str) for f in facts)  # dates as ISO text
    links = {
        (u, v, key)
        for u, v, key, attrs in graph.edges(keys=True, data=True)
        if attrs["kind"] == "link"
    }
    assert (GMBH, LTD, "DIFFERENT") in links or (LTD, GMBH, "DIFFERENT") in links
