"""`odke ground` (#99): a graph openodke did not build, grounded and summarised.

Every model call is answered from recorded responses named in a config's
`models.replay`, the path a user takes to try it with no key. The Neo4j
sources are the recording driver from `test_neo4j_sink`, answered with what
each store really returns.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import GroundingVerdict
from openodke.cli.main import app
from openodke.corroborate import CHECK
from openodke.eval.spans import load_facts
from openodke.interop import TOO_NARROW, UNSUPPORTED
from openodke.sinks import neo4j as neo4j_module
from test_interop_graphrag import _store
from test_interop_neo4j import TEXT, _driver, _facts, _foreign, _halden, _record

runner = CliRunner()
REPO = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "interop"
# Berlin is never in the text; every other claim the recordings support.
RECORDED = [
    {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
    {"match": "Claim:", "response": {"verdict": "supported"}},
]
COMPANIES = {
    "name": "companies",
    "types": {"Company": {}, "City": {}, "Person": {}},
    "predicates": {
        "office_in": {"domain": ["Company"], "range": "City"},
        "founded_in": {"domain": ["Company"], "range": "City"},
        "founded": {"domain": ["Person"], "range": "Company"},
        "foundingYear": {"domain": ["Company"], "range": "integer"},
    },
}


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding examples/triples, recorded answers and a models file."""
    shutil.copytree(REPO / "examples" / "triples", tmp_path / "triples", ignore=_no_out)
    (tmp_path / "recorded.json").write_text(json.dumps(RECORDED), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(
        "models:\n  replay:\n    ground: recorded.json\n", encoding="utf-8"
    )
    (tmp_path / "companies.json").write_text(json.dumps(COMPANIES), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _no_out(directory: str, names: list[str]) -> list[str]:
    return [name for name in names if name == "out"]


def _ground(*args: str) -> Any:
    return runner.invoke(app, ["ground", *args])


def _summary(out: Path) -> dict[str, Any]:
    summary: dict[str, Any] = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return summary


# --------------------------------------------------------------------------- #
# The example: five triples someone else wrote
# --------------------------------------------------------------------------- #

EXAMPLE = [
    "--facts",
    "triples/triples.jsonl",
    "--texts",
    "triples/texts",
    "--ontology",
    "triples/ontology.json",
    "-o",
    "out",
]


def test_the_example_grounds_on_recorded_answers_and_writes_the_facts_back(here: Path) -> None:
    # A run config serves as the models file: its `models` block is all that is read.
    result = _ground(*EXAMPLE, "--config", "triples/odke.yaml")
    assert result.exit_code == 0, result.output

    facts = load_facts(here / "out")
    assert len(facts) == 5
    verdicts = {
        f.object_entity.label if f.object_entity else f.object_value: f.verdict for f in facts
    }
    assert verdicts == {
        "Mara Quist": GroundingVerdict.SUPPORTED,
        "Lyon": GroundingVerdict.SUPPORTED,
        "Tomas Ferrand": GroundingVerdict.SUPPORTED,
        "Berlin": GroundingVerdict.NOT_FOUND,
        2012: GroundingVerdict.NOT_FOUND,
    }
    summary = _summary(here / "out")
    assert summary["verdicts"] == {
        "supported": 3,
        "contradicted": 0,
        "not_found": 2,
        "unchecked": 0,
    }
    assert summary["refused"]["not_in_text"] == 1
    assert (summary["calls"], summary["no_span"], summary["prompts"]) == (4, 3, ["ground.span@1"])
    assert summary["unsupported"]["count"] == 1 and summary["too_narrow"]["count"] == 0
    assert "Berlin" in summary["unsupported"]["examples"][0]
    assert summary["spans"]["stage"] == "spans"
    assert "odke ground\n" in result.output
    assert "wrote out/facts.jsonl, out/summary.json" in result.output

    # The facts it wrote are a run's facts, as `odke eval spans` reads them.
    spans = runner.invoke(app, ["eval", "spans", "--facts", "out"])
    assert spans.exit_code == 0 and "3 of 5 fact(s) cite no span of their own" in spans.output


def test_a_dry_run_calls_no_model_and_needs_no_key(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run built a model client")

    monkeypatch.setattr("openodke.llm.registry.resolve", refuse)
    monkeypatch.setattr("openodke.ground.llm.LLMGrounder.__init__", refuse)
    result = _ground(*EXAMPLE, "--dry-run", "--locate", "--model", "openai/gpt-5.5")
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output and "not asked: a dry run calls nothing" in result.output
    summary = _summary(here / "out")
    assert (summary["calls"], summary["tokens"], summary["cost_usd"]) == (0, 0, None)
    assert summary["verdicts"] == {
        "supported": 0,
        "contradicted": 0,
        "not_found": 1,
        "unchecked": 4,
    }
    # The locator still ran: Mara Quist's row cites, and nothing else names both ends.
    assert summary["located"] == 0 and summary["no_span"] == 3


def test_without_a_key_the_model_run_fails_with_exit_1_before_any_request(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = _ground(*EXAMPLE, "--model", "openai/gpt-5.5")
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_the_summary_names_both_failure_shapes(here: Path) -> None:
    text = (here / "triples" / "texts" / "halden.txt").read_text(encoding="utf-8")
    lyon = text.index("Lyon")
    rows = [
        # Cites the city alone: the claim cannot be read from it.
        {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in",
         "object": "Lyon", "start": lyon, "end": lyon + 4, "quote": "Lyon"},
        # Cites nothing, and the text never says it.
        {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in",
         "object": "Berlin"},
    ]  # fmt: skip
    (here / "narrow.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (here / "recorded.json").write_text(
        json.dumps([{"match": "Claim:", "response": {"verdict": "not_found"}}])
    )
    result = _ground(
        "--facts",
        "narrow.jsonl",
        "--texts",
        "triples/texts",
        "--config",
        "models.yaml",
        "-o",
        "out",
    )
    assert result.exit_code == 0, result.output
    assert f"{UNSUPPORTED}: 1" in result.output
    assert f"{TOO_NARROW}: 1" in result.output
    summary = _summary(here / "out")
    assert "Berlin" in summary["unsupported"]["examples"][0]
    assert "Lyon" in summary["too_narrow"]["examples"][0]

    # The paper's grounder reads the whole document, never the citation.
    paper = _ground(
        "--facts", "narrow.jsonl", "--texts", "triples/texts", "--config", "models.yaml",
        "--paper", "-o", "paper",
    )  # fmt: skip
    assert paper.exit_code == 0, paper.output
    assert f"{TOO_NARROW}: 0" in paper.output


def test_with_an_ontology_what_it_has_no_room_for_is_refused_for_free(here: Path) -> None:
    args = ["--adapter", "langchain", "--facts", str(FIXTURES / "langchain.graph_documents.jsonl")]
    result = _ground(*args, "--ontology", "companies.json", "--config", "models.yaml", "-o", "out")
    assert result.exit_code == 0, result.output
    # STUDIED_IN is not in the ontology: refused for free, stamped with why, never asked.
    assert "1 refused: 1 predicate not in the ontology" in result.output
    (studied,) = [f for f in load_facts(here / "out") if f.predicate == "STUDIED_IN"]
    assert studied.verdict is GroundingVerdict.UNCHECKED
    assert studied.qualifiers[CHECK]["check"] == "predicate"
    assert _summary(here / "out")["calls"] == 5


# --------------------------------------------------------------------------- #
# Every adapter, on its fixture
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("adapter", "fixture", "facts"),
    [
        ("langchain", "langchain.graph_documents.jsonl", 6),
        ("langextract", "langextract.annotated.jsonl", 8),
        ("graphrag", "graphrag.graph.json", 5),
    ],
)
def test_each_adapter_reads_its_fixture(here: Path, adapter: str, fixture: str, facts: int) -> None:
    result = _ground(
        "--adapter", adapter, "--facts", str(FIXTURES / fixture), "--config", "models.yaml",
        "-o", "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    grounded = load_facts(here / "out")
    assert len(grounded) == facts
    berlin = [f for f in grounded if f.object_entity and f.object_entity.label == "Berlin"]
    assert all(f.verdict is GroundingVerdict.NOT_FOUND for f in berlin)
    assert all(f.verdict is not GroundingVerdict.UNCHECKED for f in grounded)


def test_a_graphrag_store_is_read_through_the_driver(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = _store()
    connected: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        neo4j_module, "_connect", lambda uri, auth: connected.append((uri, auth)) or driver
    )
    monkeypatch.setenv("ODKE_TEST_SECRET", "not-a-real-password")
    result = _ground(
        "--adapter", "graphrag", "--facts", "bolt://example.invalid:7687",
        "--password-env", "ODKE_TEST_SECRET", "--config", "models.yaml", "-o", "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert connected == [("bolt://example.invalid:7687", ("neo4j", "not-a-real-password"))]
    assert "warning: 1 relationships or entities have no chunk" in result.output
    assert {mode for mode, _, _ in driver.calls} == {"read"} and driver.closed
    assert len(load_facts(here / "out")) == 4


def _neo4j(monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]], unread: int = 0) -> Any:
    driver = _driver(records, unread)
    monkeypatch.setattr(neo4j_module, "_connect", lambda uri, auth: driver)
    monkeypatch.setenv("NEO4J_PASSWORD", "not-a-real-password")
    return driver


def test_a_neo4j_graph_openodke_wrote_grounds_against_its_texts(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (here / "notes").mkdir()
    (here / "notes" / "halden.txt").write_text(TEXT, encoding="utf-8")
    driver = _neo4j(monkeypatch, [_record(f, f"r{i}") for i, f in enumerate(_facts(_halden()))])
    args = ["--adapter", "neo4j", "--facts", "bolt://example.invalid:7687", "--texts", "notes"]
    result = _ground(*args, "--config", "models.yaml", "-o", "out")
    assert result.exit_code == 0, result.output
    assert driver.writes == []  # read, never written, unless asked
    grounded = load_facts(here / "out")
    assert {f.id for f in grounded} == {"r0", "r1", "r2", "r3"}

    asked = _ground(*args, "--config", "models.yaml", "--write-verdicts", "-o", "out")
    assert asked.exit_code == 0, asked.output
    ((cypher, params),) = driver.writes
    assert "elementId(r) = row.id" in cypher
    written = {row["id"]: row["props"]["odke_verdict"] for row in params["rows"]}
    assert written == {"r0": "supported", "r1": "not_found", "r2": "supported", "r3": "supported"}
    assert "wrote odke_verdict on 2 relationships" in asked.output  # what the driver answered


def test_a_neo4j_graph_from_elsewhere_grounds_against_a_text_property(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentence = "It opened a second office in Lyon in 2019."
    _neo4j(monkeypatch, [_foreign("r1", sentence, "Lyon"), _foreign("r2", sentence, "Berlin")])
    result = _ground(
        "--adapter", "neo4j", "--facts", "neo4j://example.invalid", "--text-property",
        "sentence", "--config", "models.yaml", "-o", "out",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    verdicts = {f.id: f.verdict for f in load_facts(here / "out")}
    assert verdicts == {"r1": GroundingVerdict.SUPPORTED, "r2": GroundingVerdict.NOT_FOUND}


# --------------------------------------------------------------------------- #
# Usage errors exit 2, and say what to do
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--facts", "triples/triples.jsonl"], "give them as --texts"),
        (["--adapter", "dgraph", "--facts", "x.json"], "unknown adapter 'dgraph'"),
        (["--adapter", "neo4j", "--facts", "graph.json"], "give its URI as --facts"),
        (
            ["--adapter", "neo4j", "--facts", "bolt://example.invalid"],
            "neo4j needs --texts",
        ),
        (
            ["--adapter", "langchain", "--facts", "x.json", "--texts", "triples/texts"],
            "carries its own texts",
        ),
        (
            ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "--write-verdicts"],
            "--write-verdicts writes to the Neo4j graph",
        ),
        (
            [
                "--adapter",
                "neo4j",
                "--facts",
                "bolt://example.invalid",
                "--text-property",
                "s",
                "--write-verdicts",
                "--dry-run",
            ],
            "no verdicts to write back",
        ),  # fmt: skip
        (
            ["--adapter", "neo4j", "--facts", "bolt://x", "--text-property", "s"],
            "ODKE_UNSET_PASSWORD is not set",
        ),
        (["--facts", "missing.jsonl", "--texts", "triples/texts"], "missing.jsonl"),
        (
            ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "--config", "x.yaml"],
            "x.yaml: no such file",
        ),
    ],
)
def test_input_that_cannot_be_read_exits_2(here: Path, args: list[str], message: str) -> None:
    result = runner.invoke(
        app, ["ground", *args, "--password-env", "ODKE_UNSET_PASSWORD", "-o", "out"]
    )
    assert result.exit_code == 2, result.output
    assert message in result.output
    assert not (here / "out").exists()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("stages: {}\n", "no `models` block to read"),
        ("models: {}\nontolgy: o.json\n", "ontolgy: unknown key — did you mean 'ontology'?"),
        ("models: {grond: x/y}\n", "models.grond: unknown key — did you mean 'ground'?"),
    ],
)
def test_a_models_file_is_checked_like_a_run_config(here: Path, config: str, message: str) -> None:
    (here / "bad.yaml").write_text(config, encoding="utf-8")
    args = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "--config", "bad.yaml"]
    result = _ground(*args, "-o", "out")
    assert result.exit_code == 2, result.output
    assert message in result.output

