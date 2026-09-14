"""`odke ontology infer` and `odke ontology freeze` (#35): the bootstrap from the shell."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke.cli.main import app, ontology_infer
from openodke.infer import DEFAULT_MAX_PREDICATES, DEFAULT_MAX_TYPES, DEFAULT_SAMPLE_WORDS
from openodke.infer.build import infer_ontology
from openodke.infer.review import evidence_path, format_for, render
from openodke.llm import ReplayClient, register, unregister
from openodke.loaders import DirectoryLoader
from openodke.loaders.directory import default_loaders
from openodke.ontology import Ontology

runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures" / "llm"
PEOPLE = (
    "name,born,employer,city\n"
    "Ada Lovelace,1815-12-10,Acme Corp,London\n"
    "Grace Hopper,1906-12-09,Globex,Paris\n"
    "Alan Turing,1912-06-23,Acme Corp,London\n"
    "Linus Torvalds,1969-12-28,Globex,Paris\n"
)
NOTES = (
    "# Notes\n\n"
    "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex.\n\n"
    "A mathematician is a scientist. They used languages such as Python and Rust.\n"
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "people.csv").write_text(PEOPLE, encoding="utf-8")
    (root / "companies.csv").write_text("name\nAcme Corp\nGlobex\n", encoding="utf-8")
    (root / "notes.md").write_text(NOTES, encoding="utf-8")
    return root


def _infer(corpus: Path, out: Path, *extra: str) -> str:
    result = runner.invoke(app, ["ontology", "infer", str(corpus), "--out", str(out), *extra])
    assert result.exit_code == 0, result.output
    return result.stdout


def test_no_llm_writes_a_reviewable_draft_that_loads_back(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "draft.yaml"
    stdout = _infer(corpus, out, "--no-llm")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# INFERRED ONTOLOGY — review it before anything is extracted")
    assert "# model: none — deterministic proposers only" in text
    assert '#   "Acme Corp" — people.csv line ' in text
    loaded = Ontology.from_yaml(out)
    assert loaded == infer_ontology(DirectoryLoader().load(corpus), llm=False).ontology
    assert loaded.inferred
    evidence = json.loads(evidence_path(out).read_text(encoding="utf-8"))
    assert evidence["evidence"]["predicates.employer"]["support"] >= 4
    assert "ontology" not in evidence and evidence["sample"]["seed"] == 0
    assert "employer: Person -> Company, single  aka works_at  [support" in stdout
    assert f"then `odke ontology freeze {out}`" in stdout


def test_the_knobs_reach_the_library(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "small.json"
    _infer(corpus, out, "--no-llm", "--max-types", "1", "--seed", "3", "--sample-words", "500")
    loaded = Ontology.from_json(out)
    assert len(loaded.types) == 1 and loaded.inferred
    evidence = json.loads(evidence_path(out).read_text(encoding="utf-8"))
    assert evidence["settings"] == {
        "model": None,
        "seed": 3,
        "sample_words": 500,
        "max_types": 1,
        "max_predicates": DEFAULT_MAX_PREDICATES,
        "min_support": 1,
    }


def test_cli_defaults_are_the_library_defaults() -> None:
    params = inspect.signature(ontology_infer).parameters
    assert params["sample_words"].default.default == DEFAULT_SAMPLE_WORDS
    assert params["max_types"].default.default == DEFAULT_MAX_TYPES
    assert params["max_predicates"].default.default == DEFAULT_MAX_PREDICATES


def test_a_json_draft_has_no_comments_and_keeps_its_evidence_beside_it(
    corpus: Path, tmp_path: Path
) -> None:
    out = tmp_path / "draft.json"
    _infer(corpus, out, "--no-llm")
    assert json.loads(out.read_text(encoding="utf-8"))["inferred"] is True
    assert evidence_path(out).name == "draft.evidence.json"


def test_validate_edit_freeze_and_diff_from_the_shell(corpus: Path, tmp_path: Path) -> None:
    """#35's done-when: the whole bootstrap without writing Python."""
    draft = tmp_path / "draft.yaml"
    _infer(corpus, draft, "--no-llm")

    checked = runner.invoke(app, ["ontology", "validate", str(draft)])
    assert checked.exit_code == 0
    assert "warning: inferred:" in checked.stdout and "[unreviewed]" in checked.stdout

    edited = draft.read_text(encoding="utf-8").replace(
        "  Person:", "  Person:\n    description: Staff.", 1
    )
    draft.write_text(edited, encoding="utf-8")
    frozen_file = tmp_path / "frozen.yaml"
    frozen = runner.invoke(
        app, ["ontology", "freeze", str(draft), "--by", "Deepak Kandpal", "--out", str(frozen_file)]
    )
    assert frozen.exit_code == 0, frozen.output
    assert frozen.stdout.startswith(f"frozen: {frozen_file} (by Deepak Kandpal at ")
    text = frozen_file.read_text(encoding="utf-8")
    assert text.startswith("# Reviewed and frozen by Deepak Kandpal at ")
    assert "support:" not in text
    loaded = Ontology.from_yaml(frozen_file)
    assert (loaded.inferred, loaded.frozen_by) == (False, "Deepak Kandpal")
    assert loaded.types["Person"].description == "Staff."
    assert runner.invoke(app, ["ontology", "validate", str(frozen_file)]).stdout.strip() == (
        "0 errors, 0 warnings"
    )

    diff = runner.invoke(app, ["ontology", "diff", str(draft), str(frozen_file)])
    assert "0 breaking" in diff.stdout and "compatible changed inferred" in diff.stdout


def test_freeze_in_place_defaults_the_reviewer(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOGNAME", "reviewer")
    monkeypatch.setenv("USER", "reviewer")
    draft = tmp_path / "draft.json"
    _infer(corpus, draft, "--no-llm")
    assert runner.invoke(app, ["ontology", "freeze", str(draft)]).exit_code == 0
    assert Ontology.from_json(draft).frozen_by == "reviewer"


def test_freeze_refuses_a_file_with_errors_and_leaves_it_alone(
    corpus: Path, tmp_path: Path
) -> None:
    draft = tmp_path / "draft.yaml"
    _infer(corpus, draft, "--no-llm")
    broken = draft.read_text(encoding="utf-8").replace("range: Company", "range: Compnay")
    draft.write_text(broken, encoding="utf-8")
    result = runner.invoke(app, ["ontology", "freeze", str(draft), "--by", "Deepak"])
    assert result.exit_code == 1
    assert "[unknown-range]" in result.stdout
    assert result.stdout.splitlines()[-1] == "not frozen: 1 error"
    assert draft.read_text(encoding="utf-8") == broken


@pytest.fixture
def replay() -> Iterator[None]:
    register("replay", lambda spec: ReplayClient(FIXTURES / "infer_people.json"))
    yield
    unregister("replay")


def test_the_model_path_from_the_shell_on_a_recorded_response(tmp_path: Path, replay: None) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "people.csv").write_text(
        "name,employer\nAda Lovelace,Acme Corp\nGrace Hopper,Globex\n", encoding="utf-8"
    )
    (root / "companies.csv").write_text("name\nAcme Corp\nGlobex\n", encoding="utf-8")
    (root / "notes.md").write_text(
        "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex.\n\n"
        "Acme Corp is headquartered in London.\n",
        encoding="utf-8",
    )
    out = tmp_path / "draft.yaml"
    stdout = _infer(root, out, "--model", "replay/sonnet")
    assert "# model: replay/sonnet" in out.read_text(encoding="utf-8")
    assert "Place" in Ontology.from_yaml(out).types
    assert "rejected from the model (3)" in stdout
    assert "model calls: 1, 1250 tokens, $0.0075" in stdout


def test_bad_input_exits_2_with_a_message(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app, ["ontology", "infer", str(empty), "--out", str(tmp_path / "o.yaml")]
    )
    assert result.exit_code == 2 and "no documents" in result.output
    # The hint names every suffix the loader reads, HTML, PDF and Word included.
    for suffix in default_loaders():
        assert suffix in result.output
    result = runner.invoke(
        app, ["ontology", "infer", str(empty), "--out", str(tmp_path / "o.txt"), "--no-llm"]
    )
    assert result.exit_code == 2 and ".yaml, .yml or .json" in result.output
    missing = runner.invoke(
        app, ["ontology", "infer", str(tmp_path / "nowhere"), "--out", str(tmp_path / "o.yaml")]
    )
    assert missing.exit_code == 2 and "no such file" in missing.output


def test_render_round_trips_and_names_the_format() -> None:
    ontology = Ontology.from_dict({"name": "x", "types": {"Person": {}}, "inferred": True})
    assert format_for(Path("a.YML")) == "yaml"
    assert Ontology.from_yaml(render(ontology, fmt="yaml", header=["hi"])) == ontology
    assert Ontology.from_json(render(ontology, fmt="json")) == ontology
    with pytest.raises(ValueError):
        format_for(Path("a.toml"))
