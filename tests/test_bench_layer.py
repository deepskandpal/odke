"""`bench/layer.py`, the table of every extractor with and without the layer (#104).

No model and no key: a synthetic prepared set, recorded answers, and a
response cache in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke.llm import RecordedClient
from openodke.llm.base import Completion, Message, ModelSpec

BENCH = Path(__file__).parent.parent / "bench"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layer = _load("layer")

PAPER = {"use": "llm", "context": "document", "verdicts": "binary"}
ANSWERS = [
    {"match": "place of birth, Warsaw>", "response": {"verdict": True}},
    {"match": "place of birth, Paris>", "response": {"verdict": False}},
    {"match": "<Ada Lovelace", "response": {"verdict": True}},
]


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in data))
    else:
        path.write_text(json.dumps(data))


def _prepared(tmp_path: Path) -> Path:
    """A Re-DocRED-shaped set as `odke bench prepare` and the runs on it leave one."""
    folder = tmp_path / "redocred"
    (folder / "docs").mkdir(parents=True)
    (folder / "docs" / "d1.txt").write_text("Marie Curie was born in Warsaw. She died in Paris.")
    (folder / "docs" / "d2.txt").write_text("Name: Ada Lovelace\nPlace of birth: London\n")
    ontology = {
        "types": {"Person": {"keys": ["name"]}, "City": {}},
        "predicates": {
            "name": {"domain": ["Person"], "range": "string"},
            "place_of_birth": {"domain": ["Person"], "range": "City"},
        },
    }
    _write(folder / "ontology.json", ontology)
    labels = {"place_of_birth": "place of birth", "name": "name"}
    _write(
        folder / "dataset.json",
        {"dataset": "redocred", "split": "test", "documents": 2, "relation_labels": labels},
    )
    _write(
        folder / "gold.jsonl",
        [
            {
                "id": "d1",
                "text": "Marie Curie was born in Warsaw. She died in Paris.",
                "entities": [["Marie Curie"], ["Warsaw"], ["Paris"]],
                "facts": [[0, "place of birth", 1]],
            },
            {
                "id": "d2",
                "text": "Name: Ada Lovelace Place of birth: London",
                "entities": [["Ada Lovelace"], ["London"]],
                "facts": [[0, "place of birth", 1]],
            },
        ],
    )
    config = {
        "ontology": "ontology.json",
        "inputs": [{"path": "docs", "loader": "directory"}],
        "stages": {
            "extractor": {"use": "llm"},
            "grounder": PAPER,
            "corroborator": "signature",
            "scorer": "evidence",
            "validator": {"use": "verdict", "refuse_not_found": True},
        },
        "models": {
            "extract": "anthropic/claude-sonnet-5-5",
            "ground": "anthropic/claude-haiku-4-5",
        },
    }
    _write(folder / "odke.json", config)
    first = {"model_calls": 2, "prompt_tokens": 1000, "completion_tokens": 100, "cost_usd": 0.004}
    _write(folder / "report.json", {"breakdown": {"extraction alone": first}})
    triples = [
        ["Marie Curie", "place of birth", "Warsaw"],
        ["Marie Curie", "place of birth", "Paris"],
    ]
    _write(folder / "predictions" / "extraction-alone.jsonl", [{"id": "d1", "triples": triples}])
    lgt = {
        "doc": "d1",
        "subject": "Marie Curie",
        "subject_type": "Person",
        "predicate": "PLACE_OF_BIRTH",
        "object": "Paris",
        "object_type": "City",
    }
    _write(folder / "competitors" / "lgt" / "facts.jsonl", [lgt])
    usage = {"input_tokens": 3000, "output_tokens": 300, "calls": 2, "usd": 0.009}
    _write(folder / "competitors" / "lgt" / "usage.json", usage)
    return folder


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_prepare_reads_each_extractor_back_and_validates_it_in_the_sets_own_mode(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    made = layer.prepare([_prepared(tmp_path)], out)
    assert sorted(p.name for p in made) == ["lgt", "openodke", "pattern"]
    # Predictions keep labels and no types: the predicate and its first domain type come back.
    openodke = _rows(out / "redocred" / "openodke" / "triples.jsonl")
    assert {(r["predicate"], r["subject_type"], r["object"]) for r in openodke} == {
        ("place_of_birth", "Person", "Warsaw"),
        ("place_of_birth", "Person", "Paris"),
    }
    # The pattern extractor runs here, and cites what it read.
    pattern = _rows(out / "redocred" / "pattern" / "triples.jsonl")
    assert {(r["predicate"], r["object"], r["quote"]) for r in pattern} == {
        ("name", "Ada Lovelace", "Ada Lovelace"),
        ("place_of_birth", "London", "London"),
    }
    config = json.loads((out / "redocred" / "lgt" / "odke.json").read_text())
    assert config["stages"]["grounder"] == PAPER
    assert config["stages"]["gate"] == {"use": "verdict", "refuse_not_found": True, "schema": True}
    assert "validator" not in config["stages"]
    assert config["stages"]["normalizer"] == "passthrough"
    assert config["models"]["ground"] == layer.GROUND_MODEL
    cost = json.loads((out / "redocred" / "openodke" / "extraction.json").read_text())
    assert (cost["calls"], cost["usd"], cost["documents"]) == (2, 0.004, 2)
    assert json.loads((out / "redocred" / "lgt" / "extraction.json").read_text())["usd"] == 0.009


def test_the_estimate_prices_every_question_and_asks_none(tmp_path: Path) -> None:
    out = tmp_path / "out"
    layer.prepare([_prepared(tmp_path)], out)
    totals = layer.ground(out, cache=tmp_path / "cache", estimate=True)
    # openodke 2, LLMGraphTransformer 1, the pattern extractor 2; nothing in the cache.
    assert (totals["asked"], totals["cached"]) == (5, 0)
    assert totals["input_tokens"] > 0
    assert not list((tmp_path / "cache").glob("*/*.json"))


def test_three_rows_and_what_grounding_removed(tmp_path: Path) -> None:
    out, cache = tmp_path / "out", tmp_path / "cache"
    layer.prepare([_prepared(tmp_path)], out)
    client = RecordedClient(ANSWERS)
    totals = layer.ground(out, cache=cache, client=client)
    # openodke asks LLMGraphTransformer's question about Paris again: the cache answers it.
    assert totals["calls"] == len(client.calls) == 4
    cost = json.loads((out / "redocred" / "openodke" / "ground" / "cost.json").read_text())
    assert (cost["now"]["calls"], cost["from_cache"]["calls"], cost["list"]["calls"]) == (1, 1, 2)
    assert cost["prompts"] == ["ground.paper@1"]

    # A rerun is the cache's: no call, and the list price counts its answers.
    again = RecordedClient([])
    assert layer.ground(out, cache=cache, client=again)["calls"] == 0
    assert again.calls == []
    cost = json.loads((out / "redocred" / "lgt" / "ground" / "cost.json").read_text())
    assert (cost["now"]["calls"], cost["from_cache"]["calls"], cost["list"]["calls"]) == (0, 1, 1)

    result = layer.table(out, cache, resamples=50)
    rows = result["redocred"]["openodke"]["rows"]
    assert [rows[r]["precision"] for r in layer.ROWS] == [0.5, 1.0, 1.0]
    assert [rows[r]["recall"] for r in layer.ROWS] == [0.5, 0.5, 0.5]
    assert rows["+ grounding"]["precision_change_value"] == 0.5
    low, high = rows["+ grounding"]["precision_range"]
    assert low <= 1.0 <= high
    found = result["redocred"]["openodke"]
    assert found["removed"] == {"wrong": 1}
    assert (found["raw_right"], found["raw_wrong"]) == (1, 1)
    gone = _rows(out / "redocred" / "openodke" / "removed.jsonl")
    assert gone == [
        {"doc": "d1", "triple": ["Marie Curie", "place of birth", "Paris"], "is": "wrong"}
    ]
    # LLMGraphTransformer's one triple was wrong, and the gate removed it.
    assert result["redocred"]["lgt"]["facts"] == dict(zip(layer.ROWS, (1, 0, 0), strict=True))
    # The pattern extractor's two facts were both affirmed; one is in the gold.
    assert result["redocred"]["pattern"]["rows"]["+ grounding"]["recall"] == 0.5
    assert json.loads((out / "layer.json").read_text())["redocred"]["lgt"]["removed"] == {
        "wrong": 1
    }
    markdown = layer.markdown(result)
    assert "| openodke | raw | 2 | 2 | 50.0 [" in markdown
    assert "| Re-DocRED | openodke | 1 | 1 | 1 | 1 | 0 | 0 | 100% | 0% |" in markdown


def test_the_table_asks_no_model_for_a_question_the_cache_lacks(tmp_path: Path) -> None:
    out, cache = tmp_path / "out", tmp_path / "cache"
    layer.prepare([_prepared(tmp_path)], out)
    layer.ground(out, cache=cache, client=RecordedClient(ANSWERS))
    shutil.rmtree(cache)
    with pytest.raises(SystemExit, match="run `ground` first"):
        layer.table(out, cache, resamples=10)


class _Priced:
    """A dollar a call: the budget stops the job after the first."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
    ) -> Completion:
        self.calls += 1
        return Completion(
            text='{"verdict": true}',
            parsed={"verdict": True},
            prompt_tokens=10,
            completion_tokens=1,
            cost_usd=1.0,
        )


def test_the_budget_stops_the_job_and_the_table_refuses_what_it_left(tmp_path: Path) -> None:
    out, cache = tmp_path / "out", tmp_path / "cache"
    layer.prepare([_prepared(tmp_path)], out)
    client = _Priced()
    totals = layer.ground(out, cache=cache, budget_usd=0.5, client=client)
    # LLMGraphTransformer's one question spends the budget; openodke's first is refused.
    assert totals["stopped"] == ["redocred/openodke"]
    assert client.calls == 1
    with pytest.raises(SystemExit, match="stopped by the budget"):
        layer.table(out, cache, resamples=10)
