"""The competitor harness against a fake LiteLLM: no network, no key.

    bench/.venv/bin/python -m pytest bench/test_competitors.py

Not in CI: it needs the competitor libraries, which the package never installs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import competitors
import pytest

LGT_REPLY = json.dumps(
    [
        {
            "head": "Marie Curie",
            "head_type": "Person",
            "relation": "BORN_IN",
            "tail": "Warsaw",
            "tail_type": "City",
        }
    ]
)
NEO4J_REPLY = json.dumps(
    {
        "nodes": [
            {"id": "0", "label": "Person", "properties": {"name": "Marie Curie"}},
            {"id": "1", "label": "City", "properties": {"name": "Warsaw"}},
        ],
        "relationships": [
            {"type": "BORN_IN", "start_node_id": "0", "end_node_id": "1", "properties": {}}
        ],
    }
)


def fake_litellm(monkeypatch: pytest.MonkeyPatch, reply: str) -> list[dict[str, Any]]:
    """Patch LiteLLM to answer `reply`, recording every call it gets."""
    import litellm

    calls: list[dict[str, Any]] = []

    def response() -> Any:
        message = SimpleNamespace(content=reply)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    async def acompletion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response()

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response()

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    monkeypatch.setattr(litellm, "completion", completion)
    for key in competitors.USAGE:
        monkeypatch.setitem(competitors.USAGE, key, 0)
    return calls


def prepared(tmp_path: Path, extract: Any = None) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "d1.txt").write_text("Marie Curie was born in Warsaw.")
    (tmp_path / "docs" / "d2.txt").write_text("Nothing here.")
    ontology = {
        "types": {"Person": {}, "City": {}},
        "predicates": {"born_in": {"domain": ["Person"], "range": "City"}},
    }
    (tmp_path / "ontology.json").write_text(json.dumps(ontology))
    (tmp_path / "dataset.json").write_text("{}")
    (tmp_path / "gold.jsonl").write_text('{"id": "d1"}\n{"id": "d2"}\n')
    config: dict[str, Any] = {"stages": {"extractor": {"use": "llm"}}}
    if extract is not None:
        config["models"] = {"extract": extract}
    (tmp_path / "odke.json").write_text(json.dumps(config))
    return tmp_path


def test_the_model_is_the_sets_extractor(tmp_path: Path) -> None:
    spec = {"model": "openai/gpt-5", "max_tokens": 4000}
    assert competitors.extract_model(prepared(tmp_path, spec), None) == ("openai/gpt-5", 4000)
    assert competitors.extract_model(tmp_path, "ollama/llama3.1") == ("ollama/llama3.1", 4000)


def test_a_bare_model_string_gets_the_default_room(tmp_path: Path) -> None:
    model = competitors.extract_model(prepared(tmp_path, "gemini/gemini-2.5-pro"), None)
    assert model == ("gemini/gemini-2.5-pro", competitors.DEFAULT_MAX_TOKENS)


def test_no_model_anywhere_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        competitors.extract_model(prepared(tmp_path), None)


def test_langchain_prompt_becomes_chat_messages() -> None:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("system", "rules"), ("human", "{x}")])
    messages = competitors.as_messages(prompt.invoke({"x": "text"}))
    assert messages == [{"role": "system", "content": "rules"}, {"role": "user", "content": "text"}]


def test_llmgraphtransformer_runs_on_any_litellm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_litellm(monkeypatch, LGT_REPLY)
    rows = asyncio.run(
        competitors.run_lgt(
            [("d1", "Marie Curie was born in Warsaw.")],
            ["City", "Person"],
            [("Person", "BORN_IN", "City")],
            "openai/gpt-5",
            4000,
        )
    )
    assert [(r["subject"], r["predicate"], r["object"]) for r in rows] == [
        ("Marie Curie", "BORN_IN", "Warsaw")
    ]
    assert calls[0]["model"] == "openai/gpt-5"
    assert calls[0]["max_tokens"] == 4000
    assert competitors.USAGE["input_tokens"] == 100


def test_neo4j_graphrag_runs_on_any_litellm_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_litellm(monkeypatch, NEO4J_REPLY)
    rows = asyncio.run(
        competitors.run_neo4j(
            [("d1", "Marie Curie was born in Warsaw.")],
            ["City", "Person"],
            [("Person", "BORN_IN", "City")],
            "ollama/llama3.1",
            2000,
        )
    )
    assert [(r["subject"], r["predicate"], r["object"]) for r in rows] == [
        ("Marie Curie", "BORN_IN", "Warsaw")
    ]
    assert calls[0]["model"] == "ollama/llama3.1"
    assert competitors.USAGE["output_tokens"] == 20


def test_a_limited_run_scores_only_its_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_litellm(monkeypatch, LGT_REPLY)
    out = competitors.extract("lgt", prepared(tmp_path, "openai/gpt-5"), limit=1)
    assert [p.name for p in (out / "docs").iterdir()] == ["d1.txt"]
    assert (out / "gold.jsonl").read_text() == '{"id": "d1"}\n'
    usage = json.loads((out / "usage.json").read_text())
    assert usage["model"] == "openai/gpt-5"
    assert usage["documents"] == 1
    config = json.loads((out / "odke.json").read_text())
    assert config["stages"]["extractor"] == {
        "use": "triples",
        "path": str((out / "facts.jsonl").resolve()),
        "extractor": "lgt",
    }
