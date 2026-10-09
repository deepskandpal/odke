"""The prompt registry: every prompt sent is `id@version`, hashed, and never edited in place."""

from __future__ import annotations

import dataclasses

import pytest

from openodke import prompts
from openodke.extract import llm as extract_llm
from openodke.ground import llm as ground_llm
from openodke.infer import llm as infer_llm
from openodke.prompts import get, registered

# The SHA-256 of each prompt as it stood as a module constant, before the
# registry existed (release/1.0.0 at 9cea049). The move had to change no byte.
BEFORE_THE_REGISTRY = {
    "ground.span@1": "dbc23eec8779777579cedc48e047c29073a180de167e9a4df68913b277c700f9",
    "ground.paper@1": "1fc40142c704528b97f44a05ad8331382a5770a4a5b8ed208ff81fbb3a01693a",
    "extract@1": "976a43141ed35c33b86d1cfd0355f936e28c25574189dddffb1a871deff36002",
    "extract.repair@1": "b39233e97ad2ecda0e4885dba15121592709319b33947f3837bf57a227462ebd",
    "infer@1": "2165e6dacc504fb2557ad73a38b9800fdcacd2c77288729deca2b0f71869991c",
    "infer.repair@1": "f6ff413a8915b7a6e802afd7ebda695db62562c1cffbd3b5e555035044ebdf0a",
}


def test_the_move_into_the_registry_changed_no_byte() -> None:
    for key, digest in BEFORE_THE_REGISTRY.items():
        assert get(key).sha256 == digest, key


def test_the_old_constants_are_the_registered_texts() -> None:
    assert get("ground.span", 1).text == ground_llm.SYSTEM_PROMPT
    assert get("ground.paper", 1).text == ground_llm.PAPER_PROMPT
    assert get("extract", 1).text == extract_llm._INSTRUCTIONS
    assert get("extract.repair", 1).text == extract_llm._REPAIR
    assert get("infer", 1).text == infer_llm._INSTRUCTIONS
    assert get("infer.repair", 1).text == infer_llm._REPAIR


def test_every_prompt_says_where_it_came_from() -> None:
    assert {p.key: p.source for p in registered()} == {
        "extract.repair@1": "openodke",
        "extract@1": "openodke",
        "ground.paper@1": "ODKE+ App. B, verbatim",
        "ground.span@1": "openodke",
        "infer.repair@1": "openodke",
        "infer@1": "openodke",
    }


def test_a_prompt_is_frozen_and_named_by_its_key_and_hash() -> None:
    prompt = get("ground.span")
    assert prompt.key == "ground.span@1"
    assert len(prompt.sha256) == 64
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.text = "something else"  # type: ignore[misc]


def test_get_gives_the_latest_without_a_version_and_reads_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_REGISTRY", {})
    first = prompts._register("demo", 1, "Say yes.", source="test")
    second = prompts._register("demo", 2, "Say yes or no.", source="test")
    assert get("demo") is second
    assert get("demo", 1) is first
    assert get("demo@1") is first
    # A new version is a new entry: the old one stays, for whoever named it.
    assert [p.key for p in registered()] == ["demo@1", "demo@2"]


def test_an_unknown_prompt_or_version_names_what_there_is() -> None:
    with pytest.raises(LookupError, match="registered: extract, extract.repair"):
        get("summarise")
    with pytest.raises(LookupError, match="ground.span has version 1"):
        get("ground.span", 9)


def test_a_text_is_never_registered_twice_under_one_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts, "_REGISTRY", {})
    prompts._register("demo", 1, "Say yes.", source="test")
    with pytest.raises(ValueError, match="a new text is a new version"):
        prompts._register("demo", 1, "Say no.", source="test")
    with pytest.raises(ValueError, match="starts at 1"):
        prompts._register("demo", 0, "Say no.", source="test")
