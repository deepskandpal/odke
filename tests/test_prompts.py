"""The prompt registry: every prompt sent is `id@version`, hashed, and never edited in place."""

from __future__ import annotations

import dataclasses
import json
import re
import warnings
from collections.abc import Mapping
from pathlib import Path

import pytest

from openodke import prompts
from openodke.corroborate import judge
from openodke.extract import llm as extract_llm
from openodke.ground import llm as ground_llm
from openodke.infer import llm as infer_llm
from openodke.prompts import Prompt, check, get, read_lock, registered

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
    assert get("pair", 1) is judge.PROMPT and get("pair.user", 1) is judge.FRAME


# The entity-pair judge's lineage, which both its keys carry.
PAIR = (
    "openodke, after LLM entity matching (Peeters, Steiner & Bizer 2023, arXiv 2310.11244), "
    "asked in both orders for position bias (Zheng et al. 2023, arXiv 2306.05685 §3.4)"
)


def test_every_prompt_says_where_it_came_from() -> None:
    assert {p.key: p.source for p in registered()} == {
        "extract.repair@1": "openodke",
        "extract@1": "openodke",
        "ground.paper@1": "ODKE+ App. B, verbatim",
        "ground.span@1": "openodke",
        "infer.repair@1": "openodke",
        "infer@1": "openodke",
        "pair.user@1": PAIR,
        "pair@1": PAIR,
        "reextract@1": (
            "openodke, after GraphRAG's gleaning pass (Edge et al. 2024, arXiv 2404.16130), "
            "scoped to one window and grounded after"
        ),
    }


def test_the_reextract_rules_are_extract_s_unchanged() -> None:
    """The same reply shape and span checks as extraction, so the same parser reads it."""
    rules = get("extract", 1).text
    rules = rules[rules.index('Every fact needs "quote"') :]
    text = get("reextract", 1).text
    assert text.startswith("You find facts that an earlier extraction missed in one passage.")
    assert text.endswith("\n\n" + rules)


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


# --------------------------------------------------------------------------- #
# The lock: a text edited in place fails the build, by name
# --------------------------------------------------------------------------- #


def test_every_registered_prompt_matches_its_hash_in_the_lock() -> None:
    problems = check()
    assert not problems, "\n".join(problems)


def test_a_text_edited_in_place_is_caught_and_named() -> None:
    span = get("ground.span", 1)
    edited = dataclasses.replace(span, text=span.text + " Be brief.")
    others = [p for p in registered() if p.key != span.key]
    (problem,) = check([edited, *others])
    assert problem.startswith("bump the version: ground.span ")


def test_a_new_version_needs_its_own_entry_and_an_old_one_may_not_go() -> None:
    span = get("ground.span", 1)
    second = Prompt("ground.span", 2, span.text + " Be brief.", "openodke")
    (missing,) = check([*registered(), second])
    assert missing.startswith("ground.span@2 is not in the lock")
    assert f'"ground.span@2": "{second.sha256}"' in missing
    (removed,) = check([p for p in registered() if p.key != span.key])
    assert removed.startswith("ground.span@1 is in the lock but no longer registered")


def test_the_lock_sits_beside_the_registry_and_names_every_key() -> None:
    assert prompts.LOCK.parent == Path(prompts.__file__).parent
    assert set(read_lock()) == {p.key for p in registered()}


# --------------------------------------------------------------------------- #
# Leakage: no prompt may contain a passage from a benchmark gate split
# --------------------------------------------------------------------------- #

# Where the gate splits live: `bench/labels/**/*gate*.jsonl`, one passage per row
# in `text`. A prompt that quotes one has been tuned on the test.
BENCH_LABELS = Path(__file__).resolve().parents[1] / "bench" / "labels"
# Twelve words in a row is a copied passage, not a shared phrase.
WINDOW = 12


def _words(text: str) -> list[str]:
    # Words alone, case folded: re-punctuating a copied passage does not hide it.
    return re.findall(r"\w+", text.casefold())


def leaks(labels: Path, texts: Mapping[str, str], window: int = WINDOW) -> list[str]:
    """Every gate passage that shares `window` consecutive words with a prompt text.

    The prompts' windows go in a set, which is small, and each passage is read
    once against it, so a large split costs one pass.
    """
    shingles: dict[tuple[str, ...], str] = {}
    for key, text in texts.items():
        words = _words(text)
        for at in range(len(words) - window + 1):
            shingles.setdefault(tuple(words[at : at + window]), key)
    found: list[str] = []
    for path in sorted(labels.glob("**/*gate*.jsonl")):
        with path.open(encoding="utf-8") as rows:
            for line_number, line in enumerate(rows, 1):
                passage = json.loads(line).get("text") if line.strip() else None
                if not isinstance(passage, str):
                    continue
                words = _words(passage)
                for at in range(len(words) - window + 1):
                    key = shingles.get(tuple(words[at : at + window]))
                    if key is not None:
                        quoted = " ".join(words[at : at + window])
                        found.append(
                            f"{path.relative_to(labels)}:{line_number} is in {key}: {quoted!r}"
                        )
                        break
    return found


def test_no_prompt_contains_a_passage_from_a_benchmark_gate_split() -> None:
    texts = {p.key: p.text for p in registered()}
    if not any(BENCH_LABELS.glob("**/*gate*.jsonl")):
        warnings.warn(
            "no benchmark gate split under bench/labels yet, so no prompt was checked for leakage",
            stacklevel=1,
        )
        return
    found = leaks(BENCH_LABELS, texts)
    assert not found, "\n".join(found)


def test_twelve_copied_words_are_a_leak_and_eleven_are_not(tmp_path: Path) -> None:
    words = _words(get("extract@1").text)[20:32]
    # Re-cased and re-punctuated, which is still a copy.
    copied = "; ".join(words).upper()
    split = tmp_path / "text2kgbench" / "dbpedia.gate.jsonl"
    split.parent.mkdir()
    rows = [
        {"id": "a", "text": "Ada Lovelace was born in London in 1815."},
        {"id": "b", "text": f"A passage that says: {copied} and goes on."},
        {"id": "c", "text": " ".join(words[:11])},
        {"id": "d"},
    ]
    split.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    # A split that is not a gate split is not this check's business.
    (tmp_path / "train.jsonl").write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")

    texts = {p.key: p.text for p in registered()}
    (found,) = leaks(tmp_path, texts)
    assert found.startswith("text2kgbench/dbpedia.gate.jsonl:2 is in extract@1: ")
