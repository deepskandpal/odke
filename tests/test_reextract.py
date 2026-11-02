"""The re-extract hook (#102): the gaps go back to the extractor, and what returns is grounded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Span,
)
from openodke.extract import HybridExtractor, LLMExtractor
from openodke.ground import LLMGrounder
from openodke.ground.llm import render_claim
from openodke.llm import ModelSpec, RecordedClient, ScriptedClient
from openodke.prompts import get as get_prompt
from openodke.reextract import REEXTRACT, Reextract, Reextractor, relations_for
from openodke.run import ConfigError, execute, parse_config
from openodke.run.build import build

ONTOLOGY = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {}, "Company": {}, "City": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "importance": 0.9},
            "born_in": {"domain": ["Person"], "range": "City", "importance": 0.8},
            "founded": {"domain": ["Company"], "range": "integer", "importance": 0.5},
            "headquarters": {"domain": ["Company"], "range": "City", "importance": 0.1},
        },
    }
)
FIRST = "Ada Lovelace worked for Analytical Engines in London."
# The planted gap: two known entities, and no fact cites it.
GAP = "Ada Lovelace was born in London."
TEXT = f"{FIRST} {GAP} The weather was poor."
DOC = Document(id="d1", text=TEXT)
SONNET = ModelSpec(model="anthropic/claude-sonnet-5")

ADA = Entity(key="Person:ada lovelace", type="Person", label="Ada Lovelace")
ENGINES = Entity(key="Company:analytical engines", type="Company", label="Analytical Engines")
LONDON = Entity(key="City:london", type="City", label="London")


def _cited(quote: str, subject: Entity, predicate: str, obj: Entity) -> Fact:
    start = TEXT.index(quote)
    span = Span(doc_id=DOC.id, start=start, end=start + len(quote), quote=quote)
    return Fact(
        subject=subject,
        predicate=predicate,
        object_entity=obj,
        evidence=(Evidence(doc_id=DOC.id, span=span),),
    )


EMPLOYED = _cited(FIRST, ADA, "employer", ENGINES)
BASED = _cited(FIRST, ENGINES, "headquarters", LONDON)
BORN = _cited(GAP, ADA, "born_in", LONDON)


def _item(predicate: str, value: str, quote: str) -> dict[str, Any]:
    return {
        "predicate": predicate,
        "value": value,
        "quote": quote,
        "start": 0,
        "mention": "",
        "polarity": "asserted",
        "qualifiers": {},
    }


FIRST_PASS = {
    "entities": [
        {
            "type": "Person",
            "name": "Ada Lovelace",
            "facts": [_item("employer", "Analytical Engines", FIRST)],
        },
        {
            "type": "Company",
            "name": "Analytical Engines",
            "facts": [_item("headquarters", "London", FIRST)],
        },
    ]
}
GLEANED = {
    "entities": [
        {"type": "Person", "name": "Ada Lovelace", "facts": [_item("born_in", "London", GAP)]}
    ]
}


class _Planted:
    """An extractor that finds the first sentence's facts, and records what it is asked again."""

    def __init__(self, gleaned: list[Fact] | None = None) -> None:
        self.gleaned = [BORN] if gleaned is None else gleaned
        self.asked: list[tuple[Chunk, list[str], list[Fact]]] = []

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return [EMPLOYED, BASED] if chunk.doc_id == DOC.id else []

    def reextract(
        self, window: Chunk, relations: list[str], already: list[Fact], ontology: Ontology
    ) -> list[Fact]:
        self.asked.append((window, relations, already))
        return list(self.gleaned)


def _grounder(verdict: str = "supported") -> LLMGrounder:
    return LLMGrounder(
        client=RecordedClient([{"match": "Claim:", "response": {"verdict": verdict}}])
    )


def test_the_reference_extractors_are_reextractors() -> None:
    assert isinstance(LLMExtractor(client=ScriptedClient([])), Reextractor)
    assert isinstance(HybridExtractor(LLMExtractor(client=ScriptedClient([]))), Reextractor)
    assert isinstance(_Planted(), Reextractor)


def test_off_by_default_and_inert() -> None:
    planted = _Planted()
    plain = Pipeline(ONTOLOGY, planted, grounder=_grounder(), coverage=True).run([DOC])
    assert planted.asked == []
    assert "reextract" not in plain.stats
    assert {f.predicate for f in plain.facts} == {"employer", "headquarters"}


def test_a_planted_gap_goes_back_and_what_returns_is_grounded() -> None:
    planted = _Planted()
    kg = Pipeline(ONTOLOGY, planted, grounder=_grounder(), reextract=Reextract()).run([DOC])

    ((window, relations, already),) = planted.asked
    assert window.text == GAP and DOC.text[window.start : window.end] == GAP
    # The relations a Person and a City could hold, and nothing else.
    assert relations == ["born_in"]
    # The facts naming an entity the window names, so they are not returned again.
    assert [f.id for f in already] == [EMPLOYED.id, BASED.id]

    born = next(f for f in kg.facts if f.predicate == "born_in")
    assert born.verdict is GroundingVerdict.SUPPORTED
    assert born.qualifiers[REEXTRACT] == {"window": [window.start, window.end]}
    assert kg.stats["reextract"] == {
        "windows": 1,
        "returned": 1,
        "duplicates": 0,
        "kept": 1,
        "refused": 0,
        "verdicts": {"supported": 1},
    }
    # Coverage is measured to find the gaps, and reported only when asked for.
    assert "coverage" not in kg.stats


def test_what_grounding_refuses_is_counted_as_refused() -> None:
    kg = Pipeline(ONTOLOGY, _Planted(), grounder=_grounder("not_found"), reextract=Reextract()).run(
        [DOC]
    )
    stats = kg.stats["reextract"]
    assert (stats["kept"], stats["refused"], stats["verdicts"]) == (0, 1, {"not_found": 1})
    # Refused is a count; the gate decides what is written, as for any fact.
    assert any(f.predicate == "born_in" for f in kg.facts)


def test_a_fact_already_held_is_dropped_and_counted() -> None:
    planted = _Planted(gleaned=[EMPLOYED, BORN, BORN])
    kg = Pipeline(ONTOLOGY, planted, grounder=_grounder(), reextract=Reextract()).run([DOC])
    stats = kg.stats["reextract"]
    assert (stats["returned"], stats["duplicates"], stats["kept"]) == (3, 2, 1)
    assert sum(f.predicate == "employer" for f in kg.facts) == 1


def test_each_window_is_asked_once_and_the_windows_are_capped() -> None:
    sentences = [f"Ada Lovelace was born in London in {1815 + i}." for i in range(5)]
    doc = Document(id="d1", text=f"{FIRST} " + " ".join(sentences))
    planted = _Planted(gleaned=[])
    Pipeline(ONTOLOGY, planted, reextract=Reextract(windows=2)).run([doc])
    assert [w.text for w, _, _ in planted.asked] == sentences[:2]
    with pytest.raises(ValueError, match="windows"):
        Reextract(windows=0)


def test_a_pipeline_whose_extractor_cannot_reextract_is_refused() -> None:
    class Plain:
        def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
            return []

    with pytest.raises(TypeError, match="has no reextract"):
        Pipeline(ONTOLOGY, Plain(), reextract=Reextract())
    hook = _Planted()
    Pipeline(ONTOLOGY, Plain(), reextract=Reextract(hook=hook)).run([DOC])
    assert hook.asked == []  # Plain found nothing, so nothing is known and nothing is a gap


def test_relations_fit_the_types_a_window_names() -> None:
    assert relations_for(ONTOLOGY, {"Person", "City"}) == ["born_in"]
    assert relations_for(ONTOLOGY, {"Person", "Company", "City"}) == [
        "employer",
        "born_in",
        "founded",
        "headquarters",
    ]
    assert relations_for(ONTOLOGY, {"City"}) == []


# --------------------------------------------------------------------------- #
# The reference extractor's hook: reextract@1
# --------------------------------------------------------------------------- #


def _window() -> Chunk:
    start = TEXT.index(GAP)
    return Chunk(doc_id=DOC.id, start=start, end=start + len(GAP), text=GAP, index=0)


def test_llm_extractor_asks_with_the_registered_prompt_and_the_flagged_snippets() -> None:
    client = ScriptedClient([GLEANED])
    extractor = LLMExtractor(client=client, spec=SONNET, documents=[DOC], snippet_limit=1)
    (fact,) = extractor.reextract(_window(), ["born_in"], [EMPLOYED], ONTOLOGY)

    ((messages, _, schema),) = client.calls
    system, user = messages[0].content, messages[1].content
    prompt = get_prompt("reextract")
    assert system.startswith(prompt.text)
    # The flagged property alone, though it is past the snippet limit.
    assert "- born_in (City, single)" in system and "employer" not in system
    assert user == (
        f"Already extracted from this passage:\n- {render_claim(EMPLOYED)}\n\nPassage:\n{GAP}"
    )
    assert schema is not None
    assert extractor.calls[0].prompt == "reextract@1"
    # Read as extraction reads a reply: a span into the document, checked.
    (evidence,) = fact.evidence
    assert evidence.span is not None and evidence.span.is_faithful(DOC)
    assert evidence.span.quote == GAP and (fact.subject.key, fact.predicate) == (
        "Person:ada lovelace",
        "born_in",
    )


def test_an_empty_answer_is_the_expected_one() -> None:
    extractor = LLMExtractor(client=ScriptedClient([{"entities": []}]), spec=SONNET)
    assert extractor.reextract(_window(), ["born_in"], [], ONTOLOGY) == []
    assert extractor.empty_extractions == 0
    nothing = LLMExtractor(client=ScriptedClient([]), spec=SONNET)
    assert nothing.reextract(_window(), ["no_such_relation"], [], ONTOLOGY) == []
    assert nothing.calls == []


def test_the_whole_loop_through_llm_extractor_and_llm_grounder() -> None:
    client = RecordedClient(
        [
            {"match": "Already extracted from this passage", "response": GLEANED},
            {"match": "Extract facts from the passage", "response": FIRST_PASS},
        ]
    )
    extractor = LLMExtractor(client=client, spec=SONNET, documents=[DOC])
    kg = Pipeline(
        ONTOLOGY, extractor, grounder=_grounder(), coverage=True, reextract=Reextract()
    ).run([DOC])
    assert kg.stats["coverage"]["uncovered"] == 1
    assert kg.stats["reextract"]["kept"] == 1
    assert {f.predicate for f in kg.facts} == {"employer", "headquarters", "born_in"}
    assert extractor.prompts == ["extract@1", "reextract@1"]


# --------------------------------------------------------------------------- #
# odke run
# --------------------------------------------------------------------------- #


def _project(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    (tmp_path / "corpus").mkdir(parents=True)
    (tmp_path / "ontology.json").write_text(ONTOLOGY.model_dump_json(), encoding="utf-8")
    (tmp_path / "corpus" / "d1.txt").write_text(TEXT, encoding="utf-8")
    return {
        "ontology": "ontology.json",
        "inputs": ["corpus"],
        "stages": {"extractor": "test_reextract:AnyDoc"},
        **extra,
    }


class AnyDoc(_Planted):
    """`test_reextract:AnyDoc`: the planted facts, under whatever id the run gives the document."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return [
            f.model_copy(
                update={
                    "evidence": tuple(
                        e.model_copy(
                            update={
                                "doc_id": chunk.doc_id,
                                "span": e.span.model_copy(update={"doc_id": chunk.doc_id})
                                if e.span
                                else None,
                            }
                        )
                        for e in f.evidence
                    )
                }
            )
            for f in (EMPLOYED, BASED)
        ]

    def reextract(
        self, window: Chunk, relations: list[str], already: list[Fact], ontology: Ontology
    ) -> list[Fact]:
        self.asked.append((window, relations, already))
        span = Span(doc_id=window.doc_id, start=window.start, end=window.end, quote=window.text)
        return [BORN.model_copy(update={"evidence": (Evidence(doc_id=window.doc_id, span=span),)})]


def test_odke_run_turns_it_on_from_the_config(tmp_path: Path) -> None:
    config = _project(tmp_path, reextract={"windows": 2})
    result = execute(parse_config(config, base_dir=tmp_path))
    assert result.stats["reextract"]["windows"] == 1
    assert result.stats["reextract"]["kept"] == 1
    line = next(row for row in result.render().splitlines() if row.startswith("reextract"))
    assert "1 windows asked, 1 facts returned (0 already held), 1 kept" in line
    json.dumps(result.stats["reextract"])  # what a manifest writes

    off = execute(parse_config(_project(tmp_path / "off"), base_dir=tmp_path / "off"))
    assert "reextract" not in off.stats
    on = parse_config(_project(tmp_path / "on", reextract=True), base_dir=tmp_path / "on")
    assert on.reextract is not None and on.reextract.windows == 3


def test_odke_run_refuses_an_extractor_that_cannot_reextract(tmp_path: Path) -> None:
    config = _project(tmp_path, reextract=True)
    config["stages"]["extractor"] = {"use": "pattern"}
    with pytest.raises(ConfigError, match="reextract: the extractor"):
        build(parse_config(config, base_dir=tmp_path))
    with pytest.raises(ConfigError, match="windows"):
        parse_config(_project(tmp_path / "zero", reextract={"windows": 0}), base_dir=tmp_path)
