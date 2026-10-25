"""Widen and retry (#102): a narrow `not_found` gets one more reading, against its sentence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Chunk, Document, Entity, Evidence, Fact, GroundingVerdict, Ontology, Span
from openodke.cli.main import app
from openodke.corroborate import WIDEN
from openodke.ground import LLMGrounder
from openodke.ground.widen import widen
from openodke.llm import RecordedClient, ScriptedClient
from openodke.run import ConfigError, execute, parse_config
from openodke.run.build import build

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
# The enumerating case of #77: one sentence states three regions, and an
# extractor that cites the region alone loses the fact to its own citation.
SENTENCE = (
    "Acme Cloud runs regional data centres in Ireland, Singapore and Virginia, "
    "and its console is available in English and German."
)
TEXT = f"Acme Cloud is a hosting company. {SENTENCE} It was founded in 2009."
DOC = Document(id="cloud", text=TEXT)
ACME = Entity(key="Provider:acme cloud", type="Provider", label="Acme Cloud")


def _narrow(word: str = "Ireland", *, mention: str | None = None) -> Fact:
    """A fact citing one word of the sentence, as 0.1.0's extractor did."""
    start = TEXT.index(word)
    span = Span(doc_id=DOC.id, start=start, end=start + len(word), quote=word)
    marked = None
    if mention is not None:
        at = TEXT.index(mention)
        marked = Span(doc_id=DOC.id, start=at, end=at + len(mention), quote=mention)
    return Fact(
        subject=ACME,
        predicate="operates_in",
        object_value=word,
        evidence=(Evidence(doc_id=DOC.id, span=span, mention=marked),),
    )


def _recorded() -> RecordedClient:
    # The sentence is supported, a bare word is not_found: #77's own fixture.
    return RecordedClient.from_fixture(FIXTURES / "grounding_enumerating.json")


def test_off_by_default_and_inert() -> None:
    """Today's grounder, unchanged: the narrow citation is refused, once, and nothing is added."""
    default, explicit = _recorded(), _recorded()
    fact = _narrow()
    plain = LLMGrounder(client=default).ground(fact, DOC)
    off = LLMGrounder(client=explicit, widen=False)
    grounded = off.ground(fact, DOC)
    assert plain == grounded
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert len(default.calls) == len(explicit.calls) == 1
    assert grounded.evidence == fact.evidence and WIDEN not in grounded.qualifiers
    assert "widen" not in off.stats
    assert off.stats["calls"] == 1 and off.stats["not_found"] == 1


def test_a_narrow_not_found_is_recovered_against_its_sentence() -> None:
    client = _recorded()
    grounder = LLMGrounder(client=client, widen=True)
    fact = _narrow()
    grounded = grounder.ground(fact, DOC)

    assert grounded.verdict is GroundingVerdict.SUPPORTED
    (evidence,) = grounded.evidence
    assert evidence.span is not None and evidence.span.quote == SENTENCE
    assert evidence.span.is_faithful(DOC)
    # The narrow citation is kept as the mention: the word that tells this fact apart.
    assert evidence.mention == fact.evidence[0].span
    start = TEXT.index(SENTENCE)
    word = TEXT.index("Ireland")
    assert grounded.qualifiers[WIDEN] == {
        "from": [word, word + len("Ireland")],
        "to": [start, start + len(SENTENCE)],
        "verdict": "supported",
    }
    # Two calls: the narrow span, then the sentence.
    passages = [messages[1].content.split("Passage:\n")[1] for messages, _, _ in client.calls]
    assert passages == ["Ireland", SENTENCE]
    stats = grounder.stats
    assert (stats["calls"], stats["supported"], stats["not_found"]) == (2, 1, 0)
    assert stats["widen"]["retried"] == stats["widen"]["recovered"] == 1
    assert stats["widen"]["calls"] == 1
    assert (
        stats["widen"]["prompt_tokens"] > 0
        and stats["prompt_tokens"] > stats["widen"]["prompt_tokens"]
    )


def test_a_mention_already_on_the_evidence_stays() -> None:
    fact = _narrow("regional data centres in Ireland", mention="Ireland")
    grounded = LLMGrounder(client=_recorded(), widen=True).ground(fact, DOC)
    assert grounded.verdict is GroundingVerdict.SUPPORTED
    assert grounded.evidence[0].mention == fact.evidence[0].mention


def test_still_not_found_keeps_the_span_and_records_the_attempt() -> None:
    client = RecordedClient([{"match": "Passage:", "response": {"verdict": "not_found"}}])
    grounder = LLMGrounder(client=client, widen=True)
    fact = _narrow()
    grounded = grounder.ground(fact, DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert grounded.evidence == fact.evidence
    assert grounded.qualifiers[WIDEN]["verdict"] == "not_found"
    assert (grounder.stats["widen"]["retried"], grounder.stats["widen"]["recovered"]) == (1, 0)
    assert grounder.stats["not_found"] == 1 and len(client.calls) == 2


def test_contradicted_on_the_sentence_keeps_not_found() -> None:
    """Only `supported` changes the fact; the retry's answer is in the record."""
    client = ScriptedClient([{"verdict": "not_found"}, {"verdict": "contradicted"}])
    grounded = LLMGrounder(client=client, widen=True).ground(_narrow(), DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert grounded.qualifiers[WIDEN]["verdict"] == "contradicted"


def test_a_failed_retry_leaves_the_first_answer() -> None:
    client = RecordedClient(
        [
            {"match": "Passage:\nIreland", "response": {"verdict": "not_found"}},
            {"match": "Passage:", "error": "HTTP 400 bad request"},
        ]
    )
    grounder = LLMGrounder(client=client, widen=True)
    grounded = grounder.ground(_narrow(), DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert grounded.qualifiers[WIDEN]["verdict"] == "unchecked"
    assert grounder.stats["widen"]["failed"] == 1 and grounder.stats["failed"] == 0


@pytest.mark.parametrize("first", ["supported", "contradicted"])
def test_only_not_found_is_retried(first: str) -> None:
    client = ScriptedClient([{"verdict": first}])
    grounder = LLMGrounder(client=client, widen=True)
    assert grounder.ground(_narrow(), DOC).verdict.value == first
    assert len(client.calls) == 1 and grounder.stats["widen"]["retried"] == 0


def test_a_span_as_wide_as_its_sentence_is_not_retried() -> None:
    client = RecordedClient([{"match": "Passage:", "response": {"verdict": "not_found"}}])
    grounder = LLMGrounder(client=client, widen=True)
    # The sentence without its full stop adds no word, so there is nothing to widen to.
    fact = _narrow(SENTENCE[:-1])
    grounded = grounder.ground(fact, DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert WIDEN not in grounded.qualifiers
    assert len(client.calls) == 1 and grounder.stats["widen"]["retried"] == 0


def test_widening_reaches_every_sentence_a_span_crosses() -> None:
    crossing = "company. Acme Cloud runs"
    (evidence,) = _narrow(crossing).evidence
    wider = widen(evidence, DOC)
    assert wider is not None and wider.span is not None
    assert wider.span.quote == TEXT[: TEXT.index(SENTENCE) + len(SENTENCE)]
    assert widen(Evidence(doc_id=DOC.id), DOC) is None


def test_the_batched_path_widens_too() -> None:
    grounder = LLMGrounder(client=_recorded(), widen=True, max_workers=4)
    facts = [_narrow(word) for word in ("Ireland", "Singapore", "Virginia", "German")]
    grounded = grounder.ground_many(facts, DOC)
    assert {f.verdict for f in grounded} == {GroundingVerdict.SUPPORTED}
    assert grounder.stats["widen"]["recovered"] == 4 and grounder.stats["calls"] == 8


def test_widen_needs_a_span_to_widen() -> None:
    with pytest.raises(ValueError, match="whole document"):
        LLMGrounder(client=_recorded(), context="document", widen=True)


def test_the_retries_go_through_their_own_client() -> None:
    first = RecordedClient([{"match": "Passage:", "response": {"verdict": "not_found"}}])
    second = _recorded()
    grounder = LLMGrounder(client=first, widen=True, widen_client=second)
    assert grounder.ground(_narrow(), DOC).verdict is GroundingVerdict.SUPPORTED
    assert (len(first.calls), len(second.calls)) == (1, 1)


# --------------------------------------------------------------------------- #
# odke run: opt in, and the extra calls are a cost row of their own
# --------------------------------------------------------------------------- #


class Narrow:
    """`test_ground_widen:Narrow`: the 0.1.0 citation of each region, for any chunk."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        out = []
        for word in ("Ireland", "Singapore"):
            at = chunk.text.find(word)
            if at < 0:
                continue
            start = chunk.start + at
            span = Span(doc_id=chunk.doc_id, start=start, end=start + len(word), quote=word)
            out.append(
                Fact(
                    subject=ACME,
                    predicate="operates_in",
                    object_value=word,
                    evidence=(Evidence(doc_id=chunk.doc_id, span=span),),
                )
            )
        return out


def _project(tmp_path: Path, grounder: Any) -> dict[str, Any]:
    ontology = {
        "name": "cloud",
        "types": {"Provider": {}},
        "predicates": {"operates_in": {"domain": ["Provider"], "cardinality": "multi"}},
    }
    (tmp_path / "corpus").mkdir(parents=True)
    (tmp_path / "ontology.json").write_text(json.dumps(ontology), encoding="utf-8")
    (tmp_path / "corpus" / "cloud.txt").write_text(TEXT, encoding="utf-8")
    fixture = (FIXTURES / "grounding_enumerating.json").read_text(encoding="utf-8")
    (tmp_path / "ground.json").write_text(fixture, encoding="utf-8")
    return {
        "ontology": "ontology.json",
        "inputs": ["corpus"],
        "models": {"replay": {"ground": "ground.json"}, "meter": True},
        "stages": {"extractor": "test_ground_widen:Narrow", "grounder": grounder},
    }


def test_odke_run_widens_when_asked_and_meters_the_retries_apart(tmp_path: Path) -> None:
    off = execute(parse_config(_project(tmp_path / "off", "llm"), base_dir=tmp_path / "off"))
    assert {f.verdict for f in off.graph.facts} == {GroundingVerdict.NOT_FOUND}
    assert "widen" not in off.stats["stages"]["grounder"]
    assert set(off.stats["cost"]["stages"]) == {"ground"}

    config = _project(tmp_path / "on", {"use": "llm", "widen": True})
    on = execute(parse_config(config, base_dir=tmp_path / "on"))
    assert {f.verdict for f in on.graph.facts} == {GroundingVerdict.SUPPORTED}
    widened = on.stats["stages"]["grounder"]["widen"]
    assert (widened["retried"], widened["recovered"]) == (2, 2)
    cost = on.stats["cost"]["stages"]
    assert (cost["ground"]["calls"], cost["ground.widen"]["calls"]) == (2, 2)
    assert "widen (retried 2, recovered 2" in on.render()


def test_the_widen_flag_needs_the_model_grounder(tmp_path: Path) -> None:
    config = parse_config(_project(tmp_path, "span"), base_dir=tmp_path)
    with pytest.raises(ConfigError, match="--widen needs the model grounder"):
        config.with_widen()
    widened = parse_config(_project(tmp_path / "x", "llm"), base_dir=tmp_path / "x").with_widen()
    assert widened.stages.grounder is not None
    assert widened.stages.grounder.options == {"widen": True}
    with pytest.raises(ConfigError, match="widen_client: set by odke run"):
        bad = _project(tmp_path / "y", {"use": "llm", "widen_client": "x"})
        build(parse_config(bad, base_dir=tmp_path / "y"))


def test_odke_run_takes_a_widen_flag(tmp_path: Path) -> None:
    path = tmp_path / "odke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_project(tmp_path, "llm")), encoding="utf-8")
    result = CliRunner().invoke(app, ["run", str(path), "--widen", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "widen (retried 2, recovered 2" in result.output
    plain = CliRunner().invoke(app, ["run", str(path), "--dry-run"])
    assert plain.exit_code == 0 and "widen" not in plain.output
