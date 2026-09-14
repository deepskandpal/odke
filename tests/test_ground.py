"""Grounding — the cheapest rejection first, then the second model's question."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pytest

from odke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    Grounder,
    GroundingVerdict,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    Polarity,
    Span,
    ValidationVerdict,
)
from odke.ground import (
    LLMGrounder,
    SpanGrounder,
    SpanStatus,
    build_messages,
    check_span,
    located,
    parse_verdict,
    render_claim,
)
from odke.ground.llm import GROUNDING_SCHEMA, SYSTEM_PROMPT
from odke.llm import Completion, ModelRoles, RecordedClient, ScriptedClient

FIXTURE = Path(__file__).parent / "fixtures" / "llm" / "grounding.json"

DOC = Document(id="d1", text="Ada Lovelace wrote the first algorithm. It ran on paper.")
ADA = Entity(key="p1", type="Person", label="Ada Lovelace")


def _fact(*evidence: Evidence, **overrides: object) -> Fact:
    fields: dict[str, object] = {
        "subject": ADA,
        "predicate": "wrote",
        "object_value": "the first algorithm",
        "evidence": evidence,
    }
    fields.update(overrides)
    return Fact.model_validate(fields)


def _cite(start: int, end: int, quote: str | None = None, doc_id: str = "d1") -> Evidence:
    return Evidence(doc_id=doc_id, span=Span(doc_id=doc_id, start=start, end=end, quote=quote))


HONEST = _cite(0, 39, "Ada Lovelace wrote the first algorithm.")
INVENTED = _cite(0, 39, "Alan Turing wrote the first algorithm.")


# --------------------------------------------------------------------------- #
# #15 — free span verification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("evidence", "status"),
    [
        (HONEST, SpanStatus.LOCATED),
        (_cite(0, 12), SpanStatus.LOCATED),  # offsets alone: nothing to contradict
        (INVENTED, SpanStatus.QUOTE_MISMATCH),
        (Evidence(doc_id="d1"), SpanStatus.NO_SPAN),
        (_cite(0, 12, "Ada Lovelace", doc_id="d2"), SpanStatus.FOREIGN),
        (_cite(0, 500, "Ada"), SpanStatus.OUT_OF_RANGE),
        (_cite(-1, 3), SpanStatus.OUT_OF_RANGE),
        (_cite(10, 4), SpanStatus.OUT_OF_RANGE),
        (_cite(5, 5), SpanStatus.EMPTY),
    ],
)
def test_check_span_classifies_every_way_a_citation_can_go(
    evidence: Evidence, status: SpanStatus
) -> None:
    assert check_span(evidence, DOC) is status


def test_only_the_three_bad_shapes_are_rejections() -> None:
    """A document-level citation or another document's span is unverifiable, not wrong."""
    assert {s for s in SpanStatus if s.rejected} == {
        SpanStatus.OUT_OF_RANGE,
        SpanStatus.EMPTY,
        SpanStatus.QUOTE_MISMATCH,
    }


def test_a_span_running_past_the_text_is_rejected_even_when_the_quote_matches() -> None:
    """Slicing truncates silently; the offsets are still wrong."""
    doc = Document(id="d1", text="abc")
    assert check_span(_cite(0, 10, "abc"), doc) is SpanStatus.OUT_OF_RANGE


def test_a_faithful_span_passes_through_untouched() -> None:
    """Nothing to reject and nothing to decide: the model gets asked next."""
    fact = _fact(HONEST)
    assert SpanGrounder().ground(fact, DOC) is fact
    assert fact.verdict is GroundingVerdict.UNCHECKED


def test_an_invented_quote_is_rejected_for_free_and_the_fact_is_not_found() -> None:
    """The cheapest possible demonstration of the differentiator: no token spent."""
    grounded = SpanGrounder().ground(_fact(INVENTED), DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert grounded.evidence == ()


def test_a_bad_span_is_stripped_and_a_good_one_keeps_the_fact_in_play() -> None:
    grounded = SpanGrounder().ground(_fact(INVENTED, HONEST), DOC)
    assert grounded.verdict is GroundingVerdict.UNCHECKED
    assert grounded.evidence == (HONEST,)


def test_a_fact_with_no_evidence_at_all_is_not_found() -> None:
    assert SpanGrounder().ground(_fact(), DOC).verdict is GroundingVerdict.NOT_FOUND


def test_unverifiable_evidence_is_kept_but_does_not_locate_the_claim() -> None:
    """Nothing showed it wrong, so it stays; nothing showed it right, so it is not found here."""
    doc_level = Evidence(doc_id="d1")
    elsewhere = _cite(0, 12, "Ada Lovelace", doc_id="d2")
    grounded = SpanGrounder().ground(_fact(doc_level, elsewhere), DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert grounded.evidence == (doc_level, elsewhere)


def test_located_picks_the_first_evidence_that_really_cites_the_document() -> None:
    assert located(_fact(INVENTED, Evidence(doc_id="d1"), HONEST), DOC) is HONEST
    assert located(_fact(INVENTED), DOC) is None


def test_a_verdict_already_on_the_fact_stands() -> None:
    """Re-running over a partially grounded set resumes rather than re-deciding."""
    decided = _fact(INVENTED, verdict=GroundingVerdict.SUPPORTED)
    grounder = SpanGrounder()
    assert grounder.ground(decided, DOC) is decided
    assert grounder.stats["facts"] == 0


def test_the_grounder_never_mutates_and_never_drops() -> None:
    fact = _fact(INVENTED)
    grounded = SpanGrounder().ground(fact, DOC)
    assert fact.evidence == (INVENTED,) and fact.verdict is GroundingVerdict.UNCHECKED
    assert grounded.id == fact.id and grounded.signature == fact.signature


def test_stats_count_every_status_and_the_rejection_reason_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run report: a high rejection rate is a prompt problem, not a model problem."""
    grounder = SpanGrounder()
    with caplog.at_level(logging.DEBUG, logger="odke.ground"):
        grounder.ground(_fact(INVENTED, HONEST), DOC)
        grounder.ground(_fact(_cite(0, 500)), DOC)
        grounder.ground(_fact(), DOC)
    stats = grounder.stats
    assert stats["facts"] == 3
    assert stats["located"] == 1
    assert stats["quote_mismatch"] == 1
    assert stats["out_of_range"] == 1
    assert stats["rejected"] == 2
    assert stats["not_found"] == 2
    reasons = [r.getMessage() for r in caplog.records]
    assert any("quote_mismatch" in r and "d1[0:39]" in r for r in reasons)
    assert any("out_of_range" in r for r in reasons)


def test_span_grounder_satisfies_the_protocol() -> None:
    assert isinstance(SpanGrounder(), Grounder)


class _HallucinatingExtractor:
    """Cites a real span for the first claim and invents a quote for the second."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        real = Span(doc_id=chunk.doc_id, start=0, end=12, quote=chunk.text[0:12])
        fake = Span(doc_id=chunk.doc_id, start=0, end=12, quote="Charles Babbage")
        person = Entity(key=f"p:{chunk.doc_id}", type="Person")
        return [
            Fact(
                subject=person,
                predicate="name",
                object_value="Ada Lovelace",
                evidence=(Evidence(doc_id=chunk.doc_id, span=real),),
            ),
            Fact(
                subject=person,
                predicate="name",
                object_value="Charles Babbage",
                evidence=(Evidence(doc_id=chunk.doc_id, span=fake),),
            ),
        ]


def test_in_the_pipeline_the_hallucination_is_stamped_before_any_model_exists() -> None:
    """End to end with no model configured: the invented quote is `NOT_FOUND`, the
    real one is still a candidate, and the counts say which was which."""
    grounder = SpanGrounder()
    kg = Pipeline(Ontology(), _HallucinatingExtractor(), grounder=grounder).run([DOC])
    by_value = {f.object_value: f.verdict for f in kg.facts}
    assert by_value == {
        "Ada Lovelace": GroundingVerdict.UNCHECKED,
        "Charles Babbage": GroundingVerdict.NOT_FOUND,
    }
    assert grounder.stats["rejected"] == 1


# --------------------------------------------------------------------------- #
# #16 — LLMGrounder: one fact, one span, one verdict
# --------------------------------------------------------------------------- #

CORPUS = Document(
    id="c1",
    text=(
        "Ada Lovelace wrote the first algorithm in 1843. She was born in London. "
        "Babbage never finished the Analytical Engine."
    ),
)
LONDON = Entity(key="london", type="Place", label="London")
PARIS = Entity(key="paris", type="Place", label="Paris")
BABBAGE = Entity(key="p2", type="Person", label="Charles Babbage")
ENGINE = Entity(key="m1", type="Machine", label="Analytical Engine")


def _cites(doc: Document, sentence: str) -> Evidence:
    """Evidence for the one sentence, offsets found rather than hand-counted."""
    start = doc.text.index(sentence)
    span = Span(doc_id=doc.id, start=start, end=start + len(sentence), quote=sentence)
    return Evidence(doc_id=doc.id, span=span)


WROTE_IT = _cites(CORPUS, "Ada Lovelace wrote the first algorithm in 1843.")
BORN_LONDON = _cites(CORPUS, "She was born in London.")
NEVER_FINISHED = _cites(CORPUS, "Babbage never finished the Analytical Engine.")

WROTE = _fact(WROTE_IT)
BORN_IN_PARIS = _fact(BORN_LONDON, predicate="birth_place", object_value=None, object_entity=PARIS)
BORN_IN_1815 = _fact(BORN_LONDON, predicate="birth_date", object_value="1815")
DID_NOT_FINISH = Fact(
    subject=BABBAGE,
    predicate="finished",
    object_entity=ENGINE,
    polarity=Polarity.DENIED,
    evidence=(NEVER_FINISHED,),
)


def _recorded() -> RecordedClient:
    return RecordedClient.from_fixture(FIXTURE)


def _grounder(client: RecordedClient | ScriptedClient | None = None) -> LLMGrounder:
    return LLMGrounder(client=client if client is not None else _recorded())


def test_render_claim_reads_as_one_stable_line() -> None:
    assert render_claim(WROTE) == 'Ada Lovelace (Person) — wrote — "the first algorithm".'
    assert render_claim(BORN_IN_PARIS) == "Ada Lovelace (Person) — birth place — Paris (Place)."
    assert render_claim(WROTE) == render_claim(WROTE.model_copy(update={"confidence": 0.9}))


def test_render_claim_spells_out_polarity() -> None:
    """A denial and its assertion are the same triple and opposite claims."""
    assert render_claim(DID_NOT_FINISH) == (
        "It is NOT the case that: "
        "Charles Babbage (Person) — finished — Analytical Engine (Machine)."
    )
    partial = DID_NOT_FINISH.model_copy(update={"polarity": Polarity.PARTIAL})
    assert render_claim(partial).startswith("Only partially, or under a stated condition: ")


def test_render_claim_shows_identity_qualifiers_and_hides_reconcilable_ones() -> None:
    """p95 is part of the claim; 'since 2024' is a view of it that the corroborator reconciles."""
    uptime = Fact(
        subject=Entity(key="acme", type="Company"),
        predicate="has_uptime",
        object_value=99.9,
        qualifiers={"percentile": "p95", "start_time": "2024"},
        identity_keys=("percentile",),
    )
    assert render_claim(uptime) == 'acme (Company) — has uptime — 99.9 (percentile = "p95").'


def test_the_prompt_is_the_claim_and_the_cited_passage_and_nothing_more() -> None:
    """Not the document: the question is whether *this span* supports the claim."""
    system, user = build_messages(WROTE, WROTE_IT.span.resolve(CORPUS))  # type: ignore[union-attr]
    assert system.role == "system" and system.content == SYSTEM_PROMPT
    assert user.content == (
        'Claim: Ada Lovelace (Person) — wrote — "the first algorithm".\n\n'
        "Passage:\nAda Lovelace wrote the first algorithm in 1843."
    )
    assert "born in London" not in user.content
    # The cost story: this is sent once per fact and must stay a fraction of an
    # extraction prompt, which carries an ontology snippet and a whole chunk.
    assert len(SYSTEM_PROMPT) < 600


@pytest.mark.parametrize(
    ("fact", "verdict"),
    [
        (WROTE, GroundingVerdict.SUPPORTED),
        (BORN_IN_PARIS, GroundingVerdict.CONTRADICTED),
        (BORN_IN_1815, GroundingVerdict.NOT_FOUND),
        (DID_NOT_FINISH, GroundingVerdict.SUPPORTED),
    ],
)
def test_each_verdict_is_read_from_a_recorded_answer(fact: Fact, verdict: GroundingVerdict) -> None:
    client = _recorded()
    grounded = _grounder(client).ground(fact, CORPUS)
    assert grounded.verdict is verdict
    (call,) = client.calls
    assert call[2] == GROUNDING_SCHEMA


def test_the_grounder_uses_the_ground_role_and_sets_only_the_verdict() -> None:
    """`confidence` belongs to the scorer; the model is the cheap one on purpose."""
    roles = ModelRoles(ground=ModelRoles.single("ollama/qwen2.5:3b").ground)
    client = _recorded()
    grounded = LLMGrounder(roles, client=client).ground(WROTE, CORPUS)
    assert client.calls[0][1] == roles.ground
    assert grounded.confidence == 0.0
    before = WROTE.model_dump(exclude={"verdict"})
    assert grounded.model_dump(exclude={"verdict"}) == before


def test_the_span_check_runs_first_and_an_invented_quote_costs_no_call() -> None:
    client = _recorded()
    grounder = _grounder(client)
    grounded = grounder.ground(_fact(INVENTED), DOC)
    assert grounded.verdict is GroundingVerdict.NOT_FOUND
    assert client.calls == []
    assert grounder.stats["calls"] == 0
    assert grounder.stats["span"]["rejected"] == 1


def test_a_bad_span_is_stripped_and_the_located_one_is_what_the_model_sees() -> None:
    invented = Evidence(
        doc_id="c1", span=Span(doc_id="c1", start=0, end=47, quote="Alan Turing wrote it.")
    )
    client = _recorded()
    grounded = _grounder(client).ground(_fact(invented, WROTE_IT), CORPUS)
    assert grounded.verdict is GroundingVerdict.SUPPORTED
    assert grounded.evidence == (WROTE_IT,)
    assert client.calls[0][0][1].content.endswith("Ada Lovelace wrote the first algorithm in 1843.")


def test_a_bare_word_answer_is_still_read() -> None:
    """A model that ignored the schema but said the word is not a failed call."""
    fact = _fact(BORN_LONDON, predicate="born_in", object_value=None, object_entity=LONDON)
    assert _grounder().ground(fact, CORPUS).verdict is GroundingVerdict.SUPPORTED


def test_prose_is_never_guessed_at(caplog: pytest.LogCaptureFixture) -> None:
    """'not supported' must not become SUPPORTED; an unreadable answer leaves UNCHECKED."""
    fact = _fact(BORN_LONDON, predicate="died_in", object_value=None, object_entity=LONDON)
    grounder = _grounder()
    with caplog.at_level(logging.WARNING, logger="odke.ground"):
        grounded = grounder.ground(fact, CORPUS)
    assert grounded.verdict is GroundingVerdict.UNCHECKED
    assert grounder.stats["unparseable"] == 1
    assert any("unreadable grounding answer" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        (Completion(text="", parsed={"verdict": "Supported"}), GroundingVerdict.SUPPORTED),
        (Completion(text="not found"), GroundingVerdict.NOT_FOUND),
        (Completion(text=' "CONTRADICTED".'), GroundingVerdict.CONTRADICTED),
        (Completion(text="not supported"), None),
        (Completion(text="", parsed={"verdict": "unchecked"}), None),
        (Completion(text="", parsed={"verdict": 1}), None),
        (Completion(text="The claim is supported by the passage."), None),
    ],
)
def test_parse_verdict_reads_exactly_one_of_the_three(
    completion: Completion, expected: GroundingVerdict | None
) -> None:
    assert parse_verdict(completion) is expected


def test_a_verdict_already_on_the_fact_costs_no_call() -> None:
    client = _recorded()
    grounder = _grounder(client)
    decided = WROTE.model_copy(update={"verdict": GroundingVerdict.CONTRADICTED})
    assert grounder.ground(decided, CORPUS) is decided
    assert client.calls == []
    assert grounder.stats["skipped"] == 1


def test_stats_are_the_stage_run_report() -> None:
    grounder = _grounder()
    for fact in (WROTE, BORN_IN_PARIS, BORN_IN_1815, _fact(INVENTED)):
        grounder.ground(fact, CORPUS if fact.evidence and fact.evidence[0].doc_id == "c1" else DOC)
    stats = grounder.stats
    assert stats["facts"] == 4
    assert stats["calls"] == 3
    assert (stats["supported"], stats["contradicted"], stats["not_found"]) == (1, 1, 1)
    assert stats["prompt_tokens"] > 0 and stats["completion_tokens"] > 0
    assert stats["cost_usd"] is None
    assert stats["span"]["not_found"] == 1


def test_llm_grounder_satisfies_the_protocol() -> None:
    assert isinstance(_grounder(), Grounder)


class _AdaExtractor:
    """One supported claim and one the span contradicts, both citing real spans."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        return [WROTE, BORN_IN_PARIS]


class _GateOnVerdict:
    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        if fact.verdict is GroundingVerdict.CONTRADICTED:
            return ValidationVerdict(action="refuse", reason="contradicted by its own span")
        return ValidationVerdict(action="accept")


class _Collect:
    def __init__(self) -> None:
        self.graphs: list[KnowledgeGraph] = []

    def write(self, kg: KnowledgeGraph) -> None:
        self.graphs.append(kg)


def test_end_to_end_a_contradicted_fact_never_reaches_the_sink() -> None:
    """Stamp, then gate: the grounder marks it, the validator refuses it (DECISIONS #20)."""
    sink = _Collect()
    kg = Pipeline(
        Ontology(),
        _AdaExtractor(),
        grounder=_grounder(),
        validator=_GateOnVerdict(),
        sinks=[sink],
    ).run([CORPUS])
    assert [f.predicate for f in kg.facts] == ["wrote"]
    assert kg.facts[0].verdict is GroundingVerdict.SUPPORTED
    assert kg.stats["refused"] == 1
    assert sink.graphs == [kg]
