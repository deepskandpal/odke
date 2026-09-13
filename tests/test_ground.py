"""Grounding — the cheapest rejection first, then the second model's question."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pytest

from odke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    Grounder,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Span,
)
from odke.ground import SpanGrounder, SpanStatus, check_span, located

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
