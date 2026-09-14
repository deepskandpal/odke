"""Scoring: a documented, monotone, bounded number — and the whole M3 chain at once."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from odke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Pipeline,
    Predicate,
    Scorer,
    SourceTier,
    Span,
)
from odke.corroborate import (
    CONFLICT,
    SCORE,
    NativeResolver,
    SignatureCorroborator,
    ValueNormalizer,
)
from odke.corroborate.score import EvidenceScorer, combine

V = GroundingVerdict
ACME = Entity(key="c1", type="Company", label="Acme")
# Weakest to strongest, the order the score must respect.
VERDICT_ORDER = (V.CONTRADICTED, V.NOT_FOUND, V.UNCHECKED, V.SUPPORTED)


def _fact(confidence: float = 0.8, verdict: V = V.SUPPORTED, *docs: str, **kw: object) -> Fact:
    evidence = tuple(Evidence(doc_id=d) for d in docs or ("d1",))
    kw.setdefault("object_value", "Jane Doe")
    return Fact(
        subject=ACME,
        predicate="ceo",
        confidence=confidence,
        verdict=verdict,
        evidence=evidence,
        **kw,  # type: ignore[arg-type]
    )


def test_the_documented_worked_values_hold() -> None:
    scorer = EvidenceScorer()
    assert scorer.score(_fact(0.8, V.SUPPORTED)).confidence == pytest.approx(0.80)
    assert scorer.score(_fact(0.8, V.UNCHECKED)).confidence == pytest.approx(0.48)
    assert scorer.score(_fact(0.8, V.NOT_FOUND)).confidence == pytest.approx(0.20)
    three = scorer.score(_fact(0.8, V.SUPPORTED, "d1", "d2", "d3"))
    assert three.support == 3
    assert three.confidence == pytest.approx(0.97, abs=0.005)
    assert combine(0.8, 1.0, 1, 0.35) == pytest.approx(0.28)


def test_the_score_is_monotone_in_every_input_and_bounded() -> None:
    confidences = [0.05, 0.3, 0.5, 0.8, 0.99, 1.0]
    supports = [1, 2, 3, 10, 100]
    conflicts = [0.0, 0.2, 0.7, 1.0]
    weights = [EvidenceScorer().verdict_weights[v] for v in VERDICT_ORDER]
    for c in confidences:
        for s in supports:
            for g in weights:
                for k in conflicts:
                    value = combine(c, g, s, k)
                    assert 0.0 <= value <= 1.0
    for s in supports:
        for g in weights:
            row = [combine(c, g, s) for c in confidences]
            assert all(a <= b for a, b in pairwise(row)), "not monotone in confidence"
    for c in confidences:
        for g in weights:
            row = [combine(c, g, s) for s in supports]
            assert all(a <= b for a, b in pairwise(row)), "not monotone in support"
        for s in supports:
            row = [combine(c, g, s) for g in weights]
            assert all(a <= b for a, b in pairwise(row)), "not monotone in verdict"
            row = [combine(c, 1.0, s, k) for k in conflicts]
            assert all(a <= b for a, b in pairwise(row)), "not monotone in conflict"


def test_no_number_of_sources_lifts_a_contradicted_claim() -> None:
    docs = [f"d{i}" for i in range(100)]
    scored = EvidenceScorer().score(_fact(0.99, V.CONTRADICTED, *docs))
    assert scored.support == 100
    assert scored.confidence <= 0.05


def test_an_unreported_confidence_uses_the_prior_and_says_so() -> None:
    scored = EvidenceScorer(prior=0.5).score(_fact(0.0, V.SUPPORTED))
    assert scored.confidence == pytest.approx(0.5)
    assert scored.qualifiers[SCORE]["prior_used"] is True
    assert scored.qualifiers[SCORE]["extractor"] == 0.0


def test_support_is_populated_even_without_a_corroborator() -> None:
    fact = _fact(0.8, V.SUPPORTED).model_copy(
        update={
            "evidence": (
                Evidence(doc_id="a", uri="https://one.example/x"),
                Evidence(doc_id="b", uri="https://two.example/y"),
                Evidence(doc_id="c", uri="https://two.example/z"),
            )
        }
    )
    assert fact.support == 1
    assert EvidenceScorer().score(fact).support == 2


def test_a_contest_loser_scores_lower_and_the_score_says_by_how_much() -> None:
    ontology = Ontology(predicates={"ceo": Predicate(name="ceo", cardinality="single")})
    now = datetime(2026, 9, 1, tzinfo=UTC)
    curated = _fact(0.8, V.SUPPORTED).model_copy(
        update={
            "evidence": (
                Evidence(
                    doc_id="reg", tier=SourceTier.CURATED, retrieved_at=now - timedelta(days=700)
                ),
            )
        }
    )
    scraped = _fact(0.8, V.SUPPORTED, object_value="John Roe").model_copy(
        update={"evidence": (Evidence(doc_id="scrape", retrieved_at=now),)}
    )
    corroborated = SignatureCorroborator(ontology).corroborate([curated, scraped])
    scored = {f.object_value: EvidenceScorer().score(f) for f in corroborated}
    winner, loser = scored["Jane Doe"], scored["John Roe"]
    ratio = loser.qualifiers[CONFLICT]["ratio"]
    assert winner.confidence == pytest.approx(0.8)
    assert loser.confidence == pytest.approx(0.8 * ratio)
    assert loser.qualifiers[SCORE]["conflict"] == ratio
    # The extractor's number is what was scored, not the corroborator's penalty.
    assert loser.qualifiers[SCORE]["extractor"] == 0.8


def test_scoring_is_idempotent() -> None:
    scorer = EvidenceScorer()
    once = scorer.score(_fact(0.6, V.UNCHECKED, "d1", "d2"))
    twice = scorer.score(once)
    assert twice.confidence == once.confidence
    assert twice.qualifiers == once.qualifiers


def test_the_score_can_be_recomputed_from_what_it_recorded() -> None:
    """What #57's calibration needs: the inputs and the output, on the fact, after JSON."""
    scored = EvidenceScorer().score(_fact(0.7, V.NOT_FOUND, "d1", "d2", "d3"))
    (back,) = KnowledgeGraph.model_validate_json(
        KnowledgeGraph(facts=(scored,)).model_dump_json()
    ).facts
    inputs = back.qualifiers[SCORE]
    assert combine(
        inputs["extractor"], inputs["verdict"], inputs["support"], inputs["conflict"]
    ) == pytest.approx(back.confidence)


def test_an_unbounded_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="prior"):
        EvidenceScorer(prior=0.0)
    with pytest.raises(ValueError, match="missing"):
        EvidenceScorer(verdict_weights={V.SUPPORTED: 1.0})
    with pytest.raises(ValueError, match="unbounded"):
        EvidenceScorer(verdict_weights={v: 2.0 for v in V})


class _Births:
    """Three documents, three spellings of one birth date, three keys for one person."""

    SPELLINGS = {"d1": "10 December 1815", "d2": "1815-12-10", "d3": "Dec 10, 1815"}

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        subject = Entity(
            key=f"ada:{chunk.doc_id}",
            type="Person",
            label="Ada Lovelace",
            external_id="wikidata:Q7259",
        )
        span = Span(doc_id=chunk.doc_id, start=0, end=len(chunk.text))
        return [
            Fact(
                subject=subject,
                predicate="born",
                object_value=self.SPELLINGS[chunk.doc_id],
                evidence=(Evidence(doc_id=chunk.doc_id, span=span),),
                confidence=0.7,
                verdict=V.SUPPORTED,
            )
        ]


def test_normalise_resolve_corroborate_score_in_the_pipeline() -> None:
    """The M3 chain: three spellings, three keys — one fact, support 3, a higher score."""
    ontology = Ontology(predicates={"born": Predicate(name="born", range="date")})
    scorer = EvidenceScorer()
    assert isinstance(scorer, Scorer)
    kg = Pipeline(
        ontology,
        _Births(),
        normalizer=ValueNormalizer(ontology),
        resolver=NativeResolver(),
        corroborator=SignatureCorroborator(ontology),
        scorer=scorer,
    ).run([Document(id=d, text=t) for d, t in _Births.SPELLINGS.items()])

    (fact,) = kg.facts
    assert fact.object_value == "1815-12-10"
    assert fact.subject.key == "ada:d1"
    assert fact.support == 3
    assert fact.confidence == pytest.approx(combine(0.7, 1.0, 3))
    assert fact.confidence > 0.7
    assert {link.kind for link in kg.links} == {LinkKind.SAME_AS}
    back = KnowledgeGraph.model_validate_json(kg.model_dump_json())
    assert back.facts[0].confidence == fact.confidence
