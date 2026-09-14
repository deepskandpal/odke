"""Extraction: per-predicate P/R/F1, and every miss sorted into one of four kinds.

Expected numbers are hand-computed from tests/fixtures/eval/extract.*.jsonl —
five gold facts in two documents, not a benchmark:

    d1 ada born 1815          <- p1 born "1815"          correct (value normalised)
    d1 ada name Ada Lovelace  <- p2 name "ada  lovelace" correct (value normalised)
    d1 ada employer analytical<- p3 employer cambridge   wrong entity
    d2 babbage born 1791      <- p4 born 1792            wrong value
    d2 babbage employer cambridge                         missing
                                 p5 babbage field        spurious

tp 2, fp 3 (wrong value, wrong entity, spurious), fn 3 (wrong value, wrong
entity, missing): P = R = F1 = 2/5.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from odke import Chunk, Document, Entity, Evidence, Fact, Ontology, Polarity
from odke.eval import GoldFact, load_jsonl
from odke.eval.extraction import (
    evaluate_extraction,
    match_extraction,
    normalise_value,
    run_extract,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
GOLD = load_jsonl(FIXTURES / "extract.labels.jsonl", GoldFact)
PREDICTIONS = load_jsonl(FIXTURES / "extract.predictions.jsonl", Fact)

ADA = Entity(key="ada", type="Person")
BABBAGE = Entity(key="babbage", type="Person")


def _in(doc_id: str, fact: Fact) -> Fact:
    return fact.model_copy(update={"evidence": (Evidence(doc_id=doc_id),)})


def test_totals_on_the_fixture() -> None:
    report = evaluate_extraction(GOLD, PREDICTIONS)
    m = report.metrics
    assert report.stage == "extract" and report.n == 5
    assert (m["tp"], m["fp"], m["fn"]) == (2, 3, 3)
    assert m["precision"] == pytest.approx(0.4)
    assert m["recall"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(0.4)
    assert (m["wrong_value"], m["wrong_entity"], m["missing"], m["spurious"]) == (1, 1, 1, 1)
    assert m["predicates"] == 4


def test_the_breakdown_is_per_predicate_not_only_an_aggregate() -> None:
    """The aggregate 0.4 hides that `name` is perfect and `employer` never works."""
    b = evaluate_extraction(GOLD, PREDICTIONS).breakdown
    assert list(b) == ["born", "employer", "field", "name"]
    assert (b["born"]["precision"], b["born"]["recall"], b["born"]["wrong_value"]) == (0.5, 0.5, 1)
    assert (b["name"]["precision"], b["name"]["recall"], b["name"]["f1"]) == (1.0, 1.0, 1.0)
    # employer: tp 0, fp 1 (wrong entity), fn 2 (wrong entity, missing).
    assert (b["employer"]["fp"], b["employer"]["fn"], b["employer"]["f1"]) == (1, 2, 0.0)
    assert (b["employer"]["wrong_entity"], b["employer"]["missing"]) == (1, 1)
    # field was never labelled: precision 0, recall undefined — not zero.
    assert (b["field"]["precision"], b["field"]["recall"], b["field"]["f1"]) == (0.0, None, 0.0)
    assert b["field"]["support"] == 0


def test_macro_f1_and_the_note_that_names_the_dead_predicates() -> None:
    report = evaluate_extraction(GOLD, PREDICTIONS)
    assert report.metrics["macro_f1"] == pytest.approx((0.5 + 0.0 + 0.0 + 1.0) / 4)
    assert "F1 is zero for: employer, field" in report.notes
    assert "employer" in report.render()


@pytest.mark.parametrize(
    ("a", "b"),
    [(1815, "1815"), (1815.0, "1815"), ("Ada  Lovelace", "ada lovelace"), (3.50, "3.5")],
)
def test_literal_values_match_after_normalisation(a: object, b: object) -> None:
    assert normalise_value(a) == normalise_value(b)


def test_normalisation_does_not_merge_different_values() -> None:
    assert normalise_value(1791) != normalise_value(1792)
    assert normalise_value(True) != normalise_value(1)
    assert normalise_value("nan") == "nan"


def test_a_prediction_with_no_evidence_is_matched_across_documents() -> None:
    uncited = Fact(
        subject=BABBAGE,
        predicate="employer",
        object_entity=Entity(key="c:cambridge", type="Company"),
    )
    report = evaluate_extraction(GOLD, [*PREDICTIONS, uncited])
    assert report.metrics["missing"] == 0
    assert report.metrics["tp"] == 3
    (hit,) = [o for o in match_extraction(GOLD, [uncited]) if o.kind == "correct"]
    assert hit.doc_id == "d2"


def test_a_prediction_from_an_unlabelled_document_is_not_scored() -> None:
    stray = _in("d99", Fact(subject=ADA, predicate="born", object_value=1815))
    report = evaluate_extraction(GOLD, [*PREDICTIONS, stray])
    assert report.metrics["spurious"] == 1
    assert any("1 prediction(s) cite only documents" in n for n in report.notes)


def test_a_cited_prediction_does_not_match_gold_in_another_document() -> None:
    """Babbage's employer in d1 is not evidence for Babbage's employer in d2."""
    wrong_doc = _in(
        "d1",
        Fact(
            subject=BABBAGE,
            predicate="employer",
            object_entity=Entity(key="c:cambridge", type="Company"),
        ),
    )
    kinds = [o.kind for o in match_extraction(GOLD, [wrong_doc])]
    assert "correct" not in kinds


def test_a_flipped_polarity_is_missing_and_spurious_never_a_near_miss() -> None:
    gold = [GoldFact(doc_id="d", fact=Fact(subject=ADA, predicate="sells", object_value="data"))]
    denied = _in("d", gold[0].fact.model_copy(update={"polarity": Polarity.DENIED}))
    assert sorted(o.kind for o in match_extraction(gold, [denied])) == ["missing", "spurious"]


def test_identity_qualifiers_keep_two_measurements_apart() -> None:
    p50 = Fact(
        subject=Entity(key="svc", type="Service"),
        predicate="uptime",
        object_value="99.9%",
        qualifiers={"percentile": "p50"},
        identity_keys=("percentile",),
    )
    p95 = _in("d", p50.model_copy(update={"qualifiers": {"percentile": "p95"}}))
    assert "correct" not in {
        o.kind for o in match_extraction([GoldFact(doc_id="d", fact=p50)], [p95])
    }
    # A reconcilable qualifier does not split the claim.
    later = _in("d", p50.model_copy(update={"qualifiers": {"percentile": "P50", "start": "2025"}}))
    (outcome,) = match_extraction([GoldFact(doc_id="d", fact=p50)], [later])
    assert outcome.kind == "correct"


def test_the_right_claim_about_the_wrong_subject_is_a_wrong_entity() -> None:
    gold = [GoldFact(doc_id="d", fact=Fact(subject=ADA, predicate="born", object_value=1815))]
    (outcome,) = match_extraction(
        gold, [_in("d", Fact(subject=BABBAGE, predicate="born", object_value=1815))]
    )
    assert outcome.kind == "wrong_entity"
    assert outcome.gold is not None and outcome.predicted is not None


class _YearExtractor:
    """Finds 'born in YYYY' and nothing else."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        words = chunk.text.rstrip(".").split()
        if "born" in words:
            yield Fact(
                subject=Entity(key=words[0].lower(), type="Person"),
                predicate="born",
                object_value=words[-1],
            )


def test_run_extract_cites_each_document_and_scores_the_extractor_alone() -> None:
    docs = [
        Document(id="d1", text="Ada was born in 1815."),
        Document(id="d2", text="Babbage was born in 1791."),
    ]
    predictions = run_extract(_YearExtractor(), docs, Ontology())
    assert [p.evidence[0].doc_id for p in predictions] == ["d1", "d2"]
    report = evaluate_extraction(GOLD, predictions)
    # Both birth years right; name and both employers missing.
    assert report.breakdown["born"]["f1"] == 1.0
    assert (report.metrics["tp"], report.metrics["missing"], report.metrics["spurious"]) == (
        2,
        3,
        0,
    )
