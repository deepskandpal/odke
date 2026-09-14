"""Validation agreement, and sink idempotency — write a graph twice, count once.

Validation numbers are hand-computed from tests/fixtures/eval/validate.*.jsonl —
six facts, not a benchmark:

    v1 accept   -> accept     v4 refuse   -> refuse
    v2 accept   -> refuse     v5 refuse   -> accept
    v3 accept   -> accept     v6 conflict -> conflict

Agreement 4/6. Both sides say accept 3, refuse 2, conflict 1 times, so chance
agreement is (9 + 4 + 1) / 36 = 14/36 and kappa = (24/36 - 14/36) / (22/36) = 5/11.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openodke import (
    Delegated,
    Entity,
    EntityLink,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Predicate,
    ValidationVerdict,
    Validator,
)
from openodke.eval import ValidationLabel, ValidationPrediction, load_jsonl
from openodke.eval.sinks import assert_idempotent, check_idempotency, jsonl_counts
from openodke.eval.validation import evaluate_validation, run_validate
from openodke.sinks import JsonlSink

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
LABELS = load_jsonl(FIXTURES / "validate.labels.jsonl", ValidationLabel)
PREDICTIONS = load_jsonl(FIXTURES / "validate.predictions.jsonl", ValidationPrediction)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_agreement_and_kappa_by_hand() -> None:
    report = evaluate_validation(LABELS, PREDICTIONS)
    assert report.stage == "validate" and report.n == 6
    assert report.metrics["agreement"] == pytest.approx(4 / 6)
    assert report.metrics["kappa"] == pytest.approx(5 / 11)


def test_per_action_and_the_costly_disagreements() -> None:
    report = evaluate_validation(LABELS, PREDICTIONS)
    b = report.breakdown
    # accept: tp v1 v3, fp v5, fn v2. refuse: tp v4, fp v2, fn v5. conflict: tp v6.
    assert b["accept"]["precision"] == pytest.approx(2 / 3)
    assert b["accept"]["recall"] == pytest.approx(2 / 3)
    assert (b["refuse"]["precision"], b["refuse"]["recall"]) == (0.5, 0.5)
    assert (b["conflict"]["precision"], b["conflict"]["recall"]) == (1.0, 1.0)
    assert report.metrics["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert report.metrics["wrongly_refused"] == 1
    assert report.metrics["wrongly_written"] == 1
    assert report.metrics["conflicts_missed"] == 0
    assert "1 fact(s) a person would write were refused" in report.notes
    assert report.confusion["accept"] == {"accept": 2, "refuse": 1, "conflict": 0}


def test_unjoined_rows_are_dropped_and_said_so() -> None:
    report = evaluate_validation(
        LABELS, [*PREDICTIONS[:5], ValidationPrediction(id="x", action="accept")]
    )
    assert report.n == 5
    assert any("1 labelled fact(s) had no prediction" in n for n in report.notes)
    assert any("1 prediction(s) matched no labelled fact" in n for n in report.notes)


class _DomainValidator:
    """Refuses an unknown predicate or a subject outside its domain. Blind to conflicts."""

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        predicate = ontology.predicates.get(fact.predicate)
        if predicate is None:
            return ValidationVerdict(action="refuse", reason=f"no predicate {fact.predicate}")
        if predicate.domain and fact.subject.type not in predicate.domain:
            return ValidationVerdict(action="refuse", reason="outside domain")
        return ValidationVerdict(action="accept")


ONTOLOGY = Ontology(
    predicates={
        "born": Predicate(name="born", domain=("Person",), range="integer"),
        "name": Predicate(name="name", domain=("Person",)),
    }
)


def test_run_validate_scores_a_validator_in_process() -> None:
    validator = _DomainValidator()
    assert isinstance(validator, Validator)
    report = evaluate_validation(LABELS, run_validate(validator, LABELS, ONTOLOGY))
    # Right on v1-v5; v6 is a conflict it cannot see and accepts.
    assert report.metrics["agreement"] == pytest.approx(5 / 6)
    assert report.metrics["conflicts_missed"] == 1
    assert report.metrics["wrongly_refused"] == 0


def test_a_delegated_validator_is_scored_the_same_way() -> None:
    """A platform that prunes accepts everything here; the evaluator says what that costs."""
    predictions = run_validate(Delegated(to="shacl-store"), LABELS, ONTOLOGY)
    assert predictions[0].reason == "delegated to shacl-store"
    report = evaluate_validation(LABELS, predictions)
    assert report.metrics["agreement"] == pytest.approx(3 / 6)
    # Accept-everything agrees with the majority class and nothing more.
    assert report.metrics["kappa"] == 0.0
    assert report.metrics["wrongly_written"] == 2


# --------------------------------------------------------------------------- #
# Sink idempotency
# --------------------------------------------------------------------------- #


def _graph() -> KnowledgeGraph:
    ada = Entity(key="ada", type="Person")
    company = Entity(key="c1", type="Company")
    return KnowledgeGraph(
        entities=(ada, company),
        facts=(
            Fact(subject=ada, predicate="name", object_value="Ada"),
            Fact(subject=ada, predicate="employer", object_entity=company),
        ),
        links=(EntityLink(source_key="ada", target_key="ada-l", kind=LinkKind.SIMILAR),),
    )


def test_the_jsonl_sink_writes_twice_and_counts_once(tmp_path) -> None:
    report = assert_idempotent(JsonlSink(tmp_path), _graph(), jsonl_counts(tmp_path))
    assert report.stage == "sink"
    assert report.metrics["changed"] == 0
    assert report.breakdown["edges"] == {"after_first": 1, "after_last": 1}
    assert report.breakdown["facts.jsonl"] == {"after_first": 2, "after_last": 2}
    assert report.notes == ("idempotent: every count unchanged over 2 writes",)


class _AppendingJsonlSink(JsonlSink):
    """Rewrites its manifest correctly and appends to its facts stream: the bug to catch."""

    def write(self, kg: KnowledgeGraph) -> None:
        facts = self.directory / "facts.jsonl"
        previous = facts.read_text(encoding="utf-8") if facts.exists() else ""
        super().write(kg)
        facts.write_text(previous + facts.read_text(encoding="utf-8"), encoding="utf-8")


def test_line_counts_catch_a_sink_whose_manifest_looks_fine(tmp_path) -> None:
    sink = _AppendingJsonlSink(tmp_path)
    report = check_idempotency(sink, _graph(), jsonl_counts(tmp_path), writes=3)
    assert report.metrics["changed"] == 1
    assert report.notes == ("not idempotent: facts.jsonl went from 2 to 6 over 3 writes",)
    assert json.loads((tmp_path / "manifest.json").read_text())["facts"] == 2
    with pytest.raises(AssertionError, match="facts.jsonl went from 2 to 4"):
        assert_idempotent(
            _AppendingJsonlSink(tmp_path / "again"), _graph(), jsonl_counts(tmp_path / "again")
        )


class _MemoryStore:
    """A store with MERGE or CREATE semantics, counted the way a graph database would be."""

    def __init__(self, merge: bool) -> None:
        self.merge = merge
        self.nodes: list[str] = []
        self.edges: list[tuple[str, ...]] = []

    def write(self, kg: KnowledgeGraph) -> None:
        for entity in kg.entities:
            if not (self.merge and entity.key in self.nodes):
                self.nodes.append(entity.key)
        for fact in kg.edges:
            if not (self.merge and fact.signature in self.edges):
                self.edges.append(fact.signature)

    def counts(self) -> dict[str, int]:
        return {"nodes": len(self.nodes), "relationships": len(self.edges)}


def test_the_helper_is_generic_over_any_sink_and_any_count() -> None:
    merging = _MemoryStore(merge=True)
    assert_idempotent(merging, _graph(), merging.counts)
    creating = _MemoryStore(merge=False)
    report = check_idempotency(creating, _graph(), creating.counts)
    assert report.metrics["changed"] == 2
    assert report.breakdown["nodes"] == {"after_first": 2, "after_last": 4}


def test_one_write_is_not_an_idempotency_check() -> None:
    store = _MemoryStore(merge=True)
    with pytest.raises(ValueError, match="at least two writes"):
        check_idempotency(store, _graph(), store.counts, writes=1)
