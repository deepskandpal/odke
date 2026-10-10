"""The ontology that checked each fact, and the schema slice its extractor saw (#163).

Every fact the gate lets through is stamped, and a reader can filter facts in
hand by the stamp. Model calls answer from recorded responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    Span,
    Validator,
)
from openodke.cli.main import app
from openodke.corroborate import ONTOLOGY, SCHEMA_SLICE, checked_by, checked_under
from openodke.eval.spans import load_facts

runner = CliRunner()
TEXT = "Ada Lovelace worked for Acme from 1833. Acme is based in London."
DOC = Document(id="d1", text=TEXT)
OLD = Ontology.from_dict(
    {
        "name": "people",
        "version": "1",
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "hq": {"domain": ["Company"]},
        },
    }
)
# The same predicates, one of them narrowed: a breaking change, and a new fingerprint.
NEW = Ontology.from_dict(
    {
        **OLD.model_dump(mode="json", include={"name", "types"}),
        "version": "2",
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "cardinality": "multi"},
            "hq": {"domain": ["Company"]},
        },
    }
)
ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace")
ACME = Entity(key="c:acme", type="Company", label="Acme")


def _cited(quote: str) -> Evidence:
    start = TEXT.index(quote)
    return Evidence(doc_id="d1", span=Span(doc_id="d1", start=start, end=start + len(quote)))


FACTS = [
    Fact(
        subject=ADA,
        predicate="employer",
        object_entity=ACME,
        evidence=(_cited("Ada Lovelace worked for Acme"),),
    ),
    Fact(
        subject=ACME,
        predicate="hq",
        object_value="London",
        evidence=(_cited("Acme is based in London"),),
    ),
]


class _Replay:
    """The facts above, handed back as an extractor's."""

    def extract(self, chunk: Any, ontology: Ontology) -> list[Fact]:
        return list(FACTS) if chunk.index == 0 else []


def _graph(ontology: Ontology = OLD) -> KnowledgeGraph:
    return Pipeline(ontology, _Replay()).run([DOC])


# --------------------------------------------------------------------------- #
# Every fact, stamped
# --------------------------------------------------------------------------- #


def test_every_fact_the_gate_lets_through_names_the_ontology_that_checked_it() -> None:
    kg = _graph()
    assert len(kg.facts) == 2
    assert {checked_by(fact) for fact in kg.facts} == {OLD.fingerprint}
    assert {fact.qualifiers[ONTOLOGY] for fact in kg.facts} == {OLD.fingerprint}
    # Not part of a fact's identity: the same claim, whatever checked it.
    assert {f.signature for f in kg.facts} == {f.signature for f in FACTS}
    # A schema with nothing in it checks nothing, and stamps nothing.
    unchecked = Pipeline(Ontology(), _Replay()).run([DOC]).facts
    assert unchecked and all(ONTOLOGY not in f.qualifiers for f in unchecked)


def test_a_fact_checked_again_names_the_ontology_that_checked_it_last() -> None:
    first = _graph(OLD)
    kg, report = Validator(NEW, grounder=_Supported()).validate(list(first.facts), [DOC])
    assert report.facts_out == 2
    assert {checked_by(fact) for fact in kg.facts} == {NEW.fingerprint}


class _Supported:
    """A grounder that finds every fact supported, so the job needs no model."""

    def ground(self, fact: Fact, doc: Document) -> Fact:
        from openodke import GroundingVerdict

        return fact.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})


def test_odke_run_stamps_every_fact_and_the_model_path_the_slice_it_saw(example: Path) -> None:
    result = runner.invoke(app, ["run", str(example / "odke.yaml")])
    assert result.exit_code == 0, result.output
    facts = load_facts(example / "out")
    ontology = Ontology.from_json(example / "ontology.json")
    assert {checked_by(fact) for fact in facts} == {ontology.fingerprint}
    sliced = [fact for fact in facts if SCHEMA_SLICE in fact.qualifiers]
    # The facts the model extracted carry the slice it was shown for their subject's type.
    assert sliced and all(
        fact.qualifiers[SCHEMA_SLICE] == ontology.snippet(fact.subject.type).fingerprint
        for fact in sliced
    )


# --------------------------------------------------------------------------- #
# A reader filters by it
# --------------------------------------------------------------------------- #


def test_the_facts_one_ontology_checked_are_a_filter_away() -> None:
    old, new = _graph(OLD).facts, _graph(NEW).facts
    mixed = [*old, new[0]]
    assert checked_under(mixed, OLD) == list(old)
    assert checked_under(mixed, NEW.fingerprint) == [new[0]]
    assert checked_under(FACTS, OLD) == []
