"""Where a pipeline loses facts (#140): every miss in one bucket, in the stated order.

Each test builds the smallest case that puts a miss in one bucket and checks it
lands there, and that the buckets sum to the scorer's own misses. The planted
causes, through the pipeline paths a user runs, are in `test_eval_planted.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openodke import Entity, Evidence, Fact, Ontology, Polarity
from openodke.eval.datasets import redocred
from openodke.eval.diagnosis import (
    BUCKETS,
    PRECISION,
    RECALL,
    UNCONFIRMED,
    Claim,
    Offered,
    diagnose,
    gold_view,
    near,
    offered_by,
    read_trace,
)
from openodke.eval.extraction import document_counts
from openodke.eval.formats import GoldFact
from openodke.extract.llm import LLMExtractor
from openodke.llm import ScriptedClient

ONTOLOGY = Ontology.model_validate(
    {
        "name": "t",
        "types": {"Person": {}, "Place": {}},
        "predicates": {
            "born_in": {"domain": ["Person"], "range": "Place", "importance": 1.0},
            "works_for": {"domain": ["Person"], "range": "Place", "importance": 0.5},
            "located_in": {"domain": ["Place"], "range": "Place", "inverse_of": "contains"},
            "contains": {"domain": ["Place"], "range": "Place"},
        },
    }
)
TEXT = (
    "Ada was born in London. Ada worked for Acme. London is in England. Paris is in France. "
    "Bob married Carol. Dan arrived. Eve left. Frank was born in Rome. Gina works for Hal. "
    "Ivy works for Hal too. Gina was born in Oslo."
)


def entity(name: str, kind: str = "Person") -> Entity:
    return Entity(key=name.lower(), type=kind, label=name)


def fact(s: str, p: str, o: str, doc: str = "d1", **extra: Any) -> Fact:
    kind = "Place" if p in ("located_in", "contains") else "Person"
    return Fact(
        subject=entity(s, kind),
        predicate=p,
        object_entity=entity(o, "Place"),
        evidence=(Evidence(doc_id=doc),),
        **extra,
    )


def gold(*facts: Fact) -> list[GoldFact]:
    return [GoldFact(doc_id=f.evidence[0].doc_id, fact=f) for f in facts]


GOLD = gold(
    fact("Ada", "born_in", "London"),  # found
    fact("Ada", "works_for", "Acme"),  # written as founded: wrong relation
    fact("London", "located_in", "England"),  # written England contains London: inverse
    fact("Paris", "located_in", "France"),  # France (country): surface form
    fact("Bob", "spouse", "Carol"),  # the same triple, negated: scored apart
    fact("Dan", "knows", "Eve"),  # never in one sentence: cross-sentence
    fact("Frank", "born_in", "Rome"),  # Frank in no prediction: entity never extracted
    fact("Gina", "works_for", "Hal"),  # both predicted, never together: not linked
)
SAID = [
    fact("Ada", "born_in", "London"),
    fact("Ada", "founded", "Acme"),
    fact("England", "contains", "London"),
    fact("Paris", "located_in", "France (country)"),
    fact("Bob", "spouse", "Carol", polarity=Polarity.DENIED),
    fact("Gina", "born_in", "Oslo"),
    fact("Ivy", "works_for", "Hal"),
]
BUCKET = [
    None,
    "wrong_relation",
    "inverse",
    "surface_form",
    "scored_apart",
    "cross_sentence",
    "entity_missing",
    "not_linked",
]


def assigned(found: Any) -> list[str | None]:
    return [found.assigned.get(g.id) for g in found.view.gold]


def test_each_miss_lands_in_the_first_bucket_its_evidence_fits() -> None:
    view = gold_view(GOLD, SAID, texts={"d1": TEXT}, row="extract")
    found = diagnose(view, inverses=ONTOLOGY.inverses)
    assert assigned(found) == BUCKET
    counts = {b.bucket: b.count for b in found.buckets if b.side == "recall"}
    # Unknown, not zero: no trace says what was offered, no gate's record was kept.
    assert counts["never_offered"] is None and counts["refused"] is None
    # The buckets sum to the scorer's misses, to the fact.
    misses = sum(fn for _, _, fn in document_counts(GOLD, SAID).by_doc.values())
    assert sum(c or 0 for c in counts.values()) == misses == found.misses == 7
    shares = [b.share for b in found.buckets if b.side == "recall" and b.count]
    assert sum(s or 0 for s in shares) == pytest.approx(1.0)
    surface = next(b for b in found.buckets if b.bucket == "surface_form")
    assert surface.label == UNCONFIRMED
    (example,) = surface.examples
    assert (example.gold, example.nearest) == (
        "Paris · located_in · France",
        "Paris · located_in · France (country)",
    )


def test_the_trace_and_the_gate_come_first() -> None:
    """Never offered beats wrong relation; a refused match beats a missing entity."""
    view = gold_view(
        GOLD,
        SAID,
        refused=[(fact("Frank", "born_in", "Rome"), "grounding verdict contradicted")],
        texts={"d1": TEXT},
    )
    shown = {"Person": ("born_in", "spouse", "knows"), "Place": ("located_in", "contains")}
    offered = Offered(by_type=shown, source="trace.json")
    found = diagnose(view, offered=offered, inverses=ONTOLOGY.inverses)
    expected = list(BUCKET)
    expected[1] = expected[7] = "never_offered"  # works_for is not in the Person snippet
    expected[6] = "refused"
    assert assigned(found) == expected
    refused = next(b for b in found.buckets if b.bucket == "refused")
    assert refused.examples[0].why == "grounding verdict contradicted"
    # Rome was refused, so Frank was extracted: an entity in a refused fact counts.
    assert "frank" in {s.subject for s in view.refused or ()}


def test_without_the_ontology_s_inverse_a_reversed_relation_is_a_wrong_one() -> None:
    found = diagnose(gold_view(GOLD, SAID, texts={"d1": TEXT}))
    assert assigned(found)[2] == "wrong_relation"
    assert found.swaps[("located_in", "contains")] == 1


def test_a_judge_confirms_or_sends_a_candidate_on() -> None:
    class Judge:
        def __init__(self, answer: bool | None) -> None:
            self.answer = answer
            self.asked: list[tuple[Claim, Claim]] = []

        def equivalent(self, gold: Claim, predicted: Claim, text: str | None) -> bool | None:
            self.asked.append((gold, predicted))
            return self.answer

    view = gold_view(GOLD, SAID, texts={"d1": TEXT})
    yes, no = Judge(True), Judge(False)
    confirmed = diagnose(view, inverses=ONTOLOGY.inverses, judge=yes)
    assert assigned(confirmed)[3] == "surface_form"
    paris = ("Paris", "located_in", "France"), ("Paris", "located_in", "France (country)")
    # With a judge every pair the pre-filter makes is asked, a near name or not:
    # Ivy works for Hal shares the relation and the object with Gina's.
    gina = ("Gina", "works_for", "Hal"), ("Ivy", "works_for", "Hal")
    assert yes.asked == [paris, gina]
    assert assigned(confirmed)[7] == "surface_form"
    label = next(b.label for b in confirmed.buckets if b.bucket == "surface_form")
    assert label == "surface form"
    # Rejected, the miss goes on: France is in no prediction under its own key.
    rejected = assigned(diagnose(view, inverses=ONTOLOGY.inverses, judge=no))
    assert (rejected[3], rejected[7]) == ("entity_missing", "not_linked")


def test_near_names() -> None:
    assert near("Palamu", "Palamu region")
    assert near("The Loud Tour", "Loud Tour")
    assert near("Zoë Smith", "Zoe  Smith")
    assert not near("Rome", "Romania")
    assert not near("1815", "1816")
    assert not near("", "x")


def test_the_false_positives_split_by_what_the_gate_did() -> None:
    spurious = fact("Ivy", "born_in", "Oslo", verdict="supported")
    stray = fact("Ivy", "born_in", "Rome", verdict="not_found")
    caught = fact("Hal", "born_in", "Oslo")
    view = gold_view(GOLD, [*SAID, spurious, stray], refused=[(caught, "contradicted")])
    found = diagnose(view)
    split = {b.bucket: b.count for b in found.buckets if b.side == "precision"}
    # Six of the seven predictions are wrong; with the two strays, eight written FPs,
    # one supported by its passage, and one refused.
    assert split == {"refused_not_in_gold": 1, "written_supported": 1, "written_other": 7}

    class Adjudicator:
        def supports(self, predicted: Claim, text: str | None) -> bool | None:
            return predicted[0] == "Gina"

    adjudicated = diagnose(view, adjudicator=Adjudicator())
    written = next(b for b in adjudicated.buckets if b.bucket == "written_supported")
    assert (written.count, written.label) == (1, "written, not in gold, supported (adjudicated)")
    assert written.examples[0].nearest == "Gina · born_in · Oslo"


def test_examples_spread_across_documents() -> None:
    labels = gold(
        *(fact(f"P{d}{i}", "born_in", f"C{d}{i}", doc=f"d{d}") for d in range(3) for i in range(3))
    )
    found = diagnose(gold_view(labels, []), examples=4)
    missing = next(b for b in found.buckets if b.bucket == "entity_missing")
    assert missing.count == 9
    assert [e.doc_id for e in missing.examples] == ["d0", "d1", "d2", "d0"]


def test_offered_reads_an_extractor_s_snippets_per_type() -> None:
    extractor = LLMExtractor(client=ScriptedClient([]), snippet_limit=1)
    offered = offered_by(extractor, ONTOLOGY, "x")
    assert offered is not None and offered.by_type == {
        "Person": ("born_in",),
        "Place": ("contains",),
    }
    assert offered.shown("born_in", "Person", str)
    assert not offered.shown("works_for", "Person", str)
    # A type the trace does not name is read against every snippet together.
    assert offered.shown("contains", None, str)
    assert offered_by(object(), ONTOLOGY) is None


def test_a_trace_is_a_manifest_or_the_run_config(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"stats": {"coverage": {"not_offered": ["works_for"]}}}))
    offered, config = read_trace(manifest)
    assert offered is not None and config is None
    assert offered.withheld == ("works_for",) and not offered.shown("works_for", "Person", str)
    unknown = tmp_path / "stats.json"
    unknown.write_text(json.dumps({"coverage": {"not_offered": None}}))
    assert read_trace(unknown) == (None, None)

    (tmp_path / "ontology.json").write_text(ONTOLOGY.model_dump_json())
    (tmp_path / "texts").mkdir()
    run = {
        "ontology": "ontology.json",
        "inputs": ["texts"],
        "stages": {"extractor": {"use": "llm", "snippet_limit": 1}},
    }
    (tmp_path / "odke.json").write_text(json.dumps(run))
    offered, config = read_trace(tmp_path / "odke.json")
    assert config == run
    assert offered is not None
    assert offered.by_type == {"Person": ("born_in",), "Place": ("contains",)}

    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="a trace is a run's manifest.json"):
        read_trace(tmp_path / "other.json")


# --------------------------------------------------------------------------- #
# Re-DocRED: the dataset's own matching, evidence and types
# --------------------------------------------------------------------------- #

DOCRED = [
    {
        "id": "d1",
        "text": (
            "Washington County is in Oregon. Raleigh Hills is a place. It is in Washington County."
        ),
        "entities": [["Washington County"], ["Oregon"], ["Raleigh Hills"], ["Portland"]],
        "types": ["Location", "Location", "Location", "Location"],
        "facts": [
            [0, "located in the administrative territorial entity", 1],
            [0, "contains administrative territorial entity", 2],
            [2, "located in the administrative territorial entity", 0],
            [3, "located in the administrative territorial entity", 1],
        ],
        "evidence": [[0], [1, 2], [], []],
    }
]


def test_the_re_docred_view_scores_as_the_dataset_does() -> None:
    predicted = {
        "d1": [
            ("Washington County", "located in the administrative territorial entity", "Oregon"),
            (
                "Raleigh Hills",
                "located in the administrative territorial entity",
                "Washington County",
            ),
            ("Oregon", "capital", "Salem"),
        ]
    }
    view = redocred.diagnosis_view(DOCRED, predicted, row="extract")
    (unit,) = redocred.documents(DOCRED, predicted)
    assert sum(g.found for g in view.gold) == unit["tp"] == 2
    assert sum(s.hit for s in view.said) == 2 and len(view.said) == unit["predicted"]
    assert view.gold[0].subject_type == "Location" and view.gold[1].evidence == (1, 2)
    located = "located_in_the_administrative_territorial_entity"
    contains = "contains_administrative_territorial_entity"
    found = diagnose(view, inverses={located: contains, contains: located})
    # The fact written the other way round is an inverse, though its evidence spans
    # two sentences; Portland is named nowhere, so it is an entity never extracted.
    assert [found.assigned.get(g.id) for g in view.gold] == [
        None,
        "inverse",
        None,
        "entity_missing",
    ]
    assert view.name("capital") == "capital"
    assert view.name(view.gold[0].relation) == "located in the administrative territorial entity"


def test_a_set_prepared_with_types_and_evidence_scores_the_same() -> None:
    plain = [{k: v for k, v in DOCRED[0].items() if k not in ("types", "evidence")}]
    swapped = ("Oregon", "contains administrative territorial entity", "Washington County")
    predicted = {"d1": [swapped]}
    assert redocred.documents(plain, predicted) == redocred.documents(DOCRED, predicted)
    view = redocred.diagnosis_view(plain, predicted)
    assert view.gold[0].subject_type is None and view.gold[1].evidence == ()


def test_every_bucket_is_named_once_in_order() -> None:
    assert [name for name, _ in RECALL][:2] == ["never_offered", "refused"]
    assert [name for name, _ in RECALL][-2:] == ["entity_missing", "not_linked"]
    assert len(BUCKETS) == len(RECALL) + len(PRECISION) == 13
