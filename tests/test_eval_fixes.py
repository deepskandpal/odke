"""What should change (#141): each fix's expected gain is arithmetic on the bucket counts.

Every number here is worked by hand from the counts in the test, so a change to
the arithmetic shows up as a changed number, not as a changed opinion.
"""

from __future__ import annotations

from typing import Any

import pytest

from openodke import Entity, Evidence, Fact, Ontology
from openodke.eval.diagnosis import Claim, Offered, diagnose, gold_view
from openodke.eval.fixes import CATALOGUE, REMEDIES, fixes, render
from openodke.eval.formats import GoldFact

INVERSES = Ontology.model_validate(
    {
        "name": "t",
        "types": {"Place": {}},
        "predicates": {
            "located_in": {"domain": ["Place"], "range": "Place", "inverse_of": "contains"},
            "contains": {"domain": ["Place"], "range": "Place"},
        },
    }
).inverses


def fact(s: str, p: str, o: str, doc: str = "d1", **extra: Any) -> Fact:
    return Fact(
        subject=Entity(key=s.lower(), type="Person", label=s),
        predicate=p,
        object_entity=Entity(key=o.lower(), type="Place", label=o),
        evidence=(Evidence(doc_id=doc),),
        **extra,
    )


def gold(*facts: Fact) -> list[GoldFact]:
    return [GoldFact(doc_id=f.evidence[0].doc_id, fact=f) for f in facts]


def by_id(found: list[Any]) -> dict[str, Any]:
    return {fix.id: fix for fix in found}


def test_a_relation_never_offered_is_priced_at_the_recall_on_those_that_were() -> None:
    # Ten gold facts: four use `award`, which was never offered; of the six that
    # were, three were found. 4 × 3/6 / 10 = +20 points expected, +40 at most.
    labels = gold(
        *(fact(f"P{i}", "award", f"A{i}") for i in range(4)),
        *(fact(f"Q{i}", "born_in", f"C{i}") for i in range(6)),
    )
    said = [fact(f"Q{i}", "born_in", f"C{i}") for i in range(3)]
    offered = Offered(by_type={"Person": ("born_in",)})
    found = diagnose(gold_view(labels, said), offered=offered)
    (offer, reextract) = fixes(found)
    assert (offer.id, offer.rank, offer.misses) == ("offer-relations", 1, 4)
    assert offer.recall.expected == pytest.approx(0.2)
    assert offer.recall.ceiling == pytest.approx(0.4) and not offer.recall.exact
    assert offer.recall.basis == (
        "4 misses × 50.0% (its recall on the relations it was offered: 3 of 6) / 10 gold"
    )
    assert offer.detail == ("Person · award: 4",)
    assert offer.knob == "stages.extractor.snippet_limit"
    # The other three misses are entities never extracted, at the run's recall: 3 × 0.3 / 10.
    assert (reextract.id, reextract.misses) == ("reextract", 3)
    assert reextract.recall.expected == pytest.approx(0.09)


def test_accepting_the_refusals_is_replayed_exactly() -> None:
    labels = gold(fact("Ada", "born_in", "London"), fact("Bob", "born_in", "Rome"))
    said = [fact("Ada", "born_in", "London")]
    refused = [
        (fact("Bob", "born_in", "Rome"), "grounding verdict contradicted"),
        (fact("Bob", "born_in", "Paris"), "grounding verdict contradicted"),
    ]
    found = diagnose(gold_view(labels, said, refused=refused))
    (audit,) = fixes(found)
    assert audit.id == "audit-refusals" and audit.recall.exact
    assert audit.recall.expected == audit.recall.ceiling == 0.5
    # Precision 1/1 now; with both refusals written, 2 of 3.
    assert audit.recall.precision == pytest.approx(2 / 3 - 1)
    assert "accepting the 2 refused facts writes back 1 gold fact" in audit.recall.basis
    assert "precision 100.0% → 66.7%" in audit.recall.basis


def test_the_inverse_step_is_replayed_on_the_output() -> None:
    # The ontology declares the pair; the output wrote one fact as its inverse, and
    # two more that the gold does not hold either way.
    labels = gold(
        fact("London", "located_in", "England"),
        fact("Paris", "located_in", "France"),
    )
    said = [
        fact("England", "contains", "London"),
        fact("Paris", "located_in", "France"),
        fact("Spain", "contains", "Madrid"),
        fact("Rome", "located_in", "Italy"),
    ]
    found = diagnose(gold_view(labels, said), inverses=INVERSES)
    assert found.count("inverse") == 1
    (inverses,) = fixes(found)
    assert inverses.id == "inverses" and inverses.recall.exact
    assert inverses.recall.expected == inverses.recall.ceiling == 0.5
    # Every written edge gains a partner, four, and one of them is the missed gold
    # fact: precision 1/4 → 2/8, no change.
    assert inverses.recall.precision == pytest.approx(0.0)
    assert inverses.detail == ("contains ↔ located_in (declared): 1 gold, 4 partners",)


def test_a_relation_written_backwards_is_priced_as_symmetric() -> None:
    labels = gold(fact("Ann", "spouse", "Ben"), fact("Cy", "spouse", "Di"))
    said = [fact("Ben", "spouse", "Ann"), fact("Cy", "spouse", "Di")]
    found = diagnose(gold_view(labels, said))
    assert found.count("inverse") == 1
    (inverses,) = fixes(found)
    assert inverses.recall.expected == 0.5
    assert inverses.detail == (
        "spouse symmetric (written the other way round): 1 gold, 2 partners",
    )


def test_a_near_name_is_worth_nothing_until_a_judge_confirms_it() -> None:
    labels = gold(fact("Ada", "born_in", "Palamu"), fact("Bob", "born_in", "Rome"))
    said = [fact("Ada", "born_in", "Palamu region"), fact("Bob", "born_in", "Rome")]
    unconfirmed = by_id(fixes(diagnose(gold_view(labels, said))))["normalise-names"]
    assert (unconfirmed.recall.expected, unconfirmed.recall.ceiling) == (0.0, 0.5)
    assert unconfirmed.recall.basis.startswith("unconfirmed: no fact-equivalence judge (#143)")

    class Yes:
        def equivalent(self, gold: Claim, predicted: Claim, text: str | None) -> bool | None:
            return True

    confirmed = by_id(fixes(diagnose(gold_view(labels, said), judge=Yes())))["normalise-names"]
    assert confirmed.recall.exact and confirmed.recall.expected == 0.5


def test_a_reference_run_prices_each_fix_at_its_rate_on_the_same_facts() -> None:
    labels = gold(*(fact(f"P{i}", "award", f"A{i}") for i in range(4)), fact("Q", "born_in", "C"))
    found = diagnose(gold_view(labels, []), offered=Offered(by_type={"Person": ("born_in",)}))
    reference = gold_view(labels, [fact("P0", "award", "A0"), fact("P1", "award", "A1")])
    caught = {g.id for g in reference.gold if g.found}
    offer = by_id(fixes(found, reference=caught))["offer-relations"]
    assert offer.recall.reference == pytest.approx(2 / 5)
    assert by_id(fixes(found))["offer-relations"].recall.reference is None


def test_fixes_are_ranked_and_a_fix_with_nothing_to_recover_is_left_out() -> None:
    labels = gold(
        fact("Ada", "born_in", "London"),
        fact("Bob", "works_for", "Acme"),
        fact("Cy", "born_in", "Rome"),
        fact("Di", "born_in", "Oslo"),
    )
    said = [fact("Ada", "born_in", "London"), fact("Bob", "founded", "Acme")]
    found = fixes(diagnose(gold_view(labels, said)))
    # Two entities never extracted beat one wrong relation, both at the run's 25%.
    assert [(f.rank, f.id, f.misses) for f in found] == [
        (1, "reextract", 2),
        (2, "relation-descriptions", 1),
    ]
    assert found[1].detail == ("works_for written as founded: 1",)
    text = "\n".join(render(found))
    assert "1  reextract" in text and "+12.5" in text
    assert "2. Tell the confused relations apart" in text
    assert render([]) == []


def test_the_catalogue_names_a_knob_for_every_bucket() -> None:
    from openodke.eval.diagnosis import RECALL

    answered = {bucket for remedy in CATALOGUE for bucket in remedy.buckets}
    assert answered == {name for name, _ in RECALL}
    assert all(remedy.knob for remedy in CATALOGUE)
    assert set(REMEDIES) == {remedy.id for remedy in CATALOGUE}
