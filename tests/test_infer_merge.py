"""Merging near-duplicates (#33): one name kept, the rest as aliases, conflicts refused."""

from __future__ import annotations

from openodke.infer.candidates import PredicateCandidate, Proposals, TypeCandidate
from openodke.infer.merge import merge
from openodke.types import Span


def _spans(*doc_ids: str) -> tuple[Span, ...]:
    return tuple(Span(doc_id=d, start=0, end=1, quote="x") for d in doc_ids)


def _p(name: str, proposer: str = "records", **kw: object) -> PredicateCandidate:
    return PredicateCandidate(name=name, proposer=proposer, **kw)  # type: ignore[arg-type]


ADA, GRACE, ALAN = ("adalovelace", "acmecorp"), ("gracehopper", "globex"), ("alanturing", "initech")


def test_four_spellings_of_one_relation_become_one_predicate_with_three_aliases() -> None:
    """#33's done-when: works_at, employer, employed_by and worksFor."""
    found = Proposals(
        predicates=(
            _p(
                "employer",
                domain=("Person",),
                range="Company",
                observations=(ADA, GRACE, ALAN),
                evidence=_spans("r1", "r2", "r3"),
            ),
            _p(
                "works_at",
                "cooccurrence",
                domain=("Person",),
                range="Company",
                observations=(ADA, GRACE),
                evidence=_spans("n1"),
            ),
            _p(
                "works_for",
                aliases=("worksFor",),
                domain=("Person",),
                range="Company",
                observations=(ALAN, ADA),
                evidence=_spans("j1", "j2"),
            ),
            _p(
                "employed_by",
                "cooccurrence",
                domain=("Person",),
                range="Company",
                observations=(GRACE, ALAN),
                evidence=_spans("n2"),
            ),
        )
    )
    merged = merge(found)
    (employer,) = merged.proposals.predicates
    assert employer.name == "employer"
    assert set(employer.aliases) == {"works_at", "works_for", "employed_by"}
    assert len(employer.aliases) == 3
    assert employer.support == 7
    assert employer.proposer == "cooccurrence+records"
    (decision,) = merged.decisions
    assert decision.action == "merged" and decision.into == "employer"
    assert set(decision.names) == {"employer", "works_at", "works_for", "employed_by"}
    assert "shared (subject, value) pairs" in decision.reason


def test_two_near_duplicate_predicates_merge_with_the_loser_as_an_alias() -> None:
    found = Proposals(
        predicates=(
            _p("organisation_name", evidence=_spans("a", "b", "c")),
            _p("organization_name", "cooccurrence", evidence=_spans("d")),
        )
    )
    merged = merge(found)
    (kept,) = merged.proposals.predicates
    assert (kept.name, kept.aliases, kept.support) == (
        "organisation_name",
        ("organization_name",),
        4,
    )
    assert "similar names (0.94)" in merged.decisions[0].reason


def test_inflections_and_separators_are_one_name() -> None:
    found = Proposals(
        predicates=(
            _p("works_for", evidence=_spans("a", "b")),
            _p("worked_for", "cooccurrence", evidence=_spans("c")),
        )
    )
    (kept,) = merge(found).proposals.predicates
    assert kept.name == "works_for" and kept.aliases == ("worked_for",)
    assert "same name" in merge(found).decisions[0].reason


def test_a_range_conflict_blocks_a_merge_and_says_so() -> None:
    found = Proposals(
        predicates=(
            _p("organisation_founded", range="date", evidence=_spans("a")),
            _p("organization_founded", range="string", evidence=_spans("b")),
        )
    )
    merged = merge(found)
    assert [p.name for p in merged.proposals.predicates] == [
        "organisation_founded",
        "organization_founded",
    ]
    (decision,) = merged.decisions
    assert decision.action == "blocked"
    assert (
        "ranges conflict (organisation_founded: date, organization_founded: string)"
        in decision.reason
    )


def test_shared_evidence_does_not_merge_across_conflicting_entity_ranges() -> None:
    found = Proposals(
        predicates=(
            _p("employer", domain=("Person",), range="Company", observations=(ADA, GRACE)),
            _p("school", domain=("Person",), range="University", observations=(ADA, GRACE)),
        )
    )
    merged = merge(found)
    assert len(merged.proposals.predicates) == 2
    assert merged.decisions[0].action == "blocked"


def test_a_range_that_is_a_kind_of_the_other_widens_to_the_ancestor() -> None:
    found = Proposals(
        types=(
            TypeCandidate(name="Organisation", proposer="hearst"),
            TypeCandidate(name="Company", proposer="hearst", parents=("Organisation",)),
        ),
        predicates=(
            _p("employer", range="Company", evidence=_spans("a", "b")),
            _p("employers", range="Organisation", evidence=_spans("c")),
        ),
    )
    (kept,) = merge(found).proposals.predicates
    assert (kept.name, kept.range) == ("employer", "Organisation")


def test_numbers_and_times_widen_instead_of_conflicting() -> None:
    found = Proposals(
        predicates=(
            _p("price", range="integer", evidence=_spans("a", "b")),
            _p("prices", range="number", evidence=_spans("c")),
            _p("updated", range="date", evidence=_spans("d", "e")),
            _p("update", range="datetime", evidence=_spans("f")),
        )
    )
    ranges = {p.name: p.range for p in merge(found).proposals.predicates}
    assert ranges == {"price": "number", "updated": "datetime"}


def test_similar_names_on_different_domains_stay_two_predicates() -> None:
    found = Proposals(
        predicates=(
            _p("birth_place", domain=("Person",), evidence=_spans("a")),
            _p("berth_place", domain=("Ship",), evidence=_spans("b")),
        )
    )
    merged = merge(found)
    assert len(merged.proposals.predicates) == 2
    assert "domains do not overlap (Person / Ship)" in merged.decisions[0].reason


def test_names_that_fold_alike_cannot_both_stay() -> None:
    """`birthdate` would be matched as `birth_date` by extraction, whatever the schema said."""
    found = Proposals(
        predicates=(
            _p("birth_date", range="date", evidence=_spans("a", "b")),
            _p("birthdate", range="string", evidence=_spans("c")),
        )
    )
    merged = merge(found)
    (kept,) = merged.proposals.predicates
    assert kept.name == "birth_date"
    assert [d.action for d in merged.decisions] == ["blocked", "dropped"]


def test_the_same_name_on_two_domains_is_one_predicate_over_both() -> None:
    found = Proposals(
        predicates=(
            _p("name", domain=("Person",), evidence=_spans("a", "b")),
            _p("name", domain=("Company",), evidence=_spans("c")),
        )
    )
    (kept,) = merge(found).proposals.predicates
    assert kept.domain == ("Person", "Company")


def test_the_same_name_with_a_conflicting_range_keeps_the_better_supported_one() -> None:
    found = Proposals(
        predicates=(
            _p("employer", "cooccurrence", range="string", evidence=_spans("a")),
            _p("employer", range="Company", evidence=_spans("b", "c")),
        )
    )
    merged = merge(found)
    (kept,) = merged.proposals.predicates
    assert kept.range == "Company"
    assert [d.action for d in merged.decisions] == ["blocked", "dropped"]


def test_a_model_claim_merges_under_the_models_name() -> None:
    found = Proposals(
        predicates=(
            _p(
                "works_at",
                "cooccurrence",
                domain=("Person",),
                range="Company",
                evidence=_spans("a", "b", "c"),
            ),
            _p(
                "employer",
                "llm",
                domain=("Person",),
                range="Company",
                aliases=("works_at",),
                evidence=_spans("a"),
                rank=0,
            ),
        )
    )
    (kept,) = merge(found).proposals.predicates
    assert (kept.name, kept.aliases, kept.rank) == ("employer", ("works_at",), 0)
    assert "claimed as an alias" in merge(found).decisions[0].reason


def test_types_merge_by_name_and_rename_the_predicates_that_use_them() -> None:
    found = Proposals(
        types=(
            TypeCandidate(
                name="Company", proposer="records", keys=("name",), evidence=_spans("a", "b")
            ),
            TypeCandidate(
                name="Organisation",
                proposer="llm",
                aliases=("Company",),
                rank=0,
                evidence=_spans("a"),
            ),
            TypeCandidate(
                name="Companies", proposer="hearst", examples=("Acme",), evidence=_spans("c")
            ),
        ),
        predicates=(
            _p("employer", domain=("Person",), range="Company"),
            _p("founded", domain=("Companies",), range="integer"),
        ),
    )
    merged = merge(found).proposals
    (organisation,) = merged.types
    assert organisation.name == "Organisation"
    assert organisation.keys == ("name",) and organisation.examples == ("Acme",)
    assert set(organisation.aliases) == {"Company", "Companies"}
    assert {(p.name, p.domain, p.range) for p in merged.predicates} == {
        ("employer", ("Person",), "Organisation"),
        ("founded", ("Organisation",), "integer"),
    }


def test_a_type_is_never_merged_into_its_own_parent() -> None:
    found = Proposals(
        types=(
            TypeCandidate(name="Organisation", proposer="hearst"),
            TypeCandidate(name="Organization", proposer="hearst", parents=("Organisation",)),
        )
    )
    merged = merge(found)
    assert [t.name for t in merged.proposals.types] == ["Organisation", "Organization"]
    assert merged.decisions[0].reason.endswith("one is a kind of the other")


def test_merging_is_deterministic_and_leaves_unrelated_candidates_alone() -> None:
    found = Proposals(
        types=(
            TypeCandidate(name="Person", proposer="records"),
            TypeCandidate(name="City", proposer="records"),
        ),
        predicates=(_p("born", range="date"), _p("city", range="City")),
    )
    assert merge(found) == merge(found)
    assert merge(found).proposals == found
    assert merge(found).decisions == ()
