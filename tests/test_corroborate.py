"""Corroboration: one claim counted once per independent source, and a contested
value decided on trust, freshness and volume-normalised agreement."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from openodke import (
    Corroborator,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    Ontology,
    Polarity,
    Predicate,
    Qualifier,
    SourceTier,
)
from openodke.corroborate import (
    CONFLICT,
    SOURCE_FORM,
    SignatureCorroborator,
    ValueNormalizer,
    independent_sources,
)

NOW = datetime(2026, 9, 1, tzinfo=UTC)
ACME = Entity(key="c1", type="Company", label="Acme")
ONTOLOGY = Ontology(
    predicates={
        "ceo": Predicate(name="ceo", cardinality="single"),
        "product": Predicate(name="product", cardinality="multi"),
    }
)


def _ev(
    doc: str,
    *,
    uri: str | None = None,
    tier: SourceTier = SourceTier.UNVERIFIED,
    at: datetime = NOW,
) -> Evidence:
    return Evidence(doc_id=doc, uri=uri, tier=tier, retrieved_at=at)


def _fact(value: Any, *evidence: Evidence, predicate: str = "ceo", **kw: Any) -> Fact:
    return Fact(
        subject=ACME,
        predicate=predicate,
        object_value=value,
        evidence=evidence,
        confidence=0.8,
        **kw,
    )


def _by_value(facts: Iterable[Fact]) -> dict[Any, Fact]:
    return {f.object_value: f for f in facts}


def test_three_spellings_of_one_date_become_one_fact_with_support_three() -> None:
    """The #18 done-when, end to end: normalise, then corroborate."""
    normalizer = ValueNormalizer()
    facts = [
        normalizer.normalize(_fact(text, _ev(doc), predicate="born"))
        for doc, text in (("d1", "10 December 1815"), ("d2", "1815-12-10"), ("d3", "Dec 10, 1815"))
    ]
    (merged,) = SignatureCorroborator().corroborate(facts)
    assert merged.object_value == "1815-12-10"
    assert merged.support == 3
    assert [e.doc_id for e in merged.evidence] == ["d1", "d2", "d3"]
    assert merged.qualifiers[SOURCE_FORM] == {"object_value": ("10 December 1815", "Dec 10, 1815")}


def test_a_denial_and_its_assertion_never_merge() -> None:
    asserted = _fact("customer_data", _ev("d1"), predicate="sells")
    denied = _fact("customer_data", _ev("d2"), predicate="sells", polarity=Polarity.DENIED)
    out = SignatureCorroborator().corroborate([asserted, denied])
    assert sorted(f.polarity.value for f in out) == ["asserted", "denied"]
    assert all(f.support == 1 for f in out)
    # Equal standing: neither is chosen, and both say so.
    assert {f.qualifiers[CONFLICT]["status"] for f in out} == {"tied"}
    assert all(f.confidence == 0.8 for f in out)


def test_a_curated_denial_outranks_an_unverified_assertion() -> None:
    asserted = _fact("customer_data", _ev("d1"), predicate="sells")
    denied = _fact(
        "customer_data",
        _ev("d2", tier=SourceTier.CURATED),
        predicate="sells",
        polarity=Polarity.DENIED,
    )
    out = {f.polarity: f for f in SignatureCorroborator().corroborate([asserted, denied])}
    lost = out[Polarity.ASSERTED].qualifiers[CONFLICT]
    assert lost["status"] == "lost"
    assert lost["to"] == "not 'customer_data'"
    assert out[Polarity.ASSERTED].confidence == pytest.approx(0.8 * 0.2)
    assert out[Polarity.DENIED].qualifiers[CONFLICT]["status"] == "won"


def test_identity_qualifiers_split_and_reconcilable_ones_take_the_interval_union() -> None:
    def uptime(doc: str, **qualifiers: str) -> Fact:
        return Fact(
            subject=ACME,
            predicate="uptime",
            object_value="99.9 %",
            qualifiers=qualifiers,
            identity_keys=("percentile",),
            evidence=(_ev(doc),),
        )

    out = SignatureCorroborator().corroborate(
        [
            uptime("d1", percentile="p50", start_time="2020-03-01"),
            uptime("d2", percentile="p50", start_time="2019-01-01", end_time="2024-01-01"),
            uptime("d3", percentile="p95", start_time="2019-01-01"),
        ]
    )
    by_percentile = {f.qualifiers["percentile"]: f for f in out}
    assert (by_percentile["p50"].support, by_percentile["p95"].support) == (2, 1)
    assert by_percentile["p50"].qualifiers == {
        "percentile": "p50",
        "start_time": "2019-01-01",
        "end_time": "2024-01-01",
    }


def test_a_reconcilable_qualifier_that_is_not_a_bound_takes_the_most_trusted_value() -> None:
    low = _fact("Jane Doe", _ev("d1"), predicate="role", qualifiers={"rank": "normal"})
    high = _fact(
        "Jane Doe",
        _ev("d2", tier=SourceTier.CURATED),
        predicate="role",
        qualifiers={"rank": "preferred"},
    )
    (merged,) = SignatureCorroborator().corroborate([low, high])
    assert merged.qualifiers["rank"] == "preferred"


def test_the_valid_clock_is_reconciled_over_the_bounds_the_sources_gave() -> None:
    """'CEO since 2019' and 'CEO 2019-2024' are one claim; the end date is information."""
    since = _fact("Jane Doe", _ev("d1"), valid_from=datetime(2019, 1, 1, tzinfo=UTC))
    closed = _fact(
        "Jane Doe",
        _ev("d2"),
        valid_from=datetime(2019, 6, 1, tzinfo=UTC),
        valid_to=datetime(2024, 1, 1, tzinfo=UTC),
    )
    (merged,) = SignatureCorroborator(ONTOLOGY).corroborate([since, closed])
    assert merged.valid_from == datetime(2019, 1, 1, tzinfo=UTC)
    assert merged.valid_to == datetime(2024, 1, 1, tzinfo=UTC)
    assert CONFLICT not in merged.qualifiers


def test_support_counts_independent_sources_not_pages_or_chunks() -> None:
    pages = [_ev(f"d{i}", uri=f"https://www.scraper.example/page/{i}") for i in range(3)]
    chunks = [_ev("filing"), _ev("filing")]
    (merged,) = SignatureCorroborator().corroborate([_fact("Jane Doe", e) for e in pages + chunks])
    assert merged.support == 2
    assert independent_sources(merged) == {"scraper.example", "doc:filing"}
    assert len(merged.evidence) == 4


def test_a_curated_older_source_beats_a_fresh_unverified_scrape_and_says_why() -> None:
    """The #20 done-when, including a reason a non-author can read."""
    curated = _fact(
        "Jane Doe", _ev("registry", tier=SourceTier.CURATED, at=NOW - timedelta(days=3 * 365))
    )
    scraped = _fact("John Roe", _ev("scrape", tier=SourceTier.UNVERIFIED, at=NOW))
    out = _by_value(SignatureCorroborator(ONTOLOGY).corroborate([scraped, curated]))
    assert len(out) == 2, "the loser is kept, never dropped"

    winner, loser = out["Jane Doe"], out["John Roe"]
    assert winner.qualifiers[CONFLICT]["status"] == "won"
    assert winner.confidence == 0.8
    lost = loser.qualifiers[CONFLICT]
    assert lost["status"] == "lost"
    assert lost["to"] == "'Jane Doe'"
    assert lost["confidence_before"] == 0.8
    assert loser.confidence == pytest.approx(0.8 * lost["ratio"], abs=1e-3)
    assert loser.confidence < 0.8
    assert lost["reason"].startswith(
        "Kept 'Jane Doe' over 'John Roe': 'Jane Doe' is backed by 1 independent source (curated, "
    )
    assert "'John Roe' by 1 independent source (unverified, retrieved 2026-09-01)" in lost["reason"]
    assert winner.qualifiers[CONFLICT]["reason"] == lost["reason"]


def test_single_values_are_contested_within_the_predicate_scope_keys() -> None:
    """The grouping the store's constraint uses: a price per tier is not a conflict across tiers."""
    ontology = Ontology(
        predicates={
            "price": Predicate(
                name="price", cardinality="single", qualifiers={"tier": Qualifier(identity=True)}
            )
        }
    )
    assert ontology.predicates["price"].scope_keys == ("tier",)

    def price(value: str, level: str, doc: str, trust: SourceTier = SourceTier.UNVERIFIED) -> Fact:
        # identity_keys deliberately left unstamped: the schema scopes the contest.
        return _fact(value, _ev(doc, tier=trust), predicate="price", qualifiers={"tier": level})

    across = SignatureCorroborator(ontology).corroborate(
        [price("10 USD", "gold", "d1"), price("5 USD", "silver", "d2", SourceTier.CURATED)]
    )
    assert all(CONFLICT not in f.qualifiers for f in across)

    within = _by_value(
        SignatureCorroborator(ontology).corroborate(
            [price("10 USD", "gold", "d1"), price("12 USD", "gold", "d2", SourceTier.CURATED)]
        )
    )
    assert within["10 USD"].qualifiers[CONFLICT]["status"] == "lost"
    assert within["12 USD"].qualifiers[CONFLICT]["status"] == "won"


def test_a_multi_valued_predicate_keeps_every_value() -> None:
    out = SignatureCorroborator(ONTOLOGY).corroborate(
        [
            _fact("Widgets", _ev("d1"), predicate="product"),
            _fact("Gadgets", _ev("d2", tier=SourceTier.CURATED), predicate="product"),
        ]
    )
    assert len(out) == 2
    assert all(CONFLICT not in f.qualifiers and f.confidence == 0.8 for f in out)


def test_without_an_ontology_no_value_is_contested() -> None:
    """Picking a winner among values that may all be true is the worse mistake."""
    out = SignatureCorroborator().corroborate(
        [_fact("Jane Doe", _ev("d1", tier=SourceTier.CURATED)), _fact("John Roe", _ev("d2"))]
    )
    assert all(CONFLICT not in f.qualifiers for f in out)


def test_agreement_is_volume_normalised_so_a_prolific_source_cannot_outvote_a_filing() -> None:
    filing = _fact(
        "Jane Doe",
        _ev("filing", uri="https://registry.example.gov/acme", tier=SourceTier.AUTHORITATIVE),
    )
    hosts = [f"https://site{i}.example/acme" for i in range(10)]
    echoes = [
        _fact("John Roe", _ev(f"page{i}", uri=host, tier=SourceTier.COMMUNITY))
        for i, host in enumerate(hosts)
    ]

    # Ten independent sources that say nothing else do outweigh one filing.
    quiet = _by_value(SignatureCorroborator(ONTOLOGY).corroborate([filing, *echoes]))
    assert quiet["Jane Doe"].qualifiers[CONFLICT]["status"] == "lost"

    # The same ten, each also emitting two hundred other facts, do not.
    chatter = [
        Fact(
            subject=Entity(key=f"s{i}", type="Page"),
            predicate="mentions",
            object_value=f"item {k}",
            evidence=(_ev(f"page{i}-{k}", uri=host, tier=SourceTier.COMMUNITY),),
        )
        for i, host in enumerate(hosts)
        for k in range(200)
    ]
    loud = SignatureCorroborator(ONTOLOGY).corroborate([filing, *echoes, *chatter])
    prolific = _by_value(f for f in loud if f.predicate == "ceo")
    assert prolific["Jane Doe"].qualifiers[CONFLICT]["status"] == "won"
    assert prolific["John Roe"].qualifiers[CONFLICT]["status"] == "lost"
    # Support stays the honest count; only its weight in a contest is normalised.
    assert prolific["John Roe"].support == 10


def test_a_value_that_changed_over_time_is_not_a_conflict() -> None:
    before = _fact(
        "John Roe",
        _ev("d1"),
        valid_from=datetime(2019, 1, 1, tzinfo=UTC),
        valid_to=datetime(2024, 1, 1, tzinfo=UTC),
    )
    after = _fact(
        "Jane Doe", _ev("d2", tier=SourceTier.CURATED), valid_from=datetime(2024, 1, 1, tzinfo=UTC)
    )
    out = SignatureCorroborator(ONTOLOGY).corroborate([before, after])
    assert all(CONFLICT not in f.qualifiers and f.confidence == 0.8 for f in out)


def test_the_merged_verdict_is_the_most_informative_one() -> None:
    def merged(*verdicts: GroundingVerdict) -> GroundingVerdict:
        facts = [_fact("Jane Doe", _ev(f"d{i}"), verdict=v) for i, v in enumerate(verdicts)]
        (fact,) = SignatureCorroborator().corroborate(facts)
        return fact.verdict

    V = GroundingVerdict
    assert merged(V.CONTRADICTED, V.SUPPORTED) is V.SUPPORTED
    assert merged(V.UNCHECKED, V.CONTRADICTED) is V.CONTRADICTED
    assert merged(V.UNCHECKED, V.NOT_FOUND) is V.NOT_FOUND
    assert merged(V.UNCHECKED, V.UNCHECKED) is V.UNCHECKED


def test_corroborating_twice_changes_nothing() -> None:
    """Conflict stamps are a function of the batch; a re-run must not compound them."""
    facts = [
        _fact("Jane Doe", _ev("registry", tier=SourceTier.CURATED, at=NOW - timedelta(days=900))),
        _fact("John Roe", _ev("scrape")),
        _fact("John Roe", _ev("scrape-2")),
        Fact(subject=ACME, predicate="founded", object_value=1911, support=2),
    ]
    corroborator = SignatureCorroborator(ONTOLOGY)

    def view(fs: Iterable[Fact]) -> list[tuple[Any, ...]]:
        return [(f.signature, f.support, round(f.confidence, 9), f.qualifiers) for f in fs]

    once = corroborator.corroborate(facts)
    assert view(corroborator.corroborate(once)) == view(once)
    assert _by_value(once)[1911].support == 2


def test_the_corroborator_satisfies_its_protocol_and_its_output_survives_json() -> None:
    corroborator = SignatureCorroborator(ONTOLOGY)
    assert isinstance(corroborator, Corroborator)
    out = corroborator.corroborate(
        [_fact("Jane Doe", _ev("a", tier=SourceTier.CURATED)), _fact("John Roe", _ev("b"))]
    )
    back = KnowledgeGraph.model_validate_json(KnowledgeGraph(facts=tuple(out)).model_dump_json())
    assert [f.qualifiers for f in back.facts] == [f.qualifiers for f in out]
    again = corroborator.corroborate(back.facts)
    assert [f.confidence for f in again] == pytest.approx([f.confidence for f in out])
