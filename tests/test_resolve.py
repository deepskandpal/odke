"""Resolution: links proposed, keys changed only on proof, and the disagreement
rule that kills a match however alike the names are."""

from __future__ import annotations

from collections.abc import Iterable

from openodke import (
    Chunk,
    Document,
    Entity,
    EntityLink,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Pipeline,
    Resolution,
    Resolver,
)
from openodke.corroborate import (
    NativeResolver,
    ValueNormalizer,
    candidate_pairs,
    domain_of,
    name_key,
    name_similarity,
)

UNMERGED = Resolution(method="linker", linker="odke.native")


def _resolve(
    *entities: Entity, resolver: NativeResolver | None = None
) -> tuple[list[Fact], list[EntityLink]]:
    facts = [Fact(subject=e, predicate="name", object_value=e.label) for e in entities]
    resolved, links = (resolver or NativeResolver()).resolve(facts, {})
    return list(resolved), list(links)


def test_a_disagreeing_registration_number_kills_a_near_identical_name() -> None:
    """The rule nobody else records: evidence against beats any name similarity."""
    de = Entity(key="acme-de", type="Company", label="Acme Corporation", external_id="DE-114322")
    gb = Entity(
        key="acme-gb", type="Company", label="Acme Corporation Ltd", external_id="GB-889401"
    )
    assert name_similarity(name_key("Acme Corporation"), name_key("Acme Corporation Ltd")) == 1.0

    facts, links = _resolve(de, gb)
    (link,) = links
    assert link.kind is LinkKind.DIFFERENT
    assert link.reason == "external_id mismatch: DE-114322 vs GB-889401"
    assert link.score == 1.0
    assert [f.subject.key for f in facts] == ["acme-de", "acme-gb"]
    assert all(f.subject.resolution == UNMERGED for f in facts)

    # The same two names with no identifiers are only a proposal.
    bare = [e.model_copy(update={"external_id": None}) for e in (de, gb)]
    facts, links = _resolve(*bare)
    assert [link.kind for link in links] == [LinkKind.SIMILAR]
    assert [f.subject.key for f in facts] == ["acme-de", "acme-gb"]


def test_two_different_people_with_the_same_name_are_not_merged() -> None:
    """The #19 near-miss pair: same name, two identifiers, two people."""
    a = Entity(key="p1", type="Person", label="John Smith", external_id="orcid:0000-0001")
    b = Entity(key="p2", type="Person", label="John Smith", external_id="orcid:0000-0002")
    facts, links = _resolve(a, b)
    assert [link.kind for link in links] == [LinkKind.DIFFERENT]
    assert links[0].reason == "external_id mismatch: orcid:0000-0001 vs orcid:0000-0002"
    assert {f.subject.key for f in facts} == {"p1", "p2"}


def test_a_weak_name_match_is_similar_with_a_score_and_never_re_keys() -> None:
    a = Entity(key="c1", type="Company", label="Acme Widgets")
    b = Entity(key="c2", type="Company", label="Acme Widget")
    facts, links = _resolve(a, b)
    (link,) = links
    assert link.kind is LinkKind.SIMILAR
    assert link.score is not None and 0.9 <= link.score < 1.0
    assert (link.source_key, link.target_key) == ("c1", "c2")
    assert [f.subject.key for f in facts] == ["c1", "c2"]


def test_a_shared_external_id_is_same_as_and_re_keys_onto_one_entity() -> None:
    """Proof, not resemblance: the names share nothing and the block on the id finds them."""
    long = Entity(
        key="org-2",
        type="Company",
        label="International Business Machines",
        external_id="wikidata:Q37156",
    )
    short = Entity(
        key="org-1",
        type="Company",
        label="IBM",
        aliases=("Big Blue",),
        external_id="wikidata:Q37156",
    )
    facts, links = _resolve(long, short)
    (link,) = links
    assert link.kind is LinkKind.SAME_AS
    assert link.reason == "external_id match: wikidata:Q37156"

    merged = facts[0].subject
    assert facts[1].subject is merged
    assert merged.key == "org-1"
    assert merged.label == "IBM"
    # The losing key and label survive as aliases; nothing is destroyed.
    assert merged.aliases == ("Big Blue", "org-2", "International Business Machines")
    assert merged.resolution == Resolution(method="external_id")
    assert [f.object_value for f in facts] == ["International Business Machines", "IBM"]


def test_identifiers_compare_without_spacing_or_case() -> None:
    a = Entity(key="c1", type="Company", label="Acme", external_id="DE-114 322")
    b = Entity(key="c2", type="Company", label="Globex", external_id="de-114322")
    _, links = _resolve(a, b)
    assert [link.kind for link in links] == [LinkKind.SAME_AS]


def test_identifiers_from_different_authorities_neither_match_nor_disagree() -> None:
    a = Entity(key="c1", type="Company", label="Acme", external_id="wikidata:Q1")
    b = Entity(key="c2", type="Company", label="Acme", external_id="lei:5493001KJTIIGC8Y1R12")
    _, links = _resolve(a, b)
    assert [link.kind for link in links] == [LinkKind.SIMILAR]


def test_a_shared_domain_alias_is_same_as_decided_by_the_linker() -> None:
    assert domain_of("https://www.acme.com/about") == "acme.com"
    assert domain_of("Acme Inc.") is None
    a = Entity(key="c1", type="Company", label="Acme", aliases=("https://www.acme.com/about",))
    b = Entity(key="c2", type="Company", label="Acme Widgets Group", aliases=("acme.com",))
    facts, links = _resolve(a, b)
    (link,) = links
    assert link.kind is LinkKind.SAME_AS
    assert link.reason == "domain match: acme.com"
    assert facts[1].subject.key == "c1"
    assert facts[1].subject.resolution == Resolution(
        method="linker", linker="odke.native", score=1.0
    )


def test_evidence_against_beats_evidence_for() -> None:
    """A shared domain and a disagreeing registration number: the disagreement wins."""
    a = Entity(key="c1", type="Company", label="Acme", aliases=("acme.com",), external_id="DE-1")
    b = Entity(key="c2", type="Company", label="Globex", aliases=("acme.com",), external_id="GB-2")
    facts, links = _resolve(a, b)
    (link,) = links
    assert link.kind is LinkKind.DIFFERENT
    assert link.reason == "external_id mismatch: DE-1 vs GB-2"
    assert [f.subject.key for f in facts] == ["c1", "c2"]


def test_a_merge_that_would_join_two_disagreeing_identifiers_is_refused() -> None:
    """Transitivity is how a third entity with no id welds two disagreeing ones together."""
    a = Entity(key="a", type="Company", label="Acme", aliases=("acme.com",), external_id="DE-1")
    b = Entity(key="b", type="Company", label="Acme", aliases=("acme.com",))
    c = Entity(key="c", type="Company", label="Acme", aliases=("acme.com",), external_id="GB-2")
    facts, links = _resolve(a, b, c)
    by_pair = {(link.source_key, link.target_key): link for link in links}
    assert by_pair[("a", "b")].kind is LinkKind.SAME_AS
    assert by_pair[("a", "c")].kind is LinkKind.DIFFERENT
    refused = by_pair[("b", "c")]
    assert refused.kind is LinkKind.DIFFERENT
    assert refused.reason is not None and "DE-1 vs GB-2" in refused.reason
    assert [f.subject.key for f in facts] == ["a", "a", "c"]


def test_non_identifying_attributes_nudge_the_score_across_the_threshold() -> None:
    def pair(country_a: str, country_b: str) -> list[EntityLink]:
        return _resolve(
            Entity(
                key="c1", type="Company", label="Acme Widgets", attributes={"country": country_a}
            ),
            Entity(
                key="c2", type="Company", label="Acme Widget", attributes={"country": country_b}
            ),
        )[1]

    (agreeing,) = pair("DE", "de")
    assert agreeing.kind is LinkKind.SIMILAR and agreeing.score == 1.0
    assert pair("DE", "US") == []


def test_an_initial_is_not_a_name_at_the_default_threshold() -> None:
    normalizer = ValueNormalizer(person_types=("Person",))
    facts = [
        normalizer.normalize(
            Fact(subject=Entity(key=k, type="Person", label=label), predicate="name")
        )
        for k, label in (("p1", "J. Smith"), ("p2", "Smith, John"))
    ]
    assert NativeResolver().resolve(facts, {})[1] == []
    (loose,) = NativeResolver(threshold=0.8).resolve(facts, {})[1]
    assert loose.kind is LinkKind.SIMILAR


def test_entities_of_different_types_are_never_compared() -> None:
    a = Entity(key="x1", type="Company", label="Jaguar", external_id="Q1")
    b = Entity(key="x2", type="Animal", label="Jaguar", external_id="Q2")
    facts, links = _resolve(a, b)
    assert links == []
    assert candidate_pairs([a, b]) == set()


def test_an_identity_the_caller_decided_is_the_one_kept() -> None:
    mine = Entity(
        key="z-mine",
        type="Company",
        label="Acme",
        external_id="Q1",
        resolution=Resolution(method="caller"),
    )
    found = Entity(key="a-found", type="Company", label="Acme Inc", external_id="Q1")
    facts, _ = _resolve(found, mine)
    assert {f.subject.key for f in facts} == {"z-mine"}
    assert facts[0].subject.resolution == Resolution(method="caller")
    assert "a-found" in facts[0].subject.aliases


def test_blocking_compares_within_blocks_not_across_the_corpus() -> None:
    """10,000 entities in blocks of ten: 45,000 comparisons, not 50 million."""
    entities = [
        Entity(key=f"e{i:05d}", type="Company", label=f"t{i // 10} n{i}") for i in range(10_000)
    ]
    pairs = candidate_pairs(entities)
    assert len(pairs) == 1_000 * 45
    assert len(pairs) < 10_000 * 9_999 // 2 // 1_000


def test_an_oversized_name_block_is_split_by_country_or_skipped() -> None:
    def banks(with_country: bool) -> list[Entity]:
        return [
            Entity(
                key=f"b{i:03d}",
                type="Company",
                label=f"Bank x{i}",
                attributes={"country": ["DE", "US", "GB"][i % 3]} if with_country else {},
            )
            for i in range(120)
        ]

    assert len(candidate_pairs(banks(True), max_block=50)) == 3 * (40 * 39 // 2)
    assert candidate_pairs(banks(False), max_block=50) == set()


def test_an_oversized_exact_block_is_a_star_and_still_merges_everyone() -> None:
    entities = [
        Entity(key=f"e{i:03d}", type="Company", label=f"alias{i} x{i}", external_id="Q42")
        for i in range(60)
    ]
    assert len(candidate_pairs(entities, max_block=10)) == 59
    facts, links = _resolve(*entities, resolver=NativeResolver(max_block=10))
    assert {f.subject.key for f in facts} == {"e000"}
    assert len(links) == 59 and {link.kind for link in links} == {LinkKind.SAME_AS}


def test_entities_already_in_the_index_are_resolved_against() -> None:
    stored = Entity(key="org:ibm", type="Company", label="IBM", external_id="Q37156")
    incoming = Fact(
        subject=Entity(key="tmp:1", type="Company", label="I.B.M.", external_id="Q37156"),
        predicate="founded",
        object_value="1911",
    )
    (fact,), links = NativeResolver().resolve([incoming], {stored.key: stored})
    assert fact.subject.key == "org:ibm"
    assert [link.kind for link in links] == [LinkKind.SAME_AS]


class _TwoSpellings:
    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        label = {"d1": "IBM", "d2": "International Business Machines Corp."}[chunk.doc_id]
        subject = Entity(
            key=f"org:{chunk.doc_id}", type="Company", label=label, external_id="Q37156"
        )
        return [Fact(subject=subject, predicate="founded", object_value="1911")]


def test_resolution_in_the_pipeline_makes_the_two_spellings_one_claim() -> None:
    """Signature merges on subject.key: without this, corroboration could never count both."""
    resolver = NativeResolver()
    assert isinstance(resolver, Resolver)
    kg = Pipeline(Ontology(), _TwoSpellings(), normalizer=ValueNormalizer(), resolver=resolver).run(
        [Document(id="d1", text="IBM"), Document(id="d2", text="IBM again")]
    )
    assert {e.key for e in kg.entities} == {"org:d1"}
    assert [link.kind for link in kg.links] == [LinkKind.SAME_AS]
    assert len({f.signature for f in kg.facts}) == 1
    assert KnowledgeGraph.model_validate_json(kg.model_dump_json()).links == kg.links
