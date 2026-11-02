"""Near-duplicate sources (#154): a copied page counts once toward support.

Two documents behind one claim, from different sources, whose 5-word shingle
sets have a Jaccard similarity of at least the threshold are one source. The
evidence keeps both, and `odke.near_duplicates` says which documents were one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from openodke import Document, Entity, Evidence, Fact, KnowledgeGraph, Ontology, SourceTier
from openodke.corroborate import (
    CONFLICT,
    NEAR_DUPLICATES,
    EvidenceScorer,
    SignatureCorroborator,
    independent_sources,
)
from openodke.corroborate.duplicates import jaccard, shingles
from openodke.validator import Validator

NOW = datetime(2026, 9, 1, tzinfo=UTC)
HALDEN = Entity(key="c:halden", type="Company", label="Halden Robotics")
CLAIM = "Halden Robotics has its head office in Leeds."


def _words(start: int, count: int) -> list[str]:
    return [f"w{i}" for i in range(start, start + count)]


# A 200-word page, the same page re-saved elsewhere with one word changed, and an
# unrelated article that quotes one sentence of it.
PAGE = " ".join([CLAIM, *_words(0, 200)])
COPY = " ".join([CLAIM, *_words(0, 100), "edited", *_words(101, 99)])
QUOTES = " ".join([*_words(1000, 100), CLAIM, *_words(0, 15), *_words(2000, 100)])
OTHER = " ".join([CLAIM, *_words(3000, 200)])


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, text=text, uri=f"https://{doc_id}.example/halden")


DOCS = {d.id: d for d in (_doc("page", PAGE), _doc("copy", COPY), _doc("quotes", QUOTES))}


def _fact(*docs: str, value: str = "Leeds", tier: SourceTier = SourceTier.UNVERIFIED) -> list[Fact]:
    """One fact per document, each citing that document alone, as an extractor emits them."""
    return [
        Fact(
            subject=HALDEN,
            predicate="headquarters",
            object_value=value,
            evidence=(
                Evidence(doc_id=d, uri=f"https://{d}.example/halden", tier=tier, retrieved_at=NOW),
            ),
            confidence=0.8,
        )
        for d in docs
    ]


def test_a_copied_page_counts_once() -> None:
    """The #154 done-when: a near-duplicate pair gives support 1."""
    corroborator = SignatureCorroborator(documents=DOCS.values())
    (merged,) = corroborator.corroborate(_fact("page", "copy"))
    assert merged.support == 1
    # Both receipts are kept, and the group says why they count once.
    assert [e.doc_id for e in merged.evidence] == ["page", "copy"]
    assert merged.qualifiers[NEAR_DUPLICATES] == (("copy", "page"),)
    assert corroborator.stats["near_duplicates"] == {"compared": 1, "found": 1}
    # The scorer reads the same count, not the two hosts behind it.
    assert len(independent_sources(merged)) == 1
    assert EvidenceScorer().score(merged).support == 1


def test_two_documents_stating_the_same_fact_count_twice() -> None:
    docs = [*DOCS.values(), _doc("other", OTHER)]
    (merged,) = SignatureCorroborator(documents=docs).corroborate(_fact("page", "other"))
    assert merged.support == 2
    assert NEAR_DUPLICATES not in merged.qualifiers


def test_one_shared_sentence_does_not_make_a_copy() -> None:
    assert jaccard(shingles(PAGE), shingles(QUOTES)) < 0.1
    (merged,) = SignatureCorroborator(documents=DOCS.values()).corroborate(_fact("page", "quotes"))
    assert merged.support == 2


def test_the_threshold_is_inclusive_and_configurable() -> None:
    similarity = jaccard(shingles(PAGE), shingles(COPY))
    # One changed word changes five of 204 shingles: 199 shared of 209.
    assert similarity == pytest.approx(199 / 209)
    facts = _fact("page", "copy")

    def support(threshold: float | None) -> int:
        corroborator = SignatureCorroborator(documents=DOCS, near_duplicates=threshold)
        return corroborator.corroborate(facts)[0].support

    assert support(similarity) == 1
    assert support(similarity + 1e-9) == 2
    assert support(None) == 2
    # A tenth of the words changed is a different page at the default 0.9.
    edited = " ".join([CLAIM, *(w if i % 10 else "edited" for i, w in enumerate(_words(0, 200)))])
    docs = [_doc("page", PAGE), _doc("copy", edited)]
    assert SignatureCorroborator(documents=docs).corroborate(facts)[0].support == 2


def test_only_documents_behind_one_claim_are_compared() -> None:
    """Four copies of one page, two behind each claim: two comparisons, not six."""
    docs = [_doc(d, PAGE) for d in ("a", "b", "c", "d")]
    corroborator = SignatureCorroborator(documents=docs)
    facts = [*_fact("a", "b"), *_fact("c", "d", value="Sheffield")]
    leeds, sheffield = corroborator.corroborate(facts)
    assert (leeds.support, sheffield.support) == (1, 1)
    assert corroborator.stats["near_duplicates"] == {"compared": 2, "found": 2}

    # A pair behind several claims is compared once, and a pair from one host never.
    corroborator = SignatureCorroborator(documents=docs)
    same_host = Evidence(doc_id="b", uri="https://a.example/other", retrieved_at=NOW)
    facts = [
        *_fact("a", "b"),
        *_fact("a", "b", value="Leeds, UK"),
        *_fact("a", value="Yorkshire"),
        _fact("a", value="Yorkshire")[0].model_copy(update={"evidence": (same_host,)}),
    ]
    corroborator.corroborate(facts)
    assert corroborator.stats["near_duplicates"] == {"compared": 1, "found": 1}


def test_a_copy_counts_once_at_its_strongest_tier() -> None:
    """The group's evidence is all kept, so its trust is its best member's."""
    ontology = Ontology.from_dict(
        {"types": {"Company": {}}, "predicates": {"headquarters": {"domain": ["Company"]}}}
    )
    copies = _fact("copy") + _fact("page", tier=SourceTier.CURATED)
    rival = _fact("other", value="Sheffield", tier=SourceTier.AUTHORITATIVE)
    docs = [*DOCS.values(), _doc("other", OTHER)]
    leeds, sheffield = SignatureCorroborator(ontology, documents=docs).corroborate(copies + rival)
    assert (leeds.support, sheffield.support) == (1, 1)
    won = leeds.qualifiers[CONFLICT]
    assert won["status"] == "won"
    assert "'Leeds' is backed by 1 independent source (curated" in won["reason"]


def test_the_group_survives_json_and_a_run_without_the_texts() -> None:
    (merged,) = SignatureCorroborator(documents=DOCS).corroborate(_fact("page", "copy"))
    back = KnowledgeGraph.model_validate_json(KnowledgeGraph(facts=(merged,)).model_dump_json())
    loaded = back.facts[0]
    assert loaded.qualifiers[NEAR_DUPLICATES] == [["copy", "page"]]
    assert EvidenceScorer().score(loaded).support == 1
    # A later batch adds an independent source; the copy still counts once.
    again = SignatureCorroborator().corroborate([loaded, *_fact("other")])
    assert again[0].support == 2
    assert again[0].qualifiers[NEAR_DUPLICATES] == (("copy", "page"),)


def test_a_threshold_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(ValueError, match="near_duplicates"):
        SignatureCorroborator(near_duplicates=0.0)
    with pytest.raises(ValueError, match="near_duplicates"):
        SignatureCorroborator(near_duplicates=1.5)


def test_the_validator_hands_its_texts_to_the_corroborator() -> None:
    rows = [
        {"doc": d, "subject": "Halden Robotics", "predicate": "headquarters", "object": "Leeds",
         "quote": CLAIM}
        for d in ("page", "copy")
    ]  # fmt: skip
    kg, report = Validator().validate(rows, DOCS.values(), dry_run=True)
    (fact,) = kg.facts
    assert fact.support == 1
    assert fact.qualifiers[NEAR_DUPLICATES] == (("copy", "page"),)
    assert kg.stats["stages"]["corroborator"]["near_duplicates"] == {"compared": 1, "found": 1}


def test_odke_run_takes_the_threshold_and_hands_over_the_texts() -> None:
    from types import SimpleNamespace

    from openodke.run import ConfigError
    from openodke.run.build import BUILTINS
    from openodke.run.execute import register_documents

    factory = BUILTINS["corroborator"]["signature"]
    context = SimpleNamespace(ontology=Ontology())
    corroborator = factory({"near_duplicates": 0.95}, context, "stages.corroborator")
    assert corroborator.near_duplicates == 0.95
    register_documents(corroborator, list(DOCS.values()))
    assert set(corroborator.documents) == {"page", "copy", "quotes"}
    with pytest.raises(ConfigError, match="documents: set by odke run"):
        factory({"documents": []}, context, "stages.corroborator")


def test_a_group_is_decided_again_when_its_texts_are_in_hand() -> None:
    """A recorded group sticks only where the texts are missing, not over a new threshold."""
    (merged,) = SignatureCorroborator(documents=DOCS).corroborate(_fact("page", "copy"))
    stricter = SignatureCorroborator(documents=DOCS, near_duplicates=1.0)
    (again,) = stricter.corroborate([merged])
    assert again.support == 2
    assert NEAR_DUPLICATES not in again.qualifiers
