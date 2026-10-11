"""`bench/conflicts.py`: EnterpriseRAG-Bench's conflict cases, with no network and no model."""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from openodke import Entity, Evidence, Fact
from openodke.corroborate import SignatureCorroborator
from openodke.ontology import Ontology
from openodke.types import Polarity, SourceTier

BENCH = Path(__file__).parent.parent / "bench"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conflicts = _load("conflicts")

OLD, NEW = "dsid_5f3a672da4974781a5577b0f3d4993e9", "dsid_8ffbd9fe82df457ca67d4fbc9ff1090c"
QUESTIONS = [
    {
        "question_id": "qst_0411",
        "question_type": "conflicting_info",
        "question": "What share of burst credits is reserved for priority=high?",
        "expected_doc_ids": [OLD, NEW],
        "gold_answer": "30% (an earlier suggestion was 20%).",
    },
    {
        "question_id": "qst_0001",
        "question_type": "basic",
        "question": "Not a conflict.",
        "expected_doc_ids": ["dsid_elsewhere"],
        "gold_answer": "",
    },
]
DOCS = {
    OLD: {
        "title_field_name": "summary",
        "content_field_names": ["description", "comments"],
        "summary": "Throttle config",
        "description": "Reserve 20% for priority=high.",
        "comments": ["first", "second"],
        "created_at": "2026-03-10",
        "updated_at": "2026-03-12",
    },
    NEW: {
        "title_field_name": "title",
        "content_field_names": ["content"],
        "title": "Debug notes",
        "content": "Updated: reserve 30% for priority=high.",
        "last_modified": "2026-03-21",
    },
}


def _opener(asked: list[str]) -> Any:
    def open_url(url: str) -> io.BytesIO:
        asked.append(url)
        if url.endswith("/questions.jsonl"):
            return io.BytesIO("".join(json.dumps(q) + "\n" for q in QUESTIONS).encode())
        for dsid, doc in DOCS.items():
            if url.endswith(conflicts.PATHS[dsid]):
                return io.BytesIO(json.dumps(doc).encode())
        raise OSError(f"no fixture for {url}")

    return open_url


def test_fetch_keeps_the_conflict_questions_their_documents_and_the_canary(tmp_path: Path) -> None:
    asked: list[str] = []
    root = conflicts.fetch(tmp_path / "erb", opener=_opener(asked))
    assert all(conflicts.SHA in url for url in asked) and len(asked) == 3
    saved = json.loads((root / "questions.json").read_text())
    assert [q["question_id"] for q in saved["questions"]] == ["qst_0411"]
    assert saved["canary"] == conflicts.CANARY
    old = json.loads((root / "docs" / f"{OLD}.json").read_text())
    assert old["canary"] == conflicts.CANARY and old["source_type"] == "jira"
    conflicts.fetch(root, opener=_opener(asked))
    assert len(asked) == 4  # the documents are there already; only the questions again


def test_a_document_reads_as_the_benchmark_exports_it() -> None:
    assert conflicts.text_of(DOCS[OLD]) == (
        "Throttle config\n\nReserve 20% for priority=high.\nfirst\nsecond"
    )
    assert conflicts.time_of(DOCS[OLD]) == datetime(2026, 3, 12, tzinfo=UTC)
    slack = {"last_message_ts": "1751234640", "created_at": "2025-01-01"}
    assert conflicts.time_of(slack) == datetime.fromtimestamp(1751234640, tz=UTC)
    mail = {"last_email_at": "2026-08-29T16:45:00-07:00"}
    assert conflicts.time_of(mail) == datetime(2026, 8, 29, 23, 45, tzinfo=UTC)
    assert conflicts.time_of({}) is None
    assert conflicts.tier_of({"source_type": "confluence"}) is SourceTier.CURATED
    assert conflicts.tier_of({"source_type": "slack"}) is SourceTier.UNVERIFIED
    assert conflicts.tier_of({"source_type": "fax"}) is SourceTier.UNVERIFIED


def test_every_value_is_a_claim_about_the_cases_one_subject() -> None:
    class Inner:
        def extract(self, chunk: Any, ontology: Ontology) -> list[Fact]:
            name = Entity(key="Topic:dp-132-usw", type="Topic", label="dp-132-usw")
            return [Fact(subject=name, predicate=conflicts.PREDICATE, object_value=" 30 %. ")]

    (fact,) = conflicts.CaseSubject(Inner(), "Topic:qst_0411").extract(None, Ontology())
    # The case's key, the extractor's name for it: the grounder reads the name.
    assert (fact.subject.key, fact.subject.label) == ("Topic:qst_0411", "dp-132-usw")
    assert fact.object_value == "30 %"
    ((many,),) = conflicts.CaseSubject(Inner(), "Topic:qst_0411").extract_many([None], Ontology())
    assert (many.subject.key, many.object_value) == ("Topic:qst_0411", "30 %")


def _claim(value: str, doc: str, tier: SourceTier, when: datetime) -> Fact:
    return Fact(
        subject=Entity(key="Topic:qst_0411", type="Topic", label="qst_0411"),
        predicate=conflicts.PREDICATE,
        object_value=value,
        evidence=(Evidence(doc_id=doc, tier=tier, retrieved_at=when),),
    )


ONTOLOGY = Ontology.model_validate(conflicts.ontology_for("What share is reserved?"))
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _contest(facts: list[Fact], half_life: float = 365.0) -> list[Fact]:
    return SignatureCorroborator(ONTOLOGY, half_life_days=half_life).corroborate(facts)


def test_the_newer_and_better_document_wins() -> None:
    facts = [
        _claim("20%", OLD, SourceTier.COMMUNITY, T0),
        _claim("30%", NEW, SourceTier.AUTHORITATIVE, T0 + timedelta(days=9)),
    ]
    found = conflicts.outcome(_contest(facts), OLD, NEW)
    assert found["outcome"] == "won"
    assert found["reason"] == "ranked the newer document's value first"


def test_two_newer_values_that_tie_with_each_other_still_beat_the_older() -> None:
    facts = [
        _claim("Mondays by 08:00", OLD, SourceTier.UNVERIFIED, T0),
        _claim("Tuesdays by 07:00 PT", NEW, SourceTier.CURATED, T0 + timedelta(days=18)),
        _claim("Tuesdays by 07:00", NEW, SourceTier.CURATED, T0 + timedelta(days=18)),
    ]
    assert conflicts.outcome(_contest(facts), OLD, NEW)["outcome"] == "won"


def test_a_denial_is_not_contested_against_another_value() -> None:
    facts = [
        _claim("customer-managed keys", OLD, SourceTier.UNVERIFIED, T0).model_copy(
            update={"polarity": Polarity.DENIED}
        ),
        _claim("an add-on", NEW, SourceTier.CURATED, T0 + timedelta(days=49)),
    ]
    found = conflicts.outcome(_contest(facts), OLD, NEW)
    assert found["outcome"] == "no conflict found"
    assert "a denied value is not contested against another value" in found["reason"]


def test_a_tier_outweighs_freshness_until_the_half_life_is_short() -> None:
    """A published page beats a newer draft, unless age halves fast enough."""
    facts = [
        _claim("100k", OLD, SourceTier.CURATED, T0),
        _claim("250k", NEW, SourceTier.AUTHORITATIVE, T0 + timedelta(days=100)),
    ]
    assert conflicts.outcome(_contest(facts), OLD, NEW)["outcome"] == "lost"
    merged = _contest(facts)
    short = conflicts.recontest(merged, ONTOLOGY, conflicts.SHORT_HALF_LIFE)
    assert conflicts.outcome(short, OLD, NEW)["outcome"] == "won"


def test_one_value_from_both_documents_or_from_one_is_no_conflict() -> None:
    same = [
        _claim("30%", OLD, SourceTier.COMMUNITY, T0),
        _claim("30%", NEW, SourceTier.AUTHORITATIVE, T0),
    ]
    found = conflicts.outcome(_contest(same), OLD, NEW)
    assert found["outcome"] == "no conflict found"
    assert found["reason"] == "one value, both documents"
    alone = conflicts.outcome(_contest(same[1:]), OLD, NEW)
    assert alone["reason"] == "no value from the older document"
    assert conflicts.outcome([], OLD, NEW)["reason"] == "no value from either document"


def test_a_value_both_documents_state_is_read_by_hand() -> None:
    facts = [
        _claim("20%", OLD, SourceTier.COMMUNITY, T0),
        _claim("20%", NEW, SourceTier.AUTHORITATIVE, T0),
        _claim("30%", NEW, SourceTier.AUTHORITATIVE, T0),
    ]
    assert conflicts.outcome(_contest(facts), OLD, NEW)["outcome"] == "both"


def test_the_table_shows_each_case() -> None:
    results = {
        "cases": [
            {
                "question_id": "qst_0411",
                "documents": {
                    "older": {"source_type": "jira", "tier": "community"},
                    "newer": {"source_type": "google_drive", "tier": "authoritative"},
                },
                "same_document": False,
                "gated": [
                    {"value": "20%", "from": "older", "status": "lost", "gate": "refuse"},
                    {"value": "30%", "from": "newer", "status": "won", "gate": "accept"},
                ],
                "outcome": "won",
                "graph_holds": ["newer"],
                "half_life_30_days": "won",
            },
            {"question_id": "qst_0412", "outcome": "not run"},
        ]
    }
    lines = conflicts.table(results).splitlines()
    assert lines[2] == (
        "| qst_0411 | jira (community) → google_drive (authoritative) | "
        "20% (older, lost, refused); 30% (newer, won) | won | newer | won |"
    )
    assert lines[3] == "| qst_0412 | | | not run | | |"


def test_every_path_names_a_source_with_a_tier() -> None:
    assert len(conflicts.PATHS) == 39
    assert {path.split("/", 1)[0] for path in conflicts.PATHS.values()} <= set(conflicts.TIERS)
