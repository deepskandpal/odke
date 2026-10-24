"""Merging with the store: an incoming duplicate adds support, not an edge (#153, DECISIONS #35).

The corroborator reads what the store holds under each signature of the batch
through a `FactLookup` and merges it in as one more member of the claim, so a
support list grows and nothing else does. `JsonlSink(merge=True)` is a store
in a file. `Neo4jSink.stored` is checked here against the recording driver
(one read transaction, through the signature indexes, nothing written), and
against a real server in
`test_a_rerun_of_one_claim_from_a_second_source_adds_support_only_in_a_live_neo4j`
when `NEO4J_URI` is set.
"""

from __future__ import annotations

import json
import os
import re
import uuid
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    FactLookup,
    KnowledgeGraph,
    Ontology,
    SourceTier,
    Span,
    Validator,
)
from openodke.cli.main import app
from openodke.corroborate import (
    CONFLICT,
    DERIVED,
    SCORE,
    SignatureCorroborator,
    derived_from,
    partners,
)
from openodke.sinks import JsonlSink
from openodke.sinks.neo4j import (
    SHOW_INDEXES,
    Neo4jConstrainer,
    Neo4jSink,
    provenance_of,
    signature_of,
    stored_fact,
)
from openodke.stages import PassThroughGate, PassThroughGrounder
from test_neo4j_sink import FakeDriver

WHEN = datetime(2026, 9, 1, tzinfo=UTC)
ADA = Entity(key="p:ada", type="Person", label="Ada Lovelace")
ACME = Entity(key="c:acme", type="Company", label="Acme")
ONTOLOGY = Ontology.from_dict(
    {
        "types": {"Person": {}, "Company": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company"},
            "employs": {"domain": ["Company"], "range": "Person", "inverse_of": "employer"},
            "hq": {"domain": ["Company"]},
        },
    }
)
TEXT = "Ada Lovelace works at Acme. Acme is based in Leeds."


def _evidence(doc: str, uri: str | None = None) -> Evidence:
    return Evidence(
        doc_id=doc,
        span=Span(doc_id=doc, start=0, end=len(TEXT)),
        uri=uri,
        tier=SourceTier.COMMUNITY,
        retrieved_at=WHEN,
    )


def _employed(doc: str, **update: Any) -> Fact:
    fact = Fact(
        subject=ADA,
        predicate="employer",
        object_entity=ACME,
        evidence=(_evidence(doc),),
        confidence=0.8,
    )
    return fact.model_copy(update=update) if update else fact


def _hq(doc: str) -> Fact:
    return Fact(subject=ACME, predicate="hq", object_value="Leeds", evidence=(_evidence(doc),))


class Held:
    """A `FactLookup` over facts already in memory: what a store holds, by signature."""

    def __init__(self, *facts: Fact) -> None:
        self.facts = {fact.signature: fact for fact in facts}
        self.asked: list[int] = []

    def stored(self, facts: Sequence[Fact]) -> Mapping[tuple[Any, ...], Fact]:
        self.asked.append(len(facts))
        return {f.signature: self.facts[f.signature] for f in facts if f.signature in self.facts}


def _merged(store: Held, *facts: Fact, ontology: Ontology | None = None) -> list[Fact]:
    return list(SignatureCorroborator(ontology, store=store).corroborate(facts))


# --------------------------------------------------------------------------- #
# The corroborator
# --------------------------------------------------------------------------- #


def test_a_stored_fact_gains_the_batchs_source_and_nothing_else() -> None:
    (stored,) = SignatureCorroborator().corroborate([_employed("a")])
    store = Held(stored)
    assert isinstance(store, FactLookup)
    corroborator = SignatureCorroborator(store=store)
    (fact, hq) = corroborator.corroborate([_employed("b"), _hq("b")])
    assert [s.source for s in fact.supported_by] == ["doc:a", "doc:b"]
    assert fact.support == 2 and {e.doc_id for e in fact.evidence} == {"a", "b"}
    assert (fact.subject, fact.object_entity, fact.predicate) == (ADA, ACME, "employer")
    # A claim the store does not hold is the batch's alone.
    assert [s.source for s in hq.supported_by] == ["doc:b"]
    assert corroborator.stats["store"] == {"read": 2, "merged": 1, "dropped": 0}
    assert store.asked == [2]  # one read for the batch


def test_a_rerun_from_the_same_source_changes_nothing() -> None:
    (stored,) = SignatureCorroborator().corroborate([_employed("a")])
    (again,) = _merged(Held(stored), _employed("a"))
    assert (again.support, again.supported_by) == (stored.support, stored.supported_by)
    assert again.evidence == stored.evidence


def test_the_stored_facts_stamps_start_from_the_extractors_confidence() -> None:
    stamped = _employed(
        "a",
        confidence=0.2,
        qualifiers={SCORE: {"extractor": 0.8}, CONFLICT: {"status": "lost", "ratio": 0.25}},
    )
    (fact,) = _merged(Held(stamped), _employed("b", confidence=0.5))
    assert fact.confidence == 0.8
    assert SCORE not in fact.qualifiers and CONFLICT not in fact.qualifiers


def test_a_statement_wins_over_a_derived_twin_either_way() -> None:
    """DECISIONS #28 across the store: a stated claim stands on its own evidence."""
    stated = _employed("a")
    (partner,) = partners([stated], ONTOLOGY)
    # The store states "Acme employs Ada"; the batch only derives it: dropped.
    said = partner.model_copy(update={"qualifiers": {}})
    out = _merged(Held(said), stated, partner, ontology=ONTOLOGY)
    assert [f.predicate for f in out] == ["employer"]
    # The store derived it; the batch states it: the statement replaces it.
    statement = said.model_copy(update={"evidence": (_evidence("b"),)})
    (fact,) = _merged(Held(partner), statement, ontology=ONTOLOGY)
    assert [s.source for s in fact.supported_by] == ["doc:b"] and DERIVED not in fact.qualifiers


def test_a_derived_fact_keeps_its_parents_list_after_the_merge() -> None:
    """The store holds the parent and not its partner: the partner takes the merged list."""
    (stored,) = SignatureCorroborator().corroborate([_employed("a")])
    incoming = _employed("b")
    out = _merged(Held(stored), incoming, *partners([incoming], ONTOLOGY), ontology=ONTOLOGY)
    parent, partner = sorted(out, key=lambda f: f.predicate)
    assert derived_from(partner) == parent.signature
    assert partner.supported_by == parent.supported_by
    assert [s.source for s in partner.supported_by] == ["doc:a", "doc:b"]


# --------------------------------------------------------------------------- #
# JSONL: merging into the file is an option
# --------------------------------------------------------------------------- #


def _validate(sink: Any, doc: str, *facts: Fact) -> Any:
    validator = Validator(
        ONTOLOGY, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
    )
    return validator.validate(list(facts), [Document(id=doc, text=TEXT)])


def _lines(directory: Path) -> list[Fact]:
    text = (directory / "facts.jsonl").read_text(encoding="utf-8")
    return [Fact.model_validate_json(line) for line in text.splitlines()]


def test_jsonl_with_merge_keeps_the_file_and_grows_a_restated_facts_support(
    tmp_path: Path,
) -> None:
    sink = JsonlSink(tmp_path, merge=True)
    _validate(sink, "a", _employed("a"), _hq("a"))
    kg, report = _validate(sink, "b", _employed("b"))
    assert report.restated == 2  # the claim and its derived partner
    assert "merged        0 into a fact with the same signature; 2 into one" in report.render()
    by_predicate = {f.predicate: f for f in _lines(tmp_path)}
    assert set(by_predicate) == {"employer", "employs", "hq"}  # hq, from run 1, is kept
    assert [s.source for s in by_predicate["employer"].supported_by] == ["doc:a", "doc:b"]
    assert by_predicate["employs"].supported_by == by_predicate["employer"].supported_by
    assert [s.source for s in by_predicate["hq"].supported_by] == ["doc:a"]

    # Rerun: nothing moves.
    before = (tmp_path / "facts.jsonl").read_text(encoding="utf-8")
    _validate(sink, "b", _employed("b"))
    assert [(f.signature, f.supported_by) for f in _lines(tmp_path)] == [
        (f.signature, f.supported_by)
        for f in (Fact.model_validate_json(line) for line in before.splitlines())
    ]


def test_odke_run_with_merge_grows_a_restated_facts_support(tmp_path: Path) -> None:
    # odke run hands its signature corroborator the store, as the Validator does:
    # a second source's run adds to the stored fact rather than replacing it.
    examples = Path(__file__).parent.parent / "examples/triples"
    (tmp_path / "ontology.json").write_text(
        (examples / "ontology.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    texts = {
        "a": "Halden Robotics was founded in Leeds in 2014 by Mara Quist.",
        "b": "Mara Quist, the engineer, founded Halden Robotics after leaving Arden Labs.",
    }
    (tmp_path / "texts").mkdir()
    for doc in ("a", "b"):
        (tmp_path / "texts" / f"{doc}.txt").write_text(texts[doc], encoding="utf-8")
        row = {
            "doc": doc,
            "subject": "Halden Robotics",
            "subject_type": "Company",
            "predicate": "founder",
            "object": "Mara Quist",
            "object_type": "Person",
        }
        (tmp_path / f"{doc}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (tmp_path / f"{doc}.yaml").write_text(
            "ontology: ontology.json\n"
            "inputs:\n  - path: texts\n    loader: {use: directory}\n"
            "stages:\n"
            f"  extractor: {{use: triples, path: {doc}.jsonl}}\n"
            "  corroborator: signature\n"
            "  sink: {use: jsonl, directory: out, merge: true}\n",
            encoding="utf-8",
        )
    for doc in ("a", "b"):
        result = CliRunner().invoke(app, ["run", str(tmp_path / f"{doc}.yaml")])
        assert result.exit_code == 0, result.output
    (founder,) = [f for f in _lines(tmp_path / "out") if f.predicate == "founder"]
    assert len(founder.supported_by) == 2


def test_a_python_rerun_over_the_same_folder_is_the_same_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release audit 7: support counted reruns, and an update retracted nothing."""
    from openodke.loaders import DirectoryLoader

    (tmp_path / "texts").mkdir()
    (tmp_path / "texts" / "a.txt").write_text(TEXT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sink = JsonlSink(tmp_path / "store", merge=True)
    validator = Validator(
        ONTOLOGY, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
    )

    def rows() -> list[dict[str, str]]:
        return [{"doc": "a", "subject": "Ada Lovelace", "subject_type": "Person",
                 "predicate": "employer", "object": "Acme", "object_type": "Company"}]  # fmt: skip

    for _ in range(3):
        validator.validate(rows(), DirectoryLoader().load("texts"))
    (fact,) = [f for f in _lines(tmp_path / "store") if f.predicate == "employer"]
    assert fact.support == 1 and [s.doc_ids for s in fact.supported_by] == [("texts/a.txt",)]
    _, report = validator.validate([], DirectoryLoader().load("texts"), update=True)
    assert report.retracted is not None and report.retracted["cited"] == 2


def test_a_jsonl_file_is_never_read_half_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release audit S15: two writers into one directory crashed on a torn manifest.json.

    Each file is written beside its target and renamed into place, so whatever
    reads the directory while a write is under way finds the whole old file.
    """
    import openodke._files as files

    sink = JsonlSink(tmp_path, merge=True)
    _validate(sink, "a", _employed("a"))
    seen: list[str] = []
    real = files.os.replace

    def meanwhile(source: str, target: Any) -> None:
        held = Path(target)
        if held.is_file():
            text = held.read_text(encoding="utf-8")
            # Whole, so a second writer's read of it cannot fail.
            lines = [text] if held.suffix == ".json" else text.splitlines()
            assert all(isinstance(json.loads(line), dict) for line in lines)
            seen.append(held.name)
        real(source, target)

    monkeypatch.setattr(files.os, "replace", meanwhile)
    _validate(sink, "b", _employed("b"))
    assert {"manifest.json", "facts.jsonl", "entities.jsonl"} <= set(seen)
    assert not list(tmp_path.glob(".*.tmp"))


def test_a_file_written_whole_keeps_the_umasks_permissions(tmp_path: Path) -> None:
    """Written beside its target and renamed, a file is still the mode the umask gives.

    A temporary file from `tempfile.mkstemp` would be 0600, and the store's
    files and its manifest unreadable to the group a store is shared with.
    """
    old = os.umask(0o022)
    try:
        sink = JsonlSink(tmp_path / "store", merge=True)
        _, report = _validate(sink, "a", _employed("a"))
        assert report.manifest is not None
        report.manifest.write_into(tmp_path / "store" / "manifest.json")
        report.manifest.write(tmp_path / "run.json")
    finally:
        os.umask(old)
    written = [*sorted((tmp_path / "store").iterdir()), tmp_path / "run.json"]
    assert {p.name for p in written} >= {"manifest.json", "facts.jsonl", "entities.jsonl"}
    assert {p.name: oct(p.stat().st_mode & 0o777) for p in written} == {
        p.name: oct(0o644) for p in written
    }


def test_two_jsonl_writers_into_one_directory_at_once_do_not_crash(tmp_path: Path) -> None:
    """Release audit S15, under load: the writers race on every file, and none fails."""
    import threading

    failures: list[BaseException] = []

    def writer(name: str) -> None:
        try:
            for n in range(12):
                _validate(JsonlSink(tmp_path, merge=True), f"{name}{n}", _employed(f"{name}{n}"))
        except BaseException as exc:  # noqa: BLE001 - every failure is the finding
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(name,)) for name in "xyz"]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert _lines(tmp_path)


@pytest.mark.parametrize("first", ["supports", "contradicts"])
def test_a_contradicting_text_never_holds_a_stored_fact_up(first: str, tmp_path: Path) -> None:
    """Release audit 2, across the store, and the verifier's case: delete the supporting text.

    Text a supports the claim and b contradicts it, written in either order.
    The stored fact is supported by a alone, b is recorded against it, and once
    a is deleted the fact is retired: b, which says otherwise, does not keep it.
    """
    from openodke import GroundingVerdict, Reconciler

    sink = JsonlSink(tmp_path, merge=True)
    supports = ("a", _employed("a", verdict=GroundingVerdict.SUPPORTED))
    contradicts = ("b", _employed("b", verdict=GroundingVerdict.CONTRADICTED))
    for doc, fact in (supports, contradicts) if first == "supports" else (contradicts, supports):
        _validate(sink, doc, fact)
    (fact,) = [f for f in _lines(tmp_path) if f.predicate == "employer"]
    assert fact.verdict is GroundingVerdict.SUPPORTED
    assert [s.source for s in fact.supported_by] == ["doc:a"] and fact.support == 1
    assert fact.qualifiers[CONFLICT]["contradicted_in"] == ["b"]

    Reconciler([sink]).delete(["a"])
    (gone,) = [f for f in _lines(tmp_path) if f.predicate == "employer"]
    assert (gone.support, gone.supported_by, gone.retired_at is not None) == (0, (), True)


def test_neo4j_keeps_the_record_of_a_contradicting_text_through_a_merge() -> None:
    """The record is JSON text on the relationship, and reads back to merge again."""
    from openodke import GroundingVerdict

    (stored,) = SignatureCorroborator().corroborate(
        [_employed("a", verdict=GroundingVerdict.SUPPORTED),
         _employed("b", verdict=GroundingVerdict.CONTRADICTED)]
    )  # fmt: skip
    sink = Neo4jSink(driver=Stored(stored))
    (fact,) = SignatureCorroborator(store=sink).corroborate(
        [_employed("c", verdict=GroundingVerdict.SUPPORTED)]
    )
    assert [s.source for s in fact.supported_by] == ["doc:a", "doc:c"]
    assert fact.qualifiers[CONFLICT]["contradicted_in"] == ["b"]


def test_jsonl_without_merge_replaces_the_file_and_reads_nothing(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path)
    _validate(sink, "a", _employed("a"), _hq("a"))
    assert sink.stored([_employed("a")]) == {}
    _validate(sink, "b", _employed("b"))
    assert {f.predicate for f in _lines(tmp_path)} == {"employer", "employs"}
    (fact,) = [f for f in _lines(tmp_path) if f.predicate == "employer"]
    assert [s.source for s in fact.supported_by] == ["doc:b"]


def test_jsonl_with_merge_replaces_a_restated_entity_and_link_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    sink = JsonlSink(tmp_path, merge=True)
    sink.write(KnowledgeGraph(entities=(ADA, ACME)))
    renamed = ACME.model_copy(update={"label": "Acme Ltd"})
    sink.write(KnowledgeGraph(entities=(renamed,)))
    lines = (tmp_path / "entities.jsonl").read_text(encoding="utf-8").splitlines()
    assert [Entity.model_validate_json(line) for line in lines] == [ADA, renamed]


def test_odke_validate_merge_needs_a_directory(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["validate", "--facts", "x.jsonl", "--texts", str(tmp_path), "--merge"]
    )
    assert result.exit_code == 2
    assert "--merge merges into the JSONL at -o" in result.output


# --------------------------------------------------------------------------- #
# Neo4j, on the recording driver
# --------------------------------------------------------------------------- #


def _signature_index(predicate: str) -> dict[str, Any]:
    return {
        "name": f"odke_signature_{predicate}",
        "type": "RANGE",
        "entityType": "RELATIONSHIP",
        "labelsOrTypes": [predicate],
        "properties": ["signature"],
    }


class Stored(FakeDriver):
    """A store holding `held`, which answers each signature read with its relationship."""

    def __init__(self, *held: Fact, indexed: Sequence[str] = ("employer", "hq")) -> None:
        super().__init__({"SHOW INDEXES": [_signature_index(p) for p in indexed]})
        self.held = {signature_of(f): provenance_of(f, WHEN) for f in held}

    def answer(self, cypher: str) -> list[dict[str, Any]]:
        if "{signature: row.signature}" in cypher:
            rows = self.calls[-1][2]["rows"]
            return [
                {"signature": row["signature"], "props": self.held[row["signature"]]}
                for row in rows
                if row["signature"] in self.held
            ]
        return super().answer(cypher)


def test_neo4j_reads_a_batch_once_through_the_signature_indexes() -> None:
    (stored,) = SignatureCorroborator().corroborate([_employed("a", qualifiers={"since": 2020})])
    driver = Stored(stored)
    sink = Neo4jSink(driver=driver)
    assert isinstance(sink, FactLookup)
    found = sink.stored([_employed("b"), _hq("b")])
    # One auto-commit SHOW INDEXES, then one read transaction of one statement a predicate.
    assert [(mode, cypher) for mode, cypher, _ in driver.calls][0] == ("auto", SHOW_INDEXES)
    reads = [(cypher, params) for mode, cypher, params in driver.calls if mode == "read"]
    assert len(reads) == 2 and driver.writes == []
    assert all("UNWIND $rows AS row\nMATCH ()-[r:`" in cypher for cypher, _ in reads)
    (twin,) = found.values()
    assert twin.supported_by == stored.supported_by and twin.evidence == stored.evidence
    assert twin.qualifiers == {"since": 2020}
    assert (twin.id, twin.subject) == (stored.id, ADA)

    # The corroborator, handed the sink, merges with what it read.
    (fact,) = SignatureCorroborator(store=sink).corroborate([_employed("b")])
    assert [s.source for s in fact.supported_by] == ["doc:a", "doc:b"]


def test_a_predicate_without_a_signature_index_is_not_read_and_says_so() -> None:
    driver = Stored(indexed=("hq",))
    sink = Neo4jSink(driver=driver)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert sink.stored([_employed("b")]) == {}
    assert [str(w.message).split(",")[0] for w in caught] == [
        "the store has no index for employer.signature"
    ]
    assert [mode for mode, _, _ in driver.calls] == ["auto"]  # SHOW INDEXES, and nothing read


def test_bootstrap_makes_the_sink_read_its_indexes_again() -> None:
    driver = Stored(indexed=())
    sink = Neo4jSink(driver=driver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sink.stored([_employed("b")])
    driver.answers["SHOW INDEXES"] = [_signature_index("employer")]
    sink.stored([_employed("b")])
    assert sum(cypher == SHOW_INDEXES for _, cypher, _ in driver.calls) == 1
    sink.bootstrap(ONTOLOGY)
    sink.stored([_employed("b")])
    assert sum(cypher == SHOW_INDEXES for _, cypher, _ in driver.calls) == 2


def test_a_stored_fact_reads_back_what_provenance_wrote() -> None:
    stated = _employed("a", qualifiers={"odke.near_duplicates": (("a", "c"),), "rank": [1, 2]})
    (stored,) = SignatureCorroborator().corroborate([stated])
    twin = stored_fact(_employed("z"), provenance_of(stored, WHEN))
    # JSON, because a stamp that went in as tuples comes back as lists.
    assert twin.model_dump(mode="json") == stored.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


def _operators(plan: Mapping[str, Any] | None) -> list[str]:
    if not plan:
        return []
    found = [str(plan.get("operatorType", "")).split("@")[0]]
    for child in plan.get("children", ()):
        found += _operators(child)
    return found


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_a_rerun_of_one_claim_from_a_second_source_adds_support_only_in_a_live_neo4j() -> None:
    """Validate a claim from A, then from B: one edge, one claim node, and support grows to two.

    Types and predicates are suffixed, so a shared server is left as found.
    """
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company = f"Person_{suffix}", f"Company_{suffix}"
    employer, hq = f"employer_{suffix}", f"hq_{suffix}"
    ontology = Ontology.from_dict(
        {
            "types": {person: {}, company: {}},
            "predicates": {
                employer: {"domain": [person], "range": company},
                hq: {"domain": [company]},
            },
        }
    )
    ada = ADA.model_copy(update={"type": person, "key": f"p:ada:{suffix}"})
    acme = ACME.model_copy(update={"type": company, "key": f"c:acme:{suffix}"})

    def claims(doc: str) -> list[Fact]:
        return [
            Fact(subject=ada, predicate=employer, object_entity=acme, evidence=(_evidence(doc),)),
            Fact(subject=acme, predicate=hq, object_value="Leeds", evidence=(_evidence(doc),)),
        ]

    mine = (
        f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
        f"(n:Claim AND n.subject_type = '{company}') "
    )
    count = mine + (
        "OPTIONAL MATCH (n)-[r]-() RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
    )
    support = (
        f"MATCH (s)-[r]->() WHERE s:`{person}` OR s:`{company}` "
        "RETURN type(r) AS predicate, r.support AS support, r.support_sources AS sources, "
        "r.evidence_doc_ids AS docs"
    )
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(ontology)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth, ontology=ontology) as sink:
        driver = sink._driver
        validator = Validator(
            ontology, grounder=PassThroughGrounder(), gate=PassThroughGate(), sinks=[sink]
        )

        def run(doc: str) -> Any:
            return validator.validate(claims(doc), [Document(id=doc, text=TEXT)])[1]

        def read() -> dict[str, tuple[int, list[str], list[str]]]:
            return {
                r["predicate"]: (r["support"], list(r["sources"]), sorted(r["docs"]))
                for r in driver.execute_query(support).records
            }

        try:
            sink.bootstrap(ontology)
            driver.execute_query("CALL db.awaitIndexes(300)")
            run("doc-a")
            first = driver.execute_query(count).records[0].data()
            assert read() == {
                employer: (1, ["doc:doc-a"], ["doc-a"]),
                hq: (1, ["doc:doc-a"], ["doc-a"]),
            }

            report = run("doc-b")
            assert report.restated == 2
            # Support only: the same nodes and relationships, and both sources named.
            assert driver.execute_query(count).records[0].data() == first
            both = (2, ["doc:doc-a", "doc:doc-b"], ["doc-a", "doc-b"])
            assert read() == {employer: both, hq: both}

            # The same source again moves nothing.
            run("doc-b")
            assert driver.execute_query(count).records[0].data() == first
            assert read() == {employer: both, hq: both}

            # And the read went through the signature indexes, never a scan.
            reader = sink.lookup()
            for query in reader.fact_statements(claims("doc-c")):
                with driver.session() as session:
                    plan = session.run("EXPLAIN " + query.cypher, query.params).consume().plan
                operators = _operators(plan)
                # DirectedRelationshipUniqueIndexSeek: the constraint's own index.
                assert any("Relationship" in op and "IndexSeek" in op for op in operators), (
                    operators
                )
                assert not any("Scan" in op for op in operators), operators
        finally:
            driver.execute_query(mine + "DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")
