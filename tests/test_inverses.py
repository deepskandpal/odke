"""Inverse and symmetric partners: the other direction of a stated fact, with no model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    Span,
    VerdictGate,
)
from openodke.corroborate import DERIVED, SignatureCorroborator, derived_from, partners
from openodke.run import execute, parse_config
from openodke.sinks import JsonlSink
from openodke.sinks.neo4j import plan

ONTOLOGY = Ontology.from_dict(
    {
        "types": {"Place": {}, "Person": {}},
        "predicates": {
            "located_in": {"domain": ["Place"], "range": "Place", "inverse_of": "contains"},
            "contains": {
                "domain": ["Place"],
                "range": "Place",
                "qualifiers": {"level": {"identity": True}},
            },
            "spouse": {"domain": ["Person"], "range": "Person", "symmetric": True},
            "born_in": {"domain": ["Person"], "range": "Place"},
            "population": {"domain": ["Place"], "range": "integer"},
        },
    }
)
BRITTANY = Entity(key="brittany", type="Place", label="Brittany")
FRANCE = Entity(key="france", type="Place", label="France")
ADA = Entity(key="ada", type="Person", label="Ada")
WILLIAM = Entity(key="william", type="Person", label="William")


def _evidence(doc: str) -> Evidence:
    return Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=24, quote=None))


def _fact(subject: Entity, predicate: str, obj: Entity, doc: str = "d1", **extra: object) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object_entity=obj,
        evidence=(_evidence(doc),),
        extractor="llm",
        confidence=0.8,
        verdict=GroundingVerdict.SUPPORTED,
        **extra,  # type: ignore[arg-type]
    )


def test_an_inverse_partner_is_the_same_claim_the_other_way_round() -> None:
    stated = _fact(BRITTANY, "located_in", FRANCE, qualifiers={"level": "region"})
    (partner,) = partners([stated], ONTOLOGY)
    assert (partner.subject, partner.predicate, partner.object_entity) == (
        FRANCE,
        "contains",
        BRITTANY,
    )
    # Same receipts, same verdict: never grounded again, and the gate treats both alike.
    assert partner.evidence == stated.evidence
    assert (partner.verdict, partner.confidence, partner.extractor) == (
        GroundingVerdict.SUPPORTED,
        0.8,
        "llm",
    )
    assert partner.id != stated.id
    # The inverse's own identity keys, so its signature is the inverse's.
    assert partner.identity_keys == ("level",)
    assert partner.signature[-1] == (("level", "'region'"),)
    assert partner.qualifiers[DERIVED] == {"rule": "inverse", "of": stated.signature}
    assert derived_from(partner) == stated.signature
    assert derived_from(stated) is None


def test_a_symmetric_partner_swaps_the_ends_and_keeps_the_predicate() -> None:
    stated = _fact(ADA, "spouse", WILLIAM)
    (partner,) = partners([stated], ONTOLOGY)
    assert (partner.subject.key, partner.predicate, partner.object_entity) == (
        "william",
        "spouse",
        ADA,
    )
    assert partner.qualifiers[DERIVED]["rule"] == "symmetric"


def test_nothing_is_derived_without_a_declared_pair_or_from_a_value() -> None:
    facts = [
        _fact(ADA, "born_in", FRANCE),
        Fact(subject=FRANCE, predicate="population", object_value=68_000_000),
        # A literal on an edge predicate is malformed, and still has no inverse.
        Fact(subject=BRITTANY, predicate="located_in", object_value="France"),
    ]
    assert partners(facts, ONTOLOGY) == []
    assert partners([_fact(BRITTANY, "located_in", FRANCE)], Ontology()) == []


def test_no_partner_when_the_batch_already_states_it() -> None:
    """The stated fact stands on its own evidence: no duplicate, with or without a corroborator."""
    both = [_fact(BRITTANY, "located_in", FRANCE), _fact(FRANCE, "contains", BRITTANY, doc="d2")]
    assert partners(both, ONTOLOGY) == []
    spouses = [_fact(ADA, "spouse", WILLIAM), _fact(WILLIAM, "spouse", ADA)]
    assert partners(spouses, ONTOLOGY) == []


def test_a_derived_fact_derives_nothing() -> None:
    """Run again over its own output, the step adds nothing: not even the fact it came from."""
    (partner,) = partners([_fact(BRITTANY, "located_in", FRANCE)], ONTOLOGY)
    assert partners([partner], ONTOLOGY) == []


def test_a_contradicted_fact_hands_its_verdict_to_its_partner() -> None:
    stated = _fact(BRITTANY, "located_in", FRANCE).model_copy(
        update={"verdict": GroundingVerdict.CONTRADICTED}
    )
    (partner,) = partners([stated], ONTOLOGY)
    assert partner.verdict is GroundingVerdict.CONTRADICTED


def test_a_partner_shares_its_source_support_and_never_merges_with_it() -> None:
    """Two documents state the claim: each partner cites one, and they merge as their sources do."""
    stated = [_fact(BRITTANY, "located_in", FRANCE, doc=doc) for doc in ("d1", "d2")]
    merged = SignatureCorroborator(ONTOLOGY).corroborate([*stated, *partners(stated, ONTOLOGY)])
    by_predicate = {fact.predicate: fact for fact in merged}
    assert len(merged) == 2
    source, partner = by_predicate["located_in"], by_predicate["contains"]
    assert source.support == partner.support == 2
    assert {e.doc_id for e in partner.evidence} == {"d1", "d2"}
    assert DERIVED not in source.qualifiers
    assert derived_from(partner) == source.signature


def test_the_link_survives_json() -> None:
    stated = _fact(
        FRANCE, "contains", BRITTANY, qualifiers={"level": "region"}, identity_keys=("level",)
    )
    (partner,) = partners([stated], ONTOLOGY)
    assert partner.predicate == "located_in" and partner.identity_keys == ()
    read_back = Fact.model_validate_json(partner.model_dump_json())
    assert read_back.qualifiers[DERIVED]["of"][-1] == [["level", "'region'"]]
    assert derived_from(read_back) == stated.signature


# --------------------------------------------------------------------------- #
# In a pipeline
# --------------------------------------------------------------------------- #

TEXT = "Brittany is a region of France. Ada married William."


class _Stated:
    """Says Brittany is in France and Ada married William, once each."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return [
            _fact(BRITTANY, "located_in", FRANCE, doc=chunk.doc_id),
            _fact(ADA, "spouse", WILLIAM, doc=chunk.doc_id),
        ]


class _Contradicts:
    """The passage says otherwise about where Brittany is."""

    def ground(self, fact: Fact, doc: Document) -> Fact:
        if fact.predicate == "located_in":
            return fact.model_copy(update={"verdict": GroundingVerdict.CONTRADICTED})
        return fact


def _claims(kg: KnowledgeGraph) -> set[tuple[str, str, str, bool]]:
    return {
        (f.subject.key, f.predicate, f.object_entity.key, DERIVED in f.qualifiers)
        for f in kg.facts
        if f.object_entity is not None
    }


def test_the_pipeline_adds_partners_when_the_ontology_declares_them() -> None:
    docs = [Document(id="d1", text=TEXT)]
    kg = Pipeline(ONTOLOGY, _Stated()).run(docs)
    assert _claims(kg) == {
        ("brittany", "located_in", "france", False),
        ("france", "contains", "brittany", True),
        ("ada", "spouse", "william", False),
        ("william", "spouse", "ada", True),
    }
    assert kg.stats["derived"] == 2

    off = Pipeline(ONTOLOGY, _Stated(), inverses=False).run(docs)
    assert len(off.facts) == 2 and off.stats["derived"] == 0
    # Nothing declared, nothing to turn on.
    assert not Pipeline(Ontology(), _Stated()).inverses


def test_a_refused_fact_gets_no_partner_in_the_graph() -> None:
    """The partner inherits the verdict, so the gate refuses both or neither (DECISIONS #20)."""
    docs = [Document(id="d1", text=TEXT)]
    kg = Pipeline(ONTOLOGY, _Stated(), grounder=_Contradicts(), gate=VerdictGate()).run(docs)
    assert {(s, p) for s, p, _, _ in _claims(kg)} == {("ada", "spouse"), ("william", "spouse")}
    assert (kg.stats["derived"], kg.stats["refused"]) == (2, 2)


def test_corroboration_folds_a_stated_partner_from_another_source_into_one_fact() -> None:
    """d2 states the inverse itself: no derived duplicate, and each direction keeps its source."""

    class _BothWays:
        def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
            if chunk.doc_id == "d1":
                return [_fact(BRITTANY, "located_in", FRANCE, doc="d1")]
            return [_fact(FRANCE, "contains", BRITTANY, doc="d2")]

    docs = [Document(id="d1", text=TEXT), Document(id="d2", text=TEXT)]
    kg = Pipeline(ONTOLOGY, _BothWays(), corroborator=SignatureCorroborator(ONTOLOGY)).run(docs)
    assert _claims(kg) == {
        ("brittany", "located_in", "france", False),
        ("france", "contains", "brittany", False),
    }
    assert kg.stats["derived"] == 0


def test_derived_provenance_survives_the_jsonl_and_neo4j_sinks(tmp_path: Path) -> None:
    docs = [Document(id="d1", text=TEXT)]
    kg = Pipeline(ONTOLOGY, _Stated(), sinks=[JsonlSink(tmp_path)]).run(docs)
    stated = next(f for f in kg.facts if f.predicate == "located_in")

    lines = (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    read_back = [Fact.model_validate_json(line) for line in lines]
    (partner,) = [f for f in read_back if f.predicate == "contains"]
    assert partner.qualifiers[DERIVED]["rule"] == "inverse"
    assert derived_from(partner) == stated.signature
    assert partner.evidence == stated.evidence

    # In Neo4j the stamp is JSON text, and its signature is the stated edge's MERGE key.
    rows = {
        statement.names[1]: statement.rows[0]["props"]
        for statement in plan(kg, ontology=ONTOLOGY)
        if statement.kind == "edge"
    }
    stamp = json.loads(rows["contains"][DERIVED])
    assert stamp["rule"] == "inverse"
    canonical = json.dumps(stamp["of"], separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == rows["located_in"]["signature"]
    assert DERIVED not in rows["located_in"]


def test_odke_run_adds_partners_unless_the_config_says_inverses_false(tmp_path: Path) -> None:
    (tmp_path / "ontology.json").write_text(ONTOLOGY.model_dump_json(), encoding="utf-8")
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "geo.txt").write_text(TEXT, encoding="utf-8")
    row = {"doc": "geo", "subject": "Brittany", "predicate": "located_in", "object": "France"}
    (tmp_path / "triples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {
        "ontology": "ontology.json",
        "inputs": ["corpus"],
        "stages": {"extractor": {"use": "triples", "path": "triples.jsonl"}},
    }

    on = execute(parse_config(config, base_dir=tmp_path))
    assert [(f.predicate, DERIVED in f.qualifiers) for f in on.graph.facts] == [
        ("located_in", False),
        ("contains", True),
    ]
    assert "derived       1 inverse and symmetric partners" in on.render()

    off = execute(parse_config({**config, "inverses": False}, base_dir=tmp_path))
    assert [f.predicate for f in off.graph.facts] == ["located_in"]
    assert "derived" not in off.render()
