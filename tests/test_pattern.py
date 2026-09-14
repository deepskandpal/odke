"""The pattern extractor: exact facts from structure, each citing its cell."""

from __future__ import annotations

import pytest

from openodke import (
    Chunk,
    Document,
    EntityType,
    Extractor,
    Fact,
    Ontology,
    Predicate,
    SentenceChunker,
    SourceTier,
)
from openodke.extract import PatternExtractor, RecordMapping
from openodke.loaders import CsvLoader, JsonLoader

ONTOLOGY = Ontology(
    name="people",
    types={
        "Person": EntityType(name="Person", keys=("full_name",)),
        "Scientist": EntityType(name="Scientist", parents=("Person",)),
        "Company": EntityType(name="Company", keys=("legal_name",)),
    },
    predicates={
        "full_name": Predicate(
            name="full_name", label="Full name", aliases=("name",), domain=("Person",)
        ),
        "birth_date": Predicate(
            name="birth_date", aliases=("born", "date of birth"), domain=("Person",), range="date"
        ),
        "employer": Predicate(
            name="employer", domain=("Person",), range="Company", cardinality="multi"
        ),
        "field": Predicate(name="field", domain=("Scientist",), cardinality="multi"),
        "legal_name": Predicate(name="legal_name", aliases=("company",), domain=("Company",)),
        "uptime": Predicate(
            name="uptime", domain=("Company",), qualifiers={"percentile": {"identity": True}}
        ),
    },
)


def _whole(doc: Document) -> Chunk:
    return Chunk(doc_id=doc.id, start=0, end=len(doc.text), text=doc.text, index=0)


def _facts(extractor: PatternExtractor, doc: Document) -> list[Fact]:
    return list(extractor.extract(_whole(doc), ONTOLOGY))


def _assert_faithful(facts: list[Fact], doc: Document) -> None:
    for fact in facts:
        (evidence,) = fact.evidence
        assert evidence.span is not None and evidence.span.is_faithful(doc)
        assert (fact.extractor, fact.confidence) == ("pattern", 1.0)


def _triples(facts: list[Fact]) -> set[tuple[str, str, str]]:
    return {
        (
            f.subject.key,
            f.predicate,
            f.object_entity.key if f.object_entity else str(f.object_value),
        )
        for f in facts
    }


def test_csv_columns_matching_the_ontology_become_facts_citing_their_cells() -> None:
    (doc,) = CsvLoader(tier=SourceTier.CURATED).load(
        b"Name,Born,Employer,Shoe size\nAda Lovelace,1815-12-10,Analytical Engines Ltd,5\n"
    )
    facts = _facts(PatternExtractor(documents=[doc]), doc)
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace"),
        ("Person:ada lovelace", "birth_date", "1815-12-10"),
        ("Person:ada lovelace", "employer", "Company:analytical engines ltd"),
    }
    _assert_faithful(facts, doc)
    born = next(f for f in facts if f.predicate == "birth_date")
    (evidence,) = born.evidence
    assert evidence.span is not None and evidence.span.resolve(doc) == "1815-12-10"
    assert (evidence.tier, evidence.uri) == (SourceTier.CURATED, doc.uri)
    employer = next(f for f in facts if f.predicate == "employer")
    assert employer.is_edge and employer.object_entity is not None
    assert employer.object_entity.type == "Company"


def test_identity_keys_are_stamped_from_the_ontology() -> None:
    (doc,) = CsvLoader().load(b"company,uptime\nAcme,99.9%\n")
    facts = _facts(PatternExtractor(documents=[doc]), doc)
    uptime = next(f for f in facts if f.predicate == "uptime")
    assert uptime.identity_keys == ("percentile",)
    assert uptime.subject.key == "Company:acme"


def test_a_declarative_mapping_reads_nested_paths_and_wildcards() -> None:
    mapping = RecordMapping.model_validate_json(
        '{"subject_type": "Person", "subject": "person.name",'
        ' "predicates": {"full_name": "person.name", "employer": "$.jobs[*].company"}}'
    )
    (doc,) = JsonLoader().load(
        b'{"person": {"name": "Grace Hopper"},'
        b' "jobs": [{"company": "Remington Rand"}, {"company": "US Navy"}]}'
    )
    facts = _facts(PatternExtractor(mappings=[mapping], documents=[doc]), doc)
    assert _triples(facts) == {
        ("Person:grace hopper", "full_name", "Grace Hopper"),
        ("Person:grace hopper", "employer", "Company:remington rand"),
        ("Person:grace hopper", "employer", "Company:us navy"),
    }
    _assert_faithful(facts, doc)


def test_a_mapping_to_an_undeclared_predicate_is_a_configuration_error() -> None:
    (doc,) = JsonLoader().load(b'{"name": "Ada"}')
    bad = RecordMapping(subject_type="Person", subject="name", predicates={"shoe_size": "name"})
    with pytest.raises(ValueError, match="shoe_size"):
        _facts(PatternExtractor(mappings=[bad], documents=[doc]), doc)


def test_a_record_with_no_identifiable_subject_yields_nothing() -> None:
    """Never invent a subject: a birth date belonging to nobody is not a fact."""
    (doc,) = CsvLoader().load(b"born,employer\n1815-12-10,Acme\n")
    assert _facts(PatternExtractor(documents=[doc]), doc) == []


def test_a_record_split_across_chunks_emits_each_cell_once() -> None:
    (doc,) = CsvLoader().load(b"name,born,employer\nAda Lovelace,1815-12-10,Acme\n")
    extractor = PatternExtractor(documents=[doc])
    chunks = list(SentenceChunker(max_words=2).chunk(doc))
    assert len(chunks) == 1  # a rendering has no sentence ends: one sentence
    split = [
        Chunk(doc_id=doc.id, start=0, end=18, text=doc.text[0:18], index=0),
        Chunk(doc_id=doc.id, start=19, end=len(doc.text), text=doc.text[19:], index=1),
    ]
    per_chunk = [f for c in split for f in extractor.extract(c, ONTOLOGY)]
    assert _triples(per_chunk) == _triples(_facts(extractor, doc))
    assert len(per_chunk) == 3


def test_an_unregistered_structured_chunk_still_reads_its_key_value_lines() -> None:
    (doc,) = CsvLoader().load(b"name,born\nAda Lovelace,1815-12-10\n")
    facts = _facts(PatternExtractor(), doc)
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace"),
        ("Person:ada lovelace", "birth_date", "1815-12-10"),
    }
    # No document to look up: evidence carries the span, not the URI or tier.
    assert all(f.evidence[0].uri is None for f in facts)
    _assert_faithful(facts, doc)


TABLE = (
    "Intro line.\r\n\r\n## Staff\r\n\r\n"
    "| Full name | DATE-OF-BIRTH | Employer | Notes |\r\n"
    "|:----------|:-------------:|----------|-------|\r\n"
    "| Ada Lovelace | 1815-12-10 | Analytical Engines Ltd | a \\| b |\r\n"
    "| Alan Turing | 1912-06-23 | | |\r\n"
    "|  | 1900-01-01 | Nobody | |\r\n"
    "\r\nAfter the table.\r\n"
)


def test_a_pipe_table_becomes_one_entity_per_row() -> None:
    doc = Document(text=TABLE, modality="semi_structured")
    start = TABLE.index("## Staff")
    # A chunk that does not start at zero, so local offsets must be shifted.
    chunk = Chunk(doc_id=doc.id, start=start, end=len(TABLE), text=TABLE[start:], index=1)
    facts = list(PatternExtractor(documents=[doc]).extract(chunk, ONTOLOGY))
    assert _triples(facts) == {
        ("Person:ada lovelace", "full_name", "Ada Lovelace"),
        ("Person:ada lovelace", "birth_date", "1815-12-10"),
        ("Person:ada lovelace", "employer", "Company:analytical engines ltd"),
        ("Person:alan turing", "full_name", "Alan Turing"),
        ("Person:alan turing", "birth_date", "1912-06-23"),
    }
    _assert_faithful(facts, doc)


def test_a_key_value_block_is_named_by_its_heading_when_it_names_nobody() -> None:
    text = (
        "# Grace Hopper\n\n"
        "- **Born:** 1906-12-09\n"
        "- Employer: US Navy\n"
        "Field: computing\n\n"
        "The reason is simple: prose with a colon is not a fact.\n"
    )
    doc = Document(text=text, modality="semi_structured")
    facts = _facts(PatternExtractor(documents=[doc]), doc)
    # A scientist-only field makes Scientist the tightest type that fits all three.
    assert _triples(facts) == {
        ("Scientist:grace hopper", "birth_date", "1906-12-09"),
        ("Scientist:grace hopper", "employer", "Company:us navy"),
        ("Scientist:grace hopper", "field", "computing"),
    }
    _assert_faithful(facts, doc)


def test_a_block_names_its_subject_by_an_identity_predicate_first() -> None:
    doc = Document(text="# Staff list\n\nName: Ada Lovelace\nBorn: 1815-12-10\n")
    facts = _facts(PatternExtractor(documents=[doc]), doc)
    assert {f.subject.key for f in facts} == {"Person:ada lovelace"}
    orphan = Document(text="Born: 1815-12-10\nEmployer: Acme\n")
    assert _facts(PatternExtractor(documents=[orphan]), orphan) == []


def test_subject_type_and_confidence_are_the_callers_to_set() -> None:
    doc = Document(text="Name: Ada Lovelace\nBorn: 1815-12-10\n")
    facts = _facts(PatternExtractor(subject_type="Scientist", confidence=0.9), doc)
    assert {(f.subject.type, f.confidence) for f in facts} == {("Scientist", 0.9)}


def test_it_is_an_extractor_and_needs_no_client() -> None:
    assert isinstance(PatternExtractor(), Extractor)
    assert Document(text="").modality == "unstructured"
    assert PatternExtractor().extract(_whole(Document(text="")), ONTOLOGY) == []
