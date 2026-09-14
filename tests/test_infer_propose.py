"""The deterministic proposers (#32): records, Hearst patterns, co-occurrence."""

from __future__ import annotations

from pathlib import Path

import pytest

from openodke import Document
from openodke.infer import sample_corpus
from openodke.infer.candidates import PredicateCandidate, Proposals, TypeCandidate
from openodke.infer.names import normal_form, predicate_name, singular, type_name
from openodke.infer.propose import (
    CooccurrenceProposer,
    HearstProposer,
    RecordShapeProposer,
    propose,
)
from openodke.loaders import DirectoryLoader

PEOPLE = (
    "name,born,employer,city\n"
    "Ada Lovelace,1815-12-10,Acme Corp,London\n"
    "Grace Hopper,1906-12-09,Globex,New York\n"
    "Alan Turing,1912-06-23,Acme Corp,London\n"
    "Linus Torvalds,1969-12-28,Globex,New York\n"
)
COMPANIES = "name,founded\nAcme Corp,1901\nGlobex,1989\n"
NOTES = (
    "# Notes\n\n"
    "Ada Lovelace works at Acme Corp. Grace Hopper works at Globex.\n\n"
    "A mathematician is a scientist. Ada Lovelace was an English mathematician.\n\n"
    "She wrote about languages such as Python, Rust and Go. "
    "They saw cars, trucks and other vehicles. It is a good idea.\n"
)


def _docs(tmp_path: Path) -> list[Document]:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "people.csv").write_text(PEOPLE, encoding="utf-8")
    (root / "companies.csv").write_text(COMPANIES, encoding="utf-8")
    (root / "staff.jsonl").write_text(
        '{"name": "Ada Lovelace", "skills": ["maths", "poetry"]}\n', encoding="utf-8"
    )
    (root / "notes.md").write_text(NOTES, encoding="utf-8")
    return list(DirectoryLoader().load(root))


def _found(tmp_path: Path) -> tuple[Proposals, dict[str, Document]]:
    docs = _docs(tmp_path)
    sample = sample_corpus(docs, words=10_000)
    return propose(sample), {d.id: d for d in docs}


def _type(found: Proposals, name: str, proposer: str) -> TypeCandidate:
    (match,) = [t for t in found.types if t.name == name and t.proposer == proposer]
    return match


def _predicate(found: Proposals, name: str, domain: str) -> PredicateCandidate:
    (match,) = [p for p in found.predicates if p.name == name and p.domain == (domain,)]
    return match


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("programming languages", "ProgrammingLanguage"),
        ("people", "Person"),
        ("companies", "Company"),
        ("addresses", "Address"),
        ("status", "Status"),
        ("ProgrammingLanguage", "ProgrammingLanguage"),
    ],
)
def test_type_names_are_singular_pascal_case(phrase: str, expected: str) -> None:
    assert type_name(phrase) == expected


def test_predicate_names_and_normal_forms() -> None:
    assert predicate_name("worksFor") == predicate_name("works for") == "works_for"
    assert normal_form("worksFor") == normal_form("worked_for") == normal_form("Works-For")
    assert normal_form("employer") != normal_form("employed_by")
    assert singular("analysis") == "analysis" and singular("boxes") == "box"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def test_a_record_set_is_a_type_keyed_on_the_column_that_names_each_row(tmp_path: Path) -> None:
    found, _ = _found(tmp_path)
    person = _type(found, "Person", "records")
    assert person.keys == ("name",)
    assert set(person.examples) == {"Ada Lovelace", "Grace Hopper", "Alan Turing", "Linus Torvalds"}
    assert person.support == 4
    assert _type(found, "Company", "records").keys == ("name",)


def test_ranges_are_read_off_the_values(tmp_path: Path) -> None:
    found, _ = _found(tmp_path)
    assert _predicate(found, "born", "Person").range == "date"
    assert _predicate(found, "founded", "Company").range == "integer"
    assert _predicate(found, "name", "Person").range == "string"
    assert _predicate(found, "born", "Person").cardinality == "single"
    assert _predicate(found, "skills", "Staff").cardinality == "multi"


def test_a_column_of_another_sets_key_values_is_an_edge_to_that_type(tmp_path: Path) -> None:
    found, _ = _found(tmp_path)
    employer = _predicate(found, "employer", "Person")
    assert employer.range == "Company"
    assert ("adalovelace", "acmecorp") in employer.observations
    assert employer.support == 4


def test_recurring_capitalised_names_are_an_edge_to_a_type_named_for_the_column(
    tmp_path: Path,
) -> None:
    found, _ = _found(tmp_path)
    assert _predicate(found, "city", "Person").range == "City"
    city = _type(found, "City", "records")
    assert set(city.examples) == {"London", "New York"}


def test_record_evidence_cites_the_rendered_cell(tmp_path: Path) -> None:
    found, docs = _found(tmp_path)
    employer = _predicate(found, "employer", "Person")
    quotes = {span.quote for span in employer.evidence}
    assert quotes == {"Acme Corp", "Globex"}
    for candidate in [*found.types, *found.predicates]:
        for span in candidate.evidence:
            assert span.is_faithful(docs[span.doc_id])


# --------------------------------------------------------------------------- #
# Hearst patterns
# --------------------------------------------------------------------------- #


def _hearst(text: str) -> dict[str, TypeCandidate]:
    doc = Document(text=text)
    found = HearstProposer().propose(sample_corpus([doc], words=10_000), Proposals())
    for candidate in found.types:
        assert all(span.is_faithful(doc) for span in candidate.evidence)
    return {t.name: t for t in found.types}


def test_hearst_finds_an_is_a_parent() -> None:
    found = _hearst("A mathematician is a scientist.")
    assert found["Mathematician"].parents == ("Scientist",)
    assert found["Mathematician"].evidence[0].quote == "mathematician is a scientist"


def test_hearst_classes_proper_names_as_instances() -> None:
    found = _hearst("She wrote about programming languages such as Python, Rust and Go today.")
    assert found["ProgrammingLanguage"].examples == ("Python", "Rust", "Go")
    assert found["ProgrammingLanguage"].parents == ()
    found = _hearst("Ada Lovelace was an English mathematician and writer.")
    assert found["Mathematician"].examples == ("Ada Lovelace",)


def test_hearst_reads_and_other_backwards_without_taking_the_verb() -> None:
    found = _hearst("They saw cars, trucks and other vehicles were parked.")
    assert found["Car"].parents == ("Vehicle",)
    assert found["Truck"].parents == ("Vehicle",)
    assert "SawCar" not in found


def test_hearst_including_takes_common_nouns_as_subclasses() -> None:
    found = _hearst("The fleet has vehicles, including cars and vans.")
    assert found["Car"].parents == ("Vehicle",) and found["Van"].parents == ("Vehicle",)


@pytest.mark.parametrize(
    "text",
    [
        "It is a good idea.",
        "This was a mistake.",
        "Such as it is, the plan holds.",
        "He is a very tall man.",
    ],
)
def test_hearst_ignores_pronouns_and_praise(text: str) -> None:
    assert _hearst(text) == {}


def test_hearst_reads_every_pattern_in_a_mixed_note(tmp_path: Path) -> None:
    found, _ = _found(tmp_path)
    assert _type(found, "Mathematician", "hearst").parents == ("Scientist",)
    assert _type(found, "Language", "hearst").examples == ("Python", "Rust", "Go")
    assert _type(found, "Truck", "hearst").parents == ("Vehicle",)
    assert not [t for t in found.types if t.name in {"Idea", "GoodIdea"}]
    assert not [t for t in found.types if t.proposer == "hearst" and t.name == "Person"]


# --------------------------------------------------------------------------- #
# Co-occurrence
# --------------------------------------------------------------------------- #


def test_cooccurrence_finds_the_words_between_two_known_instances(tmp_path: Path) -> None:
    found, docs = _found(tmp_path)
    works_at = _predicate(found, "works_at", "Person")
    assert works_at.proposer == "cooccurrence"
    assert works_at.range == "Company"
    assert set(works_at.observations) == {("adalovelace", "acmecorp"), ("gracehopper", "globex")}
    assert works_at.cardinality == "single"
    assert {s.quote for s in works_at.evidence} == {
        "Ada Lovelace works at Acme Corp",
        "Grace Hopper works at Globex",
    }


def test_cooccurrence_strips_auxiliaries_and_sees_multiplicity() -> None:
    doc = Document(
        text="Ada is employed by Acme. Ada is employed by Globex. Acme and Globex compete."
    )
    known = Proposals(
        types=(
            TypeCandidate(name="Person", proposer="t", examples=("Ada",)),
            TypeCandidate(name="Company", proposer="t", examples=("Acme", "Globex")),
        )
    )
    found = CooccurrenceProposer().propose(sample_corpus([doc]), known)
    (employed_by,) = found.predicates
    assert (employed_by.name, employed_by.domain, employed_by.range) == (
        "employed_by",
        ("Person",),
        "Company",
    )
    assert employed_by.cardinality == "multi"


def test_cooccurrence_needs_instances_and_proposes_nothing_without_them() -> None:
    doc = Document(text="Ada works at Acme.")
    assert CooccurrenceProposer().propose(sample_corpus([doc]), Proposals()) == Proposals()


def test_proposing_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    sample = sample_corpus(docs, words=10_000)
    assert propose(sample) == propose(sample)
    assert RecordShapeProposer().name == "records"
