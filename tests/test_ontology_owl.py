"""Ontology.from_owl: OWL, RDFS and SKOS in, and nothing the model cannot hold dropped silently."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("rdflib")

from rdflib import Graph  # noqa: E402

from openodke import KnowledgeGraph  # noqa: E402
from openodke.ontology import (  # noqa: E402
    EntityType,
    Ontology,
    OntologyImportWarning,
    OntologyLoadError,
    Predicate,
)
from openodke.sinks.rdf import RdfSink  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "ontologies"
EXAMPLE = Path(__file__).parent.parent / "examples" / "people.ontology.json"


def _people() -> Ontology:
    """What `people.owl.ttl` says, written as an ontology."""
    return Ontology(
        name="people",
        version="2",
        types={
            "Agent": EntityType(name="Agent", description="Anything that can act."),
            "Person": EntityType(
                name="Person",
                description="An individual human being.",
                parents=("Agent",),
                aliases=("Human", "Individual"),
            ),
            "Scientist": EntityType(
                name="Scientist", description="Research scientist", parents=("Person",)
            ),
            "Company": EntityType(
                name="Company",
                description="An incorporated organisation.",
                parents=("Agent",),
                keys=("legal_name",),
            ),
        },
        predicates={
            "employer": Predicate(
                name="employer",
                label="Employer",
                description="Where they work.",
                domain=("Person",),
                range="Company",
            ),
            "past_employers": Predicate(
                name="past_employers", domain=("Person",), range="Company", cardinality="multi"
            ),
            "name": Predicate(name="name", domain=("Company", "Person"), aliases=("full name",)),
            "legal_name": Predicate(
                name="legal_name",
                label="Legal name",
                description="Name on the certificate of incorporation.",
                domain=("Company",),
            ),
            "founded": Predicate(name="founded", domain=("Company",), range="date"),
            "headcount": Predicate(name="headcount", domain=("Company",), range="integer"),
            "revenue": Predicate(
                name="revenue", domain=("Company",), range="number", cardinality="multi"
            ),
            "listed": Predicate(name="listed", domain=("Company",), range="boolean"),
            "updated_at": Predicate(name="updated_at", range="datetime"),
            "field": Predicate(name="field", domain=("Scientist",), cardinality="multi"),
        },
    )


# --------------------------------------------------------------------------- #
# OWL
# --------------------------------------------------------------------------- #


def test_an_owl_file_reads_back_as_the_ontology_it_renders() -> None:
    ontology = Ontology.from_owl(FIXTURES / "people.owl.ttl")
    assert ontology == _people()
    assert ontology.validate() == []


def test_the_result_is_usable_where_any_ontology_is() -> None:
    ontology = Ontology.from_owl(FIXTURES / "people.owl.ttl")
    rendered = ontology.snippet("Scientist").render()
    assert "- employer (Company, single): Where they work." in rendered
    assert "- field (string, multi)" in rendered
    assert Ontology.from_json(ontology.model_dump_json()) == ontology
    # Left at the default, and said so: nothing in OWL says how often a predicate is used.
    assert {p.importance for p in ontology.predicates.values()} == {0.5}


def test_every_way_of_passing_the_schema_loads_the_same() -> None:
    path = FIXTURES / "people.owl.ttl"
    text = path.read_text(encoding="utf-8")
    graph = Graph().parse(path)
    loaded = [
        Ontology.from_owl(str(path)),
        Ontology.from_owl(text),
        Ontology.from_owl(graph),
        Ontology.from_owl(graph.serialize(format="nt"), format="nt"),
        Ontology.from_owl(graph.serialize(format="xml")),
        Ontology.from_owl(graph.serialize(format="json-ld")),
    ]
    # A graph re-serialised has no owl:Ontology label order to lose; the header survives.
    assert all(o == loaded[0] for o in loaded)


def test_tagged_text_in_the_chosen_language_wins() -> None:
    english = Ontology.from_owl(FIXTURES / "people.owl.ttl")
    french = Ontology.from_owl(FIXTURES / "people.owl.ttl", language="fr")
    german = Ontology.from_owl(FIXTURES / "people.owl.ttl", language="de")
    assert english.types["Person"].description == "An individual human being."
    assert french.types["Person"].description == "Un être humain."
    # Untagged text is always kept; another language's aliases are not.
    assert english.types["Person"].aliases == ("Human", "Individual")
    assert german.types["Person"].aliases == ("Individual", "Mensch")


def test_name_and_version_come_from_the_header_unless_given() -> None:
    ontology = Ontology.from_owl(FIXTURES / "people.owl.ttl", name="hr", version="9")
    assert (ontology.name, ontology.version) == ("hr", "9")
    headless = Ontology.from_owl(FIXTURES / "library.rdfs.ttl")
    assert (headless.name, headless.version) == ("untitled", "0")


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


def _owl_view(ontology: Ontology) -> dict[str, object]:
    """The part of an ontology OWL can say: qualifiers, importance and examples it cannot."""
    literal = {"float": "number"}
    return {
        "types": {
            n: (t.description, tuple(sorted(t.parents)), t.keys, tuple(sorted(t.aliases)))
            for n, t in ontology.types.items()
        },
        "predicates": {
            n: (
                p.label,
                p.description,
                tuple(sorted(p.domain)),
                literal.get(p.range, p.range),
                p.cardinality,
                tuple(sorted(p.aliases)),
            )
            for n, p in ontology.predicates.items()
        },
    }


@pytest.mark.parametrize("fmt", ["turtle", "nt", "json-ld"])
def test_an_ontology_written_by_the_rdf_sink_reads_back_through_from_owl(
    tmp_path: Path, fmt: str
) -> None:
    original = _people().model_copy(update={"name": "untitled", "version": "0"})
    path = tmp_path / f"schema.{fmt}"
    RdfSink(path, format=fmt, ontology=original).write(KnowledgeGraph())
    back = Ontology.from_owl(path, format=fmt)
    assert _owl_view(back) == _owl_view(original)
    assert back == original


def test_the_shipped_example_survives_the_same_round_trip(tmp_path: Path) -> None:
    example = Ontology.from_json(EXAMPLE)
    path = tmp_path / "example.ttl"
    RdfSink(path, ontology=example).write(KnowledgeGraph())
    assert _owl_view(Ontology.from_owl(path)) == _owl_view(example)


def test_odd_names_survive_the_round_trip_percent_encoded(tmp_path: Path) -> None:
    original = Ontology(
        types={"Legal Entity": EntityType(name="Legal Entity")},
        predicates={"has part": Predicate(name="has part", domain=("Legal Entity",))},
    )
    path = tmp_path / "odd.ttl"
    RdfSink(path, ontology=original).write(KnowledgeGraph())
    back = Ontology.from_owl(path)
    assert set(back.types) == {"Legal Entity"}
    assert back.predicates["has part"].domain == ("Legal Entity",)


# --------------------------------------------------------------------------- #
# RDFS and SKOS
# --------------------------------------------------------------------------- #


def test_plain_rdfs_decides_edge_or_literal_by_the_range() -> None:
    ontology = Ontology.from_owl(FIXTURES / "library.rdfs.ttl")
    assert set(ontology.types) == {"Work", "Book", "Author"}
    assert ontology.types["Book"].parents == ("Work",)
    assert ontology.types["Work"].description == "A creative work."
    # A label that only repeats the name is not a description.
    assert ontology.types["Book"].description is None
    shapes = {n: (p.domain, p.range, p.cardinality) for n, p in ontology.predicates.items()}
    assert shapes == {
        "author": (("Work",), "Author", "multi"),
        "title": (("Work",), "string", "multi"),
        "pages": (("Book",), "integer", "multi"),
        "isbn": (("Book",), "string", "multi"),
    }
    assert ontology.predicates["author"].is_edge_in(ontology)


def test_skos_concepts_are_types_and_broader_or_narrower_name_parents() -> None:
    with pytest.warns(OntologyImportWarning, match="skos:related"):
        ontology = Ontology.from_owl(FIXTURES / "taxonomy.skos.ttl", strict=False)
    types = ontology.types
    assert set(types) == {"Vehicle", "Car", "ElectricCar", "Truck"}
    assert types["Car"].parents == ("Vehicle",)  # from Vehicle skos:narrower Car
    assert types["ElectricCar"].parents == ("Car",)
    assert types["Truck"].parents == ("Vehicle",)
    assert types["Vehicle"].description == "Anything that carries people or goods."
    assert types["ElectricCar"].description == "Electric car"
    assert types["Car"].aliases == ("Automobile", "Motorcar")
    assert types["ElectricCar"].aliases == ("EV",)
    assert ontology.predicates == {}


# --------------------------------------------------------------------------- #
# What cannot be mapped
# --------------------------------------------------------------------------- #

UNSUPPORTED = [
    ("org>", "owl:imports <https://example.org/base> is not followed"),
    ("Person", "subclass of an owl:Restriction on"),
    ("Person", "owl:disjointWith is not supported"),
    ("Company", "owl:equivalentClass is not supported"),
    ("employer", "owl:inverseOf is not supported"),
    ("employs", "is a owl:InverseFunctionalProperty"),
    ("manages", "is a owl:TransitiveProperty"),
    ("manages", "rdfs:subPropertyOf is not supported"),
    ("partner", "rdfs:range an owl:unionOf expression"),
    ("knows", "an object property with no rdfs:range"),
    ("born_in_year", "rdfs:range xsd:gYear has no ontology literal type"),
    ("acme", "1 individual(s) — instance data, not schema — not imported"),
    ("", "an anonymous owl:AllDisjointClasses axiom is not supported"),
]


def test_every_unsupported_construct_is_reported_by_its_subject() -> None:
    with pytest.raises(OntologyLoadError) as info:
        Ontology.from_owl(FIXTURES / "unsupported.owl.ttl")
    problems = info.value.problems
    assert len(problems) == len(UNSUPPORTED), "\n".join(problems)
    for subject, message in UNSUPPORTED:
        assert any(subject in p and message in p for p in problems), (subject, message)
    assert str(info.value).splitlines()[0].endswith(f"{len(UNSUPPORTED)} problems")


def test_not_strict_loads_what_maps_and_warns_with_the_same_problems() -> None:
    with pytest.warns(OntologyImportWarning) as caught:
        ontology = Ontology.from_owl(FIXTURES / "unsupported.owl.ttl", strict=False)
    (warning,) = [w.message for w in caught if isinstance(w.message, OntologyImportWarning)]
    assert len(warning.problems) == len(UNSUPPORTED)
    assert set(ontology.types) == {"Person", "Company", "Organisation"}
    # An edge with no single named range is not guessed at; a literal falls back to string.
    assert set(ontology.predicates) == {"employer", "employs", "manages", "born_in_year", "name"}
    assert ontology.predicates["born_in_year"].range == "string"


def test_a_clean_schema_loads_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Ontology.from_owl(FIXTURES / "people.owl.ttl", strict=False)


def test_two_iris_with_one_local_name_are_refused() -> None:
    doc = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    <https://a.example/Person> a owl:Class .
    <https://b.example/Person> a owl:Class .
    """
    with pytest.raises(OntologyLoadError, match="share 'Person'"):
        Ontology.from_owl(doc)


def test_a_schema_validate_finds_errors_in_is_refused_when_strict() -> None:
    doc = """
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix : <https://example.org/x#> .
    :Person a owl:Class ; owl:hasKey ( :ticker ) .
    :Company a owl:Class .
    :ticker a owl:DatatypeProperty ; rdfs:domain :Company .
    """
    with pytest.raises(OntologyLoadError, match="key-outside-domain"):
        Ontology.from_owl(doc)
    assert Ontology.from_owl(doc, strict=False).types["Person"].keys == ("ticker",)


def test_broken_input_is_explained() -> None:
    with pytest.raises(OntologyLoadError, match="no such file"):
        Ontology.from_owl(FIXTURES / "missing.ttl")
    with pytest.raises(OntologyLoadError, match="not valid turtle"):
        Ontology.from_owl("@prefix : <x> .\n:a :b")


def test_a_missing_rdflib_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rdflib", None)
    with pytest.raises(ImportError, match=r"openodke\[rdf\]"):
        Ontology.from_owl(FIXTURES / "people.owl.ttl")
