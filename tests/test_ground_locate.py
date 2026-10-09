"""The span locator: one or two sentences naming a fact's subject and object, or nothing."""

from __future__ import annotations

import pytest

from openodke import (
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    Span,
    SpanOrigin,
)
from openodke.corroborate import NAME_KEY
from openodke.ground import SpanLocator, check_span, locate_span

TEXT = (
    "Halden Robotics was founded in Leeds in 2014 by Mara Quist. "
    "It opened a second office in Lyon in 2019. "
    "Mara Quist left the company. "
    "She moved to Annapolis, where she founded Quist Labs."
)
DOC = Document(id="halden", text=TEXT)
HALDEN = Entity(key="Company:halden robotics", type="Company", label="Halden Robotics")


def entity(label: str, type: str = "Thing", **fields: object) -> Entity:
    return Entity.model_validate(
        {"key": f"{type}:{label.casefold()}", "type": type, "label": label, **fields}
    )


def whole(doc: Document = DOC) -> Evidence:
    """What `TriplesExtractor` gives a triple that cited nothing (DECISIONS #25)."""
    span = Span(doc_id=doc.id, start=0, end=len(doc.text))
    return Evidence(doc_id=doc.id, span=span, span_origin=SpanOrigin.CONTEXT)


def fact(subject: Entity, predicate: str, obj: Entity | object, *evidence: Evidence) -> Fact:
    fields: dict[str, object] = {"subject": subject, "predicate": predicate}
    fields["object_entity" if isinstance(obj, Entity) else "object_value"] = obj
    fields["evidence"] = evidence or (whole(),)
    return Fact.model_validate(fields)


def quote(f: Fact, doc: Document = DOC) -> str | None:
    span = locate_span(f, doc)
    return None if span is None else span.resolve(doc)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #


def test_the_sentence_naming_both_is_the_window() -> None:
    founded = fact(HALDEN, "founded_in", entity("Leeds", "City"))
    span = locate_span(founded, DOC)
    assert span is not None
    assert span.resolve(DOC) == "Halden Robotics was founded in Leeds in 2014 by Mara Quist."
    assert span.quote == span.resolve(DOC) and span.doc_id == "halden"


def test_names_split_across_two_sentences_make_a_two_sentence_window() -> None:
    """ "It opened ... Lyon": the subject is named in the sentence before."""
    assert quote(fact(HALDEN, "office_in", entity("Lyon", "City"))) == (
        "Halden Robotics was founded in Leeds in 2014 by Mara Quist. "
        "It opened a second office in Lyon in 2019."
    )


def test_the_narrowest_window_wins_and_one_sentence_beats_two() -> None:
    doc = Document(
        id="d",
        text=(
            "Mara Quist, who studied for many years at several universities across Europe, "
            "lived in Leeds. Mara Quist was in Leeds. Mara Quist rested. Leeds slept."
        ),
    )
    lived = fact(entity("Mara Quist", "Person"), "lived_in", entity("Leeds", "City"), whole(doc))
    assert quote(lived, doc) == "Mara Quist was in Leeds."


def test_two_sentences_never_span_a_paragraph_break() -> None:
    doc = Document(id="d", text="Halden Robotics\n\nIts office is in Lyon.")
    lyon = fact(HALDEN, "office_in", entity("Lyon", "City"), whole(doc))
    assert locate_span(lyon, doc) is None


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def test_aliases_and_name_keys_are_names_too() -> None:
    doc = Document(id="d", text="ACME, Inc. bought the Lyon works. Halden Ltd. moved to Leeds.")
    acme = entity("Acme Corporation", "Company", aliases=("ACME, Inc.", "https://acme.com"))
    assert quote(fact(acme, "owns", entity("Lyon", "City"), whole(doc)), doc) == (
        "ACME, Inc. bought the Lyon works."
    )
    # `name_key` drops the legal form: "Halden Ltd." is "Halden Ltd", as written or not.
    halden = entity("Halden Ltd", "Company")
    assert quote(fact(halden, "moved_to", entity("Leeds", "City"), whole(doc)), doc) == (
        "Halden Ltd. moved to Leeds."
    )
    # The normaliser's own key, when it has stamped one: a person's, here.
    doc = Document(id="d", text="Mara Quist stayed in Leeds.")
    quist = entity("Quist, Mara", "Person", attributes={NAME_KEY: "mara quist"})
    assert quote(fact(quist, "lives_in", entity("Leeds", "City"), whole(doc)), doc) == doc.text
    assert (
        locate_span(fact(entity("Quist, Mara"), "lives_in", entity("Leeds"), whole(doc)), doc)
        is None
    )


def test_case_accents_and_punctuation_are_not_names() -> None:
    doc = Document(id="d", text="LÉON  Martin was born in   saint-étienne.")
    leon = entity("Léon Martin", "Person")
    assert (
        quote(fact(leon, "born_in", entity("saint étienne", "City"), whole(doc)), doc) is not None
    )


def test_a_name_is_whole_words_never_part_of_one() -> None:
    """ "Ann" is in "Annapolis" as letters, not as a word."""
    doc = Document(id="d", text="Mara Quist moved to Annapolis.")
    ann = fact(entity("Mara Quist", "Person"), "moved_to", entity("Ann", "City"), whole(doc))
    assert locate_span(ann, doc) is None


def test_a_capitalised_name_is_not_found_inside_a_longer_one() -> None:
    doc = Document(
        id="d",
        text="Rihanna toured South Africa. Rihanna sang on The Loud Tour. Rihanna released Loud.",
    )
    rihanna = entity("Rihanna", "Person")
    africa = fact(rihanna, "visited", entity("Africa", "Continent"), whole(doc))
    assert locate_span(africa, doc) is None
    # Not inside "The Loud Tour", but found where the album is named alone.
    assert quote(fact(rihanna, "notable_work", entity("Loud", "Album"), whole(doc)), doc) == (
        "Rihanna released Loud."
    )


def test_a_name_written_with_a_capital_is_only_found_capitalised() -> None:
    doc = Document(id="d", text="Mara Quist told us about Leeds.")
    us = fact(entity("Mara Quist", "Person"), "citizen_of", entity("US", "Country"), whole(doc))
    assert locate_span(us, doc) is None


def test_an_object_named_only_inside_the_subjects_name_is_not_found() -> None:
    doc = Document(id="d", text="The Constitution of Ecuador was approved in 2008.")
    constitution = entity("Constitution of Ecuador", "Law")
    jurisdiction = fact(constitution, "applies_to", entity("Ecuador", "Country"), whole(doc))
    assert locate_span(jurisdiction, doc) is None
    # Named apart from the subject, it is found.
    assert quote(fact(constitution, "approved", "2008", whole(doc)), doc) == doc.text


# --------------------------------------------------------------------------- #
# Literal objects
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    ["July 15, 1895", "1895-07-15", "15th July 1895"],
)
def test_a_date_is_found_by_its_normalised_value(value: str) -> None:
    doc = Document(id="d", text="He was ill. Vladimir Orlov was born on 15 July 1895 in Kherson.")
    born = fact(entity("Vladimir Orlov", "Person"), "date_of_birth", value, whole(doc))
    assert quote(born, doc) == "Vladimir Orlov was born on 15 July 1895 in Kherson."


def test_numbers_and_text_literals_are_found() -> None:
    assert quote(fact(HALDEN, "founded", 2014)) == (
        "Halden Robotics was founded in Leeds in 2014 by Mara Quist."
    )
    doc = Document(id="d", text="Halden Robotics employs 1,200 people.")
    assert quote(fact(HALDEN, "employees", 1200, whole(doc)), doc) == doc.text
    assert quote(fact(HALDEN, "employees", "1200", whole(doc)), doc) == doc.text
    assert locate_span(fact(HALDEN, "employees", 1300, whole(doc)), doc) is None
    assert locate_span(fact(HALDEN, "public", True, whole(doc)), doc) is None


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def test_a_located_span_replaces_the_whole_text_and_says_so() -> None:
    locator = SpanLocator()
    founded = fact(HALDEN, "founded_in", entity("Leeds", "City"))
    placed = locator.locate(founded, DOC)
    (evidence,) = placed.evidence
    assert evidence.span_origin is SpanOrigin.LOCATED
    assert evidence.span is not None and evidence.span.end - evidence.span.start == 59
    # Everything else about the evidence and the fact is as it was.
    assert evidence.model_copy(update={"span": None, "span_origin": SpanOrigin.CONTEXT}) == (
        founded.evidence[0].model_copy(update={"span": None})
    )
    assert placed.model_copy(update={"evidence": founded.evidence}) == founded
    assert check_span(evidence, DOC).value == "located"


def test_with_no_window_the_fact_is_returned_untouched() -> None:
    locator = SpanLocator()
    nowhere = fact(HALDEN, "office_in", entity("Berlin", "City"))
    assert locator.locate(nowhere, DOC) is nowhere
    assert locator.stats == {"facts": 1, "located": 0, "not_located": 1}


def test_only_a_fact_that_cited_nothing_is_located() -> None:
    """A citation is the extractor's to make, and a verdict already given stands."""
    locator = SpanLocator()
    cited = fact(
        HALDEN,
        "founded_in",
        entity("Leeds", "City"),
        Evidence(doc_id="halden", span=Span(doc_id="halden", start=0, end=15)),
    )
    judged = fact(HALDEN, "founded_in", entity("Leeds", "City")).model_copy(
        update={"verdict": GroundingVerdict.SUPPORTED}
    )
    elsewhere = fact(
        HALDEN, "founded_in", entity("Leeds", "City"), whole(Document(id="o", text="x"))
    )
    for untouched in (cited, judged, elsewhere):
        assert locator.locate(untouched, DOC) is untouched
    assert locator.stats == {"facts": 0, "located": 0, "not_located": 0}
    locator.locate(fact(HALDEN, "founded_in", entity("Leeds", "City")), DOC)
    assert locator.stats == {"facts": 1, "located": 1, "not_located": 0}
