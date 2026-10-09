"""The coverage report: what extraction left behind, found with no model (#130)."""

from __future__ import annotations

from openodke import Document, Entity, Evidence, Fact, Ontology, Span, SpanOrigin
from openodke.coverage import NameMatcher, known_names, measure, offered_by
from openodke.extract import LLMExtractor
from openodke.llm import ScriptedClient

ONTOLOGY = Ontology.from_dict(
    {
        "name": "people",
        "types": {"Person": {}, "Company": {}, "City": {}},
        "predicates": {
            "employer": {"domain": ["Person"], "range": "Company", "importance": 0.9},
            "born_in": {"domain": ["Person"], "range": "City", "importance": 0.8},
            "founded": {"domain": ["Company"], "range": "integer", "importance": 0.5},
            "headquarters": {"domain": ["Company"], "range": "City", "importance": 0.1},
        },
    }
)
# Three sentences. The first is cited, the second names two known entities and
# no fact covers it (the planted gap), the third names only one.
TEXT = (
    "Ada Lovelace worked for Analytical Engines Ltd. "
    "Ada Lovelace was born in London. "
    "London is large."
)
FIRST = "Ada Lovelace worked for Analytical Engines Ltd."
GAP = "Ada Lovelace was born in London."
DOC = Document(id="d1", text=TEXT)


def _entity(type_: str, label: str) -> Entity:
    return Entity(key=f"{type_}:{label.casefold()}", type=type_, label=label)


ADA = _entity("Person", "Ada Lovelace")
ENGINES = _entity("Company", "Analytical Engines")
LONDON = _entity("City", "London")


def _cited(doc: Document, quote: str, subject: Entity, predicate: str, obj: Entity) -> Fact:
    start = doc.text.index(quote)
    span = Span(doc_id=doc.id, start=start, end=start + len(quote), quote=quote)
    return Fact(
        subject=subject,
        predicate=predicate,
        object_entity=obj,
        evidence=(Evidence(doc_id=doc.id, span=span),),
    )


EMPLOYED = _cited(DOC, FIRST, ADA, "employer", ENGINES)
# London is known to the batch from a fact in another document.
ELSEWHERE = Document(id="d2", text="Analytical Engines is based in London.")
BASED = _cited(ELSEWHERE, ELSEWHERE.text, ENGINES, "headquarters", LONDON)


def test_a_planted_gap_is_found_and_a_covered_sentence_is_not() -> None:
    report = measure([DOC], [EMPLOYED, BASED], ONTOLOGY)
    (record,) = report.documents
    # Two sentences name two known entities; the third names only London.
    assert record.sentences == 2
    assert [span.quote for span in record.uncovered] == [GAP]
    gap = record.uncovered[0]
    assert DOC.text[gap.start : gap.end] == GAP
    # London is mentioned here and named by no fact of this document.
    assert [span.quote for span in record.missed] == ["London"]
    assert report.uncovered == 1 and report.missed == 1 and report.sentences == 2


def test_a_fact_on_the_gap_covers_it() -> None:
    born = _cited(DOC, GAP, ADA, "born_in", LONDON)
    (record,) = measure([DOC], [EMPLOYED, BASED, born], ONTOLOGY).documents
    assert record.sentences == 2
    assert record.uncovered == ()
    assert record.missed == ()


def test_any_overlap_covers_a_sentence() -> None:
    # A citation narrower than its sentence still covers it: a narrow citation
    # is the grounder's concern, not a gap.
    born = _cited(DOC, "London", ADA, "born_in", LONDON)
    assert born.evidence[0].span is not None and born.evidence[0].span.start > TEXT.index(GAP)
    (record,) = measure([DOC], [EMPLOYED, BASED, born], ONTOLOGY).documents
    assert record.uncovered == ()


def test_a_span_nobody_chose_covers_the_sentence_naming_both_ends_of_its_fact() -> None:
    whole = Evidence(
        doc_id="d1",
        span=Span(doc_id="d1", start=0, end=len(TEXT), quote=TEXT),
        span_origin=SpanOrigin.CONTEXT,
    )
    born = Fact(subject=ADA, predicate="born_in", object_entity=LONDON, evidence=(whole,))
    (record,) = measure([DOC], [EMPLOYED, born], ONTOLOGY).documents
    assert record.uncovered == ()
    # An uncited fact whose ends no sentence names covers nothing, though its
    # span overlaps every sentence.
    elsewhere = born.model_copy(update={"object_entity": _entity("City", "Paris")})
    (record,) = measure([DOC], [EMPLOYED, elsewhere, BASED], ONTOLOGY).documents
    assert [span.quote for span in record.uncovered] == [GAP]


def test_a_span_that_does_not_resolve_covers_nothing() -> None:
    wrong = EMPLOYED.model_copy(
        update={
            "evidence": (
                Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=5, quote="nope!")),
            )
        }
    )
    (record,) = measure([DOC], [wrong, BASED], ONTOLOGY).documents
    assert [span.quote for span in record.uncovered] == [FIRST, GAP]


def test_relations_never_offered_when_the_snippet_is_limited() -> None:
    # A one-predicate snippet per type: `born_in` and `headquarters` fall off the end.
    extractor = LLMExtractor(client=ScriptedClient([]), snippet_limit=1)
    assert offered_by(extractor, ONTOLOGY) == ["founded", "employer"]
    report = measure([DOC], [EMPLOYED, BASED], ONTOLOGY, offered=offered_by(extractor, ONTOLOGY))
    assert report.not_offered == ("born_in", "headquarters")
    # Offered, and no fact used it.
    assert report.unused == ("founded",)


def test_relations_unknown_to_have_been_withheld_are_only_counted_as_unused() -> None:
    assert offered_by(object(), ONTOLOGY) is None
    report = measure([DOC], [EMPLOYED], ONTOLOGY)
    assert report.not_offered is None
    assert report.unused == ("born_in", "founded", "headquarters")


def test_names_match_as_name_keys_and_the_longest_wins() -> None:
    matcher = NameMatcher(["Zoë Smith", "Acme, Inc.", "Acme Cloud", "U.S.", "The Beatles", "A"])
    text = "ZOE SMITH met acme cloud staff and Acme in the U.S. with the Beatles."
    found = [(key, text[s:e]) for key, s, e in matcher.find(text)]
    assert found == [
        ("zoe smith", "ZOE SMITH"),
        ("acme cloud", "acme cloud"),
        ("acme", "Acme"),
        ("us", "U.S."),
        ("beatles", "Beatles"),
    ]
    # A one-character name would match nearly everything, so it is left out.
    assert len(matcher) == 5


def test_known_names_are_every_label_and_alias_in_the_batch() -> None:
    aliased = ADA.model_copy(update={"aliases": ("Countess of Lovelace",)})
    fact = EMPLOYED.model_copy(update={"subject": aliased})
    assert known_names([fact, BASED]) == [
        "Ada Lovelace",
        "Countess of Lovelace",
        "Analytical Engines",
        "London",
    ]
