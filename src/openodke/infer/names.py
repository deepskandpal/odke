"""Names for inferred types and predicates, and the forms they are compared in.

Every proposer names what it finds the same way — types `PascalCase` and
singular, predicates `snake_case` — so `people.csv`, "people such as Ada" and a
model's `Person` arrive at one spelling before anything is merged. The merge
step compares `normal_form`s, which also forget inflection: `worksFor`,
`works_for` and `worked for` are one name told three ways.

Standard library only, and deliberately small. A lemmatiser would do better and
would be a dependency (DECISIONS #1); what is wrong here is caught at review.
"""

from __future__ import annotations

import re

from openodke.extract._common import fold

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SEPARATORS = re.compile(r"[\W_]+")
_IRREGULAR = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
    "mice": "mouse",
    "criteria": "criterion",
    "phenomena": "phenomenon",
    "analyses": "analysis",
    "indices": "index",
    "vertices": "vertex",
}
# Words that end in "s" and are already singular, or have no singular.
_UNCHANGED = frozenset({"data", "media", "news", "series", "species", "staff", "status"})
_KEEPS_S = ("ss", "us", "is", "ics")


def _word_set(text: str) -> frozenset[str]:
    return frozenset(text.split())


STOPWORDS = _word_set(
    """
    a an the and or but nor of in on at to for from by with without into onto over under
    about as than then that this these those there here it its he she they them we you i me
    my our your his her their who whom whose which what when where why how is are was were
    be been being am do does did has have had having not no so such other others also very
    more most many much some any each every all both few several various another same own
    can could may might must shall should will would one ones etc
    """
)
PRONOUNS = _word_set("i you he she it we they who which that this these those")
# Adjectives that say how a writer feels about a class, not which class it is.
NOISE_ADJECTIVES = _word_set(
    """
    good great new old major leading large small big first last important popular famous
    well-known known main key top best notable prominent early late modern common typical
    certain particular different similar real true whole
    """
)


def words(name: str) -> list[str]:
    """`worksFor`, `works_for`, `Works for` → `["works", "for"]`."""
    return [w.casefold() for w in _SEPARATORS.split(_CAMEL.sub(" ", name)) if w]


def singular(word: str) -> str:
    w = word.casefold()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w in _UNCHANGED or len(w) <= 3 or w.endswith(_KEEPS_S):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "ches", "shes", "xes", "zes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def type_name(phrase: str) -> str:
    """A noun phrase as a type: the head noun made singular, the whole in PascalCase.

    `programming languages` → `ProgrammingLanguage`; `people.csv`'s stem → `Person`.
    """
    parts = words(phrase)
    if not parts:
        return ""
    parts[-1] = singular(parts[-1])
    return "".join(p[:1].upper() + p[1:] for p in parts)


def predicate_name(phrase: str) -> str:
    """A key or phrase as a predicate: `worksFor`, `works for` → `works_for`."""
    return "_".join(words(phrase))


def normal_form(name: str) -> str:
    """The form two names are compared in: case, separators, plural and tense forgotten."""
    return " ".join(_stem(w) for w in words(name))


def _stem(word: str) -> str:
    w = singular(word)
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


__all__ = [
    "NOISE_ADJECTIVES",
    "STOPWORDS",
    "fold",
    "normal_form",
    "predicate_name",
    "singular",
    "type_name",
    "words",
]
