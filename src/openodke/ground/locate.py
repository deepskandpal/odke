"""The span locator: the support for a fact whose extractor cited nothing, found for free.

Most extractors outside this package emit bare triples, and a bare triple is
grounded against the whole text it came from (DECISIONS #25). That makes every
such fact pay for the whole text in grounder input, and a verdict against a
whole document cannot say where the support is. Usually the support can be
found without a model: a fact's subject and object are names, and the sentence
that states the fact usually names both.

`locate_span` finds the narrowest window of one sentence, or two adjacent ones,
that names both. `SpanLocator` sets it as the span of a fact whose span is the
whole text, marked `located` so the diagnostics never count it as a citation.
When nothing is found nothing changes, and the grounder reads the whole text
as before.

It errs towards finding nothing. The grounder is shown the span alone (#23), so
a window that names both and does not state the fact refuses a fact the whole
text would have supported: a wrong span is worse than none. So:

- a name is the entity's label or an alias, as written or as its name key
  (`name_key`, the normaliser's `odke.name_key`), matched as whole words with
  case, accents and the punctuation between words ignored. `Ann` is not in
  `Annapolis`;
- a name written with a capital is only found capitalised: `US` is not `us`;
- a capitalised name is not found inside a longer one. A capitalised word
  joined on to either side extends it, so `Africa` is not in `South Africa`
  and `Loud` is not in `The Loud Tour`. A sentence's first word is exempt;
- a literal object is found by its words, or by its normalised value when it
  is a date or a quantity: `July 15, 1895` is found as `15 July 1895`;
- the subject and the object are found in different places, so an object
  named only inside the subject's name (`Ecuador` in `Constitution of
  Ecuador`) is not found;
- one sentence before two, then the narrower window, then the earlier. Two
  sentences never span a paragraph break.

The sentences are the chunker's, so a window never cuts one in half (#19).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openodke.chunking import _segment
from openodke.corroborate.normalize import (
    name_key,
    normalize_date,
    normalize_quantity,
    normalize_value,
)
from openodke.corroborate.provenance import NAME_KEY
from openodke.corroborate.resolve import domain_of
from openodke.ground.span import Counts
from openodke.types import Document, Entity, Evidence, Fact, GroundingVerdict, Span, SpanOrigin

# A word, a dotted initialism (`U.S.` is one word, as `name_key` reads it), or `&`.
_WORD = re.compile(r"(?:[^\W_]\.){2,}|[^\W_]+|&")
# What joins two words into one name: nothing, spaces or a hyphen.
_JOINED = re.compile(r"[\s-]*")
# The longest run of words read as one date or quantity: `10th of December 1815`.
_LONGEST_VALUE = 6


@dataclass(frozen=True, slots=True)
class _Word:
    key: str
    start: int
    end: int
    capital: bool


@dataclass(frozen=True, slots=True)
class _Sentence:
    start: int
    end: int
    opens_paragraph: bool
    words: tuple[_Word, ...]


@dataclass(frozen=True, slots=True)
class _Name:
    keys: tuple[str, ...]
    # Written with a capital, so only found capitalised.
    capital: bool


def _fold(word: str) -> str:
    """One word as `name_key` compares it: accents dropped, casefolded, dots closed up."""
    if word == "&":
        return "and"
    decomposed = unicodedata.normalize("NFKD", word)
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return bare.casefold().replace(".", "")


def _words(text: str, start: int = 0, end: int | None = None) -> tuple[_Word, ...]:
    stop = len(text) if end is None else end
    return tuple(
        _Word(_fold(m[0]), m.start(), m.end(), m[0][:1].isupper())
        for m in _WORD.finditer(text, start, stop)
    )


@lru_cache(maxsize=32)
def _sentences(text: str) -> tuple[_Sentence, ...]:
    return tuple(
        _Sentence(s.start, s.end, s.opens_paragraph, _words(text, s.start, s.end))
        for s in _segment(text)
    )


def _names_in(text: str, *, keyed: bool) -> set[_Name]:
    """The forms one written name is found by: as written, and by its name key."""
    words = _words(text)
    if not words:
        return set()
    capital = words[0].capital
    forms = {tuple(w.key for w in words)}
    if keyed:
        forms.add(tuple(name_key(text).split()))
    # A name of one character finds every initial and every stray letter.
    return {_Name(keys, capital) for keys in forms if len("".join(keys)) >= 2}


def _entity_names(entity: Entity) -> set[_Name]:
    texts = [entity.label] if entity.label else []
    # A domain or a URL names a host, not a mention the text would contain.
    texts += [alias for alias in entity.aliases if domain_of(alias) is None]
    names: set[_Name] = set()
    for text in texts:
        names |= _names_in(text, keyed=True)
    keyed = entity.attributes.get(NAME_KEY)
    if isinstance(keyed, str) and len(keyed.replace(" ", "")) >= 2:
        label = _words(entity.label) if entity.label else ()
        names.add(_Name(tuple(keyed.split()), bool(label) and label[0].capital))
    return names


def _canonical(value: Any) -> Any:
    """A literal's normalised value when it is a date or a quantity, else None.

    Plain text is found by its words, under the rules names are found by;
    comparing it by value would find `Loud` inside `The Loud Tour`.
    """
    if isinstance(value, bool):
        return None
    normal = normalize_value(value)
    if isinstance(normal, bool):
        return None
    if isinstance(normal, int | float):
        return normal
    if isinstance(normal, str) and (
        normalize_date(normal) is not None or normalize_quantity(normal) is not None
    ):
        return normal
    return None


def _joined(text: str, left: _Word, right: _Word) -> bool:
    return _JOINED.fullmatch(text, left.end, right.start) is not None


def _extends(text: str, words: Sequence[_Word], i: int, j: int) -> bool:
    """A capitalised word joined on to the match, making it part of a longer name.

    Not one the name key drops: `Halden Ltd` and `The Halden` are still `halden`.
    The sentence's first word is capitalised whatever it is, so it never extends.
    """

    def longer(start: _Word, end: _Word) -> bool:
        return name_key(text[start.start : end.end]) != name_key(text[first.start : last.end])

    first, last = words[i], words[j - 1]
    after = j < len(words) and words[j].capital and _joined(text, last, words[j])
    before = i > 1 and words[i - 1].capital and _joined(text, words[i - 1], first)
    return (after and longer(first, words[j])) or (before and longer(words[i - 1], last))


def _found(text: str, words: Sequence[_Word], names: Iterable[_Name]) -> list[tuple[int, int]]:
    """Where in a sentence's words any of `names` is found, as word ranges."""
    keys = [w.key for w in words]
    out: list[tuple[int, int]] = []
    for name in names:
        n = len(name.keys)
        for i in range(len(words) - n + 1):
            if tuple(keys[i : i + n]) != name.keys:
                continue
            if name.capital and not words[i].capital:
                continue
            if words[i].capital and _extends(text, words, i, i + n):
                continue
            out.append((i, i + n))
    return out


@lru_cache(maxsize=1024)
def _values(text: str, start: int, end: int) -> tuple[tuple[Any, int, int], ...]:
    """Every date and quantity in one sentence, normalised, with its word range."""
    words = _words(text, start, end)
    out = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + _LONGEST_VALUE, len(words)) + 1):
            value = _canonical(text[words[i].start : words[j - 1].end])
            if value is not None:
                out.append((value, i, j))
    return tuple(out)


def _apart(left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]]) -> bool:
    return any(a_end <= b or b_end <= a for a, a_end in left for b, b_end in right)


def locate_span(fact: Fact, doc: Document) -> Span | None:
    """The narrowest window of `doc` naming both the fact's subject and its object.

    One sentence, or two adjacent ones in one paragraph, with the text found
    there as its quote. None when no window names both, which is the answer
    whenever the match would be a guess (see the module's rules).
    """
    sentences = _sentences(doc.text)
    subject = _entity_names(fact.subject)
    if fact.object_entity is not None:
        target, obj = None, _entity_names(fact.object_entity)
    else:
        value = fact.object_value
        target = _canonical(value)
        obj = _names_in(value, keyed=False) if isinstance(value, str) else set()
    if not subject or not (obj or target is not None):
        return None

    subjects = [_found(doc.text, s.words, subject) for s in sentences]
    near = {k + d for k, found in enumerate(subjects) if found for d in (-1, 0, 1)}
    objects: list[list[tuple[int, int]]] = []
    for k, s in enumerate(sentences):
        found = _found(doc.text, s.words, obj) if k in near else []
        if k in near and target is not None:
            found += [(i, j) for v, i, j in _values(doc.text, s.start, s.end) if v == target]
        objects.append(found)

    # (sentences, width, start, end): one sentence first, then the narrower, then the earlier.
    windows = [
        (1, s.end - s.start, s.start, s.end)
        for k, s in enumerate(sentences)
        if subjects[k] and objects[k] and _apart(subjects[k], objects[k])
    ]
    windows += [
        (2, second.end - first.start, first.start, second.end)
        for k, (first, second) in enumerate(zip(sentences, sentences[1:], strict=False))
        if not second.opens_paragraph
        and ((subjects[k] and objects[k + 1]) or (objects[k] and subjects[k + 1]))
    ]
    if not windows:
        return None
    _, _, start, end = min(windows)
    return Span(doc_id=doc.id, start=start, end=end, quote=doc.text[start:end])


class SpanLocator:
    """Gives a fact that cited nothing the window that names its subject and object.

    `locate(fact, doc)` replaces the fact's `context` evidence in `doc` (the
    whole text, DECISIONS #25) with the window `locate_span` finds, marked
    `SpanOrigin.LOCATED`. A fact with a citation of its own, a verdict already,
    or no window is returned as it is. Pure and free: no model, no network.

    `LLMGrounder(locate=True)` runs it after the free span check and before
    the model. `stats` counts the facts it looked at, and how many it placed.
    """

    def __init__(self) -> None:
        self._counts = Counts("facts", "located", "not_located")

    def locate(self, fact: Fact, doc: Document) -> Fact:
        def whole(evidence: Evidence) -> bool:
            return evidence.span_origin is SpanOrigin.CONTEXT and evidence.doc_id == doc.id

        if fact.verdict is not GroundingVerdict.UNCHECKED or not any(map(whole, fact.evidence)):
            return fact
        self._counts.bump("facts")
        span = locate_span(fact, doc)
        if span is None:
            self._counts.bump("not_located")
            return fact
        self._counts.bump("located")
        evidence = tuple(
            e.model_copy(update={"span": span, "span_origin": SpanOrigin.LOCATED})
            if whole(e)
            else e
            for e in fact.evidence
        )
        return fact.model_copy(update={"evidence": evidence})

    @property
    def stats(self) -> dict[str, int]:
        """Facts with no span of their own that it looked at, located and not."""
        return self._counts.snapshot()


__all__ = ["SpanLocator", "locate_span"]
