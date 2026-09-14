"""Splitting documents into chunks without losing a single offset.

DECISIONS #3 made spans character offsets and #19 made the chunk the unit of the
pipeline. Together they put the weight of provenance on this module: a chunk's
`text` must be exactly `doc.text[start:end]`, or a span an extractor finds inside
a chunk points at characters that were never there and the grounder checks a
quote against nothing.

So nothing here normalises. Paragraphs and sentences are found by scanning the
original string, a chunk is a pair of indices into it, and its text is sliced
rather than rebuilt. CRLF, tabs, runs of spaces and non-ASCII text survive
because they are never touched. Offsets are Python string indices — code points
— which is what `Span` and `Chunk` already mean.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from openodke.types import Chunk, Document

# `\r(?!\n)`: without the lookahead one CRLF backtracks into two line breaks and
# every Windows line becomes a paragraph.
_LINE_BREAK = r"(?:\r\n|\r(?!\n)|\n)"
_PARAGRAPH_BREAK = re.compile(
    rf"{_LINE_BREAK}(?:[^\S\r\n]*{_LINE_BREAK})+" "|\N{PARAGRAPH SEPARATOR}"
)
_CLOSERS = "\"'”’)\\]」』"
# Western terminators need whitespace after them ("3.14", "example.com" are not
# ends); CJK full stops do not, because CJK text has no spaces to wait for.
_SENTENCE_END = re.compile(rf"[.!?…]+[{_CLOSERS}]*(?=\s|\Z)|[。！？]+[{_CLOSERS}]*")
_WORD = re.compile(r"\S+")
# Only the ones that are nearly always followed by a capitalised name. A miss
# here merges two sentences, which is the safe direction: it never splits one.
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "cf", "fig", "approx"}
)


@dataclass(frozen=True, slots=True)
class _Sentence:
    start: int
    end: int
    words: int
    opens_paragraph: bool


class SentenceChunker:
    """Packs whole sentences into chunks of at most `max_words` words.

    Boundaries are chosen in order of preference: a paragraph break, then a
    sentence break, and never anything smaller. A router asked "fact or
    narrative?" about half a sentence is asked nothing (DECISIONS #19), so a
    single sentence longer than the cap becomes a chunk of its own rather than
    being cut — and a document with no punctuation at all is one sentence per
    paragraph.

    A chunk ends at the last paragraph break inside it, unless that would leave
    it less than half full; a heading stranded as its own chunk is as useless to
    a router as half a sentence. Chunks start and end on non-whitespace, so the
    whitespace between two chunks belongs to neither.

    `overlap` repeats that many trailing sentences at the start of the next
    chunk, so a fact stated across a boundary is seen whole at least once. It is
    shed first whenever it would crowd out new text. Overlapping chunks emit the
    same fact twice; the corroborator merges them by signature.
    """

    def __init__(self, max_words: int = 200, *, overlap: int = 0) -> None:
        if max_words < 1:
            raise ValueError("max_words must be at least 1")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        self.max_words = max_words
        self.overlap = overlap

    def chunk(self, doc: Document) -> Iterator[Chunk]:
        sentences = list(_segment(doc.text))
        n = len(sentences)
        prefix = [0]
        for s in sentences:
            prefix.append(prefix[-1] + s.words)

        first = 0  # first sentence of the chunk being built
        covered = 0  # sentences [0, covered) already emitted
        index = 0
        while covered < n:
            end = first
            while end < n and prefix[end + 1] - prefix[first] <= self.max_words:
                end += 1
            if end <= covered:
                if first < covered:
                    # The carried-over overlap leaves no room for anything new.
                    first += 1
                    continue
                # One sentence over the cap: it stands alone, uncut.
                end = covered + 1

            if end < n and not sentences[end].opens_paragraph:
                for k in range(end - 1, covered, -1):
                    if sentences[k].opens_paragraph:
                        if 2 * (prefix[k] - prefix[first]) >= self.max_words:
                            end = k
                        break

            start, stop = sentences[first].start, sentences[end - 1].end
            yield Chunk(
                doc_id=doc.id, start=start, end=stop, text=doc.text[start:stop], index=index
            )
            index += 1
            covered = end
            first = max(end - self.overlap, first + 1)


def _segment(text: str) -> Iterator[_Sentence]:
    """Every sentence in `text`, in order, with whitespace trimmed from both ends."""
    cursor = 0
    for brk in [*_PARAGRAPH_BREAK.finditer(text), None]:
        stop = len(text) if brk is None else brk.start()
        start, end = _trim(text, cursor, stop)
        opens = True
        for s, e in _sentences_in(text, start, end):
            yield _Sentence(s, e, len(_WORD.findall(text, s, e)), opens)
            opens = False
        if brk is not None:
            cursor = brk.end()


def _sentences_in(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        if _is_abbreviation(text, cursor, match, end):
            continue
        yield cursor, match.end()
        cursor, _ = _trim(text, match.end(), end)
    if cursor < end:
        yield cursor, end


def _is_abbreviation(text: str, sentence_start: int, match: re.Match[str], end: int) -> bool:
    if match.group().rstrip(_CLOSERS) != ".":
        return False
    following, _ = _trim(text, match.end(), end)
    if following < end and text[following].islower():
        return True
    token_start = match.start()
    while token_start > sentence_start and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start : match.start()].lstrip(_CLOSERS + "(“‘")
    # A single letter is an initial: "J. R. R. Tolkien".
    return token.casefold() in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


__all__ = ["SentenceChunker"]
