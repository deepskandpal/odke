"""Widen and retry: one more reading for a narrow citation that came back `not_found` (#102).

The grounder is shown the cited span and nothing else (DECISIONS #3, #23). An
extractor that cites the word naming a value, `Ireland` out of a clause listing
three regions, loses a true fact to its own citation: asked whether "Acme
operates in Ireland" follows from `Ireland`, the grounder correctly answers
`not_found`. #23 asked the extractor for the clause instead, and named this as
the fallback: re-ground against the enclosing sentence.

The sentence is the chunker's (`openodke.chunking.sentences`), never one the
model names, so a widened citation is still offsets into the document and still
passes `Span.is_faithful`. The narrow citation becomes the evidence's `mention`
(or the mention it already had stays), which is what #23 says a mention is: the
words inside the clause that tell this fact from its siblings. A clause finer
than a sentence would need a parser, so the sentence is the one step taken.

`LLMGrounder(widen=True)` asks once more, once per fact, and only for a
`not_found` whose located span is narrower than its sentence. `supported` keeps
the wider span; any other answer keeps `not_found` and the original span.
Either way `qualifiers["odke.widen"]` records the attempt.
"""

from __future__ import annotations

import re

from openodke.chunking import sentences
from openodke.corroborate.provenance import WIDEN
from openodke.types import Document, Evidence, Span

_WORD = re.compile(r"\w")


def widen(evidence: Evidence, doc: Document) -> Evidence | None:
    """`evidence` re-cited to the whole sentence around its span, or None if it already is.

    A span crossing sentences widens to all of them. One whose sentence adds no
    word, only punctuation or space, is already as wide as it gets.
    """
    span = evidence.span
    if span is None:
        return None
    around = [(s, e) for s, e in sentences(doc.text) if s < span.end and span.start < e]
    if not around:
        return None
    start, end = min(around[0][0], span.start), max(around[-1][1], span.end)
    if not _WORD.search(doc.text[start : span.start] + doc.text[span.end : end]):
        return None
    wider = Span(doc_id=span.doc_id, start=start, end=end, quote=doc.text[start:end])
    mention = evidence.mention if evidence.mention is not None else span
    return evidence.model_copy(update={"span": wider, "mention": mention})


__all__ = ["WIDEN", "widen"]
