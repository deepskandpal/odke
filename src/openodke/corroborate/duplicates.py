"""Near-duplicate documents: one text at two addresses is one source.

`source_of` counts a host once, so forty pages from one site are one source. A
page copied to another site, or a document re-saved with a few words changed,
is a second host saying nothing new, and each copy would raise `support` as if
someone had checked.

The fingerprint is the set of 5-word shingles of a document's text, casefolded,
and two documents are near-duplicates when the Jaccard similarity of those sets
is at least the corroborator's threshold, 0.9 by default. Exact Jaccard, not a
MinHash estimate: only documents that back one claim are compared, so the sets
are few, and an estimate would blur the threshold. One changed word changes at
most five shingles, so a 200-word page with one edit scores 0.95. Two articles
that share a quoted sentence score near zero, and a short excerpt of a long page
scores its share of the page, so neither is a copy.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from openodke.types import Document, Evidence

SHINGLE_WORDS = 5
DEFAULT_THRESHOLD = 0.9

_WORD = re.compile(r"\w+")


def shingles(text: str, size: int = SHINGLE_WORDS) -> frozenset[str]:
    """Every run of `size` consecutive words, casefolded; a shorter text is one shingle."""
    words = _WORD.findall(text.casefold())
    if len(words) <= size:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a ∩ b| / |a ∪ b|, and 0.0 when either set is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDuplicates:
    """Which documents behind one claim are copies of one another.

    One per `corroborate()` call: each document is shingled once and each pair
    compared once, however many claims they share. `compared` counts the pairs
    that were scored. A pair from one source is never scored, because it
    already counts once, and neither is a pair whose sizes alone rule it out:
    Jaccard is at most the smaller set's size over the larger's.
    """

    def __init__(self, documents: Mapping[str, Document], threshold: float) -> None:
        self.documents = documents
        self.threshold = threshold
        self.compared = 0
        self.found = 0
        self._shingles: dict[str, frozenset[str]] = {}
        self._pairs: dict[tuple[str, str], bool] = {}

    def clusters(
        self, evidence: Iterable[Evidence], source: Callable[[Evidence], str]
    ) -> list[frozenset[str]]:
        """The groups of two or more of these documents that are near-duplicates."""
        sources: dict[str, set[str]] = {}
        for e in evidence:
            if e.doc_id in self.documents:
                sources.setdefault(e.doc_id, set()).add(source(e))
        if len(set().union(*sources.values())) < 2:
            return []
        ids = sorted(sources, key=lambda d: (len(self._of(d)), d))
        parent = {d: d for d in ids}

        def root(d: str) -> str:
            while parent[d] != d:
                parent[d] = parent[parent[d]]
                d = parent[d]
            return d

        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if len(self._of(a)) < self.threshold * len(self._of(b)):
                    break
                if sources[a] & sources[b] or root(a) == root(b):
                    continue
                if self._same(a, b):
                    parent[root(b)] = root(a)
        groups: dict[str, set[str]] = {}
        for d in ids:
            groups.setdefault(root(d), set()).add(d)
        return [frozenset(g) for g in groups.values() if len(g) > 1]

    def _of(self, doc_id: str) -> frozenset[str]:
        if doc_id not in self._shingles:
            self._shingles[doc_id] = shingles(self.documents[doc_id].text)
        return self._shingles[doc_id]

    def _same(self, a: str, b: str) -> bool:
        pair = (a, b) if a < b else (b, a)
        if pair not in self._pairs:
            self.compared += 1
            self._pairs[pair] = jaccard(self._of(a), self._of(b)) >= self.threshold
            self.found += self._pairs[pair]
        return self._pairs[pair]


__all__ = ["DEFAULT_THRESHOLD", "SHINGLE_WORDS", "NearDuplicates", "jaccard", "shingles"]
