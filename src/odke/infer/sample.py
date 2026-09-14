"""The slice of a corpus an ontology is inferred from.

Inferring from everything is slow and no better than inferring from a good
sample. A good sample here means three things.

**Stratified.** Documents are grouped by modality and length band; inside a
group, by source — the file a record came from; inside a source, by document.
Every level takes turns. A 10,000-row CSV gets one turn per round next to a
three-paragraph note, and a book-length document gives up one chunk per visit
instead of filling the budget on its own.

**Reproducible.** Every shuffle is seeded, and keyed on what a document *is* —
its source, row and text — never on `Document.id`, which a loader draws at
random. The same files and the same seed give the same sample on any machine,
so an inferred ontology can be regenerated and diffed.

**Recorded.** `CorpusSample.record` lists every chunk taken, by source, offsets
and word count, in the order it was taken. It is serialisable, and it is what
makes an inferred ontology traceable to the evidence it was inferred from.

The default budget
------------------
`DEFAULT_SAMPLE_WORDS` is 8,000 words. Schema signals repeat: a record set shows
all its columns in its first row, and a type worth a node in a lightweight
ontology — around ten types, `03-schema` — recurs across documents rather than
hiding in one paragraph. So the budget only has to be large enough for each
stratum to show its repeated shapes a few times: 8,000 words is some forty
short records and a dozen pages of prose. The deterministic proposers read that
in milliseconds. The model proposer reads a prefix of it (`prompt_words`), and a
prefix is itself a stratified sample, because chunks are kept in the order the
round-robin took them. Doubling the budget is the check that it is big enough:
the tests assert the inferred schema does not move when it doubles.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from odke.chunking import SentenceChunker
from odke.stages import Chunker
from odke.types import Chunk, Document

DEFAULT_SAMPLE_WORDS = 8_000
# Small enough that one sampled chunk is one idea, large enough to hold a
# "such as" list and the sentence around it.
DEFAULT_CHUNK_WORDS = 120
# Word counts below which a document is short, then medium; anything longer is long.
_BANDS = ((250, "short"), (2_500, "medium"))
# A chunk too large for what is left of the budget is skipped, not the end of
# sampling; this many skips in a row is.
_MAX_MISSES = 50

T = TypeVar("T")


class SampledChunk(BaseModel):
    """One chunk the sampler took, addressed so it can be found again.

    `doc_id` is the id of this load of the corpus. `doc_key` is stable across
    loads, and together with `start`/`end` is what reproducibility is asserted on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    doc_key: str
    source: str | None
    modality: str
    stratum: str
    index: int
    start: int
    end: int
    words: int


class SampleRecord(BaseModel):
    """What was sampled, in the order it was taken. Kept beside an inferred ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    budget_words: int
    words: int = 0
    documents_seen: int = 0
    documents_sampled: int = 0
    chunks: tuple[SampledChunk, ...] = ()

    @property
    def strata(self) -> dict[str, int]:
        """Chunks taken per stratum — the first thing to read when a sample looks lopsided."""
        return dict(sorted(Counter(c.stratum for c in self.chunks).items()))


@dataclass(frozen=True, slots=True)
class CorpusSample:
    """The sampled chunks, the documents they belong to, and the record of both."""

    chunks: tuple[Chunk, ...]
    documents: Mapping[str, Document]
    record: SampleRecord

    def document(self, chunk: Chunk) -> Document:
        return self.documents[chunk.doc_id]


def words_in(text: str) -> int:
    return len(text.split())


def document_key(doc: Document) -> str:
    """What a document is, independent of the id a loader drew for it.

    The row index is part of it, so two identical rows of one CSV stay two
    documents.
    """
    digest = hashlib.sha256()
    for part in (doc.uri or "", str(doc.metadata.get("row_index", "")), doc.text):
        digest.update(part.encode("utf-8", "surrogatepass"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def sample_corpus(
    corpus: Iterable[Document],
    *,
    words: int = DEFAULT_SAMPLE_WORDS,
    seed: int = 0,
    chunker: Chunker | None = None,
) -> CorpusSample:
    """A stratified, seeded sample of `corpus` of at most `words` words.

    The first chunk taken is kept even if it alone is over the budget, so a
    corpus is never sampled to nothing. Chunks come from `chunker`, by default a
    `SentenceChunker` of `DEFAULT_CHUNK_WORDS`, and only for the documents the
    round-robin actually visits.
    """
    if words < 1:
        raise ValueError("words must be at least 1")
    chunker = chunker if chunker is not None else SentenceChunker(DEFAULT_CHUNK_WORDS)
    docs = sorted(((document_key(d), d) for d in corpus), key=lambda kd: kd[0])
    rng = random.Random(seed)

    groups: dict[str, dict[str, list[tuple[str, Document]]]] = {}
    for key, doc in docs:
        stratum = f"{doc.modality}/{_band(words_in(doc.text))}"
        source = _source(doc) or f"doc:{key}"
        groups.setdefault(stratum, {}).setdefault(source, []).append((key, doc))

    def taken_from(key: str, doc: Document, stratum: str) -> Iterator[tuple[Chunk, str, str]]:
        # Per-document shuffle on its own generator, so which chunk a document
        # gives up first does not depend on when the round-robin reached it.
        chunks = [c for c in chunker.chunk(doc) if c.text.strip()]
        random.Random(f"{seed}:{key}").shuffle(chunks)
        for chunk in chunks:
            yield chunk, key, stratum

    lanes: list[Iterator[tuple[Chunk, str, str]]] = []
    for stratum in sorted(groups):
        sources = sorted(groups[stratum])
        rng.shuffle(sources)
        per_source = []
        for source in sources:
            members = list(groups[stratum][source])
            rng.shuffle(members)
            per_source.append(_interleave([taken_from(k, d, stratum) for k, d in members]))
        lanes.append(_interleave(per_source))
    rng.shuffle(lanes)

    by_id = {doc.id: doc for _, doc in docs}
    picked: list[Chunk] = []
    entries: list[SampledChunk] = []
    total = misses = 0
    for chunk, key, stratum in _interleave(lanes):
        size = words_in(chunk.text)
        if picked and total + size > words:
            misses += 1
            if misses >= _MAX_MISSES:
                break
            continue
        misses = 0
        doc = by_id[chunk.doc_id]
        picked.append(chunk)
        entries.append(
            SampledChunk(
                doc_id=chunk.doc_id,
                doc_key=key,
                source=_source(doc),
                modality=doc.modality,
                stratum=stratum,
                index=chunk.index,
                start=chunk.start,
                end=chunk.end,
                words=size,
            )
        )
        total += size
        if total >= words:
            break

    sampled = {c.doc_id: by_id[c.doc_id] for c in picked}
    record = SampleRecord(
        seed=seed,
        budget_words=words,
        words=total,
        documents_seen=len(docs),
        documents_sampled=len(sampled),
        chunks=tuple(entries),
    )
    return CorpusSample(chunks=tuple(picked), documents=sampled, record=record)


def _band(count: int) -> str:
    return next((name for limit, name in _BANDS if count < limit), "long")


def _source(doc: Document) -> str | None:
    # Records loaded from memory share no URI but are still one record set.
    if doc.uri:
        return doc.uri
    return "records" if doc.modality == "structured" else None


def _interleave(iterators: list[Iterator[T]]) -> Iterator[T]:
    """One item from each iterator in turn, dropping each as it runs out."""
    active = deque(iterators)
    while active:
        current = active.popleft()
        try:
            item = next(current)
        except StopIteration:
            continue
        yield item
        active.append(current)


__all__ = [
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_SAMPLE_WORDS",
    "CorpusSample",
    "SampleRecord",
    "SampledChunk",
    "document_key",
    "sample_corpus",
    "words_in",
]
