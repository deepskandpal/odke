"""Corpus sampling (#31): stratified, reproducible under a seed, and recorded."""

from __future__ import annotations

from pathlib import Path

import pytest

from odke import Document
from odke.infer import DEFAULT_SAMPLE_WORDS, SampleRecord, sample_corpus
from odke.loaders import DirectoryLoader, RecordsLoader


def _corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    rows = "\n".join(f"Person {i},{1900 + i},Company {i % 7}" for i in range(300))
    (root / "people.csv").write_text(f"name,born,employer\n{rows}\n", encoding="utf-8")
    for i in range(3):
        body = " ".join(
            f"Note {i} sentence {j} mentions languages such as Python and Rust." for j in range(40)
        )
        (root / f"note{i}.md").write_text(f"# Note {i}\n\n{body}\n", encoding="utf-8")
    return root


def _keys(record: SampleRecord) -> list[tuple[str, int, int]]:
    return [(c.doc_key, c.start, c.end) for c in record.chunks]


def test_the_same_corpus_and_seed_give_the_same_sample_across_loads(tmp_path: Path) -> None:
    """Loaders draw random document ids; the sample must not depend on them."""
    root = _corpus_dir(tmp_path)
    first = sample_corpus(DirectoryLoader().load(root), words=600, seed=7)
    second = sample_corpus(DirectoryLoader().load(root), words=600, seed=7)
    assert _keys(first.record) == _keys(second.record)
    assert {c.doc_id for c in first.chunks}.isdisjoint(c.doc_id for c in second.chunks)
    other = sample_corpus(DirectoryLoader().load(root), words=600, seed=8)
    assert _keys(other.record) != _keys(first.record)


def test_the_budget_holds_and_every_chunk_is_a_slice_of_its_document(tmp_path: Path) -> None:
    sample = sample_corpus(DirectoryLoader().load(_corpus_dir(tmp_path)), words=500, seed=1)
    assert 0 < sample.record.words <= 500
    assert sample.record.words == sum(c.words for c in sample.record.chunks)
    for chunk in sample.chunks:
        assert sample.document(chunk).text[chunk.start : chunk.end] == chunk.text


def test_a_big_record_set_cannot_crowd_out_the_prose(tmp_path: Path) -> None:
    """300 rows from one file are one source, taking turns with three notes."""
    sample = sample_corpus(DirectoryLoader().load(_corpus_dir(tmp_path)), words=400, seed=3)
    sources = {c.source for c in sample.record.chunks}
    notes = {s for s in sources if s and s.endswith(".md")}
    assert len(notes) == 3
    assert set(sample.record.strata) == {"structured/short", "unstructured/medium"}


def test_one_huge_document_cannot_fill_the_budget() -> None:
    huge = Document(
        text=" ".join(f"Sentence {i} is about nothing much." for i in range(3000)),
        uri="file:///huge.txt",
    )
    small = [
        Document(text=f"Note {i} is short. It has two sentences.", uri=f"file:///note{i}.txt")
        for i in range(5)
    ]
    sample = sample_corpus([huge, *small], words=400, seed=11)
    sources = {c.source for c in sample.record.chunks}
    assert {d.uri for d in small} <= sources
    assert sample.record.documents_seen == 6 and sample.record.documents_sampled == 6


def test_records_from_memory_are_one_source_not_one_each() -> None:
    rows = RecordsLoader().load({"name": f"Row {i}", "size": i} for i in range(200))
    note = Document(text="A lone note about vehicles such as cars and trucks.")
    sample = sample_corpus([*rows, note], words=60, seed=0)
    assert note.id in sample.documents


def test_the_record_round_trips_and_names_what_was_seen(tmp_path: Path) -> None:
    sample = sample_corpus(DirectoryLoader().load(_corpus_dir(tmp_path)), words=300, seed=5)
    record = sample.record
    assert SampleRecord.model_validate_json(record.model_dump_json()) == record
    assert (record.seed, record.budget_words, record.documents_seen) == (5, 300, 303)
    assert sum(record.strata.values()) == len(record.chunks)


def test_an_empty_corpus_samples_to_nothing_and_a_bad_budget_is_refused() -> None:
    empty = sample_corpus([], seed=0)
    assert empty.chunks == () and empty.record.words == 0
    assert empty.record.budget_words == DEFAULT_SAMPLE_WORDS
    with pytest.raises(ValueError, match="at least 1"):
        sample_corpus([], words=0)


def test_an_oversized_first_chunk_is_still_taken() -> None:
    doc = Document(text=" ".join(["word"] * 500) + ".")
    sample = sample_corpus([doc], words=10)
    assert len(sample.chunks) == 1 and sample.record.words == 500
