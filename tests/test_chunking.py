"""The chunker's one promise: offsets survive, and no sentence is ever cut."""

from __future__ import annotations

import random

import pytest

from odke import Chunk, Chunker, Document, Pipeline, SentenceChunker, Span

# Words that cannot trip the abbreviation rules: never one letter, never in the
# abbreviation list, and a mix of scripts, combining marks and emoji.
_WORDS = (
    "ada",
    "lovelace",
    "wrote",
    "notes",
    "engine",
    "naïve",
    "Zoë",
    "日本語",
    "αβγ",
    "e\N{COMBINING ACUTE ACCENT}te",
    "emoji🙂",
    "résumé",
    "data",
    "42",
    "x86",
    "über",
    "straße",
    "ok",
)
_TERMINATORS = (".", "!", "?", "…", '."', ".)", "?!", "。")
_INNER_SPACE = (" ", "  ", "\t", "\r\n", "\n", " \N{NO-BREAK SPACE}", " \r\n ")
_PARAGRAPH_BREAKS = ("\n\n", "\r\n\r\n", "\n \t\n", "\n\n\n\n", "\r\r", "\r\n  \r\n\r\n")


def _generated(seed: int) -> tuple[Document, list[tuple[int, int]]]:
    """A document and the true (start, end) of every sentence in it."""
    rng = random.Random(seed)
    parts: list[str] = [rng.choice(("", " ", "\r\n", "\n\n  "))]
    sentences: list[tuple[int, int]] = []
    for p in range(rng.randint(1, 6)):
        if p:
            parts.append(rng.choice(_PARAGRAPH_BREAKS))
        count = rng.randint(1, 5)
        for s in range(count):
            if s:
                parts.append(rng.choice(_INNER_SPACE))
            words = [rng.choice(_WORDS) for _ in range(rng.randint(1, 30))]
            words[0] = words[0][0].upper() + words[0][1:]
            body = words[0] + "".join(rng.choice(_INNER_SPACE) + w for w in words[1:])
            # The last sentence of a paragraph may be an unpunctuated fragment.
            if s < count - 1 or rng.random() < 0.7:
                body += rng.choice(_TERMINATORS)
            start = sum(map(len, parts))
            parts.append(body)
            sentences.append((start, start + len(body)))
    parts.append(rng.choice(("", " ", "\n", "\r\n\r\n")))
    return Document(id=f"g{seed}", text="".join(parts)), sentences


def _words(text: str) -> int:
    return len(text.split())


@pytest.mark.parametrize("seed", range(300))
def test_generated_documents_keep_offsets_and_sentences(seed: int) -> None:
    doc, sentences = _generated(seed)
    cap = random.Random(seed).choice((1, 5, 12, 40, 1000))
    chunks = list(SentenceChunker(max_words=cap).chunk(doc))
    starts = {s for s, _ in sentences}
    ends = {e for _, e in sentences}

    assert [c.index for c in chunks] == list(range(len(chunks)))
    previous_end = 0
    for c in chunks:
        assert doc.text[c.start : c.end] == c.text
        assert Span(doc_id=doc.id, start=c.start, end=c.end, quote=c.text).is_faithful(doc)
        # Boundaries fall only where a sentence begins or ends.
        assert c.start in starts and c.end in ends
        assert c.start >= previous_end
        inside = [s for s in sentences if c.start <= s[0] and s[1] <= c.end]
        assert _words(c.text) <= cap or len(inside) == 1
        previous_end = c.end
    # Every non-whitespace character is in exactly one chunk.
    covered = sum(len(c.text) - sum(ch.isspace() for ch in c.text) for c in chunks)
    assert covered == sum(not ch.isspace() for ch in doc.text)


def test_crlf_text_slices_back_exactly() -> None:
    text = "First line of one sentence\r\ncontinues here. Second one.\r\n\r\nNew paragraph.\r\n"
    doc = Document(text=text)
    chunks = list(SentenceChunker(max_words=3).chunk(doc))
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)
    # A single CRLF is a line break inside a sentence, not a paragraph break.
    assert chunks[0].text == "First line of one sentence\r\ncontinues here."
    assert [c.text for c in chunks[1:]] == ["Second one.", "New paragraph."]


def test_unicode_and_whitespace_runs_are_never_normalised() -> None:
    text = (
        "  Zoë   met\t\tthe 🙂 team.\N{NO-BREAK SPACE} Straße ist lang!"
        "\n\n\n\t日本語の文です。次の文。  "
    )
    doc = Document(text=text)
    chunks = list(SentenceChunker(max_words=3).chunk(doc))
    assert [c.text for c in chunks] == [
        "Zoë   met\t\tthe 🙂 team.",
        "Straße ist lang!",
        "日本語の文です。次の文。",
    ]
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)


def test_a_document_with_no_punctuation_is_one_sentence_per_paragraph() -> None:
    para = " ".join(["word"] * 30)
    doc = Document(text=f"{para}\n{para}\n\n{para}")
    chunks = list(SentenceChunker(max_words=10).chunk(doc))
    # Over the cap, but cutting would split a sentence; each paragraph stands alone.
    assert [c.text for c in chunks] == [f"{para}\n{para}", para]


def test_a_sentence_longer_than_the_cap_is_its_own_chunk() -> None:
    long = "This sentence has far more words than the cap allows by a wide margin."
    doc = Document(text=f"Short one. {long} Short two.")
    assert [c.text for c in SentenceChunker(max_words=5).chunk(doc)] == [
        "Short one.",
        long,
        "Short two.",
    ]


def test_sentences_pack_up_to_the_cap() -> None:
    doc = Document(text="One two three. Four five. Six seven eight. Nine.")
    assert [c.text for c in SentenceChunker(max_words=5).chunk(doc)] == [
        "One two three. Four five.",
        "Six seven eight. Nine.",
    ]


def test_a_paragraph_break_is_preferred_to_a_sentence_break() -> None:
    doc = Document(text="Aa bb cc dd. Ee ff.\n\nGg hh. Ii jj kk ll.")
    # Six words would fit "...Ee ff. Gg hh." but the paragraph ends first.
    assert [c.text for c in SentenceChunker(max_words=8).chunk(doc)] == [
        "Aa bb cc dd. Ee ff.",
        "Gg hh. Ii jj kk ll.",
    ]


def test_a_paragraph_break_that_would_strand_a_heading_is_passed_over() -> None:
    doc = Document(text="# Notes\n\nAa bb cc dd ee. Ff gg hh ii jj. Kk ll mm nn oo.")
    chunks = [c.text for c in SentenceChunker(max_words=12).chunk(doc)]
    assert chunks[0] == "# Notes\n\nAa bb cc dd ee. Ff gg hh ii jj."


def test_abbreviations_and_initials_do_not_end_a_sentence() -> None:
    doc = Document(text="Dr. Ada met J. R. Tolkien, e.g. at noon. Then she left.")
    assert [c.text for c in SentenceChunker(max_words=1).chunk(doc)] == [
        "Dr. Ada met J. R. Tolkien, e.g. at noon.",
        "Then she left.",
    ]


def test_decimals_and_domains_are_not_sentence_ends() -> None:
    doc = Document(text="Pi is 3.14 per example.com today. Next.")
    assert [c.text for c in SentenceChunker(max_words=1).chunk(doc)] == [
        "Pi is 3.14 per example.com today.",
        "Next.",
    ]


def test_overlap_repeats_trailing_sentences_and_always_makes_progress() -> None:
    doc = Document(text="Aa bb. Cc dd. Ee ff. Gg hh. Ii jj.")
    chunks = [c.text for c in SentenceChunker(max_words=4, overlap=1).chunk(doc)]
    assert chunks == ["Aa bb. Cc dd.", "Cc dd. Ee ff.", "Ee ff. Gg hh.", "Gg hh. Ii jj."]
    # An overlap wider than the cap is shed rather than looping forever.
    wide = [c.text for c in SentenceChunker(max_words=2, overlap=5).chunk(doc)]
    assert wide == ["Aa bb.", "Cc dd.", "Ee ff.", "Gg hh.", "Ii jj."]
    # When the carried sentence would crowd out the next one, it is dropped.
    shed = Document(text="Aa bb. Cc dd. Ee ff gg hh.")
    assert [c.text for c in SentenceChunker(max_words=4, overlap=1).chunk(shed)] == [
        "Aa bb. Cc dd.",
        "Ee ff gg hh.",
    ]


def test_whitespace_only_and_empty_documents_yield_no_chunks() -> None:
    assert list(SentenceChunker().chunk(Document(text=""))) == []
    assert list(SentenceChunker().chunk(Document(text=" \r\n\t\n\n "))) == []


def test_bad_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="max_words"):
        SentenceChunker(max_words=0)
    with pytest.raises(ValueError, match="overlap"):
        SentenceChunker(overlap=-1)


def test_it_is_a_chunker_and_runs_in_the_pipeline() -> None:
    from odke import Entity, Fact, Ontology

    class _Echo:
        def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
            return [Fact(subject=Entity(key=str(chunk.index), type="T"), predicate="p")]

    chunker = SentenceChunker(max_words=3)
    assert isinstance(chunker, Chunker)
    kg = Pipeline(Ontology(), _Echo(), chunker=chunker).run(
        [Document(text="Aa bb cc. Dd ee ff. Gg.")]
    )
    assert kg.stats["chunks"] == 3
