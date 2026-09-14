"""Ontology inference: a bootstrap for a corpus that arrives with no schema.

DECISIONS #8: inference is a bootstrap, not a mode. Nothing here runs on its
own — `Pipeline` never calls it — and what it returns is an ontology marked
`inferred=True`, for a person to review, edit and freeze before anything is
extracted against it.
"""

from odke.infer.sample import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_SAMPLE_WORDS,
    CorpusSample,
    SampledChunk,
    SampleRecord,
    document_key,
    sample_corpus,
)

__all__ = [
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_SAMPLE_WORDS",
    "CorpusSample",
    "SampleRecord",
    "SampledChunk",
    "document_key",
    "sample_corpus",
]
