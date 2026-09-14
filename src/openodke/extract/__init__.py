"""Extractors: one chunk to candidate facts, held to the ontology.

Each satisfies `openodke.stages.Extractor` without inheriting from it. They share
one way of keying entities and citing spans (`openodke.extract._common`), so a fact
from a table and a fact from prose are the same shape and merge by signature.
"""

from openodke.extract._common import entity_key
from openodke.extract.hybrid import HybridExtractor, PathReport, merge
from openodke.extract.llm import LLMExtractor, ModelCall, Rejection, response_schema
from openodke.extract.pattern import PatternExtractor, RecordMapping

__all__ = [
    "HybridExtractor",
    "LLMExtractor",
    "ModelCall",
    "PathReport",
    "PatternExtractor",
    "RecordMapping",
    "Rejection",
    "entity_key",
    "merge",
    "response_schema",
]
