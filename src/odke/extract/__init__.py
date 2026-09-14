"""Extractors: one chunk to candidate facts, held to the ontology.

Each satisfies `odke.stages.Extractor` without inheriting from it. They share
one way of keying entities and citing spans (`odke.extract._common`), so a fact
from a table and a fact from prose are the same shape and merge by signature.
"""

from odke.extract._common import entity_key
from odke.extract.pattern import PatternExtractor, RecordMapping

__all__ = ["PatternExtractor", "RecordMapping", "entity_key"]
