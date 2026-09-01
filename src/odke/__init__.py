"""odke — ontology-guided knowledge extraction.

Text in, a grounded knowledge graph out, ready for Neo4j or any graph store.

An independent implementation of the architecture in ODKE+ (arXiv:2509.04696),
generalised from one production knowledge graph to a general-purpose SDK. See
NOTICE for the relationship to that paper.
"""

from odke.ontology import EntityType, Ontology, OntologySnippet, Predicate
from odke.pipeline import Pipeline
from odke.types import (
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    SourceTier,
    Span,
)

__version__ = "0.0.1"

__all__ = [
    "Document",
    "Entity",
    "EntityType",
    "Evidence",
    "Fact",
    "GroundingVerdict",
    "KnowledgeGraph",
    "Ontology",
    "OntologySnippet",
    "Pipeline",
    "Predicate",
    "SourceTier",
    "Span",
    "__version__",
]
