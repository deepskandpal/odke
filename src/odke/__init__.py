"""odke — ontology-guided knowledge extraction.

Text in, a grounded knowledge graph out, ready for Neo4j or any graph store.

An independent implementation of the architecture in ODKE+ (arXiv:2509.04696),
generalised from one production knowledge graph to a general-purpose SDK. See
NOTICE for the relationship to that paper.
"""

from odke.ontology import EntityType, Ontology, OntologySnippet, Predicate, Qualifier
from odke.pipeline import Pipeline
from odke.types import (
    Document,
    Entity,
    EntityLink,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    LinkKind,
    Polarity,
    SourceTier,
    Span,
)

__version__ = "0.0.1"

__all__ = [
    "Document",
    "Entity",
    "EntityLink",
    "EntityType",
    "Evidence",
    "Fact",
    "GroundingVerdict",
    "KnowledgeGraph",
    "LinkKind",
    "Ontology",
    "OntologySnippet",
    "Pipeline",
    "Polarity",
    "Predicate",
    "Qualifier",
    "SourceTier",
    "Span",
    "__version__",
]
