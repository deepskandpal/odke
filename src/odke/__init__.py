"""odke — ontology-guided knowledge extraction.

Text in, a grounded knowledge graph out, ready for Neo4j or any graph store.

An independent implementation of the architecture in ODKE+ (arXiv:2509.04696),
generalised from one production knowledge graph to a general-purpose SDK. See
NOTICE for the relationship to that paper.
"""

from odke.corroborate import NativeResolver, ValueNormalizer
from odke.ontology import (
    Diagnostic,
    EntityType,
    Ontology,
    OntologyLoadError,
    OntologySnippet,
    Predicate,
    Qualifier,
)
from odke.pipeline import DoubleStageWarning, Pipeline
from odke.stages import (
    Chunker,
    Constrainer,
    Corroborator,
    Delegated,
    Extractor,
    Grounder,
    Inferrer,
    Loader,
    Normalizer,
    PlatformProfile,
    Resolver,
    Router,
    Scorer,
    Sink,
    Validator,
)
from odke.types import (
    Chunk,
    Document,
    Entity,
    EntityLink,
    Evidence,
    Fact,
    GroundingVerdict,
    KnowledgeGraph,
    LinkKind,
    Polarity,
    Resolution,
    RouteVerdict,
    SourceTier,
    Span,
    ValidationVerdict,
)

__version__ = "0.0.1"

__all__ = [
    "Chunk",
    "Chunker",
    "Constrainer",
    "Corroborator",
    "Delegated",
    "Diagnostic",
    "Document",
    "DoubleStageWarning",
    "Entity",
    "EntityLink",
    "EntityType",
    "Evidence",
    "Extractor",
    "Fact",
    "Grounder",
    "GroundingVerdict",
    "Inferrer",
    "KnowledgeGraph",
    "LinkKind",
    "Loader",
    "NativeResolver",
    "Normalizer",
    "Ontology",
    "OntologyLoadError",
    "OntologySnippet",
    "Pipeline",
    "PlatformProfile",
    "Polarity",
    "Predicate",
    "Qualifier",
    "Resolution",
    "Resolver",
    "RouteVerdict",
    "Router",
    "Scorer",
    "Sink",
    "SourceTier",
    "Span",
    "ValidationVerdict",
    "Validator",
    "ValueNormalizer",
    "__version__",
]
