"""openodke — ontology-guided knowledge extraction.

Text in, a grounded knowledge graph out, ready for Neo4j or any graph store.

An independent implementation of the architecture in ODKE+ (arXiv:2509.04696),
generalised from one production knowledge graph to a general-purpose SDK. See
NOTICE for the relationship to that paper.
"""

from openodke.chunking import SentenceChunker
from openodke.corroborate import (
    EvidenceScorer,
    NativeResolver,
    SignatureCorroborator,
    ValueNormalizer,
)
from openodke.extract import HybridExtractor, LLMExtractor, PatternExtractor
from openodke.ontology import (
    Diagnostic,
    EntityType,
    Ontology,
    OntologyImportWarning,
    OntologyLoadError,
    OntologySnippet,
    Predicate,
    Qualifier,
)
from openodke.pipeline import DoubleStageWarning, Pipeline
from openodke.stages import (
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
from openodke.types import (
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
from openodke.validators import VerdictValidator

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
    "EvidenceScorer",
    "Extractor",
    "Fact",
    "Grounder",
    "GroundingVerdict",
    "HybridExtractor",
    "Inferrer",
    "KnowledgeGraph",
    "LLMExtractor",
    "LinkKind",
    "Loader",
    "NativeResolver",
    "Normalizer",
    "Ontology",
    "OntologyImportWarning",
    "OntologyLoadError",
    "OntologySnippet",
    "PatternExtractor",
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
    "SentenceChunker",
    "SignatureCorroborator",
    "Sink",
    "SourceTier",
    "Span",
    "ValidationVerdict",
    "Validator",
    "ValueNormalizer",
    "VerdictValidator",
    "__version__",
]
