"""openodke — ontology-guided knowledge extraction.

Text in, a grounded knowledge graph out, ready for Neo4j or any graph store.

An independent implementation of the architecture in ODKE+ (arXiv:2509.04696),
generalised from one production knowledge graph to a general-purpose SDK. See
NOTICE for the relationship to that paper.
"""

import importlib.metadata
from typing import TYPE_CHECKING

from openodke._renamed import Renamed, module_getattr
from openodke.chunking import SentenceChunker
from openodke.corroborate import (
    EvidenceScorer,
    NativeResolver,
    SignatureCorroborator,
    ValueNormalizer,
)
from openodke.extract import HybridExtractor, LLMExtractor, PatternExtractor
from openodke.gate import VerdictGate
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
    FactLookup,
    Gate,
    Grounder,
    Inferrer,
    Loader,
    Normalizer,
    PlatformProfile,
    Resolver,
    Router,
    Scorer,
    Sink,
    StoreLookup,
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
    SpanOrigin,
    Support,
    ValidationVerdict,
)
from openodke.validator import Validated, ValidationReport, Validator

# Read from the installed distribution, so pyproject.toml is the only place the
# version is written and a wheel cannot report a release it is not.
__version__ = importlib.metadata.version("openodke")

if TYPE_CHECKING:
    # What a type checker sees; at run time the 0.2 name comes from `__getattr__`.
    VerdictValidator = VerdictGate

# The gate's 0.2 name (DECISIONS #26). `openodke.Validator` named the gate too,
# until 1.0.0 gave the name to the whole layer (#129); `stages.Validator` still
# names the gate, with a warning.
__getattr__ = module_getattr(
    __name__, {"VerdictValidator": Renamed(VerdictGate, "openodke.VerdictGate")}
)

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
    "FactLookup",
    "Gate",
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
    "SpanOrigin",
    "StoreLookup",
    "Support",
    "ValidationReport",
    "ValidationVerdict",
    "Validated",
    "Validator",
    "ValueNormalizer",
    "VerdictGate",
    "VerdictValidator",
    "__version__",
]
