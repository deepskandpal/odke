"""Ontology inference: a bootstrap for a corpus that arrives with no schema.

DECISIONS #8: inference is a bootstrap, not a mode. Nothing here runs on its
own — `Pipeline` never calls it — and what it returns is an ontology marked
`inferred=True`, for a person to review, edit and freeze before anything is
extracted against it.

Deterministic routes first (`_odke-design` §6): record shapes, Hearst patterns
and co-occurrence propose candidates with the spans that produced them; a model
through the `infer` role only names, merges and ranks them.

    from odke.infer import infer_ontology

    inference = infer_ontology(docs, llm=False)    # free, no model
    draft = inference.ontology                     # inferred=True
    frozen = draft.freeze(by="reviewer")           # after review
"""

from odke.infer.build import (
    DEFAULT_MAX_PREDICATES,
    DEFAULT_MAX_TYPES,
    DEFAULT_MIN_SUPPORT,
    Dropped,
    EntryEvidence,
    EvidenceRef,
    Inference,
    OntologyInferrer,
    infer_ontology,
)
from odke.infer.candidates import PredicateCandidate, Proposals, Proposer, TypeCandidate
from odke.infer.llm import InferenceCall, LLMProposer, ProposalRejection
from odke.infer.merge import Merged, MergeDecision, merge
from odke.infer.propose import (
    CooccurrenceProposer,
    HearstProposer,
    RecordShapeProposer,
    deterministic_proposers,
    propose,
)
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
    "DEFAULT_MAX_PREDICATES",
    "DEFAULT_MAX_TYPES",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_SAMPLE_WORDS",
    "CooccurrenceProposer",
    "CorpusSample",
    "Dropped",
    "EntryEvidence",
    "EvidenceRef",
    "HearstProposer",
    "Inference",
    "InferenceCall",
    "LLMProposer",
    "MergeDecision",
    "Merged",
    "OntologyInferrer",
    "PredicateCandidate",
    "ProposalRejection",
    "Proposals",
    "Proposer",
    "RecordShapeProposer",
    "SampleRecord",
    "SampledChunk",
    "TypeCandidate",
    "deterministic_proposers",
    "document_key",
    "infer_ontology",
    "merge",
    "propose",
    "sample_corpus",
]
