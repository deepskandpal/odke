"""After extraction: normalise, resolve, corroborate, score.

The four batch-side stages of the pipeline, each a plain class satisfying its
Protocol in `odke.stages`, with no dependency beyond the base install. Each one
records what it did on the objects it returns — under the reserved keys in
`odke.corroborate.provenance` — rather than discarding what it replaced.
"""

from odke.corroborate.merge import (
    DEFAULT_INTERVALS,
    SignatureCorroborator,
    independent_sources,
    source_of,
)
from odke.corroborate.normalize import (
    ValueNormalizer,
    name_key,
    normalize_date,
    normalize_quantity,
    normalize_value,
)
from odke.corroborate.provenance import CONFLICT, NAME_KEY, SCORE, SOURCE_FORM
from odke.corroborate.resolve import (
    LINKER,
    NativeResolver,
    candidate_pairs,
    domain_of,
    name_similarity,
)
from odke.corroborate.score import DEFAULT_VERDICT_WEIGHTS, EvidenceScorer, combine

__all__ = [
    "CONFLICT",
    "DEFAULT_INTERVALS",
    "DEFAULT_VERDICT_WEIGHTS",
    "EvidenceScorer",
    "LINKER",
    "NAME_KEY",
    "SCORE",
    "SOURCE_FORM",
    "NativeResolver",
    "SignatureCorroborator",
    "ValueNormalizer",
    "candidate_pairs",
    "combine",
    "domain_of",
    "independent_sources",
    "name_key",
    "name_similarity",
    "normalize_date",
    "normalize_quantity",
    "normalize_value",
    "source_of",
]
