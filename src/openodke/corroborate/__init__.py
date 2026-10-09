"""After extraction: normalise, resolve, corroborate, score.

The four batch-side stages of the pipeline, each a plain class satisfying its
Protocol in `openodke.stages`, with no dependency beyond the base install. Each one
records what it did on the objects it returns — under the reserved keys in
`openodke.corroborate.provenance` — rather than discarding what it replaced.
`partners` is the step between resolving and corroborating that adds each fact's
inverse or symmetric partner, marked `odke.derived` (DECISIONS #28).
`MemoryLookup` is the in-memory store a resolver looks entities up in
(DECISIONS #31).
"""

from openodke.corroborate.inverses import derived_from, partners
from openodke.corroborate.lookup import Embed, MemoryLookup
from openodke.corroborate.merge import (
    DEFAULT_INTERVALS,
    SignatureCorroborator,
    independent_sources,
    source_of,
)
from openodke.corroborate.normalize import (
    ValueNormalizer,
    name_key,
    normalize_date,
    normalize_quantity,
    normalize_value,
)
from openodke.corroborate.provenance import (
    CHECK,
    CONFLICT,
    DERIVED,
    NAME_KEY,
    NEAR_DUPLICATES,
    SCORE,
    SOURCE_FORM,
    WIDEN,
)
from openodke.corroborate.resolve import (
    LINKER,
    BlockKeys,
    NativeResolver,
    block_keys,
    candidate_pairs,
    domain_of,
    name_similarity,
)
from openodke.corroborate.score import DEFAULT_VERDICT_WEIGHTS, EvidenceScorer, combine

__all__ = [
    "BlockKeys",
    "CHECK",
    "CONFLICT",
    "DEFAULT_INTERVALS",
    "DEFAULT_VERDICT_WEIGHTS",
    "DERIVED",
    "Embed",
    "EvidenceScorer",
    "LINKER",
    "MemoryLookup",
    "NAME_KEY",
    "NEAR_DUPLICATES",
    "SCORE",
    "SOURCE_FORM",
    "WIDEN",
    "NativeResolver",
    "SignatureCorroborator",
    "ValueNormalizer",
    "block_keys",
    "candidate_pairs",
    "combine",
    "derived_from",
    "domain_of",
    "independent_sources",
    "name_key",
    "name_similarity",
    "normalize_date",
    "normalize_quantity",
    "normalize_value",
    "partners",
    "source_of",
]
