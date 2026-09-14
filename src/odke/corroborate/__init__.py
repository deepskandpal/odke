"""After extraction: normalise, resolve, corroborate, score.

The four batch-side stages of the pipeline, each a plain class satisfying its
Protocol in `odke.stages`, with no dependency beyond the base install. Each one
records what it did on the objects it returns — under the reserved keys in
`odke.corroborate.provenance` — rather than discarding what it replaced.
"""

from odke.corroborate.normalize import (
    ValueNormalizer,
    name_key,
    normalize_date,
    normalize_quantity,
    normalize_value,
)
from odke.corroborate.provenance import NAME_KEY, SOURCE_FORM

__all__ = [
    "NAME_KEY",
    "SOURCE_FORM",
    "ValueNormalizer",
    "name_key",
    "normalize_date",
    "normalize_quantity",
    "normalize_value",
]
