"""Other extractors' output, in a form openodke can check (DECISIONS #24).

`triples` is the documented input format every adapter maps onto: one triple
per JSON Lines row, with offsets, a quote, or neither.
"""

from __future__ import annotations

from openodke.interop.triples import (
    LITERAL_TYPES,
    THING,
    TripleRow,
    TriplesExtractor,
    read_triples,
    to_fact,
)

__all__ = [
    "LITERAL_TYPES",
    "THING",
    "TripleRow",
    "TriplesExtractor",
    "read_triples",
    "to_fact",
]
