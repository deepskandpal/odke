"""Other extractors' output, in a form openodke can check (DECISIONS #24).

`triples` is the documented input format every adapter maps onto: one triple
per JSON Lines row, with offsets, a quote, or neither. Each adapter turns one
library's output into those rows and the texts they cite, without importing
the library.
"""

from __future__ import annotations

from openodke.interop.langchain import from_graph_documents
from openodke.interop.langextract import attribute_triples, from_langextract
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
    "attribute_triples",
    "from_graph_documents",
    "from_langextract",
    "read_triples",
    "to_fact",
]
