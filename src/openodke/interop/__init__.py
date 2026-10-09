"""Other extractors' output, in a form openodke can check (DECISIONS #24).

`triples` is the documented input format every adapter maps onto: one triple
per JSON Lines row, with offsets, a quote, or neither. Each adapter turns one
library's output into those rows and the texts they cite, without importing
the library.
"""

from __future__ import annotations

from openodke.interop.graphrag import from_graphrag, read_graphrag
from openodke.interop.langchain import from_graph_documents
from openodke.interop.langextract import attribute_triples, from_langextract
from openodke.interop.neo4j import read_neo4j, write_verdicts
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
    "from_graphrag",
    "from_langextract",
    "read_graphrag",
    "read_neo4j",
    "read_triples",
    "to_fact",
    "write_verdicts",
]
