"""`replay:Replayed`: what directories prepared before the triples format still name.

`competitors.py` now writes `{"use": "triples", ...}`, openodke's own stage for
another extractor's triples (docs/inputs.md). This keeps a competitor directory
prepared under PR #105 runnable as it is. It builds the same facts the old
stage did, checked on both Re-DocRED competitors' output. Entity keys now carry
their type, as openodke's extractors key them, and the whole-document span is
marked `context` rather than quoted.
"""

from __future__ import annotations

from openodke.interop import TriplesExtractor


class Replayed(TriplesExtractor):
    def __init__(self, path: str, system: str = "competitor") -> None:
        super().__init__(path, extractor=system)
