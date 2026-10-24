"""What failed in a run the Evaluator scores, so a failure never becomes a number (rule R1).

A stage that fails for one document leaves it out and names it
(`KnowledgeGraph.stats["failed"]`, `ValidationReport.failed`). A model grounder
whose call fails, or whose answer cannot be read, raises nothing: it leaves the
fact `UNCHECKED` and counts the call, so one bad call never loses a batch. An
evaluator reads both, leaves those documents out of every row it scores, and
says so; with nothing left, it has nothing to score and stops (`PipelineError`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from openodke.corroborate.provenance import CHECK
from openodke.types import Fact, GroundingVerdict

UNJUDGED = "ground: a grounding call failed or came back unreadable, so a fact is unchecked"


def calls_failed(stage: Any) -> int:
    """How many of a stage's model calls failed or came back unreadable, from its `stats`."""
    stats = getattr(stage, "stats", None)
    if not isinstance(stats, Mapping):
        return 0
    return int(stats.get("failed", 0) or 0) + int(stats.get("unparseable", 0) or 0)


def answered(stage: Any) -> int | None:
    """How many verdicts a model grounder's calls gave, from its `stats`; None when it keeps none.

    The span check's own verdicts are counted apart (`stats["span"]`), so these
    are the model's answers alone.
    """
    stats = getattr(stage, "stats", None)
    if not isinstance(stats, Mapping) or "calls" not in stats:
        return None
    return sum(
        int(stats.get(v.value, 0) or 0)
        for v in GroundingVerdict
        if v is not GroundingVerdict.UNCHECKED
    )


def unjudged(facts: Iterable[Fact]) -> dict[str, str]:
    """Each document holding a fact a model grounder was asked about and left unchecked.

    Read only where the grounder's calls failed: a fact the free checks
    stamped (`odke.check`) is unchecked by design, and is left alone.
    """
    out: dict[str, str] = {}
    for fact in facts:
        if fact.verdict is GroundingVerdict.UNCHECKED and CHECK not in fact.qualifiers:
            for evidence in fact.evidence:
                out.setdefault(evidence.doc_id, UNJUDGED)
    return out


__all__: list[str] = []
