"""Scoring: one number per fact, from three signals, monotone and bounded.

    confidence = g(verdict) × (1 − (1 − c) ^ (1 + ln support)) × k

`c` is the extractor's confidence, clamped to [0, 1]. Exactly 0.0 is
`Fact.confidence`'s default and means the extractor reported nothing — a fact
its extractor believed false would not have been emitted — so `prior` stands
in for it.

`g` is what grounding found: SUPPORTED 1.0, UNCHECKED 0.6, NOT_FOUND 0.25,
CONTRADICTED 0.05. It multiplies rather than votes, so no number of agreeing
sources lifts a claim above what its own cited spans allow.

`support` is the corroborator's count of independent sources. It enters as a
noisy-OR — independent sources each believed at `c` — damped by a log in the
exponent. Two sources count for 1.69 of one and ten for 3.3, so agreement
strengthens a claim without volume alone making it certain.

`k` is 1, except for a value that lost a contest in the corroborator, where it
is the loser's rank as a fraction of the winner's.

Monotone: raising `c`, `support`, the verdict or `k` never lowers the score.
Bounded: every factor lies in [0, 1], and so does the product. The inputs are
kept on the fact under `qualifiers["odke.score"]`, so the number can be
recomputed, audited and measured.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from odke.corroborate.merge import SourceKey, independent_sources, source_of
from odke.corroborate.provenance import CONFLICT, SCORE, unstamped
from odke.types import Fact, GroundingVerdict

DEFAULT_VERDICT_WEIGHTS: Mapping[GroundingVerdict, float] = {
    GroundingVerdict.SUPPORTED: 1.0,
    GroundingVerdict.UNCHECKED: 0.6,
    GroundingVerdict.NOT_FOUND: 0.25,
    GroundingVerdict.CONTRADICTED: 0.05,
}


def combine(extractor: float, verdict_weight: float, support: int, conflict: float = 1.0) -> float:
    """The formula in this module's docstring, on its own inputs."""
    agreement = 1.0 - (1.0 - extractor) ** (1.0 + math.log(max(1, support)))
    return verdict_weight * agreement * conflict


class EvidenceScorer:
    """Sets `Fact.confidence` from extraction, grounding and corroboration.

    **What the number means.** It is an ordering, and until it has been measured
    against labels it is only that. A higher score means more reason to believe
    the fact on the evidence this pipeline saw. It is not yet a probability that
    the fact is true.

    Worked values at extractor confidence 0.8:

    - one source, span supported: 0.80
    - one source, unchecked: 0.48
    - one source, span not found: 0.20
    - three independent sources, supported: 0.97
    - a value that lost its contest at a rank ratio of 0.35: 0.28

    **Picking a threshold.** Measure it. `odke.eval` (#57) reports Brier score and
    a reliability curve for this number against a slice you label yourself.
    Choose the lowest score whose bin's observed precision clears your bar.
    Before there are labels, 0.5 is a defensible rough line. It keeps a single
    confident extraction that a span supports, and drops an unchecked
    single-source claim and every loser of a contest.

    `support` is populated too: it is the corroborator's count, or the fact's
    own independent sources when no corroborator ran, whichever is larger.
    Scoring is idempotent. It always starts from the extractor's number, never
    from its own previous output.
    """

    def __init__(
        self,
        *,
        prior: float = 0.5,
        verdict_weights: Mapping[GroundingVerdict, float] = DEFAULT_VERDICT_WEIGHTS,
        source: SourceKey = source_of,
    ) -> None:
        if not 0.0 < prior <= 1.0:
            raise ValueError(f"prior must be in (0, 1], got {prior}")
        missing = set(GroundingVerdict) - set(verdict_weights)
        if missing:
            raise ValueError(f"verdict_weights is missing {sorted(v.value for v in missing)}")
        if any(not 0.0 <= w <= 1.0 for w in verdict_weights.values()):
            raise ValueError("every verdict weight must be in [0, 1], or the score is unbounded")
        self.prior = prior
        self.verdict_weights = dict(verdict_weights)
        self.source = source

    def score(self, fact: Fact) -> Fact:
        reported = unstamped(fact).confidence
        clamped = min(1.0, max(0.0, reported))
        extractor = clamped if clamped > 0.0 else self.prior
        support = max(fact.support, len(independent_sources(fact, self.source)))

        conflict = 1.0
        stamp = fact.qualifiers.get(CONFLICT)
        if isinstance(stamp, Mapping) and stamp.get("status") == "lost":
            ratio = stamp.get("ratio")
            if isinstance(ratio, int | float):
                conflict = min(1.0, max(0.0, float(ratio)))

        weight = self.verdict_weights[fact.verdict]
        inputs: dict[str, Any] = {
            "extractor": reported,
            "prior_used": clamped == 0.0,
            "verdict": weight,
            "support": support,
            "conflict": conflict,
        }
        return fact.model_copy(
            update={
                "confidence": combine(extractor, weight, support, conflict),
                "support": support,
                "qualifiers": {**fact.qualifiers, SCORE: inputs},
            }
        )


__all__ = ["DEFAULT_VERDICT_WEIGHTS", "EvidenceScorer", "combine"]
