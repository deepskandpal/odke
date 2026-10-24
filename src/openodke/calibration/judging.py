"""Running a grounder over labelled facts, once or several times, for a card.

Provisional (DECISIONS #48), as `openodke.calibration` is: the card format and the bars settle
with the first published cards.

`odke calibrate` and `bench/calibrate.py` both do this, so it lives here once:
each labelled fact is grounded against the text its label row holds, exactly as
the Validator would ground it, and its verdict in each run is kept beside the
person's. Run 2 and after are the same question asked again on purpose
(`ModelSpec.repeat`, DECISIONS #30): a response cache keeps the draws apart,
so a cached first run never answers the second.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openodke.calibration.measure import UNCHECKED, JudgedFact
from openodke.ground.llm import LLMGrounder
from openodke.ground.span import located
from openodke.prompts import get as get_prompt
from openodke.types import Document, Fact, GroundingVerdict


class Labelled(Protocol):
    """A labelled fact as a grounding label row holds it: the text, the fact, the verdict."""

    @property
    def text(self) -> str: ...

    @property
    def fact(self) -> Fact: ...

    @property
    def verdict(self) -> str: ...

    def as_document(self) -> Document: ...


@dataclass(frozen=True)
class Judging:
    """Every labelled fact's verdicts, and the grounders that gave them, one per run."""

    facts: list[JudgedFact]
    grounders: list[LLMGrounder]

    @property
    def calls(self) -> int:
        return sum(int(g.stats["calls"]) for g in self.grounders)

    @property
    def cached(self) -> int:
        return sum(int(g.stats.get("cached", 0)) for g in self.grounders)

    @property
    def tokens(self) -> tuple[int, int]:
        """Input and output tokens over every run, as the provider counted them."""
        stats = [g.stats for g in self.grounders]
        return (
            sum(int(s["prompt_tokens"]) for s in stats),
            sum(int(s["completion_tokens"]) for s in stats),
        )

    @property
    def cost_usd(self) -> float | None:
        costs = [g.stats["cost_usd"] for g in self.grounders]
        return None if any(c is None for c in costs) else float(sum(costs))


def repeat(grounder: LLMGrounder, draw: int) -> LLMGrounder:
    """`grounder` asking its question for the `draw`-th time: the same prompt, client and model.

    A fresh grounder, so each run's counts are its own. Draw 0 is the
    grounder's own question; a cache keys any other apart from it.
    """
    spec = grounder.spec.model_copy(update={"repeat": draw})
    return LLMGrounder(
        grounder.roles.model_copy(update={"ground": spec}),
        client=grounder.client,
        max_workers=grounder.max_workers,
        retry=grounder.retry,
        context=grounder.context,
        verdicts="binary" if grounder.binary else "three_way",
        claim_format=grounder.claim_format,
        prompt=grounder.prompt.key,
    )


def passage_words(fact: Fact, doc: Document) -> int:
    """How many words the grounder reads for `fact`: its located span, or the whole text."""
    evidence = located(fact, doc)
    if evidence is not None and evidence.span is not None:
        return len(evidence.span.resolve(doc).split())
    return len(doc.text.split())


def ground_labelled(
    rows: Sequence[Labelled],
    grounder: LLMGrounder,
    *,
    runs: int = 1,
    planted: Sequence[bool] | None = None,
    ids: Sequence[str] | None = None,
) -> Judging:
    """Ground every row `runs` times and keep each verdict beside the row's label.

    A row's fact is asked as unchecked, whatever verdict it carries. `ids` name
    the items, the facts' own ids by default; `planted` flags the rows labelled
    by construction. Run 1 is `grounder` itself; each later run is `repeat`.
    """
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    if planted is not None and len(planted) != len(rows):
        raise ValueError("planted needs one flag per row")
    if ids is not None and len(ids) != len(rows):
        raise ValueError("ids needs one per row")
    docs = [row.as_document() for row in rows]
    facts = [row.fact.model_copy(update={"verdict": GroundingVerdict.UNCHECKED}) for row in rows]
    grounders = [grounder if draw == 0 else repeat(grounder, draw) for draw in range(runs)]
    verdicts: list[list[str]] = [[] for _ in rows]
    for each in grounders:
        grounded = each.ground_documents(
            [([fact], doc) for fact, doc in zip(facts, docs, strict=True)]
        )
        for at, (fact,) in enumerate(grounded):
            verdicts[at].append(_verdict(fact))
    judged = [
        JudgedFact(
            id=ids[at] if ids is not None else (row.fact.id or str(at)),
            label=row.verdict,
            runs=tuple(verdicts[at]),
            words=passage_words(facts[at], docs[at]),
            planted=bool(planted[at]) if planted is not None else False,
        )
        for at, row in enumerate(rows)
    ]
    return Judging(judged, grounders)


def _verdict(fact: Fact) -> str:
    value = fact.verdict.value
    return value if value != GroundingVerdict.UNCHECKED.value else UNCHECKED


def prompt_hashes(keys: Sequence[str]) -> dict[str, str]:
    """Each registered prompt key with its SHA-256, as a card records them."""
    return {key: get_prompt(key).sha256 for key in keys}


def describe(grounder: LLMGrounder) -> dict[str, Any]:
    """What a grounding card names about its grounder: prompt, prompts, model, claim format."""
    return {
        "prompt": grounder.prompt.key,
        "prompts": prompt_hashes(grounder.prompt_keys),
        "model": grounder.spec.model,
        "claim_format": grounder.claim_line,
    }


__all__ = [
    "Judging",
    "Labelled",
    "describe",
    "ground_labelled",
    "passage_words",
    "prompt_hashes",
    "repeat",
]
