"""Grounding: does the cited span support the claim?

The stage the release sentence rests on. It asks two questions in order, and the
order is the point:

1. *Does the span exist, and does it say what the fact claims it says?*
   `SpanGrounder` — pure, free, runs first, always. An offset that does not
   resolve to its quote is rejected before a token is spent.
2. *Does that text actually support the claim?* — the second model's question,
   and the one locatability alone cannot answer. A model can cite a span that
   genuinely exists and does not support the fact it was attached to.

Both stamp `Fact.verdict` and neither drops a fact (DECISIONS #20): the validator
is the gate, and the ablation counts what would have gone.
"""

from openodke.ground.llm import LLMGrounder, build_messages, parse_verdict, render_claim
from openodke.ground.retry import RetryPolicy, is_transient
from openodke.ground.span import SpanGrounder, SpanStatus, check_span, located

__all__ = [
    "LLMGrounder",
    "RetryPolicy",
    "SpanGrounder",
    "SpanStatus",
    "build_messages",
    "check_span",
    "is_transient",
    "located",
    "parse_verdict",
    "render_claim",
]
