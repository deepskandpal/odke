"""The entity-pair judge: a model asked about the pairs the resolver's rules leave open (#150).

The rules settle most pairs. A shared id or domain is proof, a disagreeing one
kills the match, and a name score at or above the threshold is a `SIMILAR`.
Below the threshold nothing links, and that is where the misses are: a surname
and the full name, a short form, a former name. `PairJudge` asks a model about
the pairs in a band just below the threshold, with each mention's type and the
text around it, and nothing else. Above the band the rules stand, and below it
the names are too far apart to be worth a call.

**The swap rule.** Each pair is asked twice, as (A, B) and as (B, A), because
a model reading two items in a row favours one position (Zheng et al. 2023,
§3.4). "same" counts only when both orders say same, and "different" only when
both say different. Anything else is unsure: the orders disagree, one says
unsure, or an answer could not be read. Each decision records both answers and
whether the orders disagreed.

**What a decision makes.** "same" is a `SIMILAR` link whose reason names the
judge, its prompt and its model, never a merge: only a proof re-keys
(DECISIONS #16, #31). "different" is a `DIFFERENT` link. Unsure makes no link,
and goes to the review queue when there is one.

**The review queue** is a JSONL file of `odke label make pair` rows, appended
to and never rewritten, one row a pair. A person ticks the sheets, `odke label
read` writes their labels, and `PairJudge(reviewed=...)` reads them: a pair a
person decided is decided in the judge's place, with no call, as a link whose
reason starts `person:`. A pair already in the queue is not queued again.

A mention's context is its sentence and one either side, from the document its
fact cites (`documents`, filled as the corroborator's is). An entity no fact of
the batch mentions is a stored one: it shows its aliases, and its context is
whatever `store_context` returns for it, which is meant to be the evidence of
its strongest supporting fact. Nothing in the store keeps text, so without
`store_context` it has none. A pair with a side that has no context is not
asked, since the prompt may not decide from names alone: it is unsure, and
queued for a person.

A budget stop (`openodke.llm.budget`) ends the calls, not the resolver: a pair
the stop reached is `unasked`, makes no link, is not queued, and is asked
again by the next run. `stats["stopped"]` says where the budget stood.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from openodke.chunking import sentences
from openodke.ground.retry import RetryPolicy, call_with_retry
from openodke.ground.span import Counts
from openodke.llm.base import (
    Completion,
    LLMClient,
    Message,
    MissingAPIKey,
    ModelSpec,
    ProviderNotInstalled,
)
from openodke.llm.budget import BudgetExceeded
from openodke.llm.registry import resolve as resolve_client
from openodke.llm.roles import ModelRoles
from openodke.prompts import get as get_prompt
from openodke.types import Document, Entity, EntityLink, Fact, Frozen, LinkKind

log = logging.getLogger("openodke.judge")

# The registered prompts (DECISIONS #27): the instructions, and the user
# message they frame. A calibration card names both.
PROMPT = get_prompt("pair")
FRAME = get_prompt("pair.user")

# The band's lower bound. Its upper bound is the resolver's `threshold` (0.9).
# Read off Re-DocRED's dev split, never its test split or label set R: of the
# 2,076 within-document pairs blocking lets through, 533 score in [0.7, 0.9),
# nearly a third of them one entity. The calibration card on R says whether
# it holds.
DEFAULT_LOW = 0.7
# The most text one side shows: a "sentence" with no full stop can be a page.
MAX_CONTEXT = 1000

Decision = Literal["same", "different", "unsure"]
DECISIONS: tuple[Decision, ...] = ("same", "different", "unsure")

PAIR_SCHEMA: dict[str, Any] = {
    "title": "PairDecision",
    "type": "object",
    "properties": {
        "because": {"type": "string"},
        "decision": {"type": "string", "enum": list(DECISIONS)},
    },
    "required": ["because", "decision"],
    "additionalProperties": False,
}

_FIELD = re.compile(r"\{(\w+)\}")
_BARE_WORD = re.compile(r"^\W*(same|different|unsure)\W*$", re.IGNORECASE)
_SPACE = re.compile(r"\s+")


class Mention(Frozen):
    """One side of a pair: an entity key and its type, and how and where it was written.

    `label` is the name as the text has it (the key is shown when it is
    absent); `context` is the passage it appeared in. `aliases` are a stored
    entity's other names. The judge reads exactly this, and a pair sheet
    shows exactly this.
    """

    key: str
    type: str
    label: str | None = None
    context: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def surface(self) -> str:
        return self.label or self.key


class Answer(Frozen):
    """What the model said asked in one order. `decision` is None when it could not be read."""

    decision: Decision | None = None
    because: str = ""


class PairDecision(Frozen):
    """What was decided about one pair, by whom, and on what.

    `decision` is `same`, `different`, `unsure`, `failed` when a call
    failed, or `unasked` when a budget stop came first: neither of the last
    two links or is queued, and the next run asks again. `forward` is the
    answer asked as (A, B), `backward` as (B, A); neither is set when no model
    was asked. `disagreed` is whether the two orders gave different answers.
    `why` says what made a pair unsure.
    """

    a: str
    b: str
    decision: Literal["same", "different", "unsure", "failed", "unasked"]
    by: Literal["judge", "person"] = "judge"
    score: float | None = None
    forward: Answer | None = None
    backward: Answer | None = None
    disagreed: bool = False
    why: str | None = None
    prompt: str | None = None
    model: str | None = None


def swap_rule(forward: Answer, backward: Answer) -> Decision:
    """`same` when both orders say same, `different` when both say different, else `unsure`."""
    if forward.decision == backward.decision and forward.decision in ("same", "different"):
        return forward.decision
    return "unsure"


def render_pair(a: Mention, b: Mention) -> str:
    """The user message for (A, B): the registered frame, each field filled in once."""

    def side(mention: Mention) -> dict[str, str]:
        aliases = [n for n in mention.aliases if n != mention.surface]
        return {
            "surface": _line(mention.surface),
            "type": mention.type,
            "aliases": f"; also known as: {', '.join(map(_line, aliases))}" if aliases else "",
            "context": _line(mention.context) if mention.context else "(no text)",
        }

    values = {
        f"{name}_{field}": value
        for name, mention in (("a", a), ("b", b))
        for field, value in side(mention).items()
    }
    return _FIELD.sub(lambda m: values[m[1]], FRAME.text)


def build_messages(a: Mention, b: Mention) -> list[Message]:
    """Exactly what the model is sent for the order (A, B)."""
    return [
        Message(role="system", content=PROMPT.text),
        Message(role="user", content=render_pair(a, b)),
    ]


def parse_answer(completion: Completion) -> Answer:
    """The decision in a completion, or an `Answer` with none when there is not one to read.

    Structured output is the normal path; a JSON object in the text, or the
    bare word, is read too. Prose is not: "not the same" must never be read as
    `same`.
    """
    data: Any = completion.parsed
    if data is None:
        text = completion.text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
        except ValueError:
            data = None
    value = data.get("decision") if isinstance(data, Mapping) else None
    because = data.get("because") if isinstance(data, Mapping) else None
    if not isinstance(value, str):
        match = _BARE_WORD.match(completion.text)
        value = match.group(1) if match else None
    word = value.strip().lower() if isinstance(value, str) else None
    return Answer(
        decision=next((d for d in DECISIONS if d == word), None),
        because=_line(because)[:300] if isinstance(because, str) else "",
    )


def context_around(text: str, start: int, end: int) -> str:
    """The sentence holding `[start, end)` and one sentence either side, on one line.

    Sentences are the chunker's (`openodke.chunking.sentences`). A window
    longer than `MAX_CONTEXT` is cut to that many characters around the
    mention, at spaces.
    """
    bounds = sentences(text)
    if not bounds:
        return ""
    at = next((k for k, (s, e) in enumerate(bounds) if s <= start < e), None)
    if at is None:
        at = min(range(len(bounds)), key=lambda k: abs(bounds[k][0] - start))
    lo, hi = bounds[max(0, at - 1)][0], bounds[min(len(bounds) - 1, at + 1)][1]
    if hi - lo > MAX_CONTEXT:
        half = (MAX_CONTEXT - (end - start)) // 2
        lo, hi = max(lo, start - half), min(hi, end + half)
        lo = text.find(" ", lo, start) + 1 if lo > 0 and " " in text[lo:start] else lo
        hi = text.rfind(" ", end, hi) if " " in text[end:hi] else hi
    return _line(text[lo:hi])


def _line(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _names(entity: Entity) -> list[str]:
    return [n for n in dict.fromkeys((entity.label, *entity.aliases, entity.key)) if n]


def _occurrence(names: Sequence[str], text: str, near: int) -> tuple[int, int] | None:
    """Where a name of the entity is in `text`: its label first, nearest `near`."""
    for name in names:
        found = [
            (m.start(), m.end())
            for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE)
        ]
        if found:
            return min(found, key=lambda span: (abs(span[0] - near), span[0]))
    return None


def _reviewed(given: str | Path | Iterable[Any] | None) -> dict[frozenset[str], bool]:
    """A person's decisions by unordered pair: `odke label read` output, or rows with a, b, same."""
    if given is None:
        return {}
    if isinstance(given, str | Path):
        path = Path(given)
        rows: Iterable[Any] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    else:
        rows = given
    out: dict[frozenset[str], bool] = {}
    for row in rows:
        if isinstance(row, Mapping):
            a, b, same = row.get("a"), row.get("b"), row.get("same")
        else:
            a, b, same = (getattr(row, k, None) for k in ("a", "b", "same"))
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(same, bool):
            raise ValueError(f"a reviewed pair needs a, b and same: {row!r}")
        out[frozenset((a, b))] = same
    return out


class PairJudge:
    """A model asked whether two mentions name one entity, in both orders, on ambiguous pairs.

    `NativeResolver(judge=PairJudge())` hands it every pair the rules leave
    open whose name score is at least `low` and below the resolver's
    threshold. `roles.ground` names the model, since this is the same small
    question grounding asks; `client` overrides how it is reached, and is
    resolved on the first call when not given, so a judge that only applies
    `reviewed` decisions needs no provider.

    `queue` is the JSONL file unsure pairs are appended to, for a person;
    none is written without it. `reviewed` is a person's decisions (`odke
    label read` output, or `PairLabel` rows), applied in the judge's place.
    `store_context` gives a stored entity's context. Everything it did is in
    `stats`: pairs handed in, pairs asked, calls, `swapped` (the calls in the
    (B, A) order), `disagreed` (pairs whose orders differed), each decision's
    count, `person`, `queued`, `no_context`, `failed` calls, `unparseable`
    answers, tokens, cost, and the prompt keys sent. Calls a response cache
    answered are counted under `cached` too, and calls a budget refused under
    `unasked`, with the stop under `stopped`. `decisions` keeps every
    `PairDecision`.

    A call is retried under `retry`; when it still fails the pair is `failed`.
    A missing key or adapter raises instead, since it would fail every call. A
    budget stop does not: the resolver's rules still run, and the pairs it
    reached are `unasked`.
    """

    def __init__(
        self,
        roles: ModelRoles | None = None,
        *,
        client: LLMClient | None = None,
        low: float = DEFAULT_LOW,
        queue: str | Path | None = None,
        reviewed: str | Path | Iterable[Any] | None = None,
        store_context: Callable[[Entity], str | None] | None = None,
        max_workers: int = 8,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0.0 <= low <= 1.0:
            raise ValueError(f"low must be in [0, 1], got {low}")
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.roles = roles if roles is not None else ModelRoles()
        self.spec: ModelSpec = self.roles.ground
        self._client = client
        self.low = low
        self.queue = Path(queue) if queue is not None else None
        self.reviewed = _reviewed(reviewed)
        self.store_context = store_context
        self.max_workers = max_workers
        self.retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        # Filled by whoever runs the judge, as the corroborator's is (DECISIONS #19).
        self.documents: dict[str, Document] = {}
        self.decisions: list[PairDecision] = []
        self._queued: set[frozenset[str]] | None = None
        self._lock = threading.Lock()
        self._cost: float | None = None
        # The first budget stop, as `BudgetExceeded.report()` gives it.
        self._stopped: dict[str, Any] | None = None
        self._counts = Counts(
            "pairs",
            "asked",
            "calls",
            "swapped",
            "disagreed",
            *DECISIONS,
            "person",
            "queued",
            "no_context",
            "failed",
            "unparseable",
            "retries",
            "prompt_tokens",
            "completion_tokens",
        )

    @property
    def client(self) -> LLMClient:
        with self._lock:
            if self._client is None:
                self._client = resolve_client(self.spec)
            return self._client

    # ----------------------------------------------------------------------- #
    # Mentions
    # ----------------------------------------------------------------------- #

    def mentions(self, entities: Iterable[Entity], facts: Sequence[Fact]) -> dict[str, Mention]:
        """Each entity as the judge shows it, by key, with its context.

        A batch entity's context is found through the first fact that mentions
        it and cites a document the judge has, or else a span with a quote. An
        entity no fact mentions is a stored one.
        """
        cited: dict[str, list[Fact]] = {}
        for fact in facts:
            for entity in (fact.subject, fact.object_entity):
                if entity is not None:
                    cited.setdefault(entity.key, []).append(fact)
        out: dict[str, Mention] = {}
        for entity in entities:
            if entity.key in out:
                continue
            if entity.key in cited:
                context = next(
                    (c for f in cited[entity.key] if (c := self._context(entity, f))), None
                )
                aliases: tuple[str, ...] = ()
            else:
                context = self.store_context(entity) if self.store_context else None
                context = _line(context) if context else None
                aliases = tuple(n for n in entity.aliases if n != entity.label)
            out[entity.key] = Mention(
                key=entity.key,
                type=entity.type,
                label=entity.label,
                context=context or None,
                aliases=aliases,
            )
        return out

    def _context(self, entity: Entity, fact: Fact) -> str | None:
        for evidence in fact.evidence:
            doc = self.documents.get(evidence.doc_id)
            span = evidence.span
            if doc is not None:
                near = span.start if span is not None else 0
                at = _occurrence(_names(entity), doc.text, near)
                if at is None and span is not None:
                    at = (span.start, span.end)
                if at is not None:
                    return context_around(doc.text, *at) or None
            elif span is not None and span.quote:
                return _line(span.quote)[:MAX_CONTEXT]
        return None

    # ----------------------------------------------------------------------- #
    # Deciding
    # ----------------------------------------------------------------------- #

    def judge(self, a: Mention, b: Mention, *, score: float | None = None) -> PairDecision:
        """One pair: a person's decision when there is one, else two calls and the swap rule."""
        return self.judge_many([(a, b, score)])[0]

    def judge_many(
        self, pairs: Sequence[tuple[Mention, Mention, float | None]]
    ) -> list[PairDecision]:
        """Every pair, in order; the calls of all of them share one pool of `max_workers`.

        Unsure pairs are appended to the queue, when there is one.
        """
        decided: list[PairDecision | None] = []
        asked: list[tuple[int, Mention, Mention]] = []
        for at, (a, b, score) in enumerate(pairs):
            self._counts.bump("pairs")
            person = self.reviewed.get(frozenset((a.key, b.key)))
            if person is not None:
                self._counts.bump("person")
                decided.append(
                    PairDecision(
                        a=a.key,
                        b=b.key,
                        decision="same" if person else "different",
                        by="person",
                        score=score,
                    )
                )
            elif not a.context or not b.context:
                self._counts.bump("no_context")
                decided.append(
                    PairDecision(a=a.key, b=b.key, decision="unsure", score=score, why="no context")
                )
            else:
                self._counts.bump("asked")
                decided.append(None)
                asked.append((at, a, b))

        # Both orders of every pair asked: (A, B), then (B, A), the swap.
        calls = [call for _, a, b in asked for call in ((a, b, False), (b, a, True))]
        if len(calls) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(
                min(self.max_workers, len(calls)), thread_name_prefix="openodke-judge"
            ) as pool:
                answers = list(pool.map(lambda call: self._ask(*call), calls))
        else:
            answers = [self._ask(*call) for call in calls]
        for n, (at, a, b) in enumerate(asked):
            decided[at] = self._decide(a, b, pairs[at][2], answers[2 * n], answers[2 * n + 1])

        out = [d for d in decided if d is not None]
        for decision in out:
            self.decisions.append(decision)
            if decision.decision in DECISIONS and decision.by == "judge":
                self._counts.bump(decision.decision)
        self._enqueue([(p, d) for p, d in zip(pairs, out, strict=True) if d.decision == "unsure"])
        return out

    def _decide(
        self,
        a: Mention,
        b: Mention,
        score: float | None,
        forward: Answer | Exception,
        backward: Answer | Exception,
    ) -> PairDecision:
        decision: Literal["same", "different", "unsure", "failed", "unasked"]
        if isinstance(forward, BudgetExceeded) or isinstance(backward, BudgetExceeded):
            decision, disagreed, why = "unasked", False, "the budget stopped the run first"
        elif isinstance(forward, Exception) or isinstance(backward, Exception):
            decision, disagreed, why = "failed", False, "a call failed"
        else:
            disagreed = forward.decision != backward.decision
            if disagreed:
                self._counts.bump("disagreed")
            decision = swap_rule(forward, backward)
            why = None
            if decision == "unsure":
                if None in (forward.decision, backward.decision):
                    why = "an answer could not be read"
                elif disagreed:
                    why = "the orders disagree"
                else:
                    why = "unsure in both orders"
        return PairDecision(
            a=a.key,
            b=b.key,
            decision=decision,
            score=score,
            forward=forward if isinstance(forward, Answer) else None,
            backward=backward if isinstance(backward, Answer) else None,
            disagreed=disagreed,
            why=why,
            prompt=PROMPT.key,
            model=self.spec.model,
        )

    def _ask(self, a: Mention, b: Mention, swapped: bool) -> Answer | Exception:
        """The model's answer for the order (A, B), or what stopped the call."""
        messages = build_messages(a, b)

        def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            self._counts.bump("retries")
            log.info("retrying pair call %s/%s in %.2fs: %s", a.key, b.key, wait, exc)

        def counted() -> None:
            self._counts.bump("calls")
            if swapped:
                self._counts.bump("swapped")

        def complete() -> Completion:
            try:
                completion = self.client.complete(messages, spec=self.spec, schema=PAIR_SCHEMA)
            except BudgetExceeded:
                # Refused before it was made: not a call.
                raise
            except BaseException:
                counted()
                raise
            counted()
            return completion

        try:
            completion = call_with_retry(complete, self.retry, sleep=self._sleep, on_retry=on_retry)
        except (MissingAPIKey, ProviderNotInstalled):
            raise
        except BudgetExceeded as stop:
            # The run's, not the pair's: no answer, and the next run asks.
            self._counts.bump("unasked")
            with self._lock:
                if self._stopped is None:
                    self._stopped = stop.report()
            return stop
        except Exception as exc:  # isolation: one failed call never fails the batch
            self._counts.bump("failed")
            log.warning("pair call failed for %s/%s: %s", a.key, b.key, exc)
            return exc
        self._counts.bump("prompt_tokens", completion.prompt_tokens)
        self._counts.bump("completion_tokens", completion.completion_tokens)
        if completion.cached:
            # Counted among `calls`, and only once one happens, so a run
            # without a response cache reports exactly as it did.
            self._counts.bump("cached")
        if completion.cost_usd is not None:
            with self._lock:
                self._cost = (self._cost or 0.0) + completion.cost_usd
        answer = parse_answer(completion)
        if answer.decision is None:
            self._counts.bump("unparseable")
            log.warning("unreadable pair answer for %s/%s: %r", a.key, b.key, completion.text[:200])
        return answer

    # ----------------------------------------------------------------------- #
    # The queue
    # ----------------------------------------------------------------------- #

    def _enqueue(
        self, unsure: Sequence[tuple[tuple[Mention, Mention, float | None], PairDecision]]
    ) -> None:
        if self.queue is None or not unsure:
            return
        if self._queued is None:
            self._queued = set()
            if self.queue.is_file():
                for line in self.queue.read_text(encoding="utf-8").split("\n"):
                    if line.strip():
                        row = json.loads(line)
                        self._queued.add(frozenset((row["a"]["key"], row["b"]["key"])))
        rows = []
        for (a, b, _), decision in unsure:
            pair = frozenset((a.key, b.key))
            if pair in self._queued:
                continue
            self._queued.add(pair)
            judged = decision.model_dump(mode="json", exclude={"a", "b"}, exclude_none=True)
            rows.append(
                {
                    "a": a.model_dump(mode="json", exclude_defaults=True),
                    "b": b.model_dump(mode="json", exclude_defaults=True),
                    "judge": judged,
                }
            )
        if not rows:
            return
        self.queue.parent.mkdir(parents=True, exist_ok=True)
        with self.queue.open("a", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._counts.bump("queued", len(rows))

    @property
    def stats(self) -> dict[str, Any]:
        """The judge's run report. `cost_usd` is None until a provider reports one."""
        out: dict[str, Any] = self._counts.snapshot()
        with self._lock:
            out["cost_usd"] = self._cost
            if self._stopped is not None:
                out["stopped"] = dict(self._stopped)
        out["low"] = self.low
        # The registered prompts the calls sent, by key: none when no call was made.
        out["prompts"] = [PROMPT.key, FRAME.key] if out["calls"] else []
        return out


def link_for(decision: PairDecision) -> EntityLink | None:
    """The link a decision makes: `SIMILAR` for same, `DIFFERENT` for different, else none.

    The score is the resolver's name score, as on any link it proposes; the
    reason says who decided, and on what.
    """
    if decision.decision not in ("same", "different"):
        return None
    kind = LinkKind.SIMILAR if decision.decision == "same" else LinkKind.DIFFERENT
    if decision.by == "person":
        reason = f"person: {decision.decision}, from the review queue"
    else:
        because = decision.forward.because if decision.forward else ""
        reason = (
            f"pair judge ({decision.prompt}, {decision.model}): {decision.decision} in both orders"
            + (f"; {because}" if because else "")
        )
    return EntityLink(
        source_key=decision.a,
        target_key=decision.b,
        kind=kind,
        score=decision.score,
        reason=reason,
    )


def judge_stop(resolved: Any) -> dict[str, Any] | None:
    """A budget stop the pair judge reached first, as a run's `stopped`: during resolve.

    The resolver's rules ran in full, so nothing was left unchecked; the
    judge's own `unasked` says how many calls the budget refused.
    """
    judged = resolved.get("judge") if isinstance(resolved, Mapping) else None
    stop = judged.get("stopped") if isinstance(judged, Mapping) else None
    if not isinstance(stop, Mapping):
        return None
    return {**stop, "stage": "resolve", "unextracted": 0, "unchecked": 0}


__all__ = [
    "DECISIONS",
    "DEFAULT_LOW",
    "FRAME",
    "MAX_CONTEXT",
    "PAIR_SCHEMA",
    "PROMPT",
    "Answer",
    "Mention",
    "PairDecision",
    "PairJudge",
    "build_messages",
    "context_around",
    "judge_stop",
    "link_for",
    "parse_answer",
    "render_pair",
    "swap_rule",
]
