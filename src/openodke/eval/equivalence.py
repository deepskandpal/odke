"""The fact-equivalence judge: is a prediction the gold fact in other words? (#143)

A scorer compares strings after a normaliser. "Gabby Logan" for "Gabrielle
Nicole Logan", or "track and field athlete" for "athletics competitor", is
counted missed and counted spurious, and no normaliser can know otherwise
without reading the passage. A model can. This judge is asked about one kind
of pair only, and sets two things: the diagnosis's *surface form* bucket
(#140), through `equivalent`, and a lenient score printed beside the strict
one, never instead of it (`lenient`).

**The pre-filter** decides which pairs reach it, with no model. A gold fact the
scorer counted missed and a prediction it did not match make a pair when they
have the same relation, one end equal after normalisation, and the other end
not exactly equal as written (`candidate`). Normalisation is `name_key` for a
name, against every name the gold end goes by, and `normalise_value` for a
value. The same polarity is required: a denial is the opposite claim, never a
rewording. A pair with nothing in common but the relation is not a question of
wording, and two identical triples are the scorer's question, not the judge's.

**The swap rule**, as the pair judge's (DECISIONS #34): every pair is asked
twice, with the gold fact first and with the prediction first, because a model
reading two items in a row favours a position (Zheng et al. 2023, §3.4).
"same" counts only when both orders say same, and "different" only when both
say different. Anything else is unsure. Each decision keeps both answers and
whether the orders disagreed.

**What it reads** is the registered prompt `fact_equiv@1` and its frame
`fact_equiv.user@1` (DECISIONS #27): the relation and its description from the
ontology, both facts as the grounder renders a claim (`render_claim`), and the
gold fact's evidence (`passage_for`): the sentences its span covers, else the
sentences naming both its ends, else the start of its document. The gold fact
is the reference, as in reference-guided judging (Zheng et al. 2023), and a
quantity, date or number must have the same value, as EnterpriseRAG-Bench's
correctness rule has it ("quantities must match", Onyx, arXiv 2605.05253).

It asks the `ground` role's model, the same size of question as grounding.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from openodke.chunking import sentences
from openodke.corroborate.judge import Answer, parse_answer, swap_rule
from openodke.corroborate.normalize import name_key
from openodke.eval.extraction import normalise_value
from openodke.ground.llm import render_claim
from openodke.ground.locate import locate_span
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
from openodke.ontology import Ontology
from openodke.prompts import get as get_prompt
from openodke.types import Document, Entity, Fact, Frozen, SpanOrigin

log = logging.getLogger("openodke.equivalence")

# The registered prompts (DECISIONS #27): the instructions, and the user
# message they frame. A calibration card names both.
PROMPT = get_prompt("fact_equiv")
FRAME = get_prompt("fact_equiv.user")

# The most of a document a passage shows when no sentence of it is the
# evidence: a document with no full stop can be a book.
MAX_PASSAGE = 2000
# Said in the frame for a relation the ontology does not describe.
NO_DESCRIPTION = "no description"

Decision = Literal["same", "different", "unsure"]
DECISIONS: tuple[Decision, ...] = ("same", "different", "unsure")
# Which end of a pair differs: the one the judge is asked to read past.
Side = Literal["subject", "object"]
Claim = tuple[str, str, str]
# One end of a gold fact: a name, or every name it goes by.
Names = str | Iterable[str]

EQUIV_SCHEMA: dict[str, Any] = {
    "title": "FactEquivalence",
    "type": "object",
    "properties": {"decision": {"type": "string", "enum": list(DECISIONS)}},
    "required": ["decision"],
    "additionalProperties": False,
}

_FIELD = re.compile(r"\{(\w+)\}")
_SPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# The pre-filter
# --------------------------------------------------------------------------- #


def relation_key(relation: str) -> str:
    """A relation as two spellings of it share: `place_of_birth` is `place of birth`."""
    return re.sub(r"[\W_]+", "", relation.casefold())


def surface_pair(
    gold: tuple[Names, str, Names],
    predicted: Claim,
    *,
    key: Callable[[str], str] = name_key,
    relation: Callable[[str], str] = relation_key,
) -> Side | None:
    """The end of `predicted` that differs from `gold`'s, when the pair passes the pre-filter.

    The same relation; one end equal after `key` to one of the gold end's
    names; and the other end not exactly equal, as written, to any of its
    names. None when the pair fails, which includes two identical triples.
    Each gold end is a name or every name it goes by (an entity's label,
    aliases and key; a cluster's mentions).
    """
    subjects, objects = _names(gold[0]), _names(gold[2])
    if relation(gold[1]) != relation(predicted[1]):
        return None
    return _side(
        _matches(predicted[0], subjects, key),
        predicted[0] in subjects,
        _matches(predicted[2], objects, key),
        predicted[2] in objects,
    )


def candidate(gold: Fact, predicted: Fact) -> Side | None:
    """`surface_pair` on two facts: the end of `predicted` that differs, or None.

    A name is compared by `name_key` against every name the gold entity goes
    by (label, aliases, key). Two values are compared by `normalise_value`. The
    relations must be one predicate and the polarities one polarity.
    """
    if gold.predicate != predicted.predicate or gold.polarity != predicted.polarity:
        return None
    subjects = names(gold.subject)
    said = _shown(predicted.subject)
    literal = gold.object_entity is None and predicted.object_entity is None
    key = normalise_value if literal else name_key
    objects = names(gold.object_entity) if gold.object_entity is not None else (_value(gold),)
    obj = _value(predicted) if predicted.object_entity is None else _shown(predicted.object_entity)
    return _side(
        _matches(said, subjects, name_key) or predicted.subject.key == gold.subject.key,
        said in subjects,
        _matches(obj, objects, key),
        obj in objects,
    )


def names(entity: Entity) -> tuple[str, ...]:
    """Every name an entity goes by: its label, its aliases, its key; once each."""
    return tuple(n for n in dict.fromkeys((entity.label, *entity.aliases, entity.key)) if n)


def _side(s_match: bool, s_exact: bool, o_match: bool, o_exact: bool) -> Side | None:
    if s_match and not o_exact:
        return "object"
    if o_match and not s_exact:
        return "subject"
    return None


def _names(given: Names) -> tuple[str, ...]:
    return (given,) if isinstance(given, str) else tuple(given)


def _matches(name: str, given: Sequence[str], key: Callable[[str], str]) -> bool:
    found = key(name)
    return bool(found) and any(found == key(n) for n in given)


def _shown(entity: Entity) -> str:
    return entity.label or entity.key


def _value(fact: Fact) -> str:
    value = fact.object_value
    return value if isinstance(value, str) else str(value)


# --------------------------------------------------------------------------- #
# What the judge reads
# --------------------------------------------------------------------------- #


def passage_for(fact: Fact, document: Document) -> str:
    """The gold fact's evidence in `document`, on one line.

    The sentences its cited or located spans cover, in order, with `…`
    between ones that are not adjacent. A fact that cites none is given the
    sentences naming both its ends (`locate_span`), and one whose ends no
    window names the start of the document, up to `MAX_PASSAGE` characters.
    """
    text = document.text
    spans = [
        (e.span.start, e.span.end)
        for e in fact.evidence
        if e.span is not None
        and e.span_origin is not SpanOrigin.CONTEXT
        and e.doc_id in (document.id, None)
        and 0 <= e.span.start < e.span.end <= len(text)
    ]
    if not spans:
        found = locate_span(fact, document)
        if found is not None:
            spans = [(found.start, found.end)]
    bounds = sentences(text)
    picked = sorted(
        {k for start, end in spans for k, (s, e) in enumerate(bounds) if s < end and start < e}
    )
    if not picked:
        return _cut(_line(text), MAX_PASSAGE)
    parts: list[str] = []
    for at, k in enumerate(picked):
        if at and k != picked[at - 1] + 1:
            parts.append("…")
        parts.append(_line(text[bounds[k][0] : bounds[k][1]]))
    return " ".join(parts)


def render_question(
    predicate: str, description: str | None, first: str, second: str, passage: str
) -> str:
    """The user message: the registered frame, each field filled in once."""
    values = {
        "predicate": predicate.replace("_", " "),
        "description": _line(description) if description else NO_DESCRIPTION,
        "fact_1": _line(first),
        "fact_2": _line(second),
        "passage": _line(passage) or "(no text)",
    }
    return _FIELD.sub(lambda m: values[m[1]], FRAME.text)


def render_plain(claim: Claim) -> str:
    """A claim with no types, as the grounder would render it: `s — relation — o.`"""
    subject, relation, obj = claim
    return f"{subject} — {relation.replace('_', ' ')} — {obj}."


def build_messages(
    predicate: str, description: str | None, first: str, second: str, passage: str
) -> list[Message]:
    """Exactly what the model is sent for one order: `first` is Fact 1."""
    return [
        Message(role="system", content=PROMPT.text),
        Message(
            role="user", content=render_question(predicate, description, first, second, passage)
        ),
    ]


def _line(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return (head[: head.rfind(" ")] if " " in head else head) + " …"


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #


class Equivalence(Frozen):
    """What was decided about one pair, and on what.

    `decision` is `same`, `different`, `unsure`, `failed` when a call failed,
    or `unasked` when a budget stop came first. `gold_first` is the answer
    asked with the gold fact as Fact 1, `prediction_first` the swap; neither is
    set when no model was asked. `disagreed` is whether the two orders gave
    different answers, and `why` says what made a pair unsure.
    """

    decision: Literal["same", "different", "unsure", "failed", "unasked"]
    gold_first: Answer | None = None
    prediction_first: Answer | None = None
    disagreed: bool = False
    why: str | None = None
    prompt: str | None = None
    model: str | None = None


# One question in one order: the relation, its description, Fact 1, Fact 2, the passage.
_Question = tuple[str, str | None, str, str, str]


class FactJudge:
    """A model asked whether a prediction is the gold fact in other words, in both orders.

    `roles.ground` names the model; `client` overrides how it is reached, and
    is resolved on the first call when not given. `ontology` gives each
    relation its description, for the frame. Everything it did is in `stats`:
    pairs, calls, `swapped` (the calls with the prediction first), `disagreed`
    (pairs whose orders differed), each decision's count, `failed` calls,
    `unasked` calls a budget refused, `cached` calls a response cache
    answered, `unparseable` answers, tokens, cost, and the prompt keys sent.

    `judge_many` takes facts; `equivalent` takes claims as strings, and is the
    diagnosis's `EquivalenceJudge` hook. A call is retried under `retry`; when
    it still fails the pair is `failed`. A missing key or adapter raises,
    since it would fail every call.
    """

    def __init__(
        self,
        roles: ModelRoles | None = None,
        *,
        client: LLMClient | None = None,
        ontology: Ontology | None = None,
        max_workers: int = 8,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.roles = roles if roles is not None else ModelRoles()
        self.spec: ModelSpec = self.roles.ground
        self._client = client
        self.ontology = ontology
        self.max_workers = max_workers
        self.retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep
        self._lock = threading.Lock()
        self._cost: float | None = None
        self._stopped: dict[str, Any] | None = None
        self._counts = Counts(
            "pairs",
            "calls",
            "swapped",
            "disagreed",
            *DECISIONS,
            "failed",
            "unasked",
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

    def description(self, relation: str) -> str | None:
        """The ontology's description of `relation`, matched by name, label or alias."""
        if self.ontology is None:
            return None
        found = self.ontology.predicates.get(relation)
        if found is None:
            wanted = relation_key(relation)
            found = next(
                (
                    p
                    for p in self.ontology.predicates.values()
                    if wanted in {relation_key(n) for n in (p.name, p.label or "", *p.aliases)}
                ),
                None,
            )
        return found.description if found is not None else None

    # ----------------------------------------------------------------------- #
    # Asking
    # ----------------------------------------------------------------------- #

    def judge(self, gold: Fact, predicted: Fact, passage: str) -> Equivalence:
        """One pair: two calls, gold first and prediction first, and the swap rule."""
        return self.judge_many([(gold, predicted, passage)])[0]

    def judge_many(self, pairs: Sequence[tuple[Fact, Fact, str]]) -> list[Equivalence]:
        """Every `(gold, predicted, passage)`, in order; all the calls share one pool."""
        return self._decide_all(
            [
                (
                    gold.predicate,
                    self.description(gold.predicate),
                    render_claim(gold),
                    render_claim(predicted),
                    passage,
                )
                for gold, predicted, passage in pairs
            ]
        )

    def equivalent(self, gold: Claim, predicted: Claim, text: str | None) -> bool | None:
        """The diagnosis's hook: True when both orders say same, False when both say different.

        `text` is the passage: the gold fact's evidence sentences, not its
        whole document. None, unsure, a failed call, or no text at all, is
        None: the miss goes on to the buckets after.
        """
        if not text:
            return None
        relation = gold[1]
        (decided,) = self._decide_all(
            [
                (
                    relation,
                    self.description(relation),
                    render_plain(gold),
                    render_plain(predicted),
                    text,
                )
            ]
        )
        return {"same": True, "different": False}.get(decided.decision)

    def _decide_all(self, questions: Sequence[_Question]) -> list[Equivalence]:
        calls = [(question, swapped) for question in questions for swapped in (False, True)]
        self._counts.bump("pairs", len(questions))
        if len(calls) > 1 and self.max_workers > 1:
            with ThreadPoolExecutor(
                min(self.max_workers, len(calls)), thread_name_prefix="openodke-equivalence"
            ) as pool:
                answers = list(pool.map(lambda call: self._ask(*call), calls))
        else:
            answers = [self._ask(*call) for call in calls]
        out = [self._decide(answers[2 * n], answers[2 * n + 1]) for n in range(len(questions))]
        for decided in out:
            if decided.decision in DECISIONS:
                self._counts.bump(decided.decision)
        return out

    def _decide(self, first: Answer | Exception, second: Answer | Exception) -> Equivalence:
        decision: Literal["same", "different", "unsure", "failed", "unasked"]
        disagreed, why = False, None
        if isinstance(first, BudgetExceeded) or isinstance(second, BudgetExceeded):
            decision, why = "unasked", "the budget stopped the run first"
        elif isinstance(first, Exception) or isinstance(second, Exception):
            decision, why = "failed", "a call failed"
        else:
            disagreed = first.decision != second.decision
            if disagreed:
                self._counts.bump("disagreed")
            decision = swap_rule(first, second)
            if decision == "unsure":
                if None in (first.decision, second.decision):
                    why = "an answer could not be read"
                elif disagreed:
                    why = "the orders disagree"
                else:
                    why = "unsure in both orders"
        return Equivalence(
            decision=decision,
            gold_first=first if isinstance(first, Answer) else None,
            prediction_first=second if isinstance(second, Answer) else None,
            disagreed=disagreed,
            why=why,
            prompt=PROMPT.key,
            model=self.spec.model,
        )

    def _ask(self, question: _Question, swapped: bool) -> Answer | Exception:
        """The model's answer with the gold fact first, or the prediction first when `swapped`."""
        relation, description, gold, predicted, passage = question
        first, second = (predicted, gold) if swapped else (gold, predicted)
        messages = build_messages(relation, description, first, second, passage)

        def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            self._counts.bump("retries")
            log.info("retrying fact-equivalence call in %.2fs: %s", wait, exc)

        def counted() -> None:
            self._counts.bump("calls")
            if swapped:
                self._counts.bump("swapped")

        def complete() -> Completion:
            try:
                completion = self.client.complete(messages, spec=self.spec, schema=EQUIV_SCHEMA)
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
            self._counts.bump("unasked")
            with self._lock:
                if self._stopped is None:
                    self._stopped = stop.report()
            return stop
        except Exception as exc:  # isolation: one failed call never fails the batch
            self._counts.bump("failed")
            log.warning("fact-equivalence call failed: %s", exc)
            return exc
        self._counts.bump("prompt_tokens", completion.prompt_tokens)
        self._counts.bump("completion_tokens", completion.completion_tokens)
        if completion.cached:
            self._counts.bump("cached")
        if completion.cost_usd is not None:
            with self._lock:
                self._cost = (self._cost or 0.0) + completion.cost_usd
        answer = parse_answer(completion)
        if answer.decision is None:
            self._counts.bump("unparseable")
            log.warning("unreadable fact-equivalence answer: %r", completion.text[:200])
        return Answer(decision=answer.decision)

    @property
    def stats(self) -> dict[str, Any]:
        """The judge's run report. `cost_usd` is None until a provider reports one."""
        out: dict[str, Any] = self._counts.snapshot()
        with self._lock:
            out["cost_usd"] = self._cost
            if self._stopped is not None:
                out["stopped"] = dict(self._stopped)
        # The registered prompts the calls sent, by key: none when no call was made.
        out["prompts"] = [PROMPT.key, FRAME.key] if out["calls"] else []
        return out


__all__ = [
    "DECISIONS",
    "EQUIV_SCHEMA",
    "FRAME",
    "MAX_PASSAGE",
    "PROMPT",
    "Equivalence",
    "FactJudge",
    "build_messages",
    "candidate",
    "names",
    "passage_for",
    "relation_key",
    "render_plain",
    "render_question",
    "surface_pair",
]
