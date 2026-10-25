"""The second model's question: does that text support this claim?

Locatability is not verification. A model can cite a span that genuinely exists
and does not support the fact it was attached to, and only a second reading
catches that. One fact, one verdict from a small model (DECISIONS #7a: the
`ground` role defaults to a cheaper model than `extract`, because if this pass
cost what extraction costs, people would turn it off).

By default the model sees the cited span and answers in three ways. The paper's
own grounder (ODKE+ §3.3.1, App. B) is different: it sees the whole context and
answers True or False, and only affirmed facts are kept. `context="document"`,
`verdicts="binary"` and `VerdictGate(refuse_not_found=True)` together
reproduce it, so the paper's claims can be tested as written.

`prompt=` sends one of the two challengers for the default (#134), both from
EnterpriseRAG-Bench (Onyx): `ground.onyx`, its fact validator word for word,
which answers yes or no, and `ground.statement`, the same question answered
three ways in JSON. Both read the fact as a plain sentence (`render_sentence`).
A calibration card on label set G decides which prompt is the default
(`openodke.calibration`); until then it is `ground.span`.

The span check runs first, inside this grounder, every time. The model is only
asked about a fact whose citation really resolves in the document, and never
about one the free check already settled. `locate=True` then gives a fact that
cited nothing the window naming its subject and object (`SpanLocator`), so the
model reads that window rather than the whole text. Apart from that span, the
grounder sets `Fact.verdict` and nothing else: `confidence` belongs to the scorer.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

from openodke._renamed import Renamed, module_getattr
from openodke.ground.locate import SpanLocator
from openodke.ground.retry import RetryPolicy, call_with_retry
from openodke.ground.span import Counts, SpanGrounder, located
from openodke.ground.widen import WIDEN, widen
from openodke.llm.base import (
    Completion,
    LLMClient,
    Message,
    MissingAPIKey,
    ModelSpec,
    ProviderNotInstalled,
)
from openodke.llm.budget import BudgetExceeded
from openodke.llm.roles import ModelRoles
from openodke.prompts import Prompt
from openodke.prompts import get as get_prompt
from openodke.types import Document, Entity, Fact, GroundingVerdict, Polarity

if TYPE_CHECKING:
    from openodke.calibration import CardStatus

    # What a type checker sees; at run time the 0.2 name comes from `__getattr__`.
    SYSTEM_PROMPT: str

log = logging.getLogger("openodke.ground")

# The three answers the model may give. UNCHECKED is what a fact has before it
# is asked, or after a call that failed; it is never an answer.
VERDICTS = (GroundingVerdict.SUPPORTED, GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND)

# The two prompts this grounder can send, from the registry (DECISIONS #27):
# `ground.span` by default, and the paper's `ground.paper` (ODKE+ App. B) when
# `verdicts="binary"`. The latest version of each is what is sent, and its key
# is what `stats["prompts"]` names.
_SPAN = get_prompt("ground.span")
_PAPER = get_prompt("ground.paper")
# The challengers `prompt=` selects (#134): Onyx's validator, one user message
# with no schema, and the three-way statement prompt with its own frame.
_ONYX = get_prompt("ground.onyx")
_STATEMENT = get_prompt("ground.statement")
_STATEMENT_FRAME = get_prompt("ground.statement.user")
# The prompt ids `LLMGrounder(prompt=...)` takes, any registered version of each.
# `ground.paper` is the paper's binary prompt, chosen by `verdicts="binary"`.
PROMPTS = ("ground.span", "ground.onyx", "ground.statement", "ground.paper")
# The paper's text, for anything that imports it; outside the API, since which
# version a stage sends is provisional (release audit call B).
PAPER_PROMPT = _PAPER.text

GROUNDING_SCHEMA: dict[str, Any] = {
    "title": "GroundingVerdict",
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": [v.value for v in VERDICTS]}},
    "required": ["verdict"],
    "additionalProperties": False,
}

# The paper's prompt asks for True or False. "False" is read as NOT_FOUND: the
# paper's "No" merges a context that contradicts the triple with one that is
# silent about it, so a binary run cannot tell them apart.
BINARY_SCHEMA: dict[str, Any] = {
    "title": "GroundingJudgement",
    "type": "object",
    "properties": {"verdict": {"type": "boolean"}},
    "required": ["verdict"],
    "additionalProperties": False,
}

Context = Literal["span", "document"]
Verdicts = Literal["three_way", "binary"]
# How the claim line of `ground.span`'s user message is written. "inline" is
# `render_claim`, the line the label sheets and the other judges show.
# "labelled" names the three parts and drops the types (`_render_labelled`).
# Opt-in until the grounding calibration decides the default.
ClaimFormat = Literal["inline", "labelled"]
_FIELD = re.compile(r"\{(\w+)\}")


def render_claim(fact: Fact) -> str:
    """The fact as one line a small model can judge against a passage.

    Stable across runs — the same fact renders identically — so provider-side
    prompt caching works and a run is repeatable. Only identity-bearing
    qualifiers appear: a reconcilable one (`start_time`) is a view of the same
    claim rather than part of it (DECISIONS #11, #15), and the model is being
    asked about the claim. Polarity is spelled out because "X sells data" and
    "X does not sell data" are opposite claims and the same triple.
    """
    subject = _mention(fact.subject)
    predicate = fact.predicate.replace("_", " ")
    obj = (
        _mention(fact.object_entity)
        if fact.object_entity is not None
        else _literal(fact.object_value)
    )
    claim = f"{subject} — {predicate} — {obj}"
    scoped = [
        f"{key} = {_literal(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        claim += f" ({', '.join(scoped)})"
    if fact.polarity is Polarity.DENIED:
        return f"It is NOT the case that: {claim}."
    if fact.polarity is Polarity.PARTIAL:
        return f"Only partially, or under a stated condition: {claim}."
    return f"{claim}."


def _render_labelled(fact: Fact) -> str:
    """The claim with its three parts named and no types: `A | relation | B`.

    The same claim as `render_claim` (identity qualifiers, spelled-out polarity),
    in the shape a model is told how to read. On label set G (16 Oct 2026), a
    small classifier shown the inline line (`A (Type) — rel — B (Type).`) called
    13% of the facts the dataset's gold confirms "supported"; shown this one it
    called 90% supported, and still rejected most planted false facts.
    `bench/labels/claim_format.py` runs the same check on any grounder model.
    """
    subject = fact.subject.label or fact.subject.key
    predicate = fact.predicate.replace("_", " ")
    if fact.object_entity is not None:
        obj = fact.object_entity.label or fact.object_entity.key
    else:
        obj = _literal(fact.object_value)
    claim = f"{subject} | {predicate} | {obj}"
    scoped = [
        f"{key} = {_literal(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        claim += f" ({', '.join(scoped)})"
    if fact.polarity is Polarity.DENIED:
        return f"It is NOT the case that: {claim}"
    if fact.polarity is Polarity.PARTIAL:
        return f"Only partially, or under a stated condition: {claim}"
    return claim


def render_sentence(fact: Fact) -> str:
    """The fact as one plain sentence, as the statement prompts read it (#134).

    Subject, relation and object as the text names them, with no types, quotes
    or separators: `Glory and Gore author Lorde.` Identity-bearing qualifiers
    follow in brackets, as `render_claim` keeps them. A denied fact reads "It is
    not the case that …" and a partial one "Only partially, or under a stated
    condition: …". Code, not a model call, so the same fact is the same
    sentence in every run.
    """
    subject = fact.subject.label or fact.subject.key
    predicate = fact.predicate.replace("_", " ")
    if fact.object_entity is not None:
        obj = fact.object_entity.label or fact.object_entity.key
    else:
        obj = _bare(fact.object_value)
    sentence = f"{subject} {predicate} {obj}"
    scoped = [
        f"{key.replace('_', ' ')}: {_bare(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        sentence += f" ({'; '.join(scoped)})"
    sentence += "."
    if fact.polarity is Polarity.DENIED:
        return f"It is not the case that {sentence}"
    if fact.polarity is Polarity.PARTIAL:
        return f"Only partially, or under a stated condition: {sentence}"
    return sentence


def render_triple(fact: Fact) -> str:
    """The fact in the paper's grounder shape: `<subject, predicate(qualifier: value), object>`.

    Labels rather than typed mentions and bare strings rather than JSON, as in the
    paper's worked example (`<Felton Ross, Date of Birth, May 9, 1927>`). The paper
    has no polarity; a denial is spelled into the predicate so it is not asked as
    an assertion.
    """
    subject = fact.subject.label or fact.subject.key
    predicate = fact.predicate.replace("_", " ")
    scoped = [
        f"{key}: {_bare(fact.qualifiers[key])}"
        for key in fact.identity_keys
        if key in fact.qualifiers
    ]
    if scoped:
        predicate += f"({', '.join(scoped)})"
    if fact.polarity is Polarity.DENIED:
        predicate = f"not {predicate}"
    elif fact.polarity is Polarity.PARTIAL:
        predicate = f"partially {predicate}"
    obj = (
        fact.object_entity.label or fact.object_entity.key
        if fact.object_entity is not None
        else _bare(fact.object_value)
    )
    return f"<{subject}, {predicate}, {obj}>"


def build_messages(
    fact: Fact,
    passage: str,
    *,
    binary: bool = False,
    claim_format: ClaimFormat = "inline",
    prompt: str | Prompt | None = None,
) -> list[Message]:
    """Exactly what the model is sent: the claim and the passage, nothing more.

    By default the passage is the cited span — the question is whether *this
    span* supports the claim. `binary=True` sends the paper's prompt and triple
    instead; which passage it is asked against is the grounder's `context`.
    `claim_format="labelled"` writes the claim line with its parts named
    (`ClaimFormat`); the system prompt is the same. `prompt` names a
    challenger, `ground.onyx` or `ground.statement` (a key or a `Prompt`),
    which reads the fact as `render_sentence` writes it.
    """
    chosen = _chosen(prompt)
    if chosen is not None and chosen.id == "ground.onyx":
        return [Message(role="user", content=_fill(chosen.text, passage, render_sentence(fact)))]
    if chosen is not None and chosen.id == "ground.statement":
        frame = _STATEMENT_FRAME.text
        return [
            Message(role="system", content=chosen.text),
            Message(role="user", content=_fill(frame, passage, render_sentence(fact))),
        ]
    if binary or (chosen is not None and chosen.id == "ground.paper"):
        return [
            Message(role="system", content=_PAPER.text),
            Message(
                role="user", content=f"**Context:\n{passage}\n**triple:\n{render_triple(fact)}"
            ),
        ]
    if claim_format == "labelled":
        claim = f"Claim (subject | relation | object): {_render_labelled(fact)}"
    else:
        claim = f"Claim: {render_claim(fact)}"
    system = chosen.text if chosen is not None else _SPAN.text
    return [
        Message(role="system", content=system),
        Message(role="user", content=f"{claim}\n\nPassage:\n{passage}"),
    ]


def _chosen(prompt: str | Prompt | None) -> Prompt | None:
    if prompt is None or isinstance(prompt, Prompt):
        return prompt
    return get_prompt(prompt)


def _fill(template: str, passage: str, statement: str) -> str:
    """A statement template with each field filled in once, so braces in a passage stay text."""
    values = {"answer": passage, "passage": passage, "statement": statement}
    return _FIELD.sub(lambda m: values.get(m[1], m[0]), template)


def parse_verdict(
    completion: Completion, *, binary: bool = False, prompt: str | Prompt | None = None
) -> GroundingVerdict | None:
    """The verdict in a completion, or None when there is not exactly one to read.

    Structured output is the normal path. A model that ignored the schema and
    answered with the bare word is still readable; one that wrote a sentence
    is not — "not supported" must never be read as `SUPPORTED`, so prose is a
    failed answer rather than a guess.

    `prompt` names the prompt the completion answers, when it is a challenger.
    `ground.onyx`'s reply is read as Onyx reads it: the first line naming "yes"
    is `SUPPORTED`, and one naming "no" is `NOT_FOUND`. Onyx counts anything
    else as no; here it is unreadable, as any grounder's unreadable answer is.
    `ground.statement`'s reply is also read from a JSON object in the text, in
    a code fence or not, since without a provider schema a model often wraps it.
    """
    chosen = _chosen(prompt)
    if chosen is not None and chosen.id == "ground.onyx":
        return _parse_onyx(completion.text)
    value: object = completion.parsed.get("verdict") if completion.parsed else None
    if chosen is not None and chosen.id == "ground.statement" and not isinstance(value, str):
        value = _json_verdict(completion.text)
    if binary or (chosen is not None and chosen.id == "ground.paper"):
        return _parse_binary(value, completion.text)
    if not isinstance(value, str):
        match = _BARE_WORD.match(completion.text)
        value = match.group(1) if match else None
    if not isinstance(value, str):
        return None
    # Separators dropped, not swapped: `NotFound`, `not-found` and `not found` are
    # all spellings `_BARE_WORD` accepts, and must all read as the same verdict.
    normalised = re.sub(r"[\s_-]+", "", value.strip().lower())
    return next((v for v in VERDICTS if v.value.replace("_", "") == normalised), None)


_BARE_WORD = re.compile(r"^\W*(supported|contradicted|not[\s_-]?found)\W*$", re.IGNORECASE)
_BARE_BOOL = re.compile(r"^\W*(true|false|yes|no)\W*$", re.IGNORECASE)
_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)
_OBJECT = re.compile(r"\{[^{}]*\}")


def _parse_onyx(text: str) -> GroundingVerdict | None:
    """Onyx's reading (`validate_single_fact`): "yes" on the first line, else "no"."""
    stripped = text.strip()
    first = stripped.splitlines()[0].strip() if stripped else ""
    if _YES.search(first):
        return GroundingVerdict.SUPPORTED
    if _NO.search(first):
        return GroundingVerdict.NOT_FOUND
    return None


def _json_verdict(text: str) -> object:
    """The `verdict` of the one JSON object in `text`, fenced or not; None without one."""
    found = []
    for match in _OBJECT.finditer(text):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(data, dict) and "verdict" in data:
            found.append(data["verdict"])
    return found[0] if len(found) == 1 else None


def _parse_binary(value: object, text: str) -> GroundingVerdict | None:
    """True is SUPPORTED, False is NOT_FOUND; anything else is unreadable."""
    if not isinstance(value, bool):
        word = value if isinstance(value, str) else None
        match = _BARE_BOOL.match(word if word is not None else text)
        if match is None:
            return None
        value = match.group(1).lower() in ("true", "yes")
    return GroundingVerdict.SUPPORTED if value else GroundingVerdict.NOT_FOUND


def _mention(entity: Entity) -> str:
    return f"{entity.label or entity.key} ({entity.type})"


def _bare(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)


def _literal(value: object) -> str:
    # JSON spelling: strings arrive quoted, so their boundary is unambiguous,
    # and numbers, booleans and null read the same way in every run.
    return json.dumps(value, default=str, ensure_ascii=False)


def _prompt_for(prompt: str | None, verdicts: str) -> Prompt:
    """The prompt a grounder sends: the one named, or its mode's default."""
    if prompt is None:
        return _PAPER if verdicts == "binary" else _SPAN
    chosen = get_prompt(prompt)
    if chosen.id not in PROMPTS:
        raise ValueError(f"prompt must be one of {', '.join(PROMPTS)}; got {prompt!r}")
    if (chosen.id == "ground.paper") != (verdicts == "binary"):
        raise ValueError(
            "ground.paper is the paper's binary prompt, sent with verdicts='binary'; "
            f"got prompt={prompt!r} with verdicts={verdicts!r}"
        )
    return chosen


def _schema_for(prompt: Prompt) -> dict[str, Any] | None:
    """The response schema sent with `prompt`: none for Onyx's, which answers in a word."""
    if prompt.id == "ground.paper":
        return BINARY_SCHEMA
    if prompt.id == "ground.onyx":
        return None
    return GROUNDING_SCHEMA


class LLMGrounder:
    """One fact, one span, one verdict — after the free check has had its say.

    `roles.ground` names the model; `client` overrides how it is reached, which
    is how the suite replays recorded answers. Everything the stage did is in
    `stats`: calls made and retried, calls that failed, each verdict's count,
    answers that could not be read, tokens, the key of the registered prompt the
    calls sent under `"prompts"`, and the span check's own counts under `"span"`.
    Calls a response cache answered are counted under `"cached"` too.

    A call is retried on transient errors under `retry`; when it still fails, or
    fails in a way retrying cannot fix, the fact is left `UNCHECKED` and logged.
    `ground` never raises for a provider failure — one bad call must not lose a
    run of ten thousand — and an `UNCHECKED` fact is exactly what the next run
    picks up, because a verdict already on a fact stands and costs no call. A
    missing key or adapter is not a provider failure but configuration: it fails
    every call alike, so it raises instead of leaving the whole run `UNCHECKED`.
    So does a budget stop (`openodke.llm.budget`), once the batch has finished:
    every fact the stop reached first stays `UNCHECKED`, is counted under
    `"unasked"`, and the batch's facts go with the `BudgetExceeded` as its
    `partial`.

    `locate=True` runs `SpanLocator` after the span check: a fact whose span is
    the whole text it came from (`SpanOrigin.CONTEXT`) is asked about the one
    or two sentences naming its subject and object instead, when they exist.
    Its counts are under `"locate"`.

    `widen=True` gives a `not_found` whose span is narrower than its sentence
    one more call, against the sentence (`openodke.ground.widen`, #102). Off by
    default. The extra calls go through `widen_client` when one is given, so a
    cost meter can count them apart; `stats["widen"]` counts them either way.

    `prompt` names the registered prompt to send: `ground.span` by default (or
    `ground.paper` with `verdicts="binary"`), or a challenger, `ground.onyx` or
    `ground.statement` (#134), by id or `id@version`. A challenger reads the
    fact as `render_sentence` writes it, so `claim_format` stays "inline".
    `calibration` is the grounder's standing: its calibration card's line, or
    UNCALIBRATED (`openodke.calibration`).

    `ground_many` is the batched path: a document's facts at once, with at most
    `max_workers` model calls in flight. `ground_documents` is the same across
    many documents, and is what the pipeline uses, so a corpus of one-row
    documents runs as concurrently as one long one. Both clients are
    synchronous and the wait is network I/O, so a thread pool is the right tool
    and needs nothing outside the standard library.
    """

    def __init__(
        self,
        roles: ModelRoles | None = None,
        *,
        client: LLMClient | None = None,
        max_workers: int = 8,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        context: Context = "span",
        verdicts: Verdicts = "three_way",
        locate: bool = False,
        widen: bool = False,
        widen_client: LLMClient | None = None,
        claim_format: ClaimFormat = "inline",
        prompt: str | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        if context not in ("span", "document"):
            raise ValueError(f"context must be 'span' or 'document', got {context!r}")
        if verdicts not in ("three_way", "binary"):
            raise ValueError(f"verdicts must be 'three_way' or 'binary', got {verdicts!r}")
        if claim_format not in ("inline", "labelled"):
            raise ValueError(f"claim_format must be 'inline' or 'labelled', got {claim_format!r}")
        if claim_format == "labelled" and verdicts == "binary":
            raise ValueError(
                "claim_format='labelled' rewrites ground.span's claim line; the paper's "
                "binary prompt sends its own triple, so the two do not combine"
            )
        if widen and context == "document":
            raise ValueError(
                "widen=True re-asks against the sentence around a narrow span, and "
                "context='document' already shows the model the whole document"
            )
        chosen = _prompt_for(prompt, verdicts)
        if claim_format == "labelled" and chosen.id != "ground.span":
            raise ValueError(
                f"claim_format='labelled' rewrites ground.span's claim line; {chosen.key} reads "
                "the fact as a plain sentence"
            )
        self.context: Context = context
        self.claim_format: ClaimFormat = claim_format
        self.binary = chosen.id == "ground.paper"
        self.prompt: Prompt = chosen
        # The user-message frame a challenger sends beside its instructions.
        self._frame: Prompt | None = _STATEMENT_FRAME if chosen.id == "ground.statement" else None
        self.roles = roles if roles is not None else ModelRoles()
        self.spec: ModelSpec = self.roles.ground
        self.client: LLMClient = client if client is not None else self.roles.client_for("ground")
        self.widen = widen
        self.widen_client: LLMClient = widen_client if widen_client is not None else self.client
        self.max_workers = max_workers
        self.retry = retry if retry is not None else RetryPolicy()
        self.span_grounder = SpanGrounder()
        self.locator = SpanLocator() if locate else None
        self._sleep = sleep
        # The ceiling lives on the call, not the pool, so it also holds for a
        # caller who runs several `ground_many`s — or plain `ground`s — at once.
        self._slots = threading.BoundedSemaphore(max_workers)
        self._counts = Counts(
            "facts",
            "skipped",
            "calls",
            "retries",
            "failed",
            "unparseable",
            "prompt_tokens",
            "completion_tokens",
            *(v.value for v in VERDICTS),
        )
        self._cost_lock = threading.Lock()
        self._cost: float | None = None
        # Kept apart from the counts above, so they read the same with widen off.
        self._widened = Counts(
            "retried",
            "recovered",
            "calls",
            "failed",
            "unparseable",
            "prompt_tokens",
            "completion_tokens",
        )
        self._widen_cost: float | None = None

    def ground(self, fact: Fact, doc: Document) -> Fact:
        prepared, passage = self._prepare(fact, doc)
        return prepared if passage is None else self._ask(prepared, passage, doc)

    def ground_many(self, facts: Sequence[Fact], doc: Document) -> list[Fact]:
        """One document's facts, with the model calls run concurrently.

        The span checks run first, here and in order, and only the facts that
        survive them are handed to the pool. One fact comes back for each that
        went in, in the same order: a failed call leaves its own fact
        `UNCHECKED` and the rest of the batch carries on. A client used here
        must be thread-safe — both built-in clients are; `ScriptedClient`
        answers by position and is not, which is what `RecordedClient` is for.
        """
        try:
            return self.ground_documents([(facts, doc)])[0]
        except BudgetExceeded as stop:
            stop.partial = stop.partial[0] if isinstance(stop.partial, list) else None
            raise

    def ground_documents(
        self, batches: Sequence[tuple[Sequence[Fact], Document]]
    ) -> list[list[Fact]]:
        """`ground_many` for several documents at once, their calls sharing one pool.

        Each `(facts, doc)` gets back a list of its facts in the same order, as
        `ground_many` would return it; the only difference is that the
        `max_workers` ceiling spans documents rather than stopping at each one.
        """
        out: list[list[Fact]] = []
        pending: list[tuple[int, int, Fact, str, Document]] = []
        for at, (facts, doc) in enumerate(batches):
            row: list[Fact] = []
            for index, fact in enumerate(facts):
                prepared, passage = self._prepare(fact, doc)
                row.append(prepared)
                if passage is not None:
                    pending.append((at, index, prepared, passage, doc))
            out.append(row)
        stopped: list[BudgetExceeded] = []
        if len(pending) == 1:
            at, index, fact, passage, doc = pending[0]
            try:
                out[at][index] = self._ask(fact, passage, doc)
            except BudgetExceeded as exc:
                stopped.append(exc)
                if isinstance(exc.partial, Fact):
                    out[at][index] = exc.partial
        elif pending:
            workers = min(self.max_workers, len(pending))
            with ThreadPoolExecutor(workers, thread_name_prefix="openodke-ground") as pool:
                futures = [(a, i, pool.submit(self._ask, f, p, d)) for a, i, f, p, d in pending]
                for at, index, future in futures:
                    try:
                        out[at][index] = future.result()
                    except BudgetExceeded as exc:
                        # The fact stays as the span check left it, UNCHECKED, and
                        # the rest of the batch still finishes: a cached answer is
                        # free, and every other call is refused without being made.
                        stopped.append(exc)
                        if isinstance(exc.partial, Fact):
                            out[at][index] = exc.partial
        if stopped:
            # Facts the stop reached before their call: left UNCHECKED, and counted.
            self._counts.bump("unasked", sum(not isinstance(e.partial, Fact) for e in stopped))
            stop = stopped[0]
            stop.partial = out
            raise stop
        return out

    def _prepare(self, fact: Fact, doc: Document) -> tuple[Fact, str | None]:
        """The fact after the free check, and the passage to ask about — or None
        when there is nothing left to ask."""
        if fact.verdict is not GroundingVerdict.UNCHECKED:
            self._counts.bump("skipped")
            return fact, None
        self._counts.bump("facts")
        checked = self.span_grounder.ground(fact, doc)
        if checked.verdict is not GroundingVerdict.UNCHECKED:
            return checked, None
        if self.locator is not None:
            checked = self.locator.locate(checked, doc)
        evidence = located(checked, doc)
        # The span check leaves a fact UNCHECKED only when something located it,
        # so this guard is for the type checker, not a path that runs.
        if evidence is None or evidence.span is None:  # pragma: no cover
            return checked, None
        if self.context == "document":
            return checked, doc.text
        return checked, evidence.span.resolve(doc)

    def _ask(self, fact: Fact, passage: str, doc: Document) -> Fact:
        verdict = self._judge(fact, passage)
        if verdict is None:
            return fact
        if verdict is GroundingVerdict.NOT_FOUND and self.widen:
            try:
                retried = self._retry_wider(fact, doc)
            except BudgetExceeded as stop:
                # The first answer was paid for: it stands, and the run still stops.
                self._counts.bump(verdict.value)
                stop.partial = fact.model_copy(update={"verdict": verdict})
                raise
            if retried is not None:
                return retried
        self._counts.bump(verdict.value)
        return fact.model_copy(update={"verdict": verdict})

    def _retry_wider(self, fact: Fact, doc: Document) -> Fact | None:
        """One more call against the sentence, or None when the span is already that wide."""
        cited = located(fact, doc)
        wider = widen(cited, doc) if cited is not None else None
        if cited is None or cited.span is None or wider is None or wider.span is None:
            return None
        self._widened.bump("retried")
        verdict = self._judge(fact, wider.span.resolve(doc), widened=True)
        record = {
            "from": [cited.span.start, cited.span.end],
            "to": [wider.span.start, wider.span.end],
            "verdict": (verdict or GroundingVerdict.UNCHECKED).value,
        }
        qualifiers = {**fact.qualifiers, WIDEN: record}
        if verdict is not GroundingVerdict.SUPPORTED:
            self._counts.bump(GroundingVerdict.NOT_FOUND.value)
            return fact.model_copy(
                update={"verdict": GroundingVerdict.NOT_FOUND, "qualifiers": qualifiers}
            )
        self._widened.bump("recovered")
        self._counts.bump(GroundingVerdict.SUPPORTED.value)
        evidence = tuple(wider if e is cited else e for e in fact.evidence)
        return fact.model_copy(
            update={
                "verdict": GroundingVerdict.SUPPORTED,
                "evidence": evidence,
                "qualifiers": qualifiers,
            }
        )

    def _judge(self, fact: Fact, passage: str, *, widened: bool = False) -> GroundingVerdict | None:
        """The model's verdict on `fact` against `passage`, or None when there is none to read."""
        messages = build_messages(
            fact, passage, binary=self.binary, claim_format=self.claim_format, prompt=self.prompt
        )
        counts = self._widened if widened else self._counts

        def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            self._counts.bump("retries")
            log.info(
                "retrying grounding call for fact %s in %.2fs after attempt %d: %s",
                fact.id,
                wait,
                attempt,
                exc,
            )

        try:
            completion = call_with_retry(
                lambda: self._complete(messages, widened=widened),
                self.retry,
                sleep=self._sleep,
                on_retry=on_retry,
            )
        except (MissingAPIKey, ProviderNotInstalled, BudgetExceeded):
            # Not one fact's problem: every call would fail the same way, and a
            # gate that accepts UNCHECKED would write the run as if it were checked.
            # A budget stop is the run's, and the batch hands back what it finished.
            raise
        except Exception as exc:  # isolation: one failed call never fails the batch
            counts.bump("failed")
            left = "its not_found stands" if widened else "left unchecked"
            log.warning("grounding call failed for fact %s, %s: %s", fact.id, left, exc)
            return None
        self._account(completion, widened=widened)
        verdict = parse_verdict(completion, binary=self.binary, prompt=self.prompt)
        if verdict is None:
            counts.bump("unparseable")
            log.warning(
                "unreadable grounding answer for fact %s: %r", fact.id, completion.text[:200]
            )
        return verdict

    def _complete(self, messages: list[Message], *, widened: bool = False) -> Completion:
        with self._slots:
            schema = _schema_for(self.prompt)
            client = self.widen_client if widened else self.client
            try:
                completion = client.complete(messages, spec=self.spec, schema=schema)
            except BudgetExceeded:
                # Refused before it was made: not a call.
                raise
            except BaseException:
                self._count_call(widened)
                raise
            self._count_call(widened)
            return completion

    def _count_call(self, widened: bool) -> None:
        self._counts.bump("calls")
        if widened:
            self._widened.bump("calls")

    def _account(self, completion: Completion, *, widened: bool = False) -> None:
        for counts in (self._counts, self._widened) if widened else (self._counts,):
            counts.bump("prompt_tokens", completion.prompt_tokens)
            counts.bump("completion_tokens", completion.completion_tokens)
            if completion.cached:
                # Counted among `calls`, and only once one happens, so a run
                # without a response cache reports exactly as it did.
                counts.bump("cached")
        if completion.cost_usd is not None:
            with self._cost_lock:
                self._cost = (self._cost or 0.0) + completion.cost_usd
                if widened:
                    self._widen_cost = (self._widen_cost or 0.0) + completion.cost_usd

    @property
    def prompt_keys(self) -> list[str]:
        """Every registered prompt a call sends: the instructions, then any frame."""
        return [self.prompt.key, *([self._frame.key] if self._frame is not None else [])]

    @property
    def claim_line(self) -> str:
        """How the fact is written in the message, as a calibration card names it.

        "inline" or "labelled" for `ground.span` (`claim_format`), "sentence"
        for a challenger (`render_sentence`), "triple" for the paper's prompt.
        """
        if self.binary:
            return "triple"
        if self.prompt.id == "ground.span":
            return self.claim_format
        return "sentence"

    @property
    def calibration(self) -> CardStatus:
        """This prompt and model's calibration card, as reports print it, or UNCALIBRATED."""
        from openodke.calibration import status

        return status("ground", self.prompt.key, self.spec.model, self.claim_line)

    @property
    def stats(self) -> dict[str, Any]:
        """The stage's run report. `cost_usd` is None until a provider reports one."""
        out: dict[str, Any] = self._counts.snapshot()
        with self._cost_lock:
            out["cost_usd"] = self._cost
        # The registered prompts the calls sent, by key: none when no call was made.
        out["prompts"] = self.prompt_keys if out["calls"] else []
        # How the claim line was written: code, not a registered text, so a
        # calibration card needs it beside the prompt key.
        out["claim_format"] = self.claim_format
        out["span"] = self.span_grounder.stats
        if self.locator is not None:
            out["locate"] = self.locator.stats
        if self.widen:
            # The retries' own share; their calls, tokens and cost are in the totals too.
            widened: dict[str, Any] = self._widened.snapshot()
            with self._cost_lock:
                widened["cost_usd"] = self._widen_cost
            out["widen"] = widened
        return out


# The default prompt's text under its 0.2 name, until 2.0.0: which version a stage
# sends is provisional, so the text is read from the registry (release audit call B).
__getattr__ = module_getattr(
    __name__,
    {"SYSTEM_PROMPT": Renamed(_SPAN.text, 'openodke.prompts.get("ground.span").text')},
)

__all__ = [
    "BINARY_SCHEMA",
    "GROUNDING_SCHEMA",
    "PROMPTS",
    "VERDICTS",
    "LLMGrounder",
    "build_messages",
    "parse_verdict",
    "render_claim",
    "render_sentence",
    "render_triple",
]
