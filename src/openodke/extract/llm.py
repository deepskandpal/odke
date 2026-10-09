"""The model extractor: ontology snippets in, facts with checked spans out.

The paper's central move is that the model never sees the whole schema. It sees
a snippet per entity type (DECISIONS #6), rendered as prose for the prompt and as
JSON Schema for the structured-output contract — one object, so the prompt and
the contract cannot describe different things.

The contract makes every fact carry its own evidence: a quote copied from the
passage, and where the model thinks it starts. That offset is checked, never
trusted (DECISIONS #3). A quote found where the model said stays there; a quote
that is in the passage but was miscounted is moved to where it really is; a
quote that is not in the passage at all is dropped, and the drop is recorded.
Nothing is paraphrased into place, and every span that survives has passed
`Span.is_faithful`.

The quote asked for is the *clause* that supports the claim, not the word that
names the value, because by default the grounder reads that span alone
(DECISIONS #23). The word that tells one fact from its siblings — one of three
regions in a list — is carried alongside as `Evidence.mention`, checked the same
way and dropped if it is not inside the clause.

No vendor is named here. The call goes through `LLMClient`, served by whatever
`openodke.llm.resolve` picks for the `extract` role.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from openodke._batch import incomplete, is_config_error
from openodke.extract._common import (
    ChunkContext,
    Documents,
    index_documents,
    subject_entity,
)
from openodke.ground.llm import render_claim
from openodke.ground.retry import RetryPolicy, call_with_retry
from openodke.llm.base import Completion, LLMClient, Message, ModelSpec
from openodke.llm.budget import BudgetExceeded
from openodke.llm.registry import resolve
from openodke.llm.roles import ModelRoles
from openodke.ontology import Ontology, OntologySnippet, Predicate
from openodke.prompts import Prompt
from openodke.prompts import get as get_prompt
from openodke.types import Chunk, Entity, Fact, Polarity, Span

log = logging.getLogger("openodke.extract")

# Registered prompts (DECISIONS #27): the instructions, which the ontology
# snippets follow in the system message, and the repair turn. The latest version
# of each is what is sent, and every `ModelCall` names the one it sent.
_PROMPT = get_prompt("extract")
_REPAIR_PROMPT = get_prompt("extract.repair")
# The re-extract hook's (#102); the same repair turn follows a malformed reply.
_REEXTRACT = get_prompt("reextract")
# The texts under their old names, for anything that imports them.
_INSTRUCTIONS = _PROMPT.text
_REPAIR = _REPAIR_PROMPT.text
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One call to the model: which chunk it was for, and what it cost."""

    doc_id: str
    chunk_index: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    # None when the provider did not report cost: unknown, not free.
    cost_usd: float | None
    # A second attempt after a reply that did not follow the contract.
    repair: bool = False
    # The key of the registered prompt the call sent (DECISIONS #27): `extract@1`,
    # or on a repair the repair prompt's. None only on a record built elsewhere.
    prompt: str | None = None
    # Answered from the response cache, so nothing was sent and nothing spent.
    cached: bool = False


@dataclass(frozen=True, slots=True)
class MalformedReply:
    """Every reply for one chunk that never became the contract, kept whole.

    A repair that fails used to leave a count and nothing to read, so "the model
    returned nothing" and "the parse failed twice" looked the same from outside.
    `replies` is the raw text of the first attempt and of each repair, in order.
    """

    doc_id: str
    chunk_index: int
    replies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate the extractor refused, and why. A drop is a count, not a mystery."""

    doc_id: str
    chunk_index: int
    reason: str
    predicate: str | None = None
    quote: str | None = None


def response_schema(snippets: Sequence[OntologySnippet]) -> dict[str, Any]:
    """The structured-output contract, built from each snippet's `json_schema()`.

    A snippet's schema describes one entity's values, and evidence needs a place
    per fact. So each property becomes one kind of fact item — predicate, value,
    quote, start, mention, polarity, qualifiers — whose `value` is held to
    exactly that property's schema (its item schema when multi-valued: a fact
    holds one value). The prompt renders the same snippets, so the two cannot
    drift.
    """
    entities: list[dict[str, Any]] = []
    for snippet in snippets:
        values = snippet.json_schema()["properties"]
        facts: list[dict[str, Any]] = []
        for predicate in snippet.predicates:
            value = values[predicate.name]
            # Every key is required and none is nullable. Strict structured
            # output caps both: Anthropic at 24 optional keys and 16 union-typed
            # ones per schema, which a 15-predicate snippet passes at two or three
            # per predicate; OpenAI's strict mode allows no optional key at all.
            # So "nothing" is spelled "": the parser reads "" exactly as it reads
            # a missing key, and `start` is a hint `_locate` checks, never trusts.
            properties: dict[str, Any] = {
                "predicate": {"const": predicate.name},
                "value": value["items"] if value.get("type") == "array" else value,
                "quote": {"type": "string", "minLength": 1},
                "start": {"type": "integer", "minimum": 0},
                # The distinguishing words inside the quote, or "": a clause
                # that states one fact has nothing to tell it apart from.
                "mention": {"type": "string"},
                "polarity": {"enum": [p.value for p in Polarity]},
            }
            if predicate.qualifiers:
                properties["qualifiers"] = {
                    "type": "object",
                    "properties": {name: {"type": "string"} for name in predicate.qualifiers},
                    "required": list(predicate.qualifiers),
                    "additionalProperties": False,
                }
            facts.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                }
            )
        entities.append(
            {
                "type": "object",
                "title": snippet.type_name,
                "properties": {
                    "type": {"const": snippet.type_name},
                    "name": {"type": "string", "minLength": 1},
                    "facts": {"type": "array", "items": {"anyOf": facts}},
                },
                "required": ["type", "name", "facts"],
                "additionalProperties": False,
            }
        )
    return {
        "type": "object",
        "title": "extraction",
        "properties": {"entities": {"type": "array", "items": {"anyOf": entities}}},
        "required": ["entities"],
        "additionalProperties": False,
    }


class LLMExtractor:
    """One model call per chunk, through the `extract` role, facts out with spans.

    `types` names the entity types whose snippets go in the prompt; left out,
    every type in the ontology does, which is right for a small schema and the
    first thing to narrow for a large one. `documents` is the lookup for URI and
    tier (DECISIONS #19); an unregistered chunk is still checked, against itself.

    A reply that is not the contract gets `repairs` more attempts with a repair
    prompt, then is recorded as a rejection rather than raised — one bad reply
    should not end a ten-thousand-chunk run. A transient provider error, such as
    a 429, is retried under `retry` as the grounder's are; one that outlasts it,
    or that waiting cannot fix, still raises.

    `structured=False` sends no response schema: the model is held to the JSON
    shape by the prompt alone, and the parser and repair loop do the rest. That
    is how the ODKE+ paper prompts (App. A), and the way round a provider's
    grammar limits — Anthropic's structured output refuses the typed schema of
    an ontology much past a dozen types.

    `confidence` is a prior, not a probability: the scorer calibrates it (M3).
    `calls`, `rejections`, `empty_extractions` and `malformed` accumulate across
    chunks, so cost, drop rate and silence are counts a caller can read after a
    run. Each call names the registered prompt it sent, and `prompts` lists
    them. An extraction that yields nothing also logs a WARNING naming its
    chunk: a factless passage and a dropped one are not the same event, and a
    batch job cannot measure recall if they report identically (#78).

    `extract_many` is the batched path the pipeline uses: many chunks at once,
    with at most `max_workers` model calls in flight, as `LLMGrounder` does.
    """

    name = "llm"

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        spec: ModelSpec | None = None,
        roles: ModelRoles | None = None,
        types: Sequence[str] | None = None,
        documents: Documents = None,
        snippet_limit: int = 25,
        confidence: float = 0.5,
        repairs: int = 1,
        structured: bool = True,
        retry: RetryPolicy | None = None,
        max_workers: int = 8,
    ) -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be at least 1, got {max_workers}")
        self.spec = spec if spec is not None else (roles or ModelRoles()).extract
        self.structured = structured
        self.retry = retry if retry is not None else RetryPolicy()
        self._client = client
        self.types = list(types) if types is not None else None
        self.documents = index_documents(documents)
        self.snippet_limit = snippet_limit
        self.confidence = confidence
        self.repairs = repairs
        self.calls: list[ModelCall] = []
        self.rejections: list[Rejection] = []
        # Chunks the model was asked about that yielded no fact at all.
        self.empty_extractions = 0
        self.malformed: list[MalformedReply] = []
        self.max_workers = max_workers
        # As in the grounder, the ceiling is on the call, not the pool, so it also
        # holds for a caller who runs several batches, or plain `extract`s, at once.
        self._slots = threading.BoundedSemaphore(max_workers)
        # Guards the four accumulators above, which every worker thread adds to.
        self._lock = threading.Lock()

    @property
    def client(self) -> LLMClient:
        # Resolved on first use, so building an extractor never needs a provider.
        if self._client is None:
            self._client = resolve(self.spec)
        return self._client

    @property
    def prompts(self) -> list[str]:
        """The keys of the registered prompts sent so far, in the order first sent."""
        with self._lock:
            return list(dict.fromkeys(c.prompt for c in self.calls if c.prompt))

    def snippets(self, ontology: Ontology) -> list[OntologySnippet]:
        names = self.types if self.types is not None else sorted(ontology.types)
        if not names:
            raise ValueError(
                f"ontology {ontology.name!r} declares no entity types; pass "
                "LLMExtractor(types=[...]) to say which snippets to prompt with"
            )
        found = [ontology.snippet(name, limit=self.snippet_limit) for name in names]
        return [snippet for snippet in found if snippet.predicates]

    def offered(self, ontology: Ontology) -> list[str]:
        """The predicates the prompt shows the model, in snippet order.

        The coverage report subtracts these from the ontology to name the
        relations the model was never asked about: a type left out of `types`,
        or a predicate past `snippet_limit`, is one.
        """
        return list(dict.fromkeys(p.name for s in self.snippets(ontology) for p in s.predicates))

    def messages(self, chunk: Chunk, snippets: Sequence[OntologySnippet]) -> list[Message]:
        # The passage is the whole user message, so "start" counts from its first
        # character and maps to the chunk with no arithmetic the model can miss.
        system = "\n\n".join([_PROMPT.text, *(s.render() for s in snippets)])
        return [Message(role="system", content=system), Message(role="user", content=chunk.text)]

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        if not chunk.text.strip():
            return []
        snippets = self.snippets(ontology)
        if not snippets:
            return []
        data = self._reply(chunk, self.messages(chunk, snippets), snippets, _PROMPT)
        if data is None:
            with self._lock:
                self.empty_extractions += 1
            return []
        with self._lock:
            since = len(self.rejections)
        facts = self._facts(chunk, ontology, snippets, data)
        if not facts:
            # Not an error, and not nothing: the same chunk yields five facts on
            # one call and none on the next, and only a count and a line in the
            # log tell that apart from a passage with nothing to say (#78). The
            # two numbers say which happened — an empty reply, or a reply whose
            # every candidate was refused. Other chunks' workers add to the list
            # too, so this chunk's refusals are the ones that name it.
            with self._lock:
                self.empty_extractions += 1
                refused = sum(_is_for(r, chunk) for r in self.rejections[since:])
            log.warning(
                "extraction returned no facts for chunk %s#%d: %d entities in the reply, "
                "%d candidates refused",
                chunk.doc_id,
                chunk.index,
                len(data["entities"]),
                refused,
            )
        return facts

    def reextract(
        self,
        window: Chunk,
        relations: Sequence[str],
        already: Sequence[Fact],
        ontology: Ontology,
    ) -> list[Fact]:
        """Facts the first pass missed in `window`, asked about `relations` alone (#102).

        The re-extract hook (`openodke.reextract.Reextractor`). One call with the
        registered `reextract` prompt: the flagged properties' snippets, the
        facts `already` taken from the passage, rendered as the grounder renders
        claims, and the window. The reply is read exactly as `extract` reads
        one, quotes checked against the window, and a reply with nothing in it
        is the expected answer rather than an empty extraction.
        """
        if not window.text.strip():
            return []
        snippets = self.reextract_snippets(ontology, relations)
        if not snippets:
            return []
        listed = "\n".join(f"- {render_claim(fact)}" for fact in already) or "(none)"
        system = "\n\n".join([_REEXTRACT.text, *(s.render() for s in snippets)])
        user = f"Already extracted from this passage:\n{listed}\n\nPassage:\n{window.text}"
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        data = self._reply(window, messages, snippets, _REEXTRACT)
        return [] if data is None else self._facts(window, ontology, snippets, data)

    def reextract_snippets(
        self, ontology: Ontology, relations: Sequence[str]
    ) -> list[OntologySnippet]:
        """One snippet per entity type, holding only `relations`, ranked as `snippet` ranks.

        `snippet_limit` does not apply: the relations are already the few a gap
        could hold, and one past the limit is exactly what a gap may be missing.
        """
        wanted = set(relations)
        out: list[OntologySnippet] = []
        for name in self.types if self.types is not None else sorted(ontology.types):
            found = [p for p in ontology.predicates_for(name) if p.name in wanted]
            if not found:
                continue
            ranked = sorted(found, key=lambda p: (-p.importance, p.name))
            kind = ontology.types.get(name)
            out.append(
                OntologySnippet(
                    type_name=name,
                    type_description=kind.description if kind is not None else None,
                    predicates=tuple(ranked),
                    ontology_name=ontology.name,
                    ontology_version=ontology.version,
                )
            )
        return out

    def _reply(
        self,
        chunk: Chunk,
        messages: list[Message],
        snippets: Sequence[OntologySnippet],
        prompt: Prompt,
    ) -> dict[str, Any] | None:
        """The contract from the model, after up to `repairs` repairs; None, recorded, if never."""
        schema = response_schema(snippets) if self.structured else None
        data: dict[str, Any] | None = None
        replies: list[str] = []
        for attempt in range(self.repairs + 1):
            completion = self._complete(chunk, messages, schema)
            call = ModelCall(
                doc_id=chunk.doc_id,
                chunk_index=chunk.index,
                model=completion.model or self.spec.model,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cost_usd=completion.cost_usd,
                repair=attempt > 0,
                prompt=(_REPAIR_PROMPT if attempt else prompt).key,
                cached=completion.cached,
            )
            with self._lock:
                self.calls.append(call)
            data = _contract(completion)
            if data is not None:
                return data
            replies.append(completion.text)
            messages = [
                *messages,
                Message(role="assistant", content=completion.text),
                Message(role="user", content=_REPAIR_PROMPT.text),
            ]
        self._reject(chunk, "malformed reply")
        kept = MalformedReply(doc_id=chunk.doc_id, chunk_index=chunk.index, replies=tuple(replies))
        with self._lock:
            self.malformed.append(kept)
        log.warning(
            "no reply followed the contract for chunk %s#%d after %d attempts; "
            "nothing extracted. The raw text is in LLMExtractor.malformed",
            chunk.doc_id,
            chunk.index,
            len(replies),
        )
        return None

    def _facts(
        self,
        chunk: Chunk,
        ontology: Ontology,
        snippets: Sequence[OntologySnippet],
        data: dict[str, Any],
    ) -> list[Fact]:
        ctx = ChunkContext(chunk, self.documents.get(chunk.doc_id))
        by_type = {s.type_name: s for s in snippets}
        facts: list[Fact] = []
        for entity in data["entities"]:
            facts.extend(self._entity(ctx, ontology, by_type, entity))
        return facts

    def extract_many(self, chunks: Sequence[Chunk], ontology: Ontology) -> list[list[Fact]]:
        """Many chunks, with at most `max_workers` model calls in flight.

        One list of facts comes back per chunk, in order, and `calls`,
        `rejections` and `malformed` are left in the order chunk-by-chunk
        extraction gives, so a batch reads exactly as a loop would.

        A chunk whose call fails, after its retries, costs only itself: the
        rest of the batch is extracted, and then the first failure is raised
        with the batch's `partial` (a list of facts per chunk, None where one
        failed) and its `failures` by chunk index (`openodke._batch`). A
        configuration error raises at once and drops the calls not yet
        started, as every call would fail alike. A budget stop
        (`BudgetExceeded`) is raised once the chunks in flight are done, with
        the same two attributes; None in its `partial` is also a chunk the
        stop reached first. The client must be thread-safe: both built-in
        clients are; `ScriptedClient` answers by position and is not, which is
        what `RecordedClient` is for.
        """
        if self.max_workers == 1 or len(chunks) < 2:
            done: list[list[Fact] | None] = []
            failed: dict[int, Exception] = {}
            for at, chunk in enumerate(chunks):
                try:
                    done.append(self.extract(chunk, ontology))
                except BudgetExceeded as stop:
                    stop.partial = [*done, *([None] * (len(chunks) - len(done)))]
                    stop.failures = failed
                    raise
                except Exception as exc:
                    if is_config_error(exc):
                        raise
                    _log_failure(chunk, exc)
                    done.append(None)
                    failed[at] = exc
            if failed:
                raise incomplete(failed, done)
            return [facts for facts in done if facts is not None]
        with self._lock:
            marks = len(self.calls), len(self.rejections), len(self.malformed)
        workers = min(self.max_workers, len(chunks))
        with ThreadPoolExecutor(workers, thread_name_prefix="openodke-extract") as pool:
            futures = [pool.submit(self._guarded, chunk, ontology) for chunk in chunks]
            wait(futures, return_when=FIRST_EXCEPTION)
            if any(f.done() and f.exception() is not None for f in futures):
                pool.shutdown(cancel_futures=True)
        # Only a configuration error or a budget stop is raised by a worker.
        raised = [f.exception() for f in futures if not f.cancelled() and f.exception()]
        # The first in chunk order that is not a budget stop: the pool starts
        # chunks in order, so none before a failed one was cancelled.
        if other := next((e for e in raised if not isinstance(e, BudgetExceeded)), None):
            raise other
        order = {(c.doc_id, c.index): i for i, c in enumerate(chunks)}
        with self._lock:
            _in_chunk_order(self.calls, marks[0], order)
            _in_chunk_order(self.rejections, marks[1], order)
            _in_chunk_order(self.malformed, marks[2], order)
        results = [
            f.result() if not f.cancelled() and f.exception() is None else None for f in futures
        ]
        failures = {at: r.error for at, r in enumerate(results) if isinstance(r, _Failed)}
        partial = [None if isinstance(r, _Failed) else r for r in results]
        if raised:
            budget = raised[0]
            assert isinstance(budget, BudgetExceeded)  # every one left is a budget stop
            budget.partial = partial
            budget.failures = failures
            raise budget
        if failures:
            raise incomplete(failures, partial)
        return [facts for facts in partial if facts is not None]

    def _guarded(self, chunk: Chunk, ontology: Ontology) -> list[Fact] | _Failed:
        """`extract`, with a failure that is this chunk's alone handed back, not raised."""
        try:
            return self.extract(chunk, ontology)
        except BudgetExceeded:
            raise
        except Exception as exc:
            if is_config_error(exc):
                raise
            _log_failure(chunk, exc)
            return _Failed(exc)

    def _complete(
        self, chunk: Chunk, messages: list[Message], schema: dict[str, Any] | None
    ) -> Completion:
        """One call, retried on transient errors: a 429 on chunk 6,000 is weather,
        and the sinks write only once the whole run is done."""

        def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
            log.info(
                "retrying extraction call for chunk %s#%d in %.2fs after attempt %d: %s",
                chunk.doc_id,
                chunk.index,
                wait,
                attempt,
                exc,
            )

        def call() -> Completion:
            with self._slots:
                return self.client.complete(messages, spec=self.spec, schema=schema)

        return call_with_retry(call, self.retry, on_retry=on_retry)

    def _entity(
        self,
        ctx: ChunkContext,
        ontology: Ontology,
        by_type: dict[str, OntologySnippet],
        entity: Any,
    ) -> list[Fact]:
        if not isinstance(entity, dict):
            self._reject(ctx.chunk, "malformed entity")
            return []
        raw_type = entity.get("type")
        snippet = by_type.get(raw_type) if isinstance(raw_type, str) else None
        if snippet is None:
            self._reject(ctx.chunk, f"type not in the prompt: {raw_type!r}")
            return []
        name, items = entity.get("name"), entity.get("facts")
        if not isinstance(name, str) or not name.strip() or not isinstance(items, list):
            self._reject(ctx.chunk, "malformed entity")
            return []
        subject = subject_entity(snippet.type_name, name.strip())
        allowed = {p.name: p for p in snippet.predicates}
        facts: list[Fact] = []
        for item in items:
            fact = self._fact(ctx, ontology, subject, allowed, item)
            if fact is not None:
                facts.append(fact)
        return facts

    def _fact(
        self,
        ctx: ChunkContext,
        ontology: Ontology,
        subject: Entity,
        allowed: dict[str, Predicate],
        item: Any,
    ) -> Fact | None:
        if not isinstance(item, dict):
            self._reject(ctx.chunk, "malformed fact")
            return None
        raw_predicate, quote, value = item.get("predicate"), item.get("quote"), item.get("value")
        predicate = allowed.get(raw_predicate) if isinstance(raw_predicate, str) else None
        label = str(raw_predicate) if raw_predicate is not None else None
        quoted = quote if isinstance(quote, str) else None
        if predicate is None:
            self._reject(ctx.chunk, "predicate not in the snippet", label, quoted)
            return None
        if value is None or value == "" or isinstance(value, dict | list):
            self._reject(ctx.chunk, "no value", label, quoted)
            return None
        polarity = _polarity(item.get("polarity"))
        if polarity is None:
            self._reject(ctx.chunk, "unknown polarity", label, quoted)
            return None
        start = _locate(ctx.chunk.text, quoted, item.get("start")) if quoted else None
        span = ctx.span(start, quoted) if start is not None and quoted else None
        if span is None or start is None or quoted is None:
            self._reject(ctx.chunk, "quote not in the passage", label, quoted)
            return None
        mention = _mention(ctx, item.get("mention"), quoted, start)
        raw_qualifiers = item.get("qualifiers")
        qualifiers = (
            {
                k: v
                for k, v in raw_qualifiers.items()
                if k in predicate.qualifiers and v not in (None, "")
            }
            if isinstance(raw_qualifiers, dict)
            else {}
        )
        return ctx.fact(
            ontology,
            subject=subject,
            predicate=predicate,
            value=value,
            span=span,
            mention=mention,
            extractor=self.name,
            confidence=self.confidence,
            polarity=polarity,
            qualifiers=qualifiers,
        )

    def _reject(
        self,
        chunk: Chunk,
        reason: str,
        predicate: str | None = None,
        quote: str | None = None,
    ) -> None:
        rejection = Rejection(
            doc_id=chunk.doc_id,
            chunk_index=chunk.index,
            reason=reason,
            predicate=predicate,
            quote=quote,
        )
        with self._lock:
            self.rejections.append(rejection)


@dataclass(frozen=True, slots=True)
class _Failed:
    """A chunk whose extraction raised: its own failure, kept apart from the batch."""

    error: Exception


def _log_failure(chunk: Chunk, exc: BaseException) -> None:
    log.warning(
        "extraction failed for chunk %s#%d, the rest of the batch carries on: %s",
        chunk.doc_id,
        chunk.index,
        exc,
    )


def _is_for(record: Rejection, chunk: Chunk) -> bool:
    return (record.doc_id, record.chunk_index) == (chunk.doc_id, chunk.index)


def _in_chunk_order(records: list[Any], mark: int, order: dict[tuple[str, int], int]) -> None:
    """What a batch appended after `mark`, put in the order of its chunks.

    Workers append as they finish; sorted by chunk, the list reads as a loop
    would have left it. The sort is stable, so one chunk's own entries — a call
    and its repair — keep theirs.
    """
    records[mark:] = sorted(
        records[mark:], key=lambda r: order.get((r.doc_id, r.chunk_index), len(order))
    )


def _contract(completion: Completion) -> dict[str, Any] | None:
    """The reply as `{"entities": [...]}`, or None if it is not that shape.

    Structured output arrives parsed. A model without it may still wrap the
    object in a code fence or a sentence; the object inside is taken, and
    anything that is not one is a malformed reply.
    """
    data: Any = completion.parsed
    if data is None:
        text = completion.text.strip()
        fenced = _FENCE.search(text)
        if fenced:
            text = fenced.group(1)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict) and isinstance(data.get("entities"), list):
        return data
    return None


def _mention(ctx: ChunkContext, raw: Any, quote: str, start: int) -> Span | None:
    """The narrower distinguishing span inside the cited clause, or None.

    Found by searching the quote rather than from an offset of its own: models
    count characters badly (see `_locate`), and a second offset is a second
    chance to be wrong. The result goes through `ctx.span`, so it is held to
    `Span.is_faithful` exactly as the clause was.

    A mention the clause does not contain is dropped and logged, not refused.
    The clause is what the grounder reads, and a fact is not worth losing over a
    highlight.
    """
    if not isinstance(raw, str) or not raw:
        return None
    at = quote.find(raw)
    if at == -1:
        log.debug("mention %r is not inside the cited clause %r; dropped", raw, quote)
        return None
    return ctx.span(start + at, raw)


def _polarity(raw: Any) -> Polarity | None:
    # Missing means asserted. A word we do not know is refused rather than read
    # as asserted: "negative" guessed as a positive claim is the costly mistake.
    if raw is None:
        return Polarity.ASSERTED
    try:
        return Polarity(raw)
    except ValueError:
        return None


def _locate(text: str, quote: str, hint: Any) -> int | None:
    """Where `quote` really starts in `text`, preferring the offset the model claimed.

    Models count characters badly, so a claimed offset is a hint. If the quote is
    there, it is used; if the quote occurs elsewhere, the occurrence nearest the
    claim is. The span is always the quote's true position — correcting a count,
    not inventing evidence. A quote that occurs nowhere has no position.
    """
    claimed = hint if isinstance(hint, int) and not isinstance(hint, bool) else None
    if claimed is not None and claimed >= 0 and text.startswith(quote, claimed):
        return claimed
    found: list[int] = []
    at = text.find(quote)
    while at != -1:
        found.append(at)
        at = text.find(quote, at + 1)
    if not found:
        return None
    if claimed is None:
        return found[0]
    return min(found, key=lambda s: (abs(s - claimed), s))


__all__ = ["LLMExtractor", "MalformedReply", "ModelCall", "Rejection", "response_schema"]
