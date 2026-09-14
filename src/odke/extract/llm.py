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

No vendor is named here. The call goes through `LLMClient`, served by whatever
`odke.llm.resolve` picks for the `extract` role.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from odke.extract._common import (
    ChunkContext,
    Documents,
    index_documents,
    subject_entity,
)
from odke.llm.base import Completion, LLMClient, Message, ModelSpec
from odke.llm.registry import resolve
from odke.llm.roles import ModelRoles
from odke.ontology import Ontology, OntologySnippet, Predicate
from odke.types import Chunk, Entity, Fact, Polarity

_INSTRUCTIONS = """\
Extract facts from the passage in the user message, using only the entity types \
and properties listed below. Anything else is ignored.

Every fact needs "quote": the words of the passage that state it, copied exactly, \
with the same spelling, capitals, punctuation and spacing. A fact whose quote is \
not in the passage is discarded. Give "start", the character offset where the \
quote begins (the passage's first character is 0), if you can.

"polarity" is "denied" when the passage says the fact is not so, "partial" when \
it holds only with a limitation the passage states, and "asserted" otherwise. \
For a property whose range is an entity type, "value" is that entity's name as \
the passage writes it.

Reply with one JSON object and nothing else, in this shape:
{"entities": [{"type": "...", "name": "...", "facts": [{"predicate": "...", \
"value": "...", "quote": "...", "start": 0, "polarity": "asserted", \
"qualifiers": {}}]}]}
Reply {"entities": []} when the passage states none of these properties."""

_REPAIR = (
    'That reply was not a JSON object with an "entities" array. Reply again with '
    "only that object, following the same rules."
)
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
    quote, start, polarity, qualifiers — whose `value` is held to exactly that
    property's schema (its item schema when multi-valued: a fact holds one
    value). The prompt renders the same snippets, so the two cannot drift.
    """
    entities: list[dict[str, Any]] = []
    for snippet in snippets:
        values = snippet.json_schema()["properties"]
        facts: list[dict[str, Any]] = []
        for predicate in snippet.predicates:
            value = values[predicate.name]
            properties: dict[str, Any] = {
                "predicate": {"const": predicate.name},
                "value": value["items"] if value.get("type") == "array" else value,
                "quote": {"type": "string", "minLength": 1},
                "start": {"type": "integer", "minimum": 0},
                "polarity": {"enum": [p.value for p in Polarity]},
            }
            if predicate.qualifiers:
                properties["qualifiers"] = {
                    "type": "object",
                    "properties": {name: {"type": "string"} for name in predicate.qualifiers},
                    "additionalProperties": False,
                }
            facts.append(
                {
                    "type": "object",
                    "properties": properties,
                    "required": ["predicate", "value", "quote"],
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
    should not end a ten-thousand-chunk run. Provider errors still raise.

    `confidence` is a prior, not a probability: the scorer calibrates it (M3).
    `calls` and `rejections` accumulate across chunks, so cost and drop rate are
    counts a caller can read after a run.
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
    ) -> None:
        self.spec = spec if spec is not None else (roles or ModelRoles()).extract
        self._client = client
        self.types = list(types) if types is not None else None
        self.documents = index_documents(documents)
        self.snippet_limit = snippet_limit
        self.confidence = confidence
        self.repairs = repairs
        self.calls: list[ModelCall] = []
        self.rejections: list[Rejection] = []

    @property
    def client(self) -> LLMClient:
        # Resolved on first use, so building an extractor never needs a provider.
        if self._client is None:
            self._client = resolve(self.spec)
        return self._client

    def snippets(self, ontology: Ontology) -> list[OntologySnippet]:
        names = self.types if self.types is not None else sorted(ontology.types)
        if not names:
            raise ValueError(
                f"ontology {ontology.name!r} declares no entity types; pass "
                "LLMExtractor(types=[...]) to say which snippets to prompt with"
            )
        found = [ontology.snippet(name, limit=self.snippet_limit) for name in names]
        return [snippet for snippet in found if snippet.predicates]

    def messages(self, chunk: Chunk, snippets: Sequence[OntologySnippet]) -> list[Message]:
        # The passage is the whole user message, so "start" counts from its first
        # character and maps to the chunk with no arithmetic the model can miss.
        system = "\n\n".join([_INSTRUCTIONS, *(s.render() for s in snippets)])
        return [Message(role="system", content=system), Message(role="user", content=chunk.text)]

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        if not chunk.text.strip():
            return []
        snippets = self.snippets(ontology)
        if not snippets:
            return []
        schema = response_schema(snippets)
        messages = self.messages(chunk, snippets)
        data: dict[str, Any] | None = None
        for attempt in range(self.repairs + 1):
            completion = self.client.complete(messages, spec=self.spec, schema=schema)
            self.calls.append(
                ModelCall(
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.index,
                    model=completion.model or self.spec.model,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    cost_usd=completion.cost_usd,
                    repair=attempt > 0,
                )
            )
            data = _contract(completion)
            if data is not None:
                break
            messages = [
                *messages,
                Message(role="assistant", content=completion.text),
                Message(role="user", content=_REPAIR),
            ]
        if data is None:
            self._reject(chunk, "malformed reply")
            return []

        ctx = ChunkContext(chunk, self.documents.get(chunk.doc_id))
        by_type = {s.type_name: s for s in snippets}
        facts: list[Fact] = []
        for entity in data["entities"]:
            facts.extend(self._entity(ctx, ontology, by_type, entity))
        return facts

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
        if span is None:
            self._reject(ctx.chunk, "quote not in the passage", label, quoted)
            return None
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
        self.rejections.append(
            Rejection(
                doc_id=chunk.doc_id,
                chunk_index=chunk.index,
                reason=reason,
                predicate=predicate,
                quote=quote,
            )
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


__all__ = ["LLMExtractor", "ModelCall", "Rejection", "response_schema"]
