"""The model proposer: names, merges and ranks what the deterministic proposers found.

The model is shown the candidates — each with its support and the tool that
found it — and a stratified prefix of the sample, and asked for a small
ontology back as structured output. Its job is naming, not discovery: every
entry it returns must either claim candidates, in `from`, or quote a passage
it was shown. An entry that does neither has no evidence and is rejected, and
the rejection is recorded. A quote is checked the way the extractor checks one
(DECISIONS #3): found in a passage, or discarded.

What comes back is *added* to the proposals, not substituted for them. An entry
carries the evidence of every candidate it claims and names them as aliases;
the merge step then folds the claimed candidates into it by those aliases and
that shared evidence, under the model's name, and still refuses a merge whose
ranges conflict. A candidate the model leaves out is kept as found, for the
reviewer — a model can rename a deterministic finding, but not delete it.

No vendor is named here. The call goes through `LLMClient`, served by whatever
`odke.llm.resolve` picks for the `infer` role.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from odke.infer.candidates import PredicateCandidate, Proposals, TypeCandidate
from odke.infer.names import fold, predicate_name, type_name
from odke.infer.sample import CorpusSample, words_in
from odke.llm.base import Completion, LLMClient, Message, ModelSpec
from odke.llm.registry import resolve
from odke.llm.roles import ModelRoles
from odke.ontology import _JSON_TYPES, Cardinality
from odke.types import Chunk, Span

# One prompt on a small local model: ~4,000 tokens of passages.
DEFAULT_PROMPT_WORDS = 3_000
LITERAL_RANGES = tuple(sorted(_JSON_TYPES))

_INSTRUCTIONS = """\
You are drafting a small ontology for a corpus. Deterministic tools have read a \
sample of it and proposed the candidate entity types and properties in the user \
message, each with the number of documents that support it and the tool that \
found it. Your job is to name, merge and rank those candidates, not to invent a \
schema of your own.

- Name each type in singular PascalCase and each property in snake_case, choosing \
the name a reader of these documents would expect.
- Merge candidates that mean the same thing into one entry, and list every \
candidate it covers, by the name shown, in "from". A candidate you rename goes in \
"from" too.
- A property's "domain" lists the types it describes. Its "range" is one of the \
types you return, or one of: {literals}.
- Return at most {max_types} types. List types and properties most important first.
- Propose something no candidate covers only when a passage states it, and give \
"quote": words of that passage, copied exactly. An entry with neither "from" nor \
a quote that is in the passages is discarded.
- Candidates you leave out are kept as found, for a person to review.

Reply with one JSON object and nothing else, in this shape:
{"types": [{"name": "...", "description": "...", "parents": [], "from": []}], \
"predicates": [{"name": "...", "description": "...", "domain": ["..."], \
"range": "...", "cardinality": "single", "from": []}]}"""

_REPAIR = (
    'That reply was not a JSON object with "types" and "predicates" arrays. Reply '
    "again with only that object, following the same rules."
)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class InferenceCall:
    """One call to the model, and what it cost."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    # None when the provider did not report cost: unknown, not free.
    cost_usd: float | None
    repair: bool = False


@dataclass(frozen=True, slots=True)
class ProposalRejection:
    """Something the model returned that was not taken, and why."""

    kind: str
    name: str | None
    reason: str


def response_schema(max_types: int) -> dict[str, Any]:
    """The structured-output contract for one proposal."""
    names = {"type": "array", "items": {"type": "string"}}
    quote = {"type": ["string", "null"]}
    type_item = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "parents": names,
            "from": names,
            "quote": quote,
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    predicate_item = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string"},
            "domain": names,
            "range": {"type": "string", "minLength": 1},
            "cardinality": {"enum": ["single", "multi"]},
            "from": names,
            "quote": quote,
        },
        "required": ["name", "range"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "title": "ontology_proposal",
        "properties": {
            "types": {"type": "array", "items": type_item, "maxItems": max_types},
            "predicates": {"type": "array", "items": predicate_item},
        },
        "required": ["types", "predicates"],
        "additionalProperties": False,
    }


def render_candidates(found: Proposals) -> str:
    """The candidates as the model sees them: stable order, so prompts cache and replay."""
    lines = ["Types:"]
    for t in sorted(found.types, key=lambda t: (-t.support, t.name, t.proposer)):
        line = f"- {t.name} (support {t.support}, {t.proposer})"
        if t.parents:
            line += f"; is a {', '.join(t.parents)}"
        if t.keys:
            line += f"; keyed by {', '.join(t.keys)}"
        if t.examples:
            line += f"; e.g. {', '.join(t.examples[:3])}"
        lines.append(line)
    if not found.types:
        lines.append("- (none)")
    lines.append("Properties:")
    for p in sorted(found.predicates, key=lambda p: (-p.support, p.name, p.domain, p.proposer)):
        domain = ", ".join(p.domain) or "any"
        line = f"- {p.name}: {domain} -> {p.range}, {p.cardinality}"
        line += f" (support {p.support}, {p.proposer})"
        if p.examples:
            line += f"; e.g. {', '.join(p.examples[:3])}"
        lines.append(line)
    if not found.predicates:
        lines.append("- (none)")
    return "\n".join(lines)


class LLMProposer:
    """One call through the `infer` role: candidates and passages in, named proposals out.

    `prompt_words` caps the passages: the sample's first chunks, in the order
    they were sampled, which keeps the prefix stratified. `calls` and
    `rejections` accumulate, so cost and what was discarded can be read after.
    A reply that is not the contract gets `repairs` more attempts, then is
    recorded and adds nothing; provider errors still raise.
    """

    name = "llm"

    def __init__(
        self,
        *,
        client: LLMClient | None = None,
        spec: ModelSpec | None = None,
        roles: ModelRoles | None = None,
        prompt_words: int = DEFAULT_PROMPT_WORDS,
        max_types: int = 10,
        repairs: int = 1,
    ) -> None:
        chosen = roles or ModelRoles()
        self.spec = spec if spec is not None else (chosen.infer or chosen.extract)
        self._client = client
        self.prompt_words = prompt_words
        self.max_types = max_types
        self.repairs = repairs
        self.calls: list[InferenceCall] = []
        self.rejections: list[ProposalRejection] = []

    @property
    def client(self) -> LLMClient:
        # Resolved on first use, so building a proposer never needs a provider.
        if self._client is None:
            self._client = resolve(self.spec)
        return self._client

    def passages(self, sample: CorpusSample) -> list[Chunk]:
        taken: list[Chunk] = []
        total = 0
        for chunk in sample.chunks:
            if taken and total >= self.prompt_words:
                break
            taken.append(chunk)
            total += words_in(chunk.text)
        return taken

    def messages(self, sample: CorpusSample, found: Proposals) -> list[Message]:
        system = _INSTRUCTIONS.replace("{literals}", ", ".join(LITERAL_RANGES)).replace(
            "{max_types}", str(self.max_types)
        )
        shown = [
            f"[{i}] {_source(sample, chunk)}\n{chunk.text}"
            for i, chunk in enumerate(self.passages(sample), 1)
        ]
        user = "\n\n".join(
            [
                "Candidates",
                render_candidates(found),
                "Passages",
                *(shown or ["(none)"]),
            ]
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def propose(self, sample: CorpusSample, found: Proposals | None = None) -> Proposals:
        found = found or Proposals()
        if not sample.chunks and not found.types and not found.predicates:
            return Proposals()
        schema = response_schema(self.max_types)
        messages = self.messages(sample, found)
        data: dict[str, Any] | None = None
        for attempt in range(self.repairs + 1):
            completion = self.client.complete(messages, spec=self.spec, schema=schema)
            self.calls.append(
                InferenceCall(
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
            self._reject("reply", None, "malformed reply")
            return Proposals()

        passages = self.passages(sample)
        types = _index(found.types)
        predicates = _index(found.predicates)
        proposed_types = [
            t
            for rank, item in enumerate(data["types"])
            if (t := self._type(item, rank, types, sample, passages)) is not None
        ]
        proposed_predicates = [
            p
            for rank, item in enumerate(data["predicates"])
            if (p := self._predicate(item, rank, predicates, sample, passages)) is not None
        ]
        return Proposals(types=tuple(proposed_types), predicates=tuple(proposed_predicates))

    # ------------------------------------------------------------------ #

    def _type(
        self,
        item: Any,
        rank: int,
        index: Mapping[str, list[TypeCandidate]],
        sample: CorpusSample,
        passages: Sequence[Chunk],
    ) -> TypeCandidate | None:
        if not isinstance(item, dict):
            self._reject("type", None, "malformed entry")
            return None
        raw = item.get("name")
        name = type_name(raw) if isinstance(raw, str) else ""
        if not name:
            self._reject("type", None, "no name")
            return None
        claimed = self._claimed("type", name, item.get("from"), index, type_name)
        evidence = self._evidence("type", name, claimed, item.get("quote"), sample, passages)
        if evidence is None:
            return None
        parents = [type_name(p) for p in _strings(item.get("parents"))]
        return TypeCandidate(
            name=name,
            proposer=self.name,
            description=_text(item.get("description")),
            parents=tuple(dict.fromkeys(p for p in parents if p and p != name)),
            aliases=_aliases(name, claimed),
            keys=tuple(dict.fromkeys(k for c in claimed for k in c.keys)),
            examples=tuple(dict.fromkeys(e for c in claimed for e in c.examples)),
            evidence=evidence,
            rank=rank,
        )

    def _predicate(
        self,
        item: Any,
        rank: int,
        index: Mapping[str, list[PredicateCandidate]],
        sample: CorpusSample,
        passages: Sequence[Chunk],
    ) -> PredicateCandidate | None:
        if not isinstance(item, dict):
            self._reject("predicate", None, "malformed entry")
            return None
        raw = item.get("name")
        name = predicate_name(raw) if isinstance(raw, str) else ""
        if not name:
            self._reject("predicate", None, "no name")
            return None
        domain = tuple(dict.fromkeys(d for d in map(type_name, _strings(item.get("domain"))) if d))
        claimed = [
            c
            for c in self._claimed("predicate", name, item.get("from"), index, predicate_name)
            # A claim is by name, within the domain the model gave: `name` on a
            # Person is not `name` on a Company.
            if not domain or not c.domain or set(c.domain) & set(domain)
        ]
        evidence = self._evidence("predicate", name, claimed, item.get("quote"), sample, passages)
        if evidence is None:
            return None
        range_ = _range(item.get("range")) or (claimed[0].range if claimed else "string")
        raw_cardinality = item.get("cardinality")
        cardinality: Cardinality = (
            raw_cardinality
            if raw_cardinality in ("single", "multi")
            else ("multi" if any(c.cardinality == "multi" for c in claimed) else "single")
        )
        return PredicateCandidate(
            name=name,
            proposer=self.name,
            description=_text(item.get("description")),
            domain=domain or tuple(dict.fromkeys(d for c in claimed for d in c.domain)),
            range=range_,
            cardinality=cardinality,
            aliases=_aliases(name, claimed),
            examples=tuple(dict.fromkeys(e for c in claimed for e in c.examples))[:5],
            observations=tuple(dict.fromkeys(o for c in claimed for o in c.observations)),
            evidence=evidence,
            rank=rank,
        )

    def _claimed(
        self,
        kind: str,
        name: str,
        raw_from: Any,
        index: Mapping[str, list[Any]],
        spell: Any,
    ) -> list[Any]:
        # The entry's own name claims a candidate of that name, listed or not.
        claimed: dict[int, Any] = {id(c): c for c in index.get(fold(name), ())}
        for claim in _strings(raw_from):
            found = index.get(fold(spell(claim)))
            if not found:
                self._reject(kind, name, f"'from' names no candidate: {claim!r}")
                continue
            claimed.update((id(c), c) for c in found)
        return list(claimed.values())

    def _evidence(
        self,
        kind: str,
        name: str,
        claimed: Sequence[TypeCandidate | PredicateCandidate],
        quote: Any,
        sample: CorpusSample,
        passages: Sequence[Chunk],
    ) -> tuple[Span, ...] | None:
        spans = {(s.doc_id, s.start, s.end): s for c in claimed for s in c.evidence}
        if isinstance(quote, str) and quote.strip():
            located = _locate(sample, passages, quote)
            if located is None:
                self._reject(kind, name, "quote not in the passages")
            else:
                spans.setdefault((located.doc_id, located.start, located.end), located)
        if not spans:
            if not (isinstance(quote, str) and quote.strip()):
                self._reject(kind, name, "no evidence: claims no candidate and quotes no passage")
            return None
        return tuple(spans.values())

    def _reject(self, kind: str, name: str | None, reason: str) -> None:
        self.rejections.append(ProposalRejection(kind=kind, name=name, reason=reason))


def _contract(completion: Completion) -> dict[str, Any] | None:
    """The reply as `{"types": [...], "predicates": [...]}`, or None if it is not that shape."""
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
    if not isinstance(data, dict):
        return None
    types, predicates = data.get("types", []), data.get("predicates", [])
    if not isinstance(types, list) or not isinstance(predicates, list):
        return None
    if "types" not in data and "predicates" not in data:
        return None
    return {"types": types, "predicates": predicates}


def _index(candidates: Sequence[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for candidate in candidates:
        out.setdefault(fold(candidate.name), []).append(candidate)
    return out


def _locate(sample: CorpusSample, passages: Sequence[Chunk], quote: str) -> Span | None:
    for chunk in passages:
        at = chunk.text.find(quote)
        if at == -1:
            continue
        start = chunk.start + at
        span = Span(doc_id=chunk.doc_id, start=start, end=start + len(quote), quote=quote)
        if span.is_faithful(sample.document(chunk)):
            return span
    return None


def _aliases(name: str, claimed: Sequence[TypeCandidate | PredicateCandidate]) -> tuple[str, ...]:
    spellings = (a for c in claimed for a in (c.name, *c.aliases))
    return tuple(dict.fromkeys(a for a in spellings if fold(a) != fold(name)))


def _range(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    literal = raw.strip().casefold()
    return literal if literal in _JSON_TYPES else (type_name(raw) or None)


def _strings(raw: Any) -> list[str]:
    return [s for s in raw if isinstance(s, str) and s.strip()] if isinstance(raw, list) else []


def _text(raw: Any) -> str | None:
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _source(sample: CorpusSample, chunk: Chunk) -> str:
    doc = sample.document(chunk)
    if doc.uri:
        return PurePosixPath(unquote(urlparse(doc.uri).path)).name or doc.uri
    return doc.title or "document"


__all__ = [
    "DEFAULT_PROMPT_WORDS",
    "LITERAL_RANGES",
    "InferenceCall",
    "LLMProposer",
    "ProposalRejection",
    "render_candidates",
    "response_schema",
]
