"""From merged candidates to a reviewable ontology, and the whole bootstrap composed.

`infer_ontology` is #31–#34 in order: sample the corpus, run the deterministic
proposers, let the model name what they found (unless told not to), merge
near-duplicates, then build. Building is where the ontology is kept small,
because small ontologies are the finding (`03-schema`: around ten types and
twenty-five arrow types) and because a large one degrades extraction
(`08-creation` §5):

- **support** is the number of distinct documents behind a candidate. The paper
  ranks predicates by frequency in the existing graph; with no graph yet,
  corpus frequency is the honest analogue.
- **the cap** keeps the best-supported `max_types` types and `max_predicates`
  predicates; a model's ordering breaks ties. A predicate whose domain or range
  did not make the cut goes with it, and says so.
- **importance** is support on a log scale, relative to the best-supported
  predicate kept, so the snippet ranking has something to rank by and one
  predicate backed by every row of a large table does not flatten the rest to 0.
- aliases that would point at two entries, keys that name no kept predicate and
  parents that were not kept are removed, and each removal is recorded, so the
  result passes `validate()` without an error.

Everything comes back in `Inference`: the ontology, marked `inferred=True`, and
what it was inferred from — the sample, the evidence behind every entry
(addressed by source and offsets, not by the random ids of one load), the merge
decisions, what was dropped, what the model had rejected and what it cost.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openodke.infer.candidates import PredicateCandidate, Proposals, Proposer, TypeCandidate
from openodke.infer.llm import InferenceCall, LLMProposer, ProposalRejection
from openodke.infer.merge import MergeDecision, merge
from openodke.infer.names import fold
from openodke.infer.propose import propose
from openodke.infer.sample import (
    DEFAULT_SAMPLE_WORDS,
    CorpusSample,
    SampleRecord,
    document_key,
    sample_corpus,
)
from openodke.llm.base import LLMClient, ModelSpec
from openodke.llm.roles import ModelRoles
from openodke.ontology import _JSON_TYPES, EntityType, Ontology, Predicate
from openodke.types import Document, Span

DEFAULT_MAX_TYPES = 10
DEFAULT_MAX_PREDICATES = 25
DEFAULT_MIN_SUPPORT = 1
# Spans kept per entry in the evidence record; support counts all of them.
SPANS_KEPT = 10

Candidate = TypeCandidate | PredicateCandidate


class EvidenceRef(BaseModel):
    """A span, addressed so it can be found again after this process has exited.

    `source` is the document's URI and `line` a record's physical line when the
    loader knew it; `doc_key` is `sample.document_key`, stable across loads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str | None = None
    line: int | None = None
    doc_key: str = ""
    start: int
    end: int
    quote: str | None = None


class EntryEvidence(BaseModel):
    """Why one type or predicate is in the ontology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    support: int
    proposers: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    spans: tuple[EvidenceRef, ...] = ()


class Dropped(BaseModel):
    """Something a candidate had that the ontology does not, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["type", "predicate", "parent", "key", "alias"]
    name: str
    reason: str

    def __str__(self) -> str:
        return f"dropped {self.kind} {self.name}: {self.reason}"


class Inference(BaseModel):
    """An inferred ontology and everything a reviewer needs to judge it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ontology: Ontology
    sample: SampleRecord
    # Keyed like a diagnostic path: "types.Person", "predicates.employer".
    evidence: dict[str, EntryEvidence] = Field(default_factory=dict)
    decisions: tuple[MergeDecision, ...] = ()
    dropped: tuple[Dropped, ...] = ()
    rejections: tuple[ProposalRejection, ...] = ()
    calls: tuple[InferenceCall, ...] = ()
    settings: dict[str, Any] = Field(default_factory=dict)


def infer_ontology(
    corpus: Iterable[Document],
    *,
    llm: bool = True,
    client: LLMClient | None = None,
    spec: ModelSpec | None = None,
    roles: ModelRoles | None = None,
    name: str = "inferred",
    seed: int = 0,
    sample_words: int = DEFAULT_SAMPLE_WORDS,
    max_types: int = DEFAULT_MAX_TYPES,
    max_predicates: int = DEFAULT_MAX_PREDICATES,
    min_support: int = DEFAULT_MIN_SUPPORT,
    proposers: Sequence[Proposer] | None = None,
) -> Inference:
    """Sample, propose, name, merge and build: an `inferred=True` ontology for review.

    `llm=False` runs only the deterministic proposers — no client is resolved
    and nothing is called. With `llm=True` the `infer` role is used, from
    `spec`, or `roles`, or `ModelRoles()`; `client` overrides resolution, which
    is how the tests replay recorded responses.
    """
    sample = sample_corpus(corpus, words=sample_words, seed=seed)
    found = propose(sample, proposers)
    namer: LLMProposer | None = None
    if llm:
        namer = LLMProposer(client=client, spec=spec, roles=roles, max_types=max_types)
        found = found + namer.propose(sample, found)
    merged = merge(found)
    ontology, dropped, sources = build(
        merged.proposals,
        name=name,
        max_types=max_types,
        max_predicates=max_predicates,
        min_support=min_support,
    )
    return Inference(
        ontology=ontology,
        sample=sample.record,
        evidence=_evidence(ontology, sources, sample),
        decisions=merged.decisions,
        dropped=tuple(dropped),
        rejections=tuple(namer.rejections) if namer else (),
        calls=tuple(namer.calls) if namer else (),
        settings={
            "model": namer.spec.model if namer else None,
            "seed": seed,
            "sample_words": sample_words,
            "max_types": max_types,
            "max_predicates": max_predicates,
            "min_support": min_support,
        },
    )


class OntologyInferrer:
    """The `Inferrer` stage (`openodke.stages`): a corpus in, an ontology to review out.

    A bootstrap, not a mode (DECISIONS #8): nothing calls this on its own, and
    `Pipeline` does not take one. `last` holds the full `Inference` of the most
    recent call, evidence and all.
    """

    def __init__(
        self,
        *,
        llm: bool = True,
        client: LLMClient | None = None,
        spec: ModelSpec | None = None,
        roles: ModelRoles | None = None,
        name: str = "inferred",
        seed: int = 0,
        sample_words: int = DEFAULT_SAMPLE_WORDS,
        max_types: int = DEFAULT_MAX_TYPES,
        max_predicates: int = DEFAULT_MAX_PREDICATES,
        min_support: int = DEFAULT_MIN_SUPPORT,
    ) -> None:
        self.llm = llm
        self.client = client
        self.spec = spec
        self.roles = roles
        self.name = name
        self.seed = seed
        self.sample_words = sample_words
        self.max_types = max_types
        self.max_predicates = max_predicates
        self.min_support = min_support
        self.last: Inference | None = None

    def infer(self, corpus: Iterable[Document]) -> Ontology:
        self.last = infer_ontology(
            corpus,
            llm=self.llm,
            client=self.client,
            spec=self.spec,
            roles=self.roles,
            name=self.name,
            seed=self.seed,
            sample_words=self.sample_words,
            max_types=self.max_types,
            max_predicates=self.max_predicates,
            min_support=self.min_support,
        )
        return self.last.ontology


def build(
    proposals: Proposals,
    *,
    name: str = "inferred",
    max_types: int = DEFAULT_MAX_TYPES,
    max_predicates: int = DEFAULT_MAX_PREDICATES,
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> tuple[Ontology, list[Dropped], dict[str, Candidate]]:
    """Merged proposals to an ontology: ranked, capped, and consistent enough to validate.

    Returns the ontology, what was dropped on the way, and the candidate behind
    each kept entry, keyed `types.X` / `predicates.y`.
    """
    dropped: list[Dropped] = []

    types: list[TypeCandidate] = []
    for t in sorted(proposals.types, key=_rank):
        reason = _refusal(t, min_support, {fold(k.name) for k in types})
        if reason is None and len(types) >= max_types:
            reason = f"over the cap of {max_types} types (support {t.support})"
        if reason is not None:
            dropped.append(Dropped(kind="type", name=t.name, reason=reason))
            continue
        types.append(t)
    kept_types = {t.name for t in types}

    predicates: list[PredicateCandidate] = []
    for p in sorted(proposals.predicates, key=_rank):
        domain = tuple(d for d in p.domain if d in kept_types)
        reason = _refusal(p, min_support, {fold(k.name) for k in predicates})
        if reason is None and p.domain and not domain:
            reason = f"its domain ({', '.join(p.domain)}) was not kept"
        if reason is None and p.range not in _JSON_TYPES and p.range not in kept_types:
            reason = f"its range {p.range!r} was not kept"
        if reason is None and len(predicates) >= max_predicates:
            reason = f"over the cap of {max_predicates} predicates (support {p.support})"
        if reason is not None:
            dropped.append(Dropped(kind="predicate", name=p.name, reason=reason))
            continue
        predicates.append(p.model_copy(update={"domain": domain}))

    top = max((p.support for p in predicates), default=0)
    predicate_aliases = _unique_aliases([(p.name, p.aliases) for p in predicates], dropped)
    compiled = {
        p.name: Predicate(
            name=p.name,
            description=p.description,
            domain=p.domain,
            range=p.range,
            cardinality=p.cardinality,
            aliases=predicate_aliases[p.name],
            importance=round(math.log1p(p.support) / math.log1p(top), 3) if top else 0.0,
            examples=p.examples[:3],
        )
        for p in predicates
    }

    type_aliases = _unique_aliases([(t.name, t.aliases) for t in types], dropped)
    entity_types: dict[str, EntityType] = {}
    for t in types:
        parents: list[str] = []
        for parent in t.parents:
            if parent in kept_types and parent != t.name:
                parents.append(parent)
            else:
                dropped.append(
                    Dropped(kind="parent", name=f"{t.name}.{parent}", reason="was not kept")
                )
        entity_types[t.name] = EntityType(
            name=t.name,
            description=t.description,
            parents=tuple(dict.fromkeys(parents)),
            aliases=type_aliases[t.name],
        )

    # Keys last: they name predicates, which the merge may have renamed, and
    # must apply to the type through its lineage.
    answers: dict[str, str] = {}
    for entry in compiled.values():
        answers.setdefault(fold(entry.name), entry.name)
    for entry in compiled.values():
        for alias in entry.aliases:
            answers.setdefault(fold(alias), entry.name)
    lineage = Ontology(types=entity_types).lineage
    for t in types:
        keys: list[str] = []
        for key in t.keys:
            target = answers.get(fold(key))
            if target is None:
                why = "no kept predicate answers to it"
            elif compiled[target].domain and not set(compiled[target].domain) & lineage(t.name):
                why = f"{target!r} does not apply to {t.name}"
            else:
                keys.append(target)
                continue
            dropped.append(Dropped(kind="key", name=f"{t.name}.{key}", reason=why))
        if keys:
            entity_types[t.name] = entity_types[t.name].model_copy(
                update={"keys": tuple(dict.fromkeys(keys))}
            )

    ontology = Ontology(
        name=name, version="0", types=entity_types, predicates=compiled, inferred=True
    )
    sources: dict[str, Candidate] = {f"types.{t.name}": t for t in types}
    sources.update({f"predicates.{p.name}": p for p in predicates})
    return ontology, dropped, sources


def _rank(candidate: Candidate) -> tuple[int, int, int, str]:
    return (
        -candidate.support,
        0 if candidate.rank is not None else 1,
        candidate.rank or 0,
        candidate.name,
    )


def _refusal(candidate: Candidate, min_support: int, taken: set[str]) -> str | None:
    if candidate.support < min_support:
        return f"support {candidate.support} is below the minimum of {min_support}"
    if fold(candidate.name) in taken:
        return "its name folds to one already kept"
    return None


def _unique_aliases(
    entries: Sequence[tuple[str, tuple[str, ...]]], dropped: list[Dropped]
) -> dict[str, tuple[str, ...]]:
    """Each alias kept by the first entry to claim it; a surface form names one thing."""
    owners = {fold(name): name for name, _ in entries}
    out: dict[str, tuple[str, ...]] = {}
    for name, aliases in entries:
        kept: dict[str, str] = {}
        for alias in aliases:
            key = fold(alias)
            if not key:
                continue
            owner = owners.setdefault(key, name)
            if owner != name:
                dropped.append(
                    Dropped(kind="alias", name=f"{name}.{alias}", reason=f"already names {owner!r}")
                )
            elif key != fold(name):
                kept.setdefault(key, alias)
        out[name] = tuple(kept.values())
    return out


def _evidence(
    ontology: Ontology, sources: Mapping[str, Candidate], sample: CorpusSample
) -> dict[str, EntryEvidence]:
    keys = {doc_id: document_key(doc) for doc_id, doc in sample.documents.items()}

    def ref(span: Span) -> EvidenceRef:
        doc = sample.documents.get(span.doc_id)
        line = doc.metadata.get("line") if doc is not None else None
        return EvidenceRef(
            source=doc.uri if doc is not None else None,
            line=line if isinstance(line, int) else None,
            doc_key=keys.get(span.doc_id, ""),
            start=span.start,
            end=span.end,
            quote=span.quote,
        )

    out: dict[str, EntryEvidence] = {}
    for path, candidate in sources.items():
        section, name = path.split(".", 1)
        entry = ontology.types[name] if section == "types" else ontology.predicates[name]
        out[path] = EntryEvidence(
            support=candidate.support,
            proposers=tuple(candidate.proposer.split("+")),
            aliases=entry.aliases,
            spans=tuple(ref(s) for s in candidate.evidence[:SPANS_KEPT]),
        )
    return out


__all__ = [
    "DEFAULT_MAX_PREDICATES",
    "DEFAULT_MAX_TYPES",
    "DEFAULT_MIN_SUPPORT",
    "SPANS_KEPT",
    "Dropped",
    "EntryEvidence",
    "EvidenceRef",
    "Inference",
    "OntologyInferrer",
    "build",
    "infer_ontology",
]
