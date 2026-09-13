"""The thirteen stages, each a Protocol, each with a pass-through default.

DECISIONS #5: stages are Protocols. Anyone supplies their own by writing one
method — no import of ours, no registration, no inheritance. This module is the
whole surface. The generic pipeline is thirteen stages, and a caller who wants
only extract-and-sink gets a no-op for the other eleven. That is what makes the
package a superset rather than a product: labels, ontologies, tenancy keys and
qualifier semantics all arrive as data, and nothing domain-specific enters the
code.

The defaults are the identity function for each stage, or the nearest thing to
one. They are not stubs waiting to be filled in. A pipeline built from them is a
working pipeline that routes nothing out, grounds nothing, resolves nothing and
accepts everything — the right shape for a caller who wants recall and will
filter themselves.

The paper's two front stages — the Initiator that decides what to refresh and
the Retriever that fetches it — stay declared (DECISIONS #9). An SDK is usually
handed its documents, so they are not among the thirteen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

from odke.ontology import Ontology
from odke.types import (
    Chunk,
    Document,
    Entity,
    EntityLink,
    Fact,
    Frozen,
    KnowledgeGraph,
    Resolution,
    RouteVerdict,
    ValidationVerdict,
)

# What a loader reads: a path, a URL, bytes, a row iterator. The loader decides.
Source: TypeAlias = Any
Corpus: TypeAlias = Iterable[Document]
# The entities already known, by key. A dict on a first run; a store-backed
# mapping once there is a graph to resolve against.
EntityIndex: TypeAlias = Mapping[str, Entity]
# Constraints in the store's own language — Cypher `CREATE CONSTRAINT`
# statements, SHACL shapes, SQL — one string per statement or document.
DDL: TypeAlias = Sequence[str]


# --------------------------------------------------------------------------- #
# The paper's optional front stages
# --------------------------------------------------------------------------- #


@runtime_checkable
class Initiator(Protocol):
    """Decides what needs extracting. Optional; default is 'all of it'."""

    def targets(self) -> Iterable[str]: ...


@runtime_checkable
class Retriever(Protocol):
    """Turns targets into documents. Optional; default is the caller's list."""

    def retrieve(self, targets: Iterable[str]) -> Iterable[Document]: ...


# --------------------------------------------------------------------------- #
# The thirteen
# --------------------------------------------------------------------------- #


@runtime_checkable
class Loader(Protocol):
    """Bytes to documents, with modality and tier attached."""

    def load(self, source: Source) -> Iterable[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document without losing offsets.

    Every chunk's `text` is exactly `doc.text[start:end]`, or the grounder has
    nothing to check and provenance is decorative. The user-facing size is a
    word cap; a chunker never splits mid-sentence, because the router is about
    to be asked "fact or narrative?" and half a sentence is neither.
    """

    def chunk(self, doc: Document) -> Iterable[Chunk]: ...


@runtime_checkable
class Router(Protocol):
    """Before extraction — is this chunk a fact at all, or narrative?

    Takes a `Chunk`, not a document: a document is a chunk *source*, and
    document-level routing is a chunker configured not to split. The verdict's
    `label` is the caller's word; the package ships no taxonomy.
    """

    def route(self, chunk: Chunk) -> RouteVerdict: ...


@runtime_checkable
class Extractor(Protocol):
    """One chunk to candidate facts, held to the ontology.

    Takes the whole ontology and slices it itself with `Ontology.snippet()`:
    which types a chunk mentions is something only the extractor can know.
    Implementations are hybrid by design — a pattern extractor handles tables,
    key-value blocks and JSON exactly and for free, a model handles prose — and
    both emit `Fact`, so nothing downstream branches on which one ran.

    The one stage without an identity function: there is no default.
    """

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]: ...


@runtime_checkable
class Grounder(Protocol):
    """Does the cited span actually support the claim? Sets `Fact.verdict`.

    This is the precision stage, and it is cheap by construction: a small
    model, one fact and one evidence span at a time, answering yes or no. It
    stamps rather than drops, so the ablation can count what would have gone;
    the validator is the gate.
    """

    def ground(self, fact: Fact, doc: Document) -> Fact: ...


@runtime_checkable
class Normalizer(Protocol):
    """Dates, numbers, units and name forms to one canonical shape each."""

    def normalize(self, fact: Fact) -> Fact: ...


@runtime_checkable
class Resolver(Protocol):
    """Is this entity one we already have? Proposes links; never destroys.

    Returns the facts — with keys and `Entity.resolution` settled — and the
    `EntityLink`s it decided on, `DIFFERENT` included. Runs before
    corroboration on purpose: `Fact.signature` merges on `subject.key`, so
    corroboration cannot repair a resolution failure.
    """

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[Iterable[Fact], Iterable[EntityLink]]: ...


@runtime_checkable
class Corroborator(Protocol):
    """Merge the same claim across sources; count `support`; resolve conflicts."""

    def corroborate(self, facts: Iterable[Fact]) -> Iterable[Fact]: ...


@runtime_checkable
class Scorer(Protocol):
    """Sets `Fact.confidence` — a number that means something only once it has
    been checked against labels the caller supplies."""

    def score(self, fact: Fact) -> Fact: ...


@runtime_checkable
class Validator(Protocol):
    """Domain, range, cardinality and polarity against the ontology."""

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict: ...


class PlatformProfile(Frozen):
    """What the store does itself, after the write.

    The package sits on top of any platform, and some platforms already do a
    stage: neo4j-graphrag resolves post-write, GraphPruner prunes, an RDF store
    refuses writes that break SHACL. odke must not do those twice, and must
    still be able to measure them. A sink declares its platform here; the
    pipeline warns when a stage is configured on both sides.
    """

    name: str
    # Merges entities after the write — the fuzzy pass, on names.
    resolves: bool = False
    # Enforces the schema itself — uniqueness, cardinality, SHACL.
    constrains: bool = False
    # Removes what the schema does not allow — what a Validator refuses here.
    prunes: bool = False


@runtime_checkable
class Sink(Protocol):
    """Where a finished graph goes. Neo4j, RDF, NetworkX, JSONL, or yours.

    One method. A sink may also carry a `profile: PlatformProfile` attribute
    saying which stages the store does itself after the write; the pipeline
    reads it when present and never requires it.
    """

    def write(self, kg: KnowledgeGraph) -> None: ...


@runtime_checkable
class Constrainer(Protocol):
    """Compiles the ontology into the store's own constraints.

    The same schema that shaped the prompt shapes the uniqueness and
    cardinality rules the store enforces, so nobody writes the rules twice.
    """

    def constrain(self, ontology: Ontology) -> DDL: ...


@runtime_checkable
class Inferrer(Protocol):
    """No ontology? Propose one from the corpus. A bootstrap, not a mode
    (DECISIONS #8): the result is reviewed and frozen, not re-run silently."""

    def infer(self, corpus: Corpus) -> Ontology: ...


# --------------------------------------------------------------------------- #
# Pass-through defaults
# --------------------------------------------------------------------------- #


def _as_documents(source: Source) -> Iterable[Document]:
    if isinstance(source, Document):
        return (source,)
    if isinstance(source, str):
        return (Document(text=source),)
    return source


def _whole(doc: Document) -> Chunk:
    return Chunk(doc_id=doc.id, start=0, end=len(doc.text), text=doc.text, index=0)


class PassThroughLoader:
    """The caller already has documents; hand them on. A bare string becomes
    one document. Reads no files — that is what a real loader is for."""

    def load(self, source: Source) -> Iterable[Document]:
        return _as_documents(source)


class PassThroughChunker:
    """One chunk per document: a chunker configured not to split, which is
    exactly what document-level routing is."""

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        return (_whole(doc),)


class PassThroughRouter:
    """Everything is worth extracting from. Changes nothing for a caller who
    never asked for routing."""

    def route(self, chunk: Chunk) -> RouteVerdict:
        return RouteVerdict(action="extract", scope="chunk")


class PassThroughGrounder:
    """The verdict stays `UNCHECKED`: recall by default, filter downstream."""

    def ground(self, fact: Fact, doc: Document) -> Fact:
        return fact


class PassThroughNormalizer:
    def normalize(self, fact: Fact) -> Fact:
        return fact


class PassThroughResolver:
    """Keys are taken as given, and no links are proposed."""

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[Iterable[Fact], Iterable[EntityLink]]:
        return facts, ()


class PassThroughCorroborator:
    """Every fact is its own claim; `support` stays at 1."""

    def corroborate(self, facts: Iterable[Fact]) -> Iterable[Fact]:
        return facts


class PassThroughScorer:
    def score(self, fact: Fact) -> Fact:
        return fact


class PassThroughValidator:
    """Accepts everything. The store's own constraints are still the second half."""

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        return ValidationVerdict(action="accept")


class PassThroughConstrainer:
    """No constraints: the store enforces nothing it was not told about."""

    def constrain(self, ontology: Ontology) -> DDL:
        return ()


class PassThroughInferrer:
    """Proposes nothing. The caller brought a schema, so there is nothing to infer."""

    def infer(self, corpus: Corpus) -> Ontology:
        return Ontology()


# --------------------------------------------------------------------------- #
# Delegation
# --------------------------------------------------------------------------- #


class Delegated:
    """A stage the platform does, marked as such.

    `Delegated(to="neo4j-graphrag:FuzzyMatchResolver")` satisfies every stage
    Protocol as a pass-through, so it slots in wherever the platform already
    covers the work — and it stamps who did it wherever the data model has a
    place for that. A delegated resolver sets `Entity.resolution` to
    `Resolution(method="linker", linker=to)` on every entity that arrives
    unresolved, so a merge the store makes can be read back and scored the same
    way as one made here. A delegated router or validator names `to` in the
    verdict's `reason`. The fact-to-fact stages have no provenance slot and
    pass through unstamped; no surveyed platform grounds, normalises or scores,
    and the field can be added the day one does.
    """

    def __init__(self, to: str) -> None:
        self.to = to

    def __repr__(self) -> str:
        return f"Delegated(to={self.to!r})"

    @property
    def _reason(self) -> str:
        return f"delegated to {self.to}"

    def load(self, source: Source) -> Iterable[Document]:
        return _as_documents(source)

    def chunk(self, doc: Document) -> Iterable[Chunk]:
        return (_whole(doc),)

    def route(self, chunk: Chunk) -> RouteVerdict:
        return RouteVerdict(action="extract", reason=self._reason)

    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        return ()

    def ground(self, fact: Fact, doc: Document) -> Fact:
        return fact

    def normalize(self, fact: Fact) -> Fact:
        return fact

    def resolve(
        self, facts: Iterable[Fact], index: EntityIndex
    ) -> tuple[Iterable[Fact], Iterable[EntityLink]]:
        stamp = Resolution(method="linker", linker=self.to)
        return [_stamped(f, stamp) for f in facts], ()

    def corroborate(self, facts: Iterable[Fact]) -> Iterable[Fact]:
        return facts

    def score(self, fact: Fact) -> Fact:
        return fact

    def validate(self, fact: Fact, ontology: Ontology) -> ValidationVerdict:
        return ValidationVerdict(action="accept", reason=self._reason)

    def write(self, kg: KnowledgeGraph) -> None:
        return None

    def constrain(self, ontology: Ontology) -> DDL:
        return ()

    def infer(self, corpus: Corpus) -> Ontology:
        return Ontology()


def _stamped(fact: Fact, stamp: Resolution) -> Fact:
    # An identity already decided upstream — by the caller, or by an external
    # id — keeps its provenance; the platform's pass is only claimed for the
    # entities it will actually decide.
    def mark(entity: Entity) -> Entity:
        if entity.resolution is not None:
            return entity
        return entity.model_copy(update={"resolution": stamp})

    obj = mark(fact.object_entity) if fact.object_entity is not None else None
    return fact.model_copy(update={"subject": mark(fact.subject), "object_entity": obj})


__all__ = [
    "DDL",
    "Chunker",
    "Constrainer",
    "Corpus",
    "Corroborator",
    "Delegated",
    "EntityIndex",
    "Extractor",
    "Grounder",
    "Inferrer",
    "Initiator",
    "Loader",
    "Normalizer",
    "PassThroughChunker",
    "PassThroughConstrainer",
    "PassThroughCorroborator",
    "PassThroughGrounder",
    "PassThroughInferrer",
    "PassThroughLoader",
    "PassThroughNormalizer",
    "PassThroughResolver",
    "PassThroughRouter",
    "PassThroughScorer",
    "PassThroughValidator",
    "PlatformProfile",
    "Resolver",
    "Retriever",
    "Router",
    "Scorer",
    "Sink",
    "Source",
    "Validator",
]
