"""The composition, and nothing else.

The stages are declared in `openodke.stages`, one Protocol each with a pass-through
default. This module runs them in order and holds no ideas of its own: a stage
can be swapped without touching the others, and a stage left out is the
pass-through, so the pipeline degrades to something useful rather than to an
error. With no grounder you get candidate facts with an `UNCHECKED` verdict,
which is the right default for a caller who wants recall and will filter
themselves.

The order is the ODKE+ order with the seams the paper leaves implicit made
explicit. Per chunk: route, extract. Per document: ground, normalise. Over the
batch: resolve, corroborate, score, gate. Then write. A stage that can batch
is handed the whole run at once — `extract_many` every chunk, `ground_documents`
every document — and runs its model calls concurrently; the result is the same,
in the same order, as one at a time.
Resolution runs before corroboration on purpose — `Fact.signature` merges on
`subject.key`, so corroboration cannot repair a resolution failure.

Two of the thirteen stages are not on the `run()` path. `Constrainer` compiles
the ontology into the store's own constraints and is exposed as `constraints()`
for a sink to apply before its first write. `Inferrer` is a bootstrap, not a
mode (DECISIONS #8): it produces an ontology the caller reviews and then passes
in here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

from openodke._renamed import Renamed, deprecated, module_getattr
from openodke.ontology import Ontology
from openodke.stages import (
    DDL,
    Chunker,
    Constrainer,
    Corroborator,
    Delegated,
    Extractor,
    Gate,
    Grounder,
    Initiator,
    Normalizer,
    PassThroughChunker,
    PassThroughConstrainer,
    PassThroughCorroborator,
    PassThroughGate,
    PassThroughGrounder,
    PassThroughNormalizer,
    PassThroughResolver,
    PassThroughRouter,
    PassThroughScorer,
    PlatformProfile,
    Resolver,
    Retriever,
    Router,
    Scorer,
    Sink,
)
from openodke.types import Chunk, Document, Entity, Fact, KnowledgeGraph


class DoubleStageWarning(UserWarning):
    """A stage is configured here and the sink's platform does it too.

    Warned, never refused. The caller may want both — openodke's exact pass on
    strong identifiers before the write and the platform's fuzzy pass after —
    and the evaluators, not the pipeline, are what say whether the second one
    earned its keep.
    """


class Pipeline:
    """Composes the stages. Deliberately boring — the stages hold the ideas.

    Every stage but the extractor is optional. `None` means the pass-through
    from `openodke.stages`, so a caller names only the stages they have opinions
    about and the rest are identity functions.

    `validator=` is the 0.2 name of `gate=`, and works with a warning until
    1.0.0 (DECISIONS #26).
    """

    def __init__(
        self,
        ontology: Ontology,
        extractor: Extractor,
        *,
        retriever: Retriever | None = None,
        chunker: Chunker | None = None,
        router: Router | None = None,
        grounder: Grounder | None = None,
        normalizer: Normalizer | None = None,
        resolver: Resolver | None = None,
        corroborator: Corroborator | None = None,
        scorer: Scorer | None = None,
        gate: Gate | None = None,
        constrainer: Constrainer | None = None,
        sinks: Sequence[Sink] = (),
        validator: Gate | None = None,
    ) -> None:
        if validator is not None:
            if gate is not None:
                raise TypeError("pass gate= alone; validator= is its old name")
            deprecated("Pipeline(validator=...)", "Pipeline(gate=...)")
            gate = validator
        self.ontology = ontology
        self.extractor = extractor
        self.retriever = retriever
        self.chunker: Chunker = PassThroughChunker() if chunker is None else chunker
        self.router: Router = PassThroughRouter() if router is None else router
        self.grounder: Grounder = PassThroughGrounder() if grounder is None else grounder
        self.normalizer: Normalizer = PassThroughNormalizer() if normalizer is None else normalizer
        self.resolver: Resolver = PassThroughResolver() if resolver is None else resolver
        self.corroborator: Corroborator = (
            PassThroughCorroborator() if corroborator is None else corroborator
        )
        self.scorer: Scorer = PassThroughScorer() if scorer is None else scorer
        self.gate: Gate = PassThroughGate() if gate is None else gate
        self.constrainer: Constrainer = (
            PassThroughConstrainer() if constrainer is None else constrainer
        )
        self.sinks = tuple(sinks)

        # Once, at configuration, so a long-running pipeline says it one time
        # rather than once per run. Nothing here changes what runs.
        for sink in self.sinks:
            profile = getattr(sink, "profile", None)
            if not isinstance(profile, PlatformProfile):
                continue
            # A platform that prunes is doing what a Gate refuses here.
            overlaps = (
                ("resolver", resolver, PassThroughResolver, profile.resolves, "resolves"),
                ("gate", gate, PassThroughGate, profile.prunes, "prunes"),
                (
                    "constrainer",
                    constrainer,
                    PassThroughConstrainer,
                    # A constrainer compiled for this platform is how the store
                    # learns the rules it enforces: its other half, not a rerun.
                    profile.constrains and getattr(constrainer, "platform", None) != profile.name,
                    "constrains",
                ),
            )
            for stage, configured, default, covered, verb in overlaps:
                if covered and _is_odkes_own(configured, default):
                    warnings.warn(
                        DoubleStageWarning(
                            f"{profile.name} {verb} after the write and a {stage} is "
                            f"configured in openodke too, so that stage will run twice. Pass "
                            f"Delegated(to=...) as the {stage} to hand it to the platform, "
                            f"or keep both on purpose and let the evaluator say which earned it."
                        ),
                        stacklevel=2,
                    )

    @property
    def validator(self) -> Gate:
        """The 0.2 name of `gate`; reading it warns until 1.0.0 (DECISIONS #26)."""
        deprecated("Pipeline.validator", "Pipeline.gate")
        return self.gate

    @validator.setter
    def validator(self, value: Gate) -> None:
        deprecated("Pipeline.validator", "Pipeline.gate")
        self.gate = value

    def run(self, docs: Sequence[Document]) -> KnowledgeGraph:
        # Counts, not a log: enough to see that routing or the gate did
        # something, which is the first question when a graph comes back small.
        stats = {
            "documents": len(docs),
            "chunks": 0,
            "skipped": 0,
            "deferred": 0,
            # Chunks that were extracted from and yielded nothing. A factless
            # passage and a dropped extraction report identically without this,
            # so a batch job cannot tell a bad run from a quiet corpus (#78).
            "empty_extractions": 0,
            "refused": 0,
        }
        routed: list[tuple[Document, list[Chunk]]] = []
        for doc in docs:
            chunks: list[Chunk] = []
            for chunk in self.chunker.chunk(doc):
                stats["chunks"] += 1
                verdict = self.router.route(chunk)
                if verdict.action != "extract":
                    stats["skipped" if verdict.action == "skip" else "deferred"] += 1
                    # A document-scoped verdict stops the document here; the
                    # chunks it never produced are not counted.
                    if verdict.scope == "document":
                        break
                    continue
                chunks.append(chunk)
            routed.append((doc, chunks))

        # Every chunk of the run at once, so the model calls are not one at a time.
        found = iter(_extract(self.extractor, [c for _, cs in routed for c in cs], self.ontology))
        batches: list[tuple[list[Fact], Document]] = []
        for doc, chunks in routed:
            candidates: list[Fact] = []
            for _ in chunks:
                extracted = next(found)
                if not extracted:
                    stats["empty_extractions"] += 1
                candidates.extend(extracted)
            batches.append((candidates, doc))
        facts: list[Fact] = []
        for grounded in _ground(self.grounder, batches):
            facts.extend(self.normalizer.normalize(fact) for fact in grounded)

        resolved, links = self.resolver.resolve(facts, _entities_of(facts))
        scored = [self.scorer.score(f) for f in self.corroborator.corroborate(resolved)]
        kept: list[Fact] = []
        for fact in scored:
            if self.gate.validate(fact, self.ontology).action == "refuse":
                stats["refused"] += 1
            else:
                kept.append(fact)

        kg = KnowledgeGraph(
            entities=tuple(_entities_of(kept).values()),
            facts=tuple(kept),
            links=tuple(links),
            ontology_name=self.ontology.name,
            stats=stats,
        )
        for sink in self.sinks:
            sink.write(kg)
        return kg

    def constraints(self) -> DDL:
        """The ontology compiled into the store's own constraints.

        Not applied by `run()`: a sink applies them before its first write, and
        which sink is the caller's business.
        """
        return self.constrainer.constrain(self.ontology)


def _extract(extractor: Extractor, chunks: list[Chunk], ontology: Ontology) -> list[list[Fact]]:
    """Every chunk through the extractor, batched when it can batch.

    An extractor that also has `extract_many(chunks, ontology)` gets the run's
    chunks in one call and may run its model calls concurrently; any other is
    called per chunk, as the Protocol says. Either way one list of facts comes
    back for each chunk, in order.
    """
    many = getattr(extractor, "extract_many", None)
    if not callable(many) or not chunks:
        return [list(extractor.extract(c, ontology)) for c in chunks]
    found = [list(facts) for facts in many(chunks, ontology)]
    if len(found) != len(chunks):
        raise ValueError(
            f"{type(extractor).__name__}.extract_many returned {len(found)} results for "
            f"{len(chunks)} chunks; it must return one list of facts per chunk"
        )
    return found


def _ground(grounder: Grounder, batches: list[tuple[list[Fact], Document]]) -> list[list[Fact]]:
    """Each document's candidates through the grounder, batched when it can batch.

    A grounder that has `ground_documents(batches)` gets every document's facts
    in one call, and one that has `ground_many(facts, doc)` gets each
    document's; either may run its model calls concurrently. Any other grounder
    is called per fact, as the Protocol says. Either way one fact comes back for
    each that went in: a grounder stamps, it never drops (DECISIONS #20).
    """
    # A document with no candidates costs no call, whichever path runs.
    work = [(facts, doc) for facts, doc in batches if facts]
    documents = getattr(grounder, "ground_documents", None)
    many = getattr(grounder, "ground_many", None)
    if callable(documents):
        method, answered = "ground_documents", [list(g) for g in documents(work)] if work else []
    elif callable(many):
        method, answered = "ground_many", [list(many(facts, doc)) for facts, doc in work]
    else:
        method, answered = "ground", [[grounder.ground(f, doc) for f in fs] for fs, doc in work]
    rows = iter(answered)
    out: list[list[Fact]] = []
    for facts, _ in batches:
        grounded = next(rows, []) if facts else []
        if len(grounded) != len(facts):
            raise ValueError(
                f"{type(grounder).__name__}.{method} returned {len(grounded)} facts for "
                f"{len(facts)}; a grounder stamps a verdict, it never drops a fact"
            )
        out.append(grounded)
    return out


def _is_odkes_own(stage: object, default: type) -> bool:
    """True when the caller configured a real stage — not the pass-through,
    not a `Delegated` marker, not nothing."""
    return stage is not None and not isinstance(stage, Delegated | default)


def _entities_of(facts: Sequence[Fact]) -> dict[str, Entity]:
    """Every distinct entity mentioned as a subject or an edge's object, by key."""
    seen: dict[str, Entity] = {}
    for f in facts:
        seen.setdefault(f.subject.key, f.subject)
        if f.object_entity is not None:
            seen.setdefault(f.object_entity.key, f.object_entity)
    return seen


if TYPE_CHECKING:
    # What a type checker sees; at run time the 0.2 names come from `__getattr__`.
    Validator = Gate
    PassThroughValidator = PassThroughGate

__getattr__ = module_getattr(
    __name__,
    {
        "Validator": Renamed(Gate, "openodke.stages.Gate"),
        "PassThroughValidator": Renamed(PassThroughGate, "openodke.stages.PassThroughGate"),
    },
)

# The stage Protocols used to be declared here. They are re-exported so that
# `from openodke.pipeline import Extractor` keeps working; `openodke.stages` is home.
__all__ = [
    "Chunker",
    "Constrainer",
    "Corroborator",
    "DoubleStageWarning",
    "Extractor",
    "Grounder",
    "Initiator",
    "Normalizer",
    "PassThroughRouter",
    "Pipeline",
    "Resolver",
    "Retriever",
    "Router",
    "Scorer",
    "Sink",
    "Validator",
]
