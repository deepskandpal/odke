"""The composition, and nothing else.

The stages are declared in `odke.stages`, one Protocol each with a pass-through
default. This module runs them in order and holds no ideas of its own: a stage
can be swapped without touching the others, and a stage left out is the
pass-through, so the pipeline degrades to something useful rather than to an
error. With no grounder you get candidate facts with an `UNCHECKED` verdict,
which is the right default for a caller who wants recall and will filter
themselves.

The order is the ODKE+ order with the seams the paper leaves implicit made
explicit. Per chunk: route, extract, ground, normalise. Over the batch: resolve,
corroborate, score, validate. Then write. Resolution runs before corroboration
on purpose — `Fact.signature` merges on `subject.key`, so corroboration cannot
repair a resolution failure.

Two of the thirteen stages are not on the `run()` path. `Constrainer` compiles
the ontology into the store's own constraints and is exposed as `constraints()`
for a sink to apply before its first write. `Inferrer` is a bootstrap, not a
mode (DECISIONS #8): it produces an ontology the caller reviews and then passes
in here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from odke.ontology import Ontology
from odke.stages import (
    DDL,
    Chunker,
    Constrainer,
    Corroborator,
    Delegated,
    Extractor,
    Grounder,
    Initiator,
    Normalizer,
    PassThroughChunker,
    PassThroughConstrainer,
    PassThroughCorroborator,
    PassThroughGrounder,
    PassThroughNormalizer,
    PassThroughResolver,
    PassThroughRouter,
    PassThroughScorer,
    PassThroughValidator,
    PlatformProfile,
    Resolver,
    Retriever,
    Router,
    Scorer,
    Sink,
    Validator,
)
from odke.types import Document, Entity, Fact, KnowledgeGraph


class DoubleStageWarning(UserWarning):
    """A stage is configured here and the sink's platform does it too.

    Warned, never refused. The caller may want both — odke's exact pass on
    strong identifiers before the write and the platform's fuzzy pass after —
    and the evaluators, not the pipeline, are what say whether the second one
    earned its keep.
    """


class Pipeline:
    """Composes the stages. Deliberately boring — the stages hold the ideas.

    Every stage but the extractor is optional. `None` means the pass-through
    from `odke.stages`, so a caller names only the stages they have opinions
    about and the rest are identity functions.
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
        validator: Validator | None = None,
        constrainer: Constrainer | None = None,
        sinks: Sequence[Sink] = (),
    ) -> None:
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
        self.validator: Validator = PassThroughValidator() if validator is None else validator
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
            # A platform that prunes is doing what a Validator refuses here.
            overlaps = (
                ("resolver", resolver, PassThroughResolver, profile.resolves, "resolves"),
                ("validator", validator, PassThroughValidator, profile.prunes, "prunes"),
                (
                    "constrainer",
                    constrainer,
                    PassThroughConstrainer,
                    profile.constrains,
                    "constrains",
                ),
            )
            for stage, configured, default, covered, verb in overlaps:
                if covered and _is_odkes_own(configured, default):
                    warnings.warn(
                        DoubleStageWarning(
                            f"{profile.name} {verb} after the write and a {stage} is "
                            f"configured in odke too, so that stage will run twice. Pass "
                            f"Delegated(to=...) as the {stage} to hand it to the platform, "
                            f"or keep both on purpose and let the evaluator say which earned it."
                        ),
                        stacklevel=2,
                    )

    def run(self, docs: Sequence[Document]) -> KnowledgeGraph:
        # Counts, not a log: enough to see that routing or validation did
        # something, which is the first question when a graph comes back small.
        stats = {"documents": len(docs), "chunks": 0, "skipped": 0, "deferred": 0, "refused": 0}
        facts: list[Fact] = []
        for doc in docs:
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
                for candidate in self.extractor.extract(chunk, self.ontology):
                    grounded = self.grounder.ground(candidate, doc)
                    facts.append(self.normalizer.normalize(grounded))

        resolved, links = self.resolver.resolve(facts, _entities_of(facts))
        scored = [self.scorer.score(f) for f in self.corroborator.corroborate(resolved)]
        kept: list[Fact] = []
        for fact in scored:
            if self.validator.validate(fact, self.ontology).action == "refuse":
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


# The stage Protocols used to be declared here. They are re-exported so that
# `from odke.pipeline import Extractor` keeps working; `odke.stages` is home.
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
