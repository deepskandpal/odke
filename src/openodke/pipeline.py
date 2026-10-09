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

Between the two, when the ontology declares an inverse or a symmetric
predicate, each edge on one gains its partner: the same claim the other way
round, on the same evidence, marked `odke.derived` (DECISIONS #28). It is a
step rather than a fourteenth stage, because the ontology decides everything it
does; `inverses=False` turns it off.

One document's failure is its own (#162). A document whose chunking,
extraction, grounding or normalising raises is left out whole and named, with
its reason, in `stats["failed"]`, and the rest of the batch goes on. A batching
stage says which of its items failed (`openodke._batch`); one that does not is
asked again one item at a time. A configuration error (a missing key, adapter
or extra) is nobody's document and still fails the run.

A budget stop (`openodke.llm.budget.BudgetExceeded`) ends the model calls,
not the run. What was extracted and grounded before it is kept, a fact the stop
reached first stays `UNCHECKED`, the deterministic stages run as usual, the
sinks write, and `stats["stopped"]` says where and why. An error in a stage
that runs over the whole batch (resolving, corroborating, scoring, the gate)
still fails the run.

Two of the thirteen stages are not on the `run()` path. `Constrainer` compiles
the ontology into the store's own constraints and is exposed as `constraints()`
for a sink to apply before its first write. `Inferrer` is a bootstrap, not a
mode (DECISIONS #8): it produces an ontology the caller reviews and then passes
in here.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from openodke._batch import incomplete, is_config_error, reason, told
from openodke._renamed import Renamed, deprecated, module_getattr
from openodke.corroborate.inverses import partners
from openodke.coverage import measure as measure_coverage
from openodke.coverage import offered_by
from openodke.llm.budget import BudgetExceeded
from openodke.ontology import Ontology
from openodke.reextract import Reextract, hook_for
from openodke.reextract import reextract as reextract_gaps
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
from openodke.types import Chunk, Document, Entity, Fact, GroundingVerdict, KnowledgeGraph

log = logging.getLogger("openodke.pipeline")


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

    `coverage=True` counts what extraction left behind in each document, with
    no model (`openodke.coverage`), into `KnowledgeGraph.stats["coverage"]`.
    `reextract=Reextract()` hands those gaps back to the extractor and grounds
    what returns (`openodke.reextract`), with its counts under
    `stats["reextract"]`. Both are off by default.

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
        inverses: bool | None = None,
        sinks: Sequence[Sink] = (),
        validator: Gate | None = None,
        coverage: bool = False,
        reextract: Reextract | None = None,
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
        # On whenever the ontology declares a pair, unless the caller says not.
        self.inverses = bool(ontology.inverses) if inverses is None else inverses
        self.coverage = coverage
        self.reextract = reextract
        # Checked here, so a pipeline that cannot re-extract fails before a call is made.
        self._hook = None if reextract is None else hook_for(reextract, extractor)

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
        stats: dict[str, Any] = {
            "documents": len(docs),
            "chunks": 0,
            "skipped": 0,
            "deferred": 0,
            # Chunks that were extracted from and yielded nothing. A factless
            # passage and a dropped extraction report identically without this,
            # so a batch job cannot tell a bad run from a quiet corpus (#78).
            "empty_extractions": 0,
            # Inverse and symmetric partners added, before the gate saw them.
            "derived": 0,
            "refused": 0,
        }
        # Documents left out because something raised for them alone, with why.
        failed: dict[str, str] = {}

        def fail(doc: Document, stage: str, exc: Exception) -> None:
            if doc.id not in failed:
                failed[doc.id] = reason(stage, exc)
                log.warning("document %s failed and is left out: %s", doc.id, failed[doc.id])

        routed: list[tuple[Document, list[Chunk]]] = []
        for doc in docs:
            chunks: list[Chunk] = []
            try:
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
            except Exception as exc:
                if is_config_error(exc) or isinstance(exc, BudgetExceeded):
                    raise
                fail(doc, "chunk", exc)
                continue
            routed.append((doc, chunks))

        # Every chunk of the run at once, so the model calls are not one at a time.
        everything = [c for _, cs in routed for c in cs]
        stop: BudgetExceeded | None = None
        stage = ""
        try:
            done, errors = _extract(self.extractor, everything, self.ontology)
        except BudgetExceeded as exc:
            stop, stage = exc, "extract"
            done = _rows(exc.partial, len(everything), lambda i: None)
            errors = dict(exc.failures)
        owner = {(c.doc_id, c.index): doc for doc, cs in routed for c in cs}
        for at, error in sorted(errors.items()):
            chunk = everything[at]
            fail(owner[(chunk.doc_id, chunk.index)], "extract", error)
        found = iter(done)
        unextracted = 0
        batches: list[tuple[list[Fact], Document]] = []
        for doc, chunks in routed:
            candidates: list[Fact] = []
            for _ in chunks:
                extracted = next(found)
                if doc.id in failed:
                    continue
                if extracted is None:
                    unextracted += 1
                    continue
                if not extracted:
                    stats["empty_extractions"] += 1
                candidates.extend(extracted)
            batches.append((candidates, doc))
        # After a stop too: the free checks cost nothing, a cached answer is
        # free, and every other call is refused without being made.
        try:
            grounded, errors = _ground_each(self.grounder, batches)
        except BudgetExceeded as exc:
            stop, stage = stop or exc, stage or "ground"
            grounded = _rows(exc.partial, len(batches), lambda i: list(batches[i][0]))
            errors = dict(exc.failures)
        for at, error in sorted(errors.items()):
            fail(batches[at][1], "ground", error)
        # A failed document is left out whole, its other chunks' facts with it.
        kept_rows = [
            (doc, chunks, row)
            for (doc, chunks), row in zip(routed, grounded, strict=True)
            if doc.id not in failed
        ]
        routed = [(doc, chunks) for doc, chunks, _ in kept_rows]
        grounded = [row for _, _, row in kept_rows]
        if self.coverage or self._hook is not None:
            # What extraction left behind, before anything merges or refuses.
            report = measure_coverage(
                [doc for doc, chunks in routed if chunks],
                [fact for row in grounded for fact in row],
                self.ontology,
                offered=offered_by(self.extractor, self.ontology),
                regions={doc.id: [(c.start, c.end) for c in chunks] for doc, chunks in routed},
            )
            if self.coverage:
                stats["coverage"] = report.stats()
            if self.reextract is not None and self._hook is not None and stop is None:
                # The gaps go back to the extractor; what returns is grounded here.
                try:
                    grounded, stats["reextract"] = reextract_gaps(
                        self.reextract,
                        self._hook,
                        lambda more: _ground(self.grounder, more),
                        self.ontology,
                        routed,
                        grounded,
                        report,
                    )
                except BudgetExceeded as exc:
                    # What the first pass grounded stands; the gap pass is dropped.
                    stop, stage = exc, "reextract"
        if stop is not None:
            unchecked = sum(
                f.verdict is GroundingVerdict.UNCHECKED for row in grounded for f in row
            )
            stats["stopped"] = {
                **stop.report(),
                "stage": stage,
                "unextracted": unextracted,
                "unchecked": unchecked,
            }
            log.warning(
                "%s, during %s; kept what was done: %d chunks not extracted, %d facts unchecked",
                stop,
                stage,
                unextracted,
                unchecked,
            )
        facts: list[Fact] = []
        for (doc, _), row in zip(routed, grounded, strict=True):
            try:
                facts.extend([self.normalizer.normalize(fact) for fact in row])
            except Exception as exc:
                if is_config_error(exc) or isinstance(exc, BudgetExceeded):
                    raise
                fail(doc, "normalize", exc)
        if failed:
            stats["failed"] = dict(failed)

        resolved, links = self.resolver.resolve(facts, _entities_of(facts))
        if self.inverses:
            batch = list(resolved)
            derived = partners(batch, self.ontology)
            stats["derived"] = len(derived)
            resolved = [*batch, *derived]
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


def _extract(
    extractor: Extractor, chunks: list[Chunk], ontology: Ontology
) -> tuple[list[list[Fact] | None], dict[int, Exception]]:
    """Every chunk through the extractor, batched when it can batch, each failure its own.

    An extractor that also has `extract_many(chunks, ontology)` gets the run's
    chunks in one call and may run its model calls concurrently; any other is
    called per chunk, as the Protocol says. Either way one list of facts comes
    back for each chunk, in order, or None for a chunk that failed, with the
    failures by chunk index. A batch that fails without saying which chunk
    (`openodke._batch`) is asked again one chunk at a time. A configuration
    error and a budget stop are raised.
    """
    many = getattr(extractor, "extract_many", None)
    if callable(many) and chunks:
        try:
            found = [list(facts) for facts in many(chunks, ontology)]
        except BudgetExceeded:
            raise
        except Exception as exc:
            if is_config_error(exc):
                raise
            if (got := told(exc, len(chunks))) is not None:
                partial, failures = got
                return [None if row is None else list(row) for row in partial], failures
            log.warning("%s.extract_many failed: asking chunk by chunk", type(extractor).__name__)
        else:
            if len(found) != len(chunks):
                raise ValueError(
                    f"{type(extractor).__name__}.extract_many returned {len(found)} results "
                    f"for {len(chunks)} chunks; it must return one list of facts per chunk"
                )
            return list(found), {}
    done: list[list[Fact] | None] = []
    failed: dict[int, Exception] = {}
    for at, chunk in enumerate(chunks):
        try:
            done.append(list(extractor.extract(chunk, ontology)))
        except BudgetExceeded as stop:
            stop.partial = [*done, *([None] * (len(chunks) - len(done)))]
            stop.failures = failed
            raise
        except Exception as exc:
            if is_config_error(exc):
                raise
            done.append(None)
            failed[at] = exc
    return done, failed


def _ground(grounder: Grounder, batches: list[tuple[list[Fact], Document]]) -> list[list[Fact]]:
    """`_ground_each`, raising the first document's failure (`openodke._batch`) if any did."""
    rows, failures = _ground_each(grounder, batches)
    if failures:
        raise incomplete(failures, rows)
    return rows


def _ground_each(
    grounder: Grounder, batches: list[tuple[list[Fact], Document]]
) -> tuple[list[list[Fact]], dict[int, Exception]]:
    """Each document's candidates through the grounder, batched when it can batch.

    A grounder that has `ground_documents(batches)` gets every document's facts
    in one call, and one that has `ground_many(facts, doc)` gets each
    document's; either may run its model calls concurrently. Any other grounder
    is called per fact, as the Protocol says. Either way one fact comes back for
    each that went in: a grounder stamps, it never drops (DECISIONS #20).

    A document the grounder failed on comes back as it went in, and is named in
    the failures, by its index in `batches`. A batch that fails without saying
    which document is asked again one document at a time. A configuration error
    is raised, and so is a budget stop, with every document's row as its
    `partial`: what was grounded, then the rest as it came in, `UNCHECKED`.
    """
    # A document with no candidates costs no call, whichever path runs.
    keep = [at for at, (facts, _) in enumerate(batches) if facts]
    work = [batches[at] for at in keep]
    method = next(
        (
            name
            for name in ("ground_documents", "ground_many")
            if callable(getattr(grounder, name, None))
        ),
        "ground",
    )
    try:
        answered, failures = _answer(grounder, method, work)
    except BudgetExceeded as stop:
        rows = _rows(stop.partial, len(work), lambda i: list(work[i][0]))
        stop.partial = _spread(batches, keep, rows)
        stop.failures = {keep[at]: exc for at, exc in stop.failures.items()}
        raise
    if len(answered) != len(work):
        raise ValueError(
            f"{type(grounder).__name__}.{method} returned {len(answered)} rows for "
            f"{len(work)} documents; a grounder stamps a verdict, it never drops a fact"
        )
    for (facts, _), grounded in zip(work, answered, strict=True):
        if len(grounded) != len(facts):
            raise ValueError(
                f"{type(grounder).__name__}.{method} returned {len(grounded)} facts for "
                f"{len(facts)}; a grounder stamps a verdict, it never drops a fact"
            )
    return _spread(batches, keep, answered), {keep[at]: exc for at, exc in failures.items()}


def _answer(
    grounder: Grounder, method: str, work: list[tuple[list[Fact], Document]]
) -> tuple[list[list[Fact]], dict[int, Exception]]:
    """One row per document of `work`, and the documents that failed, by index."""
    if method == "ground_documents":
        documents = grounder.ground_documents  # type: ignore[attr-defined]
        try:
            return ([list(g) for g in documents(work)] if work else []), {}
        except BudgetExceeded as stop:
            stop.partial = _rows(stop.partial, len(work), lambda i: list(work[i][0]))
            raise
        except Exception as exc:
            if is_config_error(exc):
                raise
            if (got := told(exc, len(work))) is not None:
                partial, said = got
                return [
                    list(work[at][0]) if row is None else list(row)
                    for at, row in enumerate(partial)
                ], said
            log.warning(
                "%s.ground_documents failed: asking document by document", type(grounder).__name__
            )

        def ask(facts: list[Fact], doc: Document) -> list[Fact]:
            try:
                return list(documents([(facts, doc)])[0])
            except Exception as exc:
                # One document's row, not a batch of one.
                partial = getattr(exc, "partial", None)
                if isinstance(partial, list) and len(partial) == 1:
                    setattr(exc, "partial", partial[0])  # noqa: B010
                raise

    elif method == "ground_many":

        def ask(facts: list[Fact], doc: Document) -> list[Fact]:
            return list(grounder.ground_many(facts, doc))  # type: ignore[attr-defined]

    else:

        def ask(facts: list[Fact], doc: Document) -> list[Fact]:
            row: list[Fact] = []
            try:
                for fact in facts:
                    row.append(grounder.ground(fact, doc))
            except BudgetExceeded as stop:
                # The fact it was asking about, when it says, in that fact's place.
                asked = stop.partial if isinstance(stop.partial, Fact) else facts[len(row)]
                stop.partial = [*row, asked, *facts[len(row) + 1 :]]
                raise
            return row

    rows: list[list[Fact]] = []
    failures: dict[int, Exception] = {}
    for at, (facts, doc) in enumerate(work):
        try:
            rows.append(ask(facts, doc))
        except BudgetExceeded as stop:
            row = (
                stop.partial
                if isinstance(stop.partial, list) and len(stop.partial) == len(facts)
                else facts
            )
            stop.partial = [*rows, list(row), *(list(f) for f, _ in work[at + 1 :])]
            stop.failures = failures
            raise
        except Exception as exc:
            if is_config_error(exc):
                raise
            rows.append(list(facts))
            failures[at] = exc
    return rows, failures


def _spread(
    batches: list[tuple[list[Fact], Document]], keep: list[int], rows: list[list[Fact]]
) -> list[list[Fact]]:
    """`rows`, one per kept batch, back in place among the batches with no facts."""
    out: list[list[Fact]] = [[] for _ in batches]
    for at, row in zip(keep, rows, strict=True):
        out[at] = row
    return out


def _rows(partial: Any, length: int, default: Any) -> list[Any]:
    """`partial` when it is a list of `length` rows, else `default(i)` for each row."""
    if isinstance(partial, list) and len(partial) == length:
        return [default(i) if row is None else row for i, row in enumerate(partial)]
    return [default(i) for i in range(length)]


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
