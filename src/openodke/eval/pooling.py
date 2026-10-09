"""Recall relative to a pool of pipelines, with no gold (#146).

With no gold nobody knows how many facts a document states, so recall cannot
be computed. Two or more pipelines run on the same documents can still be
compared on it, the way TREC compares retrieval systems it has no complete
judgements for: pool what they found, and score each against the pool.

- **The pool** is every supported fact any run wrote, per document,
  deduplicated by its signature after normalisation. Values go through
  `ValueNormalizer` and are then compared with case and spacing folded, as the
  extraction matcher compares them, and each entity is its name key
  (`name_key` of its label), not its key, so two pipelines that key "Acme
  Inc." and "acme" differently still agree on one fact.
- **A run's relative recall** is its supported facts in the pool over the
  pool's: |S ∩ P| / |P|, which is |S| / |P| because the pool holds all of S.
- **The range** is a bootstrap over the documents the pool cites
  (`openodke.eval.bootstrap`), every run's from the same draws.

**It overstates true recall.** The pool misses whatever every run missed, so
its size is a floor on what the documents state and each run's share of it a
ceiling on that run's recall; how far apart they are is the subject of
Zobel's study of TREC's pools (SIGIR 1998). Adding a run can only lower every
number. So the caveat is printed beside the numbers, and each run's coverage
report with them, which needs neither a pool nor gold.

Only a supported fact is pooled: an unchecked, contradicted or not-found one
is in no pool, so each run must have been grounded (`odke ground`, `odke
validate`, or a run whose grounder ran). One run can be openodke's reference
extractor, run with `odke run` on the same documents, as a second opinion.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openodke.corroborate.normalize import ValueNormalizer
from openodke.corroborate.provenance import NAME_KEY
from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED, bootstrap
from openodke.eval.eval_report import (
    Bootstrap,
    Dataset,
    Estimate,
    EvalReport,
    PooledRecall,
    PoolMember,
    Run,
)
from openodke.eval.extraction import normalise_value, per_document
from openodke.eval.ppi import coverage_totals
from openodke.ontology import Ontology
from openodke.types import Document, Entity, Fact, GroundingVerdict

TITLE = "pool"
CAVEAT = (
    "relative recall overstates true recall: the pool holds only what some run found, so "
    "a fact every run missed is in no denominator"
)
Claim = tuple[Any, ...]

DESCRIPTION = """\
odke eval pool RUN_A RUN_B [RUN ...] [--documents DOCS] [--ontology FILE]

Recall with no gold, relative to a pool. Each RUN is a run's facts after
grounding: a facts.jsonl, or the directory a sink wrote. Every supported fact
any run wrote is pooled, per document, once per signature after
normalisation (values normalised, entities by their name key). Each run's
relative recall is its supported facts over the pool's, with a 95% bootstrap
range over the documents.

It overstates true recall: the pool misses what every run missed. With
--documents (the texts the runs read) each run's coverage report is shown
beside it, which needs no pool. A run is named by its directory, or by its
file without the suffix."""


def claim(fact: Fact, normalizer: ValueNormalizer | None = None) -> Claim:
    """What two runs must share to state one fact: the signature, normalised.

    `Fact.signature` with each entity's name key in place of its key, the
    value after `ValueNormalizer` and the matcher's folding, and the
    identity-bearing qualifiers folded the same way.
    """
    fact = (normalizer or ValueNormalizer()).normalize(fact)
    obj = (
        ("entity", _name(fact.object_entity))
        if fact.object_entity is not None
        else ("value", normalise_value(fact.object_value))
    )
    scoped = tuple(
        sorted(
            (key, normalise_value(fact.qualifiers[key]))
            for key in fact.identity_keys
            if key in fact.qualifiers
        )
    )
    return (
        _name(fact.subject),
        fact.subject.type,
        fact.predicate,
        obj,
        fact.polarity.value,
        scoped,
    )


def _name(entity: Entity) -> str:
    return str(entity.attributes.get(NAME_KEY) or entity.key)


def pool(
    runs: Sequence[tuple[str, Sequence[Fact]]],
    *,
    ontology: Ontology | None = None,
    documents: Sequence[Document] | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[PooledRecall, list[str]]:
    """Each `(name, facts)` run's recall relative to the pool of all of them, and notes.

    A supported fact that cites no document is in no document's pool, and is
    left out and counted. With `documents`, each run's coverage report comes
    too, with `ontology`'s relations when one is given.
    """
    if len(runs) < 2:
        raise ValueError(f"a pool needs two runs or more, got {len(runs)}")
    names = [name for name, _ in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"each run needs its own name; got {', '.join(names)}")
    normalizer = ValueNormalizer(ontology)
    found: list[dict[str, set[Claim]]] = []
    notes = []
    for name, facts in runs:
        by_doc: dict[str, set[Claim]] = {}
        uncited = 0
        for fact in per_document(facts):
            if fact.verdict is not GroundingVerdict.SUPPORTED:
                continue
            doc_id = next((e.doc_id for e in fact.evidence), None)
            if doc_id is None:
                uncited += 1
                continue
            by_doc.setdefault(doc_id, set()).add(claim(fact, normalizer))
        found.append(by_doc)
        if uncited:
            notes.append(f"{name}: {uncited} supported fact(s) cite no document and are not pooled")
        if not by_doc:
            notes.append(f"{name}: no supported fact; was it grounded?")
    docs = sorted({doc for by_doc in found for doc in by_doc})
    pooled = {doc: set().union(*(by_doc.get(doc, set()) for by_doc in found)) for doc in docs}
    units = [(len(pooled[doc]), *(len(by_doc.get(doc, ())) for by_doc in found)) for doc in docs]

    def statistic(draw: Sequence[tuple[int, ...]]) -> dict[str, float | None]:
        total = sum(unit[0] for unit in draw)
        return {
            name: sum(unit[at] for unit in draw) / total if total else None
            for at, name in enumerate(names, start=1)
        }

    point = statistic(units)
    ranges = bootstrap(units, statistic, resamples=resamples, seed=seed, level=level)
    members = []
    for at, ((name, facts), by_doc) in enumerate(zip(runs, found, strict=True)):
        others = [run for k, run in enumerate(found) if k != at]
        unique = sum(
            len(claims - set().union(*(run.get(doc, set()) for run in others)))
            for doc, claims in by_doc.items()
        )
        members.append(
            PoolMember(
                name=name,
                facts=len(facts),
                supported=sum(len(claims) for claims in by_doc.values()),
                unique=unique,
                relative_recall=Estimate.of(point[name], ranges[name]),
                coverage=(
                    coverage_totals(documents, facts, ontology) if documents is not None else None
                ),
            )
        )
    if not docs:
        notes.append("the pool is empty: no run has a supported fact citing a document")
    section = PooledRecall(
        pool=sum(len(claims) for claims in pooled.values()),
        documents=len(docs),
        runs=tuple(members),
        bootstrap=Bootstrap(units=len(docs), resamples=resamples, seed=seed, level=level),
        caveat=CAVEAT,
    )
    return section, notes


def run_name(path: str | Path) -> str:
    """A run's name from where its facts are: the directory, or the file without its suffix.

    A file called `facts.jsonl` is named by its directory, as a sink writes it.
    """
    source = Path(path)
    if source.is_dir() or source.name == "facts.jsonl":
        folder = source if source.is_dir() else source.parent
        return folder.resolve().name or str(source)
    return source.stem


def report_pool(
    runs: Sequence[tuple[str, Sequence[Fact]]],
    *,
    ontology: Ontology | None = None,
    documents: Sequence[Document] | None = None,
    dataset: Dataset | None = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> EvalReport:
    """`pool` as the eval report."""
    section, notes = pool(
        runs,
        ontology=ontology,
        documents=documents,
        resamples=resamples,
        seed=seed,
        level=level,
    )
    return EvalReport(
        title=TITLE,
        n=section.pool,
        run=Run(dataset=dataset),
        notes=tuple(notes),
        pooled_recall=section,
    )


__all__ = ["CAVEAT", "DESCRIPTION", "claim", "pool", "report_pool", "run_name"]
