"""Gold adjudication: predictions the gold lacks that the grounder supports (#145).

Gold is incomplete. Text2KGBench and Re-DocRED both leave out facts their
texts state, so a prediction missing from the gold may be a fact the gold
missed, and precision against that gold understates the pipeline by however
many there are. Nobody can say how many without reading them.

So each prediction the gold lacks is grounded three times against the
document it was scored in. One the grounder calls `supported` in at least two
of the three runs is listed as **possibly missing from gold**. The
adjudicated precision counts those as hits. It is printed beside the strict
precision, which never changes, and the list is written out, every
prediction the gold lacks with its three verdicts, for a person to audit.

- **Missing from the gold** is the matcher's call (`match_extraction`): a
  prediction that is not a hit, whether spurious, a wrong value or a wrong
  entity. A spurious prediction that cites no document has nothing to be
  grounded against, and stays a miss.
- **Three runs, three calls.** One call at a provider's default temperature is
  one draw, and two of three is a majority of draws. A response cache keys a
  request on what it asks, which is the same question each time, so each
  run's call carries its run index as `ModelSpec.repeat` (0, 1, 2), and
  `openodke.llm.cache` keys it: three runs are three entries. Run 0 keys as an
  ordinary call of the question does, so a run that already grounded it pays
  for two. A grounder with no model answers alike each time, and its three
  runs agree.
- **The same question is asked once** for every row it appears in: the
  pipeline's prediction and the Validator's copy of it share their verdicts.

Neither number is the truth. The strict one counts every fact the gold missed
as wrong; the adjudicated one trusts the grounder on exactly those facts, and
on the Validator's row it is the grounder vouching for what it already let
through. How often that trust is deserved is measured on label set G's
not-in-gold items, once a person has labelled them (`bench/adjudication.py`).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openodke.eval.bootstrap import LEVEL, RESAMPLES, SEED, bootstrap
from openodke.eval.eval_report import AdjudicatedRow, Adjudication, Estimate, micro
from openodke.eval.extraction import document_counts, match_extraction, per_document
from openodke.eval.formats import GoldFact
from openodke.ground.llm import render_claim
from openodke.llm.base import ModelSpec
from openodke.stages import Grounder
from openodke.types import Document, Fact, GroundingVerdict

RUNS = 3
NEEDED = 2

Key = tuple[str, tuple[Any, ...]]


def for_run(grounder: Grounder, run: int) -> Grounder:
    """The grounder whose calls carry `run` as `ModelSpec.repeat`; itself when it has no spec.

    A shallow copy, so its counts, its pool and its client are the original's,
    and only the spec it sends differs.
    """
    spec = getattr(grounder, "spec", None)
    if not isinstance(spec, ModelSpec) or spec.repeat == run:
        return grounder
    again = copy.copy(grounder)
    again.spec = spec.model_copy(update={"repeat": run})  # type: ignore[attr-defined]
    return again


def ask(
    questions: Sequence[tuple[Fact, Document]], grounder: Grounder, *, runs: int = RUNS
) -> list[tuple[GroundingVerdict, ...]]:
    """Each fact's verdict in each of `runs` runs, grounded afresh against its document.

    A verdict already on a fact is cleared first, since a grounder keeps one it
    finds. A grounder that grounds many documents at once (`LLMGrounder`) gets
    each run as one batch.
    """
    if runs < 1:
        raise ValueError(f"adjudication needs at least one run, got {runs}")
    fresh = [
        (fact.model_copy(update={"verdict": GroundingVerdict.UNCHECKED}), doc)
        for fact, doc in questions
    ]
    found: list[list[GroundingVerdict]] = [[] for _ in fresh]
    for run in range(runs):
        judge = for_run(grounder, run)
        for at, fact in enumerate(_ground(judge, fresh)):
            found[at].append(fact.verdict)
    return [tuple(verdicts) for verdicts in found]


def _ground(judge: Grounder, questions: Sequence[tuple[Fact, Document]]) -> list[Fact]:
    batched = getattr(judge, "ground_documents", None)
    if not callable(batched):
        return [judge.ground(fact, doc) for fact, doc in questions]
    order: dict[str, list[int]] = {}
    for at, (_, doc) in enumerate(questions):
        order.setdefault(doc.id, []).append(at)
    docs = {doc.id: doc for _, doc in questions}
    batches = [([questions[at][0] for at in ats], docs[doc_id]) for doc_id, ats in order.items()]
    out: list[Fact | None] = [None] * len(questions)
    for ats, grounded in zip(order.values(), batched(batches), strict=True):
        for at, fact in zip(ats, grounded, strict=True):
            out[at] = fact
    return [fact for fact in out if fact is not None]


@dataclass
class Audit:
    """One prediction the gold lacks, its document, and what the grounder said in each run."""

    doc_id: str
    fact: Fact
    kind: str
    rows: list[str] = field(default_factory=list)
    verdicts: tuple[GroundingVerdict, ...] = ()
    needed: int = NEEDED

    @property
    def supported(self) -> int:
        return sum(v is GroundingVerdict.SUPPORTED for v in self.verdicts)

    @property
    def listed(self) -> bool:
        return self.supported >= self.needed

    def row(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "claim": render_claim(self.fact),
            "kind": self.kind,
            "rows": self.rows,
            "verdicts": [v.value for v in self.verdicts],
            "supported": self.supported,
            "possibly_missing_from_gold": self.listed,
            "fact": self.fact.model_dump(mode="json"),
        }


def adjudicate(
    configurations: Sequence[tuple[str, Sequence[Fact]]],
    gold: Sequence[GoldFact],
    documents: Sequence[Document],
    grounder: Grounder,
    *,
    runs: int = RUNS,
    needed: int = NEEDED,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[Adjudication, list[Audit]]:
    """Each `(name, facts)` row's strict and adjudicated precision, and the audit list.

    The strict precision is `extraction_rows`'s, range and all, from the same
    documents and draws; the adjudicated one moves each listed prediction from
    a miss to a hit in the document it was scored in, and is resampled on the
    same draws.
    """
    if not 1 <= needed <= runs:
        raise ValueError(f"needed is between 1 and runs ({runs}), got {needed}")
    texts = {doc.id: doc for doc in documents}
    asked: dict[Key, Audit] = {}
    misses: dict[str, list[tuple[str | None, Key]]] = {}
    for name, facts in configurations:
        found = misses[name] = []
        for outcome in match_extraction(gold, per_document(facts)):
            if outcome.predicted is None or outcome.kind == "correct":
                continue
            doc_id = outcome.doc_id if outcome.doc_id in texts else None
            key: Key = (doc_id or "", outcome.predicted.signature)
            found.append((doc_id, key))
            if doc_id is None:
                continue
            entry = asked.setdefault(
                key, Audit(doc_id, outcome.predicted, outcome.kind, needed=needed)
            )
            if name not in entry.rows:
                entry.rows.append(name)
    entries = list(asked.values())
    for entry, verdicts in zip(
        entries, ask([(e.fact, texts[e.doc_id]) for e in entries], grounder, runs=runs), strict=True
    ):
        entry.verdicts = verdicts
    listed = {key for key, entry in asked.items() if entry.listed}

    rows = []
    for name, facts in configurations:
        counts = document_counts(gold, facts)
        moved = dict.fromkeys(counts.by_doc, 0)
        for doc_id, key in misses[name]:
            if doc_id in moved and key in listed:
                moved[doc_id] += 1
        strict = list(counts.by_doc.values())
        lenient = [(tp + moved[d], fp - moved[d], fn) for d, (tp, fp, fn) in counts.by_doc.items()]
        fixed = (0, counts.uncited, 0)

        def precision(
            units: Sequence[tuple[int, int, int]], fixed: tuple[int, int, int] = fixed
        ) -> dict[str, float | None]:
            value = micro([*units, fixed])["precision"]
            return {"precision": None if value is None else float(value)}

        estimates = [
            Estimate.of(
                precision(units)["precision"],
                bootstrap(units, precision, resamples=resamples, seed=seed, level=level)[
                    "precision"
                ],
            )
            for units in (strict, lenient)
        ]
        rows.append(
            AdjudicatedRow(
                name=name,
                not_in_gold=len(misses[name]),
                asked=sum(doc_id is not None for doc_id, _ in misses[name]),
                possibly_missing=sum(
                    doc_id is not None and key in listed for doc_id, key in misses[name]
                ),
                strict=estimates[0],
                adjudicated=estimates[1],
            )
        )
    section = Adjudication(runs=runs, needed=needed, questions=len(entries), rows=tuple(rows))
    return section, entries


def write_audit(path: str | Path, entries: Sequence[Audit]) -> Path:
    """Every prediction the gold lacks, one JSON line each, the listed ones first."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda e: not e.listed)
    target.write_text(
        "".join(json.dumps(e.row(), ensure_ascii=False) + "\n" for e in ordered),
        encoding="utf-8",
        newline="\n",
    )
    return target


__all__ = ["NEEDED", "RUNS", "Audit", "adjudicate", "ask", "for_run", "write_audit"]
