"""Evaluating the Evaluator, the other half (#147): a planted cause lands in its bucket.

The A/A half is `test_eval_aa.py`: the same pipeline twice is inconclusive about
95% of the time. Here each test breaks a correct pipeline in one known way,
runs it through the path a user runs, and checks that the diagnosis names that
cause and that the fix ranked first is the one for it:

- a relation hidden from the extractor's snippet: relation never offered, and
  "offer it" first;
- the pair written the other way round, as the inverse relation, with the
  inverse step off: inverse direction, and "add inverse partners" first;
- the output capped per document: output saturation, and "smaller chunks" first;
- a gate that refuses a true fact: refused by the Validator, and "audit the
  refusals" first.

Nothing calls a model: the extractor's replies are scripted, and the grounder is
a stage in this file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from openodke import Document, Ontology, Pipeline
from openodke.eval.diagnosis import read_trace
from openodke.eval.eval_report import Dataset, EvalReport
from openodke.eval.formats import GoldFact
from openodke.eval.harness import Corpus, evaluate_pipeline, score
from openodke.extract.llm import LLMExtractor
from openodke.interop.triples import TripleRow, to_fact
from openodke.llm import ScriptedClient

ONTOLOGY = Ontology.model_validate(
    {
        "name": "planted",
        "types": {"Person": {}, "City": {}, "Company": {}},
        "predicates": {
            "born_in": {"domain": ["Person"], "range": "City", "importance": 1.0},
            "works_for": {"domain": ["Person"], "range": "Company", "importance": 0.8},
            "married_to": {"domain": ["Person"], "range": "Person", "importance": 0.1},
            "located_in": {"domain": ["City"], "range": "City", "inverse_of": "contains"},
            "contains": {"domain": ["City"], "range": "City"},
        },
    }
)


def counts(report: EvalReport) -> dict[str, int | None]:
    return {b.bucket: b.count for b in report.diagnosis if b.side == "recall"}


def planted(report: EvalReport, bucket: str) -> None:
    """Every miss in `bucket`, and nothing else counted anywhere."""
    found = counts(report)
    misses = report.rows[-1].counts.under_extraction
    assert found[bucket] == misses > 0, found
    assert sum(n or 0 for name, n in found.items() if name != bucket) == 0, found


# --------------------------------------------------------------------------- #
# A corpus as files, and a pipeline that is right until it is broken
# --------------------------------------------------------------------------- #


def files(tmp_path: Path, texts: dict[str, str], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Texts, their gold (the rows, read as `odke eval pipeline` reads them) and the ontology."""
    (tmp_path / "texts").mkdir()
    docs = {}
    for name, text in texts.items():
        (tmp_path / "texts" / f"{name}.txt").write_text(text)
        docs[name] = Document(id=f"{name}.txt", text=text)
    gold = [
        GoldFact(
            doc_id=f"{row['doc']}.txt", fact=to_fact(TripleRow(**row), docs[row["doc"]], ONTOLOGY)
        )
        for row in rows
    ]
    (tmp_path / "gold.jsonl").write_text("".join(g.model_dump_json() + "\n" for g in gold))
    (tmp_path / "ontology.json").write_text(ONTOLOGY.model_dump_json())
    return {
        "labels": tmp_path / "gold.jsonl",
        "documents": tmp_path / "texts",
        "ontology": tmp_path / "ontology.json",
    }


def row(doc: str, subject: str, predicate: str, obj: str) -> dict[str, Any]:
    return {"doc": doc, "subject": subject, "predicate": predicate, "object": obj}


# --------------------------------------------------------------------------- #
# 1. A relation hidden from the extractor
# --------------------------------------------------------------------------- #

PEOPLE = [
    ("Ada", "Paris", "Acme", "Bob"),
    ("Cy", "Rome", "Initech", "Di"),
    ("Ed", "Oslo", "Hooli", "Flo"),
]


def _reply(person: str, city: str, company: str, spouse: str) -> str:
    def item(predicate: str, value: str, quote: str) -> dict[str, Any]:
        return {"predicate": predicate, "value": value, "quote": quote, "start": 0,
                "mention": "", "polarity": "asserted"}  # fmt: skip

    facts = [
        item("born_in", city, f"{person} was born in {city}."),
        item("works_for", company, f"{person} works for {company}."),
        item("married_to", spouse, f"{person} is married to {spouse}."),
    ]
    return json.dumps({"entities": [{"type": "Person", "name": person, "facts": facts}]})


def test_a_relation_hidden_from_the_snippet_is_never_offered(tmp_path: Path) -> None:
    texts = {
        f"d{i}": f"{p} was born in {c}. {p} works for {w}. {p} is married to {s}."
        for i, (p, c, w, s) in enumerate(PEOPLE)
    }
    rows = [
        row(f"d{i}", p, predicate, value)
        for i, (p, c, w, s) in enumerate(PEOPLE)
        for predicate, value in (("born_in", c), ("works_for", w), ("married_to", s))
    ]
    paths = files(tmp_path, texts, rows)
    # The model states all three; the snippet's limit of two hides married_to.
    client = ScriptedClient([_reply(*people) for people in PEOPLE])
    extractor = LLMExtractor(client=client, snippet_limit=2, max_workers=1)
    documents = [Document(id=f"{name}.txt", text=text) for name, text in texts.items()]
    kg = Pipeline(ONTOLOGY, extractor, coverage=True).run(documents)
    assert sum(r.reason == "predicate not in the snippet" for r in extractor.rejections) == 3

    # The run's own coverage report is the trace.
    (tmp_path / "manifest.json").write_text(json.dumps({"stats": kg.stats}))
    facts = tmp_path / "facts.jsonl"
    facts.write_text("".join(f.model_dump_json() + "\n" for f in kg.facts))
    report = evaluate_pipeline(predictions=facts, trace=tmp_path / "manifest.json", **paths)
    planted(report, "never_offered")
    first = report.fixes[0]
    assert (first.id, first.knob, first.misses) == (
        "offer-relations",
        "stages.extractor.snippet_limit",
        3,
    )
    # Everything it was offered, it found: the expected gain is the ceiling.
    assert first.recall.expected == first.recall.ceiling == pytest.approx(3 / 9)

    # The same from the extractor itself: per type, as its snippets are.
    offered, _ = read_trace(tmp_path / "manifest.json")
    assert offered is not None and offered.withheld == ("married_to",)


# --------------------------------------------------------------------------- #
# 2. The pair written the other way round
# --------------------------------------------------------------------------- #

PLACES = {
    "d0": [("Lyon", "France"), ("Nice", "France")],
    "d1": [("Turin", "Italy"), ("Bari", "Italy")],
}


def test_a_pair_written_backwards_is_an_inverse_direction(tmp_path: Path) -> None:
    texts = {d: " ".join(f"{a} is in {b}." for a, b in pairs) for d, pairs in PLACES.items()}
    truth = [row(d, a, "located_in", b) for d, pairs in PLACES.items() for a, b in pairs]
    truth.append(row("d0", "Ada", "born_in", "Lyon"))
    texts["d0"] += " Ada was born in Lyon."
    paths = files(tmp_path, texts, truth)

    def broken(documents: list[Document]) -> list[dict[str, Any]]:
        # The bug: every located_in written as its inverse, the ends swapped.
        return [
            row(r["doc"], r["object"], "contains", r["subject"])
            if r["predicate"] == "located_in"
            else r
            for r in truth
        ]

    report = evaluate_pipeline(function=broken, **paths)
    planted(report, "inverse")
    first = report.fixes[0]
    assert first.id == "inverses" and first.recall.exact
    assert first.recall.expected == pytest.approx(4 / 5)
    assert first.detail == ("contains ↔ located_in (declared): 4 gold, 4 partners",)


# --------------------------------------------------------------------------- #
# 3. An output cap
# --------------------------------------------------------------------------- #


def test_an_output_cap_is_output_saturation(tmp_path: Path) -> None:
    # Documents of 1 to 32 sentences, one fact each; the pipeline writes at most four.
    sizes = {f"d{n}": n for n in (1, 2, 4, 8, 16, 32)}
    people = {d: [(f"Pers{d}x{i}", f"Firm{d}x{i}") for i in range(n)] for d, n in sizes.items()}
    texts = {d: " ".join(f"{p} works for {c}." for p, c in pairs) for d, pairs in people.items()}
    truth = [row(d, p, "works_for", c) for d, pairs in people.items() for p, c in pairs]
    paths = files(tmp_path, texts, truth)

    def capped(documents: list[Document]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in people:
            out += [r for r in truth if r["doc"] == d][:4]
        return out

    report = evaluate_pipeline(function=capped, **paths)
    planted(report, "saturation")
    assert counts(report)["saturation"] == 4 + 12 + 28
    note = next(b.note for b in report.diagnosis if b.bucket == "saturation")
    assert note is not None and note.startswith("flagged: ") and "at most 4 per document" in note
    first = report.fixes[0]
    assert first.id == "smaller-chunks"
    # The short documents were found whole, so the long ones are priced at 100%.
    assert first.recall.expected == first.recall.ceiling == pytest.approx(44 / 63)

    # The same documents written whole: nothing flat, nothing flagged.
    whole = evaluate_pipeline(function=lambda documents: truth, **paths)
    assert whole.rows[0].counts.under_extraction == 0 and whole.fixes == ()
    assert next(b.note for b in whole.diagnosis if b.bucket == "saturation").startswith(
        "not flagged"
    )


# --------------------------------------------------------------------------- #
# 4. A gate that refuses a true fact
# --------------------------------------------------------------------------- #

STAGES = '''
from openodke.types import GroundingVerdict


class ContradictAda:
    """A grounder that says the passage contradicts where Ada works, and supports the rest."""

    def ground(self, fact, doc):
        wrong = fact.subject.label == "Ada" and fact.predicate == "works_for"
        verdict = GroundingVerdict.CONTRADICTED if wrong else GroundingVerdict.SUPPORTED
        return fact.model_copy(update={"verdict": verdict})
'''


def test_a_gate_that_refuses_a_true_fact_is_refused_by_the_validator(tmp_path: Path) -> None:
    texts = {"d0": "Ada works for Acme. Ada was born in Paris. Bob works for Hooli."}
    truth = [
        row("d0", "Ada", "works_for", "Acme"),
        row("d0", "Ada", "born_in", "Paris"),
        row("d0", "Bob", "works_for", "Hooli"),
    ]
    paths = files(tmp_path, texts, truth)
    stray = row("d0", "Bob", "born_in", "Paris")
    (tmp_path / "triples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [*truth, stray]))
    (tmp_path / "planted_stages.py").write_text(STAGES)
    config = {
        "ontology": "ontology.json",
        "pythonpath": ["."],
        "inputs": ["texts"],
        "stages": {
            "extractor": {"use": "triples", "path": "triples.jsonl"},
            "grounder": "planted_stages:ContradictAda",
            "gate": "verdict",
        },
    }
    (tmp_path / "odke.json").write_text(json.dumps(config))
    report = evaluate_pipeline(
        predictions=tmp_path / "triples.jsonl",
        validator=True,
        config=tmp_path / "odke.json",
        **paths,
    )
    pipeline, checked = report.rows
    assert pipeline.counts.under_extraction == 0 and checked.counts.under_extraction == 1
    planted(report, "refused")
    refused = next(b for b in report.diagnosis if b.bucket == "refused")
    assert refused.examples[0].why is not None and "contradicted" in refused.examples[0].why
    first = report.fixes[0]
    assert (first.id, first.recall.exact) == ("audit-refusals", True)
    assert first.recall.expected == pytest.approx(1 / 3)
    # The stray, supported and written, is on the precision side; nothing else was refused.
    split = {b.bucket: b.count for b in report.diagnosis if b.side == "precision"}
    assert split == {"refused_not_in_gold": 0, "written_supported": 1, "written_other": 0}


def test_the_corpus_scores_in_process_too() -> None:
    """`score` diagnoses what it is handed, with no files: the library path."""
    documents = [Document(id="d0", text="Lyon is in France.")]
    gold = [
        GoldFact(
            doc_id="d0",
            fact=to_fact(
                TripleRow(**row("d0", "Lyon", "located_in", "France")), documents[0], ONTOLOGY
            ),
        )
    ]
    said = [to_fact(TripleRow(**row("d0", "France", "contains", "Lyon")), documents[0], ONTOLOGY)]
    corpus = Corpus(documents=documents, ontology=ONTOLOGY, dataset=Dataset(name="x"), gold=gold)
    report = score(corpus, said)
    planted(report, "inverse")
