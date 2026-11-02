"""R3. A metric is computed in one place.

The same facts must get the same number whichever command scores them. Two
commands once gave one merged fact two recalls (9 Oct audit, finding 11), and
`odke eval pipeline --bench --validator` normalised values a prepared set
leaves as written, so one config scored 1.000 in `odke bench run`'s terms
and 0.000 in the harness's (release audit, finding 16). The bench scripts
copy the Wilson interval with z = 1.96 instead of importing it (S29).

The check, both ways round:

- one input through every path that scores it gives one number;
- a share's interval is defined once, in `openodke.eval.stats`: a definition
  anywhere else fails.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from _support import BENCH, SRC, LiveModel, answer_every_model_with

from openodke import Entity, Evidence, Fact, Span
from openodke.eval.extraction import evaluate_extraction, per_document
from openodke.eval.formats import GoldFact

# --------------------------------------------------------------------------- #
# One input, one number
# --------------------------------------------------------------------------- #


def test_a_fact_merged_across_documents_scores_once_in_each_whichever_way_it_comes() -> None:
    """`odke eval extract` reads a sink's merged facts; the ablation splits them first."""

    def cite(doc: str) -> Evidence:
        return Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=4))

    acme = Entity(key="Company:acme", type="Company", label="Acme")
    merged = Fact(
        subject=acme, predicate="hq", object_value="Leeds", evidence=(cite("a"), cite("b"))
    )
    claim = Fact(subject=acme, predicate="hq", object_value="Leeds")
    gold = [GoldFact(doc_id=doc, fact=claim) for doc in "ab"]
    as_merged = evaluate_extraction(gold, [merged]).metrics
    as_split = evaluate_extraction(gold, per_document([merged])).metrics
    assert as_merged == as_split and as_merged["recall"] == 1.0, (
        f"One merged fact scored {as_merged['recall']} recall as written and "
        f"{as_split['recall']} split by document. Score per document in one place, "
        "`match_extraction`, so every command that scores facts agrees."
    )


# A sentence whose gold keeps its date as written, as Text2KGBench's does.
SENTENCE = "Pride Story is a film first shown on 10 December 1815."
DATED = {"sub": "Pride Story", "rel": "publication date", "obj": "10 December 1815"}


def _dated_set(tmp_path: Path) -> Path:
    from openodke.eval.datasets import text2kgbench
    from test_datasets import ONTOLOGY, _fake_opener, _jsonl

    gold = [{"id": "ont_1_movie_test_1", "sent": SENTENCE, "triples": [DATED]}]
    # The movie ontology, its publication date typed as a date.
    relations = [
        {**rel, "range": "date"} if rel["label"] == DATED["rel"] else rel
        for rel in ONTOLOGY["relations"]
    ]
    opener = _fake_opener(
        {
            "1_movie_ontology.json": json.dumps({**ONTOLOGY, "relations": relations}).encode(),
            "ont_1_movie_ground_truth.jsonl": _jsonl(gold),
        }
    )
    raw = text2kgbench.fetch(tmp_path / "raw", ontologies=["ont_1_movie"], opener=opener)
    return text2kgbench.prepare(raw, "ont_1_movie", tmp_path / "set")


def test_the_validator_row_scores_a_bench_set_as_the_set_scores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openodke.eval.harness import PIPELINE, VALIDATOR, evaluate_pipeline

    prepared = _dated_set(tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    row = {"doc": "ont_1_movie_test_1", "subject": DATED["sub"], "predicate": DATED["rel"],
           "object": DATED["obj"], "object_type": "date"}  # fmt: skip
    predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
    # No model is reached; one asked supports every fact, so the gate refuses nothing.
    live = LiveModel({"verdict": "supported"})
    answer_every_model_with(monkeypatch, live)

    report = evaluate_pipeline(predictions=predictions, bench=prepared, validator=True)
    precision = {r.name: r.performance.precision.value for r in report.rows}
    assert precision[VALIDATOR] == precision[PIPELINE] == 1.0, (
        f"One bench set, one supported fact: the pipeline row scores {precision[PIPELINE]} and "
        f"the Validator's {precision[VALIDATOR]}, because the Validator rewrote the date the "
        "set's gold keeps as written. A set's stages are defined in one place, its config: "
        "name the stages it leaves out on purpose (`normalizer: passthrough`, and the "
        "resolver it means) so every command that runs it runs the same ones."
    )


# --------------------------------------------------------------------------- #
# One definition
# --------------------------------------------------------------------------- #


def test_a_shares_interval_is_defined_once() -> None:
    home = SRC / "eval" / "stats.py"
    copies = []
    for path in sorted([*SRC.rglob("*.py"), *BENCH.rglob("*.py")]):
        source = path.read_text(encoding="utf-8")
        if path == home or ("wilson" not in source.lower() and "1.96" not in source):
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            named = isinstance(node, ast.FunctionDef) and "wilson" in node.name.lower()
            fixed_z = isinstance(node, ast.Constant) and node.value == 1.96
            if named or fixed_z:
                where = f"{path.relative_to(SRC.parents[1]).as_posix()}:{node.lineno}"
                copies.append(f"{where} ({'def ' + node.name if named else 'z = 1.96'})")
    assert not copies, (
        f"A Wilson interval is computed outside openodke.eval.stats: {', '.join(copies)}. "
        "A copy drifts (z = 1.96 is not the exact z, and moves a published figure): import "
        "`openodke.eval.stats.wilson` instead."
    )
