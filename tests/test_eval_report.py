"""The versioned eval report (#139): its schema, its rows, and the two forms it prints.

Every number checked here is one another evaluator already computes; the report
adds ranges and a fixed shape, never a second arithmetic. The fixtures are
examples of the formats, not a benchmark.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from openodke import Entity, Evidence, Fact, Ontology
from openodke.eval import evaluate_extraction, load_jsonl
from openodke.eval.eval_report import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    EvalReport,
    Row,
    check_report,
    conformance,
    extraction_rows,
    extraction_units,
    from_stage,
    read_report,
    schema,
)
from openodke.eval.formats import GoldFact

FIXTURES = Path(__file__).parent / "fixtures" / "eval"


def extract_report() -> EvalReport:
    """The extract fixture scored, as `odke eval extract` reports it."""
    gold = load_jsonl(FIXTURES / "extract.labels.jsonl", GoldFact)
    facts = load_jsonl(FIXTURES / "extract.predictions.jsonl", Fact)
    rows, how = extraction_rows([("extract", facts, None)], gold)
    return from_stage(evaluate_extraction(gold, facts), rows=rows, bootstrap=how)


def rendered_rows(text: str, names: list[str]) -> dict[str, list[str]]:
    """Each row's cells as printed: a number with its range is one cell."""
    found = {}
    for name in names:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"  {name} "))
        cells = re.findall(r"\d+\.\d{3} \[\d+\.\d{3}, \d+\.\d{3}\]|—|\S+", line[len(name) + 3 :])
        found[name] = cells
    return found


def expected_cells(row: Row) -> list[str]:
    """A row's cells as the JSON holds them, formatted the way the text promises."""

    def estimate(e: Any) -> str:
        return f"{e.value:.3f} [{e.low:.3f}, {e.high:.3f}]"

    p, c = row.performance, row.counts
    cells = [estimate(p.precision), estimate(p.recall), estimate(p.f1)]
    cells += [str(c.hits), str(c.over_extraction), str(c.under_extraction)]
    if row.conformance is not None:
        cells.append(f"{row.conformance.rate:.3f}")
    if row.hallucination is not None:
        cells.append(str(row.hallucination.hallucinated))
    if row.cost is not None and row.latency is not None:
        usd = "—" if row.cost.usd is None else f"{row.cost.usd:.4f}"
        cells += [str(row.cost.calls), usd, f"{row.latency.seconds:.1f}"]
    return cells


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def test_the_extract_row_is_evaluate_extraction_with_ranges() -> None:
    report = extract_report()
    (stage,) = report.stages
    (row,) = report.rows
    for name in ("precision", "recall", "f1"):
        assert getattr(row.performance, name).value == stage.metrics[name]
    c = row.counts
    assert (c.hits, c.over_extraction, c.under_extraction) == (
        stage.metrics["tp"],
        stage.metrics["fp"],
        stage.metrics["fn"],
    )
    assert (c.predicted, c.gold, c.documents) == (5, 5, 2)
    assert report.bootstrap is not None and report.bootstrap.units == 2
    # Seeded: the same labels give byte-identical JSON.
    assert report.as_json() == extract_report().as_json()


def test_a_wrong_value_is_one_over_and_one_under_in_its_document() -> None:
    ada = Entity(key="ada", type="Person")
    gold = [
        GoldFact(doc_id="d1", fact=Fact(subject=ada, predicate="born", object_value=1815)),
        GoldFact(doc_id="d2", fact=Fact(subject=ada, predicate="died", object_value=1852)),
    ]
    wrong = Fact(
        subject=ada, predicate="born", object_value=1816, evidence=(Evidence(doc_id="d1"),)
    )
    stray = Fact(subject=ada, predicate="likes", object_value="tea")  # cites nothing
    elsewhere = Fact(
        subject=ada, predicate="born", object_value=1, evidence=(Evidence(doc_id="x"),)
    )
    units, unscored = extraction_units(gold, [wrong, stray, elsewhere])
    # d1: a wrong value; d2: missing; the uncited stray is a unit of its own.
    assert units == [(0, 1, 1), (0, 0, 1), (0, 1, 0)]
    assert unscored == 1  # cites only a document nobody labelled
    rows, how = extraction_rows([("x", [wrong, stray], None)], gold)
    assert how.units == 3 and rows[0].counts.over_extraction == 2


def test_a_range_never_excludes_its_number() -> None:
    """Three documents, one of them all wrong: the draws decide the range, the number stays in."""
    ada = Entity(key="ada", type="Person")
    gold = [
        GoldFact(doc_id=d, fact=Fact(subject=ada, predicate="p", object_value=d))
        for d in ("a", "b", "c")
    ]
    said = [
        Fact(subject=ada, predicate="p", object_value=v, evidence=(Evidence(doc_id=d),))
        for d, v in (("a", "a"), ("b", "b"), ("c", "wrong"))
    ]
    (row,), _ = extraction_rows([("x", said, None)], gold, resamples=200, seed=3)
    for e in (row.performance.precision, row.performance.recall, row.performance.f1):
        assert e.low is not None and e.high is not None and e.value is not None
        assert e.low <= e.value <= e.high


def test_one_document_is_counted_as_one() -> None:
    ada = Entity(key="ada", type="Person")
    gold = [GoldFact(doc_id="d", fact=Fact(subject=ada, predicate="p", object_value=1))]
    said = [Fact(subject=ada, predicate="p", object_value=1, evidence=(Evidence(doc_id="d"),))]
    rows, how = extraction_rows([("x", said, None)], gold)
    text = from_stage(evaluate_extraction(gold, said), rows=rows, bootstrap=how).render()
    assert "95% ranges: 1 document resampled 1000 times" in text


def test_conformance_checks_the_relation_and_both_ends(people: Ontology) -> None:
    ada = Entity(key="ada", type="Person")
    acme = Entity(key="acme", type="Company")
    facts = [
        Fact(subject=ada, predicate="employer", object_entity=acme),  # fits
        Fact(subject=ada, predicate="employer", object_value="Acme"),  # a value for an edge
        Fact(subject=acme, predicate="birth_date", object_value="1815"),  # outside the domain
        Fact(subject=ada, predicate="likes", object_value="tea"),  # not a predicate
    ]
    found = conformance(facts, people)
    assert found is not None
    assert (found.conformant, found.facts, found.rate) == (1, 4, 0.25)
    assert found.checks == ("predicate", "domain", "range")
    assert conformance(facts, None) is None and conformance(facts, Ontology()) is None
    assert extraction_rows([("x", facts, None)], [], ontology=people)[0][0].conformance == found


# --------------------------------------------------------------------------- #
# The schema
# --------------------------------------------------------------------------- #


def test_the_schema_ships_inside_the_package() -> None:
    assert SCHEMA_PATH.parent.name == "eval" and SCHEMA_PATH.is_file()
    assert schema()["properties"]["schema_version"]["pattern"] == r"^1\.[0-9]+$"
    assert SCHEMA_VERSION == "1.0"


def test_the_check_names_what_is_wrong() -> None:
    data = extract_report().model_dump(mode="json")
    assert check_report(data) == []
    broken = json.loads(json.dumps(data))
    del broken["fixes"]
    broken["schema_version"] = "2.0"
    broken["rows"][0]["counts"]["hits"] = -1
    broken["rows"][0]["performance"]["average"] = "weighted"
    broken["stages"][0]["metrics"]["f1"] = "high"
    broken["extra"] = 1
    problems = check_report(broken)
    assert "$: missing 'fixes'" in problems
    assert "$: unexpected 'extra'" in problems
    assert any(p.startswith("$.schema_version: does not match") for p in problems)
    assert "$.rows[0].counts.hits: below the minimum 0" in problems
    assert any(p.startswith("$.rows[0].performance.average: must be one of") for p in problems)
    assert any(p.startswith("$.stages[0].metrics.f1: expected") for p in problems)


def test_the_reserved_sections_are_empty_and_typed() -> None:
    data = extract_report().model_dump(mode="json")
    assert (data["diagnosis"], data["fixes"], data["comparison"], data["calibration"]) == (
        [],
        [],
        None,
        [],
    )
    data["diagnosis"] = ["not an object"]
    assert check_report(data) == ["$.diagnosis[0]: expected object, got str"]


def test_the_schema_agrees_with_a_full_json_schema_validator() -> None:
    """Not a dependency: where `jsonschema` happens to be installed, it must agree."""
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator
    validator.check_schema(schema())
    data = extract_report().model_dump(mode="json")
    validator(schema()).validate(data)
    data["rows"][0]["counts"]["gold"] = None
    assert not validator(schema()).is_valid(data)
    assert check_report(data)


def test_a_written_report_reads_back_and_another_major_version_is_refused(
    tmp_path: Path,
) -> None:
    report = extract_report()
    path = report.write(tmp_path / "nested" / "report.json")
    assert read_report(path) == report
    data = json.loads(path.read_text())
    data["schema_version"] = "2.0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema_version '2.0'"):
        read_report(path)


def test_a_report_that_does_not_fit_is_never_written(tmp_path: Path) -> None:
    bad = extract_report().model_copy(update={"schema_version": "one"})
    with pytest.raises(ValueError, match="does not match its schema"):
        bad.write(tmp_path / "report.json")
    assert not (tmp_path / "report.json").exists()


# --------------------------------------------------------------------------- #
# Render and JSON
# --------------------------------------------------------------------------- #


def test_render_prints_the_numbers_the_json_holds() -> None:
    report = extract_report()
    data = EvalReport.model_validate_json(report.as_json())
    text = report.render()
    (row,) = data.rows
    assert rendered_rows(text, ["extract"])["extract"] == expected_cells(row)
    assert "95% ranges: 2 documents resampled 1000 times (percentile bootstrap, seed 0)" in text
    assert f"openodke {data.run.openodke} · eval report 1.0" in text


def test_render_keeps_a_stage_s_own_table_when_it_is_not_the_rows() -> None:
    text = extract_report().render()
    assert text.startswith("extract  (n=5)\n")
    assert text.count("extract  (n=5)") == 1
    # The per-predicate table and the four-way split are still there.
    assert "employer" in text and "wrong_entity" in text
    assert "F1 is zero for: employer, field" in text
