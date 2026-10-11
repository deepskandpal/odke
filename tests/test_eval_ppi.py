"""Precision with no gold (#144): a judge's number, corrected by a labelled random sample.

The simulation is the test that matters. A judged set with a known true
precision and a judge biased in a known direction: over many seeds, the
prediction-powered interval must cover the truth about 95% of the time, the
judge's own number must miss it the way the judge leans, and the corrected
interval must be narrower than the labels' own. The rest checks the
arithmetic by hand, the sample, the sheets and the command.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Span
from openodke.cli.main import app
from openodke.eval.eval_report import (
    CoverageTotals,
    EvalReport,
    JudgedPrecision,
    check_report,
    read_report,
    schema,
)
from openodke.eval.formats import GroundingLabel, load_jsonl
from openodke.eval.ppi import (
    MINIMUM_LABELS,
    judged_precision,
    prediction_powered,
    report_precision,
    sample,
    write_sample,
)
from openodke.eval.sheets import read_sheets

runner = CliRunner()
TRUTH, FACTS, LABELS, REPS = 0.80, 2000, 300, 400


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("judge", "says_yes_if_true", "says_yes_if_false", "leans"),
    [
        # Lets four in ten false facts through: reads about 84.6% against a true 80%.
        ("lenient", 0.95, 0.43, +0.046),
        # Misses one true fact in ten: reads about 72.4%.
        ("strict", 0.90, 0.02, -0.076),
    ],
)
def test_the_corrected_interval_covers_the_truth_and_the_judge_alone_does_not(
    judge: str, says_yes_if_true: float, says_yes_if_false: float, leans: float
) -> None:
    """400 seeded judged sets of 2,000 facts, 300 labelled: coverage near 95%, narrower.

    Measured at these seeds: 94.75% (lenient) and 96.0% (strict). Offline, at
    5,000 sets each: 95.2% for the lenient judge, PPI ±4.1 points against the
    labels' ±4.5; 94.8% for the strict one, ±3.4 against ±4.5.
    """
    rng = random.Random(144 if judge == "lenient" else 145)
    covered = narrower = 0
    judged_means, corrected_widths, label_widths = [], [], []
    for _ in range(REPS):
        truth = [rng.random() < TRUTH for _ in range(FACTS)]
        said = [rng.random() < (says_yes_if_true if y else says_yes_if_false) for y in truth]
        picked = rng.sample(range(FACTS), LABELS)
        alone, corrected, labels = prediction_powered(said, [(said[i], truth[i]) for i in picked])
        assert corrected.low is not None and corrected.high is not None
        assert labels.low is not None and labels.high is not None
        assert alone.value is not None and alone.low is None
        covered += corrected.low <= TRUTH <= corrected.high
        narrower += corrected.high - corrected.low < labels.high - labels.low
        judged_means.append(alone.value)
        corrected_widths.append(corrected.high - corrected.low)
        label_widths.append(labels.high - labels.low)
    tolerance = 3 * math.sqrt(0.95 * 0.05 / REPS)
    assert abs(covered / REPS - 0.95) <= tolerance, f"{covered} of {REPS} covered"
    # The judge alone is biased, and by what its leniency or strictness predicts.
    bias = sum(judged_means) / REPS - TRUTH
    assert abs(bias - leans) < 0.005, bias
    assert all((m - TRUTH) * leans > 0 for m in judged_means)
    # The corrected interval is narrower than the labels' own, on average and almost always.
    assert sum(corrected_widths) / sum(label_widths) < 0.95
    assert narrower / REPS > 0.9


# --------------------------------------------------------------------------- #
# The arithmetic, by hand
# --------------------------------------------------------------------------- #


def test_the_correction_is_the_judge_plus_the_mean_gap_on_the_sample() -> None:
    # 100 facts, 70 judged supported. 20 labelled: the judge let two false
    # facts through and lost one true one.
    said = [True] * 70 + [False] * 30
    labelled = [(True, True)] * 12 + [(True, False)] * 2 + [(False, True)] + [(False, False)] * 5
    alone, corrected, labels = prediction_powered(said, labelled)
    assert alone.value == 0.7
    assert corrected.value == pytest.approx(0.7 + (1 - 2) / 20)
    assert labels.value == pytest.approx(13 / 20)
    # Over the 20: Var(y) = 20·0.65·0.35/19 and Var(y − f) = (3 − 20·0.05²)/19.
    # The variance is Var(y)/100 + Var(y − f)·(1/20 − 1/100).
    z = 1.959963984540054
    var_y, var_gap = 20 * 0.65 * 0.35 / 19, (3 - 20 * 0.05**2) / 19
    half = z * math.sqrt(var_y / 100 + var_gap * (1 / 20 - 1 / 100))
    assert (corrected.low, corrected.high) == pytest.approx((0.65 - half, 0.65 + half))
    alone_half = z * math.sqrt(var_y / 20)
    assert (labels.low, labels.high) == pytest.approx((0.65 - alone_half, 0.65 + alone_half))
    assert half < alone_half


def test_with_every_fact_labelled_it_is_the_labels_mean_and_their_interval() -> None:
    said = [True, True, False, True, False, True]
    truth = [True, False, False, True, True, True]
    _, corrected, labels = prediction_powered(said, list(zip(said, truth, strict=True)))
    assert corrected == labels


def test_no_labels_leaves_the_judge_alone_and_one_label_gives_no_interval() -> None:
    alone, corrected, labels = prediction_powered([True, False, True, True], [])
    assert (alone.value, corrected.value, labels.value) == (0.75, None, None)
    _, corrected, labels = prediction_powered([True, False, True, True], [(True, False)])
    assert (corrected.value, corrected.low, labels.value, labels.low) == (0.0, None, 0.0, None)
    assert prediction_powered([], []) == tuple(prediction_powered([], []))
    with pytest.raises(ValueError, match="the labels are a sample of the facts"):
        prediction_powered([True], [(True, True), (True, True)])
    with pytest.raises(ValueError, match="level"):
        prediction_powered([True], [], level=95)


def test_the_corrected_number_and_its_interval_stay_between_0_and_1() -> None:
    # A judge that says supported to nearly everything, and labels that agree: near 1.
    said = [True] * 99 + [False]
    labelled = [(True, True)] * 20 + [(False, True)]
    _, corrected, _ = prediction_powered(said, labelled)
    assert corrected.value is not None and corrected.high is not None
    assert corrected.value <= 1.0 and corrected.high == 1.0


# --------------------------------------------------------------------------- #
# Facts and labels
# --------------------------------------------------------------------------- #

TEXT = "Ada Lovelace was born in London in 1815. She worked with Charles Babbage."
ADA = Entity(key="ada", type="Person", label="Ada Lovelace")


def _fact(i: int, verdict: str, predicate: str = "born") -> Fact:
    span = Span(doc_id="d1", start=0, end=40)
    return Fact(
        id=f"f{i}",
        subject=ADA,
        predicate=predicate,
        object_value=1800 + i,
        evidence=(Evidence(doc_id="d1", span=span),),
        verdict=GroundingVerdict(verdict),
    )


def _label(fact: Fact, verdict: str) -> GroundingLabel:
    return GroundingLabel(text=TEXT, fact=fact, verdict=verdict)  # type: ignore[arg-type]


def test_labels_join_the_judged_facts_on_their_ids() -> None:
    facts = [_fact(i, "supported") for i in range(6)] + [
        _fact(6, "not_found"),
        _fact(7, "unchecked"),
    ]
    labels = [
        _label(facts[0], "supported"),
        _label(facts[1], "not_found"),  # the judge let a false fact through
        _label(facts[6], "supported"),  # and lost a true one
        _label(facts[1], "supported"),  # a second label on f1: left out
        _label(_fact(99, "supported"), "supported"),  # no such judged fact
    ]
    section, notes = judged_precision(facts, labels)
    stage = report_precision(facts, labels).stages[0]
    assert stage.n == 3 and stage.metrics["false_support"] == section.false_support == 1
    assert (section.facts, section.supported, section.labels) == (8, 6, 3)
    assert (section.labelled_supported, section.false_support, section.lost_support) == (2, 1, 1)
    assert section.judge_only.value == 0.75
    assert section.corrected.value == pytest.approx(0.75 + (0 - 1 + 1) / 3)
    assert not section.calibrated and section.minimum == MINIMUM_LABELS
    assert section.coverage is None
    assert notes[0] == (
        "judged: supported 6, contradicted 0, not_found 1, unchecked 1; "
        "only supported counts as correct"
    )
    assert any("1 fact(s) were never judged" in note for note in notes)
    assert "1 label(s) name no judged fact and were left out" in notes
    assert "1 label(s) repeat a fact already labelled and were left out" in notes
    assert notes[-1].startswith("recall is not claimed")


def test_facts_sharing_an_id_are_refused() -> None:
    with pytest.raises(ValueError, match="appear more than once"):
        judged_precision([_fact(1, "supported"), _fact(1, "not_found")])


def test_the_coverage_report_stands_in_for_recall() -> None:
    # The fact cites the first sentence; the second names both people and no fact covers it.
    text = "Ada Lovelace was born in 1815. Ada Lovelace worked with Charles Babbage."
    babbage = Entity(key="babbage", type="Person", label="Charles Babbage")
    cited = (Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=30)),)
    facts = [
        Fact(id="w", subject=ADA, predicate="worked_with", object_entity=babbage, evidence=cited)
    ]
    section, notes = judged_precision(facts, documents=[Document(id="d1", text=text)])
    assert section.coverage == CoverageTotals(
        documents=1, sentences=1, uncovered=1, missed_entities=0, not_offered=None, unused=()
    )
    report = report_precision(facts, documents=[Document(id="d1", text=text)])
    assert (
        "coverage, in place of recall: 1 of 1 sentences naming two known entities uncovered"
        in report.render()
    )
    assert notes[-1].startswith("recall is not claimed")


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def _report(labels: int, *, supported_share: float = 0.8) -> EvalReport:
    rng = random.Random(labels)
    facts = [
        _fact(i, "supported" if rng.random() < supported_share else "not_found") for i in range(300)
    ]
    rows = [
        _label(f, "supported" if rng.random() < 0.9 else "contradicted") for f in facts[:labels]
    ]
    return report_precision(facts, rows)


def test_under_100_labels_the_report_says_uncalibrated_estimate(tmp_path: Path) -> None:
    few, enough = _report(60), _report(120)
    assert few.judged_precision is not None and not few.judged_precision.calibrated
    assert enough.judged_precision is not None and enough.judged_precision.calibrated
    text = few.render()
    assert "uncalibrated estimate: 60 labels, under the 100 a calibrated one needs" in text
    assert "uncalibrated" not in enough.render()
    lines = text.splitlines()
    assert lines[0] == "precision  (n=300)"
    assert lines[2] == "precision without gold  (300 facts judged, 60 labels)"
    assert lines[3].split()[:2] == ["judge", "only"]
    assert "prediction-powered (PPI), 95%" in lines[4]
    assert "the labels alone, 95%" in lines[5]
    # The judge on the sample, as `odke eval ground` scores it.
    assert [s.stage for s in few.stages] == ["the judge on the labels"]
    assert few.stages[0].n == 60
    data = few.model_dump(mode="json")
    assert check_report(data) == []
    assert read_report(few.write(tmp_path / "r.json")) == few


def test_with_no_labels_only_the_judge_s_number_is_printed() -> None:
    report = _report(0)
    section = report.judged_precision
    assert section is not None and section.corrected.value is None
    assert report.stages == ()
    text = report.render()
    assert "300 facts judged, no labels" in text
    assert "label a random sample of the facts to correct the judge" in text


def test_the_schema_names_every_field_of_the_section() -> None:
    """The schema and the models it describes cannot drift: same fields, all required."""
    defs = schema()["$defs"]
    for name, model in (
        ("judged_precision", JudgedPrecision),
        ("coverage_totals", CoverageTotals),
    ):
        assert set(defs[name]["properties"]) == set(model.model_fields), name
        assert set(defs[name]["required"]) == set(model.model_fields), name
    data = _report(10).model_dump(mode="json")
    data["judged_precision"]["method"] = "bootstrap"
    assert check_report(data) == ["$.judged_precision.method: must be 'ppi-closed-form'"]


def test_a_1_0_report_without_the_section_still_reads(tmp_path: Path) -> None:
    data = EvalReport(title="extract", n=0).model_dump(mode="json")
    del data["judged_precision"]
    data["schema_version"] = "1.0"
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    assert read_report(path).judged_precision is None


# --------------------------------------------------------------------------- #
# The sample and the sheets
# --------------------------------------------------------------------------- #


def test_a_sample_is_seeded_and_keeps_the_facts_order() -> None:
    facts = [_fact(i, "supported") for i in range(50)]
    drawn = sample(facts, 10, seed=3)
    assert drawn == sample(facts, 10, seed=3) != sample(facts, 10, seed=4)
    assert len({f.id for f in drawn}) == 10
    assert [facts.index(f) for f in drawn] == sorted(facts.index(f) for f in drawn)
    assert sample(facts, 80) == facts
    with pytest.raises(ValueError, match="at least one"):
        sample(facts, 0)


def test_by_predicate_each_predicate_gets_its_share() -> None:
    # 60 born, 30 died, 10 lived: 15 drawn are 9, 4.5 and 1.5, so 9, 5 and 1
    # (largest remainder, ties to the predicate named first: died before lived).
    facts = (
        [_fact(i, "supported", "born") for i in range(60)]
        + [_fact(100 + i, "supported", "died") for i in range(30)]
        + [_fact(200 + i, "supported", "lived") for i in range(10)]
    )
    drawn = sample(facts, 15, seed=0, by=lambda f: f.predicate)
    counts = {p: sum(f.predicate == p for f in drawn) for p in ("born", "died", "lived")}
    assert counts == {"born": 9, "died": 5, "lived": 1}


def test_the_sample_goes_out_blind_and_comes_back_as_labels(tmp_path: Path) -> None:
    facts = [_fact(i, "supported" if i % 4 else "not_found") for i in range(12)]
    doc = Document(id="d1", text=TEXT)
    drawn, made = write_sample(facts, [doc], tmp_path / "sheets", n=5, seed=1)
    assert len(drawn) == 5 and made.items == 5
    rows = [json.loads(line) for line in (tmp_path / "sheets" / "sample.jsonl").open()]
    assert [r["fact"]["id"] for r in rows] == [f.id for f in drawn]
    # Nothing beside the sheet says what the judge said.
    assert {r["fact"]["verdict"] for r in rows} == {"unchecked"}
    assert {r["doc_id"] for r in rows} == {"d1"} and rows[0]["text"] == TEXT
    sheet = made.sheets[0]
    sheet.write_text(sheet.read_text().replace("- [ ] supported", "- [x] supported"))
    labels = [GroundingLabel.model_validate(row) for row in read_sheets(sheet.parent).labels]
    report = report_precision(facts, labels)
    assert report.judged_precision is not None and report.judged_precision.labels == 5
    with pytest.raises(ValueError, match="already has a sample or sheets"):
        write_sample(facts, [doc], tmp_path / "sheets", n=5)


def test_a_fact_citing_none_of_the_documents_cannot_be_sampled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1 judged fact\\(s\\) cite none of the documents"):
        write_sample([_fact(1, "supported")], [Document(id="other", text="x")], tmp_path)


# --------------------------------------------------------------------------- #
# odke eval precision
# --------------------------------------------------------------------------- #


def _write(path: Path, rows: list[Fact]) -> Path:
    path.write_text("".join(f.model_dump_json() + "\n" for f in rows))
    return path


def test_odke_eval_precision_from_the_shell(tmp_path: Path) -> None:
    facts = _write(tmp_path / "facts.jsonl", [_fact(i, "supported") for i in range(8)])
    docs = tmp_path / "docs.jsonl"
    docs.write_text(Document(id="d1", text=TEXT).model_dump_json() + "\n")
    sheets = tmp_path / "sheets"
    made = runner.invoke(
        app,
        ["eval", "precision", "--facts", str(facts), "--documents", str(docs)]
        + ["--make-sheet", str(sheets), "--n", "4", "--seed", "2"],
    )
    assert made.exit_code == 0, made.output
    assert "drew 4 of 8 judged facts (seed 2)" in _flat(made.output)
    assert "wrote 1 sheet to" in _flat(made.output)
    assert "judge only" in made.output
    sheet = sheets / "sheet-01.md"
    sheet.write_text(sheet.read_text().replace("- [ ] not found", "- [x] not found"))
    labels = tmp_path / "labels.jsonl"
    assert runner.invoke(app, ["label", "read", str(sheets), "-o", str(labels)]).exit_code == 0
    assert len(load_jsonl(labels, GroundingLabel)) == 4
    report = tmp_path / "report.json"
    scored = runner.invoke(
        app,
        ["eval", "precision", "--facts", str(facts), "--labels", str(labels)]
        + ["--report", str(report)],
    )
    assert scored.exit_code == 0, scored.output
    assert "uncalibrated estimate: 4 labels" in scored.output
    written = read_report(report)
    assert written.judged_precision is not None
    assert (written.judged_precision.judge_only.value, written.judged_precision.labels) == (1.0, 4)
    # The judge said supported to all four, and the person said not found to all four.
    assert written.judged_precision.corrected.value == 0.0
    as_json = runner.invoke(
        app, ["eval", "precision", "--facts", str(facts), "--labels", str(labels), "--json"]
    )
    assert json.loads(as_json.output)["judged_precision"]["false_support"] == 4


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "extract", "--labels", "x", "--n", "5"], "--n: these are for precision"),
        (["eval", "extract", "--labels", "x", "--by-predicate"], "these are for precision"),
        (["eval", "precision"], "precision needs --facts"),
        (
            ["eval", "precision", "--facts", "f", "--labels", "l", "--make-sheet", "d"],
            "one step at a time",
        ),
        (["eval", "precision", "--facts", "f", "--predictions", "p"], "nothing else"),
    ],
)
def test_precision_mistakes_exit_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in _flat(result.output)


def test_make_sheet_needs_the_documents(tmp_path: Path) -> None:
    facts = _write(tmp_path / "facts.jsonl", [_fact(1, "supported")])
    result = runner.invoke(
        app, ["eval", "precision", "--facts", str(facts), "--make-sheet", str(tmp_path / "s")]
    )
    assert result.exit_code == 2
    assert "--make-sheet needs --documents" in _flat(result.output)


def test_describe_says_what_precision_reads() -> None:
    result = runner.invoke(app, ["eval", "precision", "--describe"])
    assert result.exit_code == 0
    assert "prediction-powered inference" in _flat(result.output)
