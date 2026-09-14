"""`odke eval ablation`: one config run three ways, scored against labels.

The numbers come from the end-to-end example: 36 gold facts in 8 documents, and
responses written by hand. They check the arithmetic and the wiring, and say
nothing about any model — a demonstration on recorded responses, not a
benchmark. Worked by hand:

- extraction alone: 36 candidates, 32 correct. Two dates copied as written
  ("10 March 2014", "1 June 2019") and the GmbH's invented Leeds head office are
  wrong values; Maya Okafor's invented GmbH employer is spurious; Maya's name in
  the Corvid note was never extracted. P = R = 32/36.
- + grounding: the Leeds head office is contradicted and refused. Its gold
  (Munich) is now missing rather than wrong. P = 32/35, R = 32/36.
- + corroboration: normalisation fixes both dates. P = 34/35, R = 34/36.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from odke import Entity, Evidence, Fact, Span
from odke.cli.main import app
from odke.eval import StageReport, load_jsonl, per_document, run_ablation
from odke.eval.formats import GoldFact
from odke.run import load_config, parse_config

runner = CliRunner()


def _report(example: Path) -> StageReport:
    gold = load_jsonl(example / "gold.jsonl", GoldFact)
    return run_ablation(load_config(example / "odke.yaml"), gold)


def test_the_example_s_three_configurations(example: Path) -> None:
    report = _report(example)
    rows = report.breakdown
    assert list(rows) == ["extraction alone", "+ grounding", "+ corroboration"]
    counts = {
        name: (r["tp"], r["fp"], r["fn"], r["facts"], r["model_calls"]) for name, r in rows.items()
    }
    assert counts == {
        "extraction alone": (32, 4, 4, 36, 3),
        "+ grounding": (32, 3, 4, 35, 39),
        "+ corroboration": (34, 1, 2, 25, 39),
    }
    assert rows["extraction alone"]["precision"] == pytest.approx(32 / 36)
    assert rows["+ grounding"]["precision"] == pytest.approx(32 / 35)
    assert rows["+ corroboration"]["recall"] == pytest.approx(34 / 36)
    # Recorded responses carry no price: unknown, never 0.0.
    assert rows["+ grounding"]["cost_usd"] is None
    assert report.n == 36


def test_the_example_readme_prints_the_table_the_command_computes(example: Path) -> None:
    """A number in a README that no test recomputes is a number that drifts."""
    rows = _report(example).breakdown
    readme = (example / "README.md").read_text(encoding="utf-8")
    assert "not a benchmark" in readme
    for name, row in rows.items():
        line = next(ln for ln in readme.splitlines() if ln.startswith(f"| {name}"))
        cells = [cell.strip() for cell in line.strip("|").split("|")][1:]
        expected = [f"{row[k]:.3f}" for k in ("precision", "recall", "f1")]
        expected += [str(row[k]) for k in ("tp", "fp", "fn", "facts", "model_calls")]
        assert cells == expected, name


def test_the_notes_say_what_moved_and_what_the_gate_traded(example: Path) -> None:
    notes = _report(example).notes
    assert notes[0] == "grounding moved precision from 0.889 to 0.914; recall 0.889 → 0.889"
    assert notes[1].startswith(
        "normalising, resolving and corroborating moved precision from 0.914 to 0.971"
    )
    assert (
        "of the 36 extracted facts, 32 are true: the gate kept 32 of them and 3 of the 4 false ones"
        in notes
    )
    assert (
        "refusing not_found as well would keep 19 of 32 true facts and 1 of 4 false ones "
        "(precision 0.950)" in notes
    )


def test_grounding_that_moves_nothing_is_reported_as_such(example: Path) -> None:
    """The issue's rule: if grounding does not move precision, the report says that."""
    data = yaml.safe_load((example / "odke.yaml").read_text())
    del data["stages"]["grounder"]
    data["models"]["replay"].pop("ground")
    config = parse_config(data, base_dir=example)
    report = run_ablation(config, load_jsonl(example / "gold.jsonl", GoldFact))
    assert (
        report.notes[0]
        == "grounding did not move precision on these labels (0.889); recall 0.889 → 0.889"
    )
    assert "no grounder is configured, so + grounding differs only by the gate" in report.notes
    assert report.breakdown["+ grounding"]["model_calls"] == 3


def test_the_command_prints_the_table_and_writes_nothing(example: Path) -> None:
    args = [
        "eval",
        "ablation",
        "--config",
        str(example / "odke.yaml"),
        "--labels",
        str(example / "gold.jsonl"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.output.startswith("ablation  (n=36)")
    assert "+ corroboration" in result.output
    assert not (example / "out").exists()

    as_json = runner.invoke(app, [*args, "--json"])
    assert StageReport.model_validate_json(as_json.output).metrics[
        "precision_grounding"
    ] == pytest.approx(32 / 35)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--predictions", "p.jsonl"], "takes --config and --labels"),
        (["--run", "odke.stages:PassThroughRouter"], "takes --config and --labels"),
    ],
)
def test_ablation_takes_a_config_and_labels_only(
    example: Path, extra: list[str], message: str
) -> None:
    base = [
        "eval",
        "ablation",
        "--config",
        str(example / "odke.yaml"),
        "--labels",
        str(example / "gold.jsonl"),
    ]
    result = runner.invoke(app, [*base, *extra])
    assert result.exit_code == 2
    assert message in result.output


def test_ablation_without_a_config_says_what_it_needs() -> None:
    result = runner.invoke(app, ["eval", "ablation", "--labels", "gold.jsonl"])
    assert result.exit_code == 2
    assert "ablation needs --config and --labels" in result.output
    described = runner.invoke(app, ["eval", "ablation", "--describe"])
    assert described.exit_code == 0
    assert "extraction alone" in described.output and "GoldFact" in described.output


def test_config_is_only_for_ablation() -> None:
    result = runner.invoke(app, ["eval", "route", "--labels", "x.jsonl", "--config", "odke.yaml"])
    assert result.exit_code == 2
    assert "--config is for ablation" in result.output


def test_a_merged_fact_is_scored_once_in_each_document_it_cites() -> None:
    def cite(doc: str) -> Evidence:
        return Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=4))

    subject = Entity(key="Company:acme", type="Company")
    merged = Fact(
        subject=subject,
        predicate="hq",
        object_value="Leeds",
        evidence=(cite("a"), cite("b"), cite("a")),
    )
    single = Fact(subject=subject, predicate="founded", object_value="2014", evidence=(cite("a"),))
    split = per_document([merged, single])
    assert [[e.doc_id for e in f.evidence] for f in split] == [["a", "a"], ["b"], ["a"]]
    assert {f.id for f in split} == {merged.id, single.id}
