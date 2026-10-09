"""Recall relative to a pool (#146): two or more runs, their supported facts pooled, no gold."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Polarity, Span
from openodke.cli.main import app
from openodke.eval.eval_report import (
    EvalReport,
    PooledRecall,
    PoolMember,
    check_report,
    read_report,
    schema,
)
from openodke.eval.pooling import CAVEAT, claim, pool, report_pool, run_name

runner = CliRunner()

D1 = "Acme Inc. was founded in 1999 by Ada Lovelace. Acme Inc. is based in Leeds."
D2 = "Babbage Ltd makes engines."


def _fact(
    subject: str,
    predicate: str,
    value: object,
    doc: str = "d1",
    verdict: str = "supported",
    *,
    label: str | None = None,
) -> Fact:
    return Fact(
        subject=Entity(key=subject, type="Company", label=label or subject),
        predicate=predicate,
        object_value=value,
        evidence=(Evidence(doc_id=doc, span=Span(doc_id=doc, start=0, end=10)),),
        verdict=GroundingVerdict(verdict),
    )


A = [
    _fact("acme", "founded", 1999, label="Acme Inc."),
    _fact("acme", "based_in", "Leeds", label="Acme Inc."),
    _fact("acme", "founder", "Ada Lovelace", verdict="not_found", label="Acme Inc."),
    _fact("babbage", "makes", "engines", doc="d2", label="Babbage Ltd"),
]
B = [
    # A's first fact, keyed and spelled another way.
    _fact("Company:acme", "founded", "1999", label="ACME, Inc."),
    _fact("Company:acme", "founder", "ada lovelace", label="Acme"),
]


def test_one_fact_spelled_two_ways_is_one_claim() -> None:
    assert claim(A[0]) == claim(B[0])
    dated = [
        _fact("acme", "founded", "10 December 1815"),
        _fact("x", "founded", "1815-12-10", label="Acme"),
    ]
    assert claim(dated[0]) == claim(dated[1])
    assert claim(A[0]) != claim(A[1])
    denied = A[0].model_copy(update={"polarity": Polarity.DENIED})
    assert claim(denied) != claim(A[0])


def test_each_run_s_share_of_the_pool() -> None:
    section, notes = pool([("a", A), ("b", B)], resamples=200)
    # d1: founded (both), based_in (a), founder (b: a's is not_found); d2: makes (a).
    assert (section.pool, section.documents) == (4, 2)
    a, b = section.runs
    assert (a.name, a.facts, a.supported, a.unique) == ("a", 4, 3, 2)
    assert (b.name, b.facts, b.supported, b.unique) == ("b", 2, 2, 1)
    assert (a.relative_recall.value, b.relative_recall.value) == (3 / 4, 2 / 4)
    for member in (a, b):
        estimate = member.relative_recall
        assert estimate.low is not None and estimate.high is not None
        assert estimate.low <= estimate.value <= estimate.high  # type: ignore[operator]
    assert section.caveat == CAVEAT and "overstates true recall" in CAVEAT
    assert (section.bootstrap.units, section.bootstrap.resamples) == (2, 200)
    assert notes == []


def test_a_third_run_can_only_lower_every_number() -> None:
    two, _ = pool([("a", A), ("b", B)])
    extra = [_fact("acme", "ceo", "Charles Babbage", label="Acme Inc.")]
    three, _ = pool([("a", A), ("b", B), ("c", extra)])
    for before, after in zip(two.runs, three.runs, strict=False):
        assert after.relative_recall.value is not None
        assert before.relative_recall.value is not None
        assert after.relative_recall.value <= before.relative_recall.value
    assert three.pool == two.pool + 1


def test_unsupported_and_uncited_facts_are_not_pooled() -> None:
    uncited = Fact(
        subject=Entity(key="acme", type="Company"),
        predicate="p",
        object_value=1,
        verdict=GroundingVerdict.SUPPORTED,
    )
    unchecked = _fact("acme", "founded", 1999, verdict="unchecked")
    section, notes = pool([("a", A), ("b", [unchecked, uncited])])
    assert section.runs[1].supported == 0
    assert notes == [
        "b: 1 supported fact(s) cite no document and are not pooled",
        "b: no supported fact; was it grounded?",
    ]
    empty, notes = pool([("a", [unchecked]), ("b", [])])
    assert empty.pool == 0 and empty.runs[0].relative_recall.value is None
    assert notes[-1].startswith("the pool is empty")


def test_a_pool_needs_two_named_runs() -> None:
    with pytest.raises(ValueError, match="two runs or more"):
        pool([("a", A)])
    with pytest.raises(ValueError, match="its own name"):
        pool([("a", A), ("a", B)])


def test_the_coverage_report_sits_beside_each_run() -> None:
    docs = [Document(id="d1", text=D1), Document(id="d2", text=D2)]
    report = report_pool([("a", A), ("b", B)], documents=docs)
    section = report.pooled_recall
    assert section is not None
    assert all(run.coverage is not None for run in section.runs)
    text = report.render()
    assert text.startswith("pool  (n=4)")
    assert "recall relative to a pool  (2 runs, 4 supported facts pooled over 2 documents)" in text
    assert CAVEAT in text
    assert "coverage, which needs no pool:" in text
    assert "95% ranges: 2 documents resampled 2000 times" in text


def test_the_section_fits_the_schema_and_reads_back(tmp_path: Path) -> None:
    defs = schema()["$defs"]
    for name, model in (("pooled_recall", PooledRecall), ("pool_member", PoolMember)):
        assert set(defs[name]["properties"]) == set(model.model_fields), name
        assert set(defs[name]["required"]) == set(model.model_fields), name
    report = report_pool([("a", A), ("b", B)])
    data = report.model_dump(mode="json")
    assert check_report(data) == []
    assert read_report(report.write(tmp_path / "r.json")) == report
    data["pooled_recall"]["runs"] = data["pooled_recall"]["runs"][:1]
    assert check_report(data) == ["$.pooled_recall.runs: fewer than 2 items"]
    assert isinstance(report, EvalReport)


def test_a_run_is_named_by_its_directory_or_its_file(tmp_path: Path) -> None:
    (tmp_path / "lgt").mkdir()
    assert run_name(tmp_path / "lgt") == "lgt"
    assert run_name(tmp_path / "lgt" / "facts.jsonl") == "lgt"
    assert run_name(tmp_path / "neo4j.jsonl") == "neo4j"


# --------------------------------------------------------------------------- #
# odke eval pool
# --------------------------------------------------------------------------- #


def _write(path: Path, facts: list[Fact]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f.model_dump_json() + "\n" for f in facts))
    return path


def test_odke_eval_pool_from_the_shell(tmp_path: Path) -> None:
    a = _write(tmp_path / "a" / "out" / "facts.jsonl", A)
    b = _write(tmp_path / "b" / "out" / "facts.jsonl", B)
    docs = tmp_path / "docs.jsonl"
    docs.write_text(
        Document(id="d1", text=D1).model_dump_json()
        + "\n"
        + Document(id="d2", text=D2).model_dump_json()
        + "\n"
    )
    report = tmp_path / "report.json"
    args = ["eval", "pool", str(a), str(b.parent), "--documents", str(docs)]
    result = runner.invoke(app, [*args, "--report", str(report), "--resamples", "300"])
    assert result.exit_code == 0, result.output
    assert "overstates true recall" in result.output
    written = read_report(report).pooled_recall
    assert written is not None
    # Both runs sit in directories called out: the second is out~2.
    assert [r.name for r in written.runs] == ["out", "out~2"]
    assert written.bootstrap.resamples == 300
    shown = runner.invoke(app, [*args, "--json"])
    assert json.loads(shown.output)["pooled_recall"]["pool"] == 4


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "pool", "a.jsonl"], "pool takes two or more runs' facts"),
        (["eval", "pool", "a.jsonl", "b.jsonl", "--labels", "x"], "nothing else"),
        (["eval", "extract", "a.jsonl", "--labels", "x"], "only compare and pool take runs"),
        (
            ["eval", "extract", "--labels", "x", "--resamples", "300"],
            "--resamples: for compare and pool only",
        ),
    ],
)
def test_pool_mistakes_exit_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())


def test_describe_says_it_overstates() -> None:
    result = runner.invoke(app, ["eval", "pool", "--describe"])
    assert result.exit_code == 0 and "overstates true recall" in result.output
