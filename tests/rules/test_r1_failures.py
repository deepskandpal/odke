"""R1. A failure never becomes a number.

openodke sells numbers people trust, so a run that failed must say so, not
publish a score. The audits found failures read as answers: a failed
competitor run published as a score of zero (9 Oct audit, finding 9), a bench
script scoring a sentence that failed as an empty answer and a bad key as "no
conflict found" (release audit, finding 12), and `--update` retiring a failed
text's facts with exit 0 (finding 13).

The check: make every model call fail, or hand in a run that failed. Every
command that scores exits non-zero or leaves the failed part out and says
so; none writes a number made from the failure.

What it covers: `odke bench run`, `odke eval ablation`, `odke eval pipeline`
with `--validator`, `--adjudicate` and `--lenient`, `odke validate --update`,
and the bench scripts behind the published tables: `conflicts.py`,
`competitors.py`, `layer.py`, `volume.py` and `tables.py`. A new command that
scores belongs in this file.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from _support import RESOLVERS, answer_every_model_with, bindings_of_resolve, copy_example
from typer.testing import CliRunner

from openodke.cli.main import app
from openodke.llm import Completion, Message, ModelSpec, ProviderError

runner = CliRunner()

RELEASE_12 = pytest.mark.xfail(strict=True, reason="release audit finding 12; fixed in fix PR 5")


def _found_here(what: str, pr: int) -> pytest.MarkDecorator:
    """A failure this rule found that no audit listed: the maintainer places it."""
    return pytest.mark.xfail(
        strict=True,
        reason=f"found by R1 in PR 0, not in the release audit: {what}; proposed for fix PR "
        f"{pr}, the maintainer's call",
    )


# A 401 is not weather: nothing retries it, so a test spends no time in backoff.
FAILURE = "401 unauthorized: rule R1 makes every model call fail"


class _Refused:
    """Every model call fails. `calls` says the command did try."""

    calls = 0

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        _Refused.calls += 1
        raise ProviderError(FAILURE)


@pytest.fixture
def every_model_call_fails(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_Refused]]:
    """Recorded, scripted and live clients alike: every answer is a failure."""
    from openodke.llm import testing

    _Refused.calls = 0
    for client in (testing.RecordedClient, testing.ReplayClient, testing.ScriptedClient):
        monkeypatch.setattr(client, "complete", _Refused.complete)
    answer_every_model_with(monkeypatch, _Refused())
    yield _Refused


def test_every_place_a_client_is_resolved_is_made_to_fail() -> None:
    """A module that binds `resolve` and is not patched would reach a real provider here."""
    found = bindings_of_resolve()
    assert found == RESOLVERS, (
        f"{sorted(set(found.items()) ^ set(RESOLVERS.items()))}: list every module that binds "
        "the registry's `resolve`, under the name it binds, in RESOLVERS in "
        "tests/rules/_support.py, so the rules can make every model call fail, or answer it, "
        "without a network."
    )


def _no_score(command: str, result: Any, written: Path | None = None) -> None:
    assert _Refused.calls, f"`{command}` asked no model, so this proves nothing: {result.output}"
    leaked = written is not None and written.exists()
    assert result.exit_code != 0 and not leaked, (
        f"`{command}` exited {result.exit_code} after every model call failed"
        + (f", and wrote {written.name}" if leaked else "")
        + ". A failure never becomes a number: a command that scores exits non-zero when "
        "the calls behind its score failed, and writes no report."
        + f"\n\nIt printed:\n{result.output[-1500:]}"
    )


# --------------------------------------------------------------------------- #
# Commands that score, with every model call failing
# --------------------------------------------------------------------------- #


@_found_here("`odke bench run` publishes zeros when every call failed", 5)
def test_odke_bench_run(every_model_call_fails: Any, tmp_path: Path) -> None:
    from openodke.eval.datasets import text2kgbench
    from test_datasets import _t2k_raw

    prepared = text2kgbench.prepare(_t2k_raw(tmp_path), "ont_1_movie", tmp_path / "set")
    result = runner.invoke(app, ["bench", "run", "text2kgbench", str(prepared)])
    _no_score("odke bench run", result, prepared / "report.json")


def test_odke_eval_ablation(every_model_call_fails: Any, example: Path) -> None:
    args = ["--config", str(example / "odke.yaml"), "--labels", str(example / "gold.jsonl")]
    result = runner.invoke(app, ["eval", "ablation", *args])
    _no_score("odke eval ablation", result)


def test_odke_eval_pipeline_validator(
    every_model_call_fails: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_example("triples", tmp_path)
    monkeypatch.chdir(tmp_path)
    args = ["--predictions", "triples/triples.jsonl", "--documents", "triples/texts"]
    args += ["--labels", "triples/gold.jsonl", "--validator", "--config", "triples/odke.yaml"]
    result = runner.invoke(app, ["eval", "pipeline", *args])
    _no_score("odke eval pipeline --validator", result)


@pytest.mark.parametrize(
    "judge",
    ["--adjudicate", "--lenient"],
)
def test_odke_eval_pipeline_judged(
    judge: str, every_model_call_fails: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_example("triples", tmp_path)
    monkeypatch.chdir(tmp_path)
    args = ["--predictions", "triples/triples.jsonl", "--documents", "triples/texts"]
    args += ["--labels", "triples/gold.jsonl", "--config", "triples/odke.yaml"]
    result = runner.invoke(app, ["eval", "pipeline", *args, judge, "judged.jsonl"])
    _no_score(f"odke eval pipeline {judge}", result)


# --------------------------------------------------------------------------- #
# The bench scripts behind the published tables
# --------------------------------------------------------------------------- #


@RELEASE_12
def test_a_conflict_case_whose_every_call_failed_is_no_outcome(
    every_model_call_fails: Any, tmp_path: Path
) -> None:
    from test_bench_conflicts import _opener, conflicts

    root = conflicts.fetch(tmp_path / "erb", opener=_opener([]))
    models = {"extract": "anthropic/claude-sonnet-5-5", "ground": "anthropic/claude-haiku-4-5"}
    try:
        summary = conflicts.run(root, tmp_path / "out", models=models, budget_usd=1.0)
    except SystemExit as stopped:
        assert stopped.code not in (0, None), "conflicts.py exited 0 after every call failed"
        return
    outcomes = {case["outcome"] for case in summary["cases"]}
    assert _Refused.calls and outcomes <= {"failed", "not run"}, (
        f"With every model call failing, conflicts.py decided {sorted(outcomes)}: a bad key "
        "reads as a result. A case whose calls failed is a `failed` outcome, counted apart "
        "from the cases the contest decided."
    )


def test_a_competitor_run_that_lost_documents_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_bench_scripts import _prepared, competitors

    for key in competitors.USAGE:
        monkeypatch.setitem(competitors.USAGE, key, 0)

    async def every_call_fails(docs: list[tuple[str, str]], *args: Any) -> list[dict[str, Any]]:
        competitors.USAGE["failed_documents"] += len(docs)
        return []

    monkeypatch.setitem(competitors.SYSTEMS, "lgt", every_call_fails)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-rule1-fake")
    with pytest.raises(SystemExit) as stopped:
        competitors.extract("lgt", _prepared(tmp_path))
    assert stopped.value.code not in (0, None), (
        "competitors.py exited 0 after every document failed: an empty answer scores as a "
        "result. Exit non-zero, naming how many documents failed."
    )


def _failed_competitor(tmp_path: Path) -> Path:
    """`test_bench_layer`'s prepared set, its LLMGraphTransformer run having lost a document."""
    from test_bench_layer import _prepared

    folder = _prepared(tmp_path)
    usage = folder / "competitors" / "lgt" / "usage.json"
    saved = json.loads(usage.read_text(encoding="utf-8"))
    usage.write_text(json.dumps({**saved, "failed_documents": 1}), encoding="utf-8")
    return folder


@RELEASE_12
def test_layer_never_scores_a_run_that_lost_documents(tmp_path: Path) -> None:
    from test_bench_layer import layer

    out = tmp_path / "out"
    layer.prepare([_failed_competitor(tmp_path)], out)
    scored = sorted(p.parent.name for p in out.glob("**/triples.jsonl"))
    assert "lgt" not in scored, (
        "layer.py prepared LLMGraphTransformer's set for scoring though its usage.json says a "
        "document failed, so the failure is scored as an empty answer. Skip a set with failed "
        "documents and say so, as tables.py does."
    )


@RELEASE_12
def test_volume_leaves_a_failed_document_out_of_the_cost() -> None:
    from test_bench_volume import volume

    ran = [
        {"source_type": "slack", "run": True, "chars": 1000, "usd": 0.01, "facts": 4},
        {"source_type": "slack", "run": True, "chars": 3000, "usd": 0.03, "facts": 8},
    ]
    failed = {"source_type": "slack", "run": True, "chars": 4000, "usd": 0.0, "facts": 0,
              "failed": {"d9": "extract: ProviderError: 401"}}  # fmt: skip
    mean, shares = {"slack": 2000.0}, {"slack": 1.0}
    with_failure = volume.per_thousand([*ran, failed], mean, shares)["slack"]["usd_per_1000"]
    without = volume.per_thousand(ran, mean, shares)["slack"]["usd_per_1000"]
    assert with_failure == without, (
        f"A document that failed moved the cost per 1,000 from ${without} to ${with_failure}. "
        "Skip failed rows in per_thousand, as stopped ones are."
    )


def test_tables_drop_a_set_with_failed_documents(tmp_path: Path) -> None:
    from test_bench_scripts import _report, _write, tables

    verdicts = "supported 4, not_found 0, contradicted 0, unchecked 0"
    for name, lost in (("ont_1_movie", 0), ("ont_2_music", 20)):
        folder = tmp_path / name
        _write(folder, "report.json", _report(0.5, (4, 4, 4), verdicts))
        precision = 0.0 if lost else 0.8
        _write(
            folder / "competitors" / "lgt", "report.json", _report(precision, (4, 4, 4), verdicts)
        )
        _write(folder / "competitors" / "lgt", "usage.json", {"failed_documents": lost})
    lines = tables.table("t", sorted(tmp_path.glob("ont_*")), ["precision"], ["triples"])
    assert any("1 of 2 sets dropped" in line for line in lines), (
        "tables.py averaged in a competitor set whose documents failed. Drop it, and say so."
    )


# --------------------------------------------------------------------------- #
# Run paths
# --------------------------------------------------------------------------- #

STAGES = '''
from openodke import GroundingVerdict
from openodke.llm import ProviderError


class FailsOn:
    """Grounds every fact as supported, except that every call about `doc` fails.

    A 401, not a timeout: a timeout is retried with backoff, and the wait proves nothing.
    """

    def __init__(self, doc: str = "") -> None:
        self.doc = doc

    def ground(self, fact, doc):
        if self.doc and self.doc in doc.id:
            raise ProviderError("401 unauthorized")
        return fact.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})
'''


def test_an_update_whose_text_failed_exits_non_zero(
    every_model_call_fails: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = copy_example("triples", tmp_path)
    (project / "rule1_stages.py").write_text(STAGES, encoding="utf-8")
    (project / "texts" / "corvid.txt").write_text(
        "Corvid Analytics was founded by Ines Varga.", encoding="utf-8"
    )
    with (project / "triples.jsonl").open("a", encoding="utf-8") as rows:
        row = {"doc": "corvid", "subject": "Corvid Analytics", "subject_type": "Company",
               "predicate": "founder", "object": "Ines Varga", "object_type": "Person"}  # fmt: skip
        rows.write(json.dumps(row) + "\n")
    config = yaml.safe_load((project / "odke.yaml").read_text(encoding="utf-8"))
    config["pythonpath"] = ["."]
    config["models"].pop("replay")
    for name, doc in (("ok", ""), ("fails", "corvid")):
        config["stages"]["grounder"] = {"use": "rule1_stages:FailsOn", "doc": doc}
        (project / f"{name}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(project)
    sys.modules.pop("rule1_stages", None)
    monkeypatch.setattr(sys, "path", list(sys.path))

    first = runner.invoke(app, ["validate", "--config", "ok.yaml", "-o", "store"])
    assert first.exit_code == 0, first.output
    again = runner.invoke(app, ["validate", "--config", "fails.yaml", "-o", "store", "--update"])
    sys.modules.pop("rule1_stages", None)
    assert "corvid" in again.output, again.output
    assert "error: texts/corvid.txt: retracted, not rewritten; run it again" in again.output
    assert again.exit_code != 0, (
        "`odke validate --update` exited 0 though a text failed: its old facts were retracted "
        "and nothing replaced them, so they read as retired. With --update, a failed text "
        "makes the exit non-zero and the report says `<text>: retracted, not rewritten; run it "
        f"again`.\n\nIt printed:\n{again.output}"
    )
