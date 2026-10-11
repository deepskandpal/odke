"""Gold adjudication (#145): what the gold lacks and the grounder supports, in 2 of 3 runs.

The grounder here is the real `LLMGrounder` with a scripted client that answers
each claim by the run its call carries (`ModelSpec.repeat`), so the three runs
give different answers on purpose. No model is called.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Span
from openodke.cli.main import app
from openodke.eval.adjudication import adjudicate, ask, for_run, write_audit
from openodke.eval.eval_report import (
    AdjudicatedRow,
    Adjudication,
    EvalReport,
    check_report,
    extraction_rows,
    read_report,
    schema,
)
from openodke.eval.formats import GoldFact
from openodke.eval.harness import evaluate_pipeline
from openodke.ground import LLMGrounder
from openodke.llm import CachedClient, Completion, MemoryCache, Message, ModelRoles, ModelSpec
from openodke.llm.litellm_client import LiteLLMClient
from openodke.llm.openai_compat import OpenAICompatClient

TRIPLES = Path(__file__).parent.parent / "examples" / "triples"
runner = CliRunner()

TEXT = (
    "Ada Lovelace was born in London. She worked with Charles Babbage on the "
    "Analytical Engine. She died in 1852."
)
ADA = Entity(key="ada", type="Person", label="Ada Lovelace")
WHOLE = (Evidence(doc_id="d1", span=Span(doc_id="d1", start=0, end=len(TEXT))),)


def _fact(predicate: str, value: object, *, cited: bool = True) -> Fact:
    return Fact(
        subject=ADA, predicate=predicate, object_value=value, evidence=WHOLE if cited else ()
    )


GOLD = [
    GoldFact(doc_id="d1", fact=_fact("born_in", "London")),
    GoldFact(doc_id="d1", fact=_fact("died", 1852)),
]
PREDICTED = [
    _fact("born_in", "London"),  # a hit
    _fact("worked_with", "Charles Babbage"),  # spurious: supported in runs 0 and 1
    _fact("worked_on", "Analytical Engine"),  # spurious: supported in run 0 only
    _fact("died", 1853),  # a wrong value: never supported
    _fact("likes", "tea", cited=False),  # spurious and citing nothing: never asked
]
ANSWERS = {
    "worked with": ["supported", "supported", "not_found"],
    "worked on": ["supported", "not_found", "not_found"],
    "died": ["contradicted", "contradicted", "contradicted"],
}


class PerRun:
    """Answers a claim by the run its call carries: `answers[claim][spec.repeat]`."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        claim = messages[-1].content.splitlines()[0]
        with self._lock:
            self.calls.append((claim, spec.repeat))
        said = next(runs for name, runs in self.answers.items() if f"— {name} —" in claim)
        verdict = said[spec.repeat]
        return Completion(text=json.dumps({"verdict": verdict}), parsed={"verdict": verdict})


class BlindCache:
    """A response cache whose key leaves the run index out, as #156's did before #145."""

    def __init__(self, inner: PerRun) -> None:
        self.inner = inner
        self.store: dict[tuple[Any, ...], Completion] = {}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        key = (tuple((m.role, m.content) for m in messages), spec.model)
        if key not in self.store:
            self.store[key] = self.inner.complete(messages, spec=spec, schema=schema)
        return self.store[key]


def _grounder(client: Any) -> LLMGrounder:
    return LLMGrounder(ModelRoles.single("test/judge"), client=client)


def test_two_of_three_runs_list_a_prediction_as_possibly_missing_from_gold() -> None:
    client = PerRun(ANSWERS)
    grounder = _grounder(client)
    section, entries = adjudicate(
        [("pipeline", PREDICTED)], GOLD, [Document(id="d1", text=TEXT)], grounder
    )
    # Three predictions cite the document and are asked, each once in each run.
    assert sorted(client.calls) == sorted(
        (claim, run) for claim in {c for c, _ in client.calls} for run in (0, 1, 2)
    )
    assert len(client.calls) == 9 and grounder.stats["calls"] == 9
    by_claim = {e.fact.predicate: e for e in entries}
    assert set(by_claim) == {"worked_with", "worked_on", "died"}
    assert [v.value for v in by_claim["worked_with"].verdicts] == [
        "supported",
        "supported",
        "not_found",
    ]
    assert [e.fact.predicate for e in entries if e.listed] == ["worked_with"]
    assert by_claim["died"].kind == "wrong_value"
    (row,) = section.rows
    assert (row.not_in_gold, row.asked, row.possibly_missing) == (4, 3, 1)
    assert (row.strict.value, row.adjudicated.value) == (1 / 5, 2 / 5)
    assert (section.runs, section.needed, section.questions) == (3, 2, 3)


def test_the_strict_precision_is_the_row_s_own_to_the_last_digit() -> None:
    section, _ = adjudicate(
        [("pipeline", PREDICTED)],
        GOLD,
        [Document(id="d1", text=TEXT)],
        _grounder(PerRun(ANSWERS)),
    )
    (strict,), _ = extraction_rows([("pipeline", PREDICTED, None)], GOLD)
    assert section.rows[0].strict == strict.performance.precision


def test_a_question_two_rows_share_is_asked_once() -> None:
    client = PerRun(ANSWERS)
    section, entries = adjudicate(
        [("pipeline", PREDICTED), ("+ validator", PREDICTED[:3])],
        GOLD,
        [Document(id="d1", text=TEXT)],
        _grounder(client),
    )
    assert len(client.calls) == 9
    checked = section.rows[1]
    assert (checked.name, checked.not_in_gold, checked.possibly_missing) == ("+ validator", 2, 1)
    assert (checked.strict.value, checked.adjudicated.value) == (1 / 3, 2 / 3)
    assert next(e for e in entries if e.fact.predicate == "worked_with").rows == [
        "pipeline",
        "+ validator",
    ]


def test_the_response_cache_keeps_the_three_runs_apart() -> None:
    """Three runs through `CachedClient` are three answers, not one replayed; a rerun is free."""
    docs = [Document(id="d1", text=TEXT)]
    inner = PerRun(ANSWERS)
    cached = CachedClient(inner, MemoryCache())
    _, entries = adjudicate([("p", PREDICTED)], GOLD, docs, _grounder(cached))
    assert [e.fact.predicate for e in entries if e.listed] == ["worked_with"]
    assert by_predicate(entries)["worked_on"] == ["supported", "not_found", "not_found"]
    assert cached.stats == {"hits": 0, "misses": 9, "failed": 0}
    assert len(inner.calls) == 9 and {run for _, run in inner.calls} == {0, 1, 2}
    # The same adjudication again: every answer from the store, the same three per question.
    _, again = adjudicate([("p", PREDICTED)], GOLD, docs, _grounder(cached))
    assert cached.stats["hits"] == 9 and len(inner.calls) == 9
    assert by_predicate(again) == by_predicate(entries)


def test_a_key_without_the_run_index_would_replay_run_0_three_times() -> None:
    """Why the cache keys `repeat`: without it, worked_on's one "supported" is counted thrice."""
    blind = BlindCache(PerRun(ANSWERS))
    _, collapsed = adjudicate(
        [("p", PREDICTED)], GOLD, [Document(id="d1", text=TEXT)], _grounder(blind)
    )
    assert len(blind.store) == 3
    assert sorted(e.fact.predicate for e in collapsed if e.listed) == ["worked_on", "worked_with"]


def by_predicate(entries: Sequence[Any]) -> dict[str, list[str]]:
    return {e.fact.predicate: [v.value for v in e.verdicts] for e in entries}


def test_a_grounder_with_no_model_is_its_own_run_and_agrees_with_itself() -> None:
    class Stamp:
        def ground(self, fact: Fact, doc: Document) -> Fact:
            return fact.model_copy(update={"verdict": GroundingVerdict.SUPPORTED})

    stamp = Stamp()
    assert for_run(stamp, 2) is stamp  # type: ignore[arg-type]
    doc = Document(id="d1", text=TEXT)
    found = ask([(PREDICTED[1], doc), (PREDICTED[2], doc)], stamp, runs=3)  # type: ignore[arg-type]
    assert found == [(GroundingVerdict.SUPPORTED,) * 3] * 2
    grounder = _grounder(PerRun(ANSWERS))
    again = for_run(grounder, 1)
    assert isinstance(again, LLMGrounder) and again is not grounder and again.spec.repeat == 1
    assert grounder.spec.repeat == 0 and for_run(grounder, 0) is grounder
    with pytest.raises(ValueError, match="at least one run"):
        ask([], grounder, runs=0)
    with pytest.raises(ValueError, match="needed is between 1 and runs"):
        adjudicate([], GOLD, [], grounder, needed=4)


def test_the_run_index_is_never_sent_to_a_provider() -> None:
    """`ModelSpec.repeat` numbers a draw for whatever keys requests; a provider never sees it."""
    spec = ModelSpec(model="ollama/llama3.1", repeat=2)
    sent: dict[str, Any] = {}

    def opener(request: Any, timeout: float | None = None) -> Any:
        sent.update(json.loads(request.data))
        body = {"choices": [{"message": {"content": "ok"}}]}
        return contextlib.nullcontext(io.BytesIO(json.dumps(body).encode()))

    OpenAICompatClient(opener=opener).complete([Message(content="hi")], spec=spec)
    assert "repeat" not in sent and sent["model"] == "llama3.1"
    handed: dict[str, Any] = {}

    def completion(**kwargs: Any) -> dict[str, Any]:
        handed.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    LiteLLMClient(completion_fn=completion).complete([Message(content="hi")], spec=spec)
    assert "repeat" not in handed
    with pytest.raises(ValueError):
        ModelSpec(model="x/y", repeat=-1)


def test_the_list_is_written_for_audit_listed_first(tmp_path: Path) -> None:
    _, entries = adjudicate(
        [("pipeline", PREDICTED)], GOLD, [Document(id="d1", text=TEXT)], _grounder(PerRun(ANSWERS))
    )
    path = write_audit(tmp_path / "audit" / "adjudicated.jsonl", entries)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["possibly_missing_from_gold"] for r in rows] == [True, False, False]
    first = rows[0]
    assert first["claim"] == 'Ada Lovelace (Person) — worked with — "Charles Babbage".'
    assert (first["doc_id"], first["kind"], first["rows"]) == ("d1", "spurious", ["pipeline"])
    assert (first["verdicts"], first["supported"]) == (["supported", "supported", "not_found"], 2)
    assert Fact.model_validate(first["fact"]).predicate == "worked_with"


# --------------------------------------------------------------------------- #
# The report and the harness
# --------------------------------------------------------------------------- #


def test_the_schema_names_every_field_of_the_section() -> None:
    defs = schema()["$defs"]
    for name, model in (("adjudication", Adjudication), ("adjudicated_row", AdjudicatedRow)):
        assert set(defs[name]["properties"]) == set(model.model_fields), name
        assert set(defs[name]["required"]) == set(model.model_fields), name


@pytest.fixture
def judged(tmp_path: Path) -> Path:
    """The triples example's config, grounding from a cassette that changes its mind on Berlin."""
    copy = tmp_path / "triples"
    shutil.copytree(TRIPLES, copy, ignore=shutil.ignore_patterns("out"))
    berlin = [
        {
            "match": {"contains": ["Berlin"]},
            "response": {"text": json.dumps({"verdict": verdict})},
        }
        for verdict in ("supported", "supported", "not_found")
    ]
    (copy / "recorded" / "adjudicate.json").write_text(json.dumps({"interactions": berlin}))
    config = yaml.safe_load((copy / "odke.yaml").read_text())
    config["models"]["replay"] = {"ground": "recorded/adjudicate.json"}
    path = copy / "adjudicate.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _labels() -> dict[str, Any]:
    return {"labels": TRIPLES / "gold.jsonl", "documents": TRIPLES / "texts"}


def test_odke_eval_pipeline_adjudicates_what_the_gold_lacks(judged: Path, tmp_path: Path) -> None:
    audit = tmp_path / "adjudicated.jsonl"
    report = evaluate_pipeline(
        predictions=TRIPLES / "triples.jsonl", config=judged, adjudicate=audit, **_labels()
    )
    (row,) = report.rows
    section = report.adjudication
    assert section is not None and section.audit == str(audit)
    (adjudicated,) = section.rows
    # Berlin is spurious and supported in two runs; 2012 is a wrong value whose
    # quote the text does not hold, so the span check refuses it every time.
    assert (adjudicated.not_in_gold, adjudicated.asked, adjudicated.possibly_missing) == (2, 2, 1)
    assert adjudicated.strict == row.performance.precision
    assert (adjudicated.strict.value, adjudicated.adjudicated.value) == (0.6, 0.8)
    listed = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [(r["claim"], r["verdicts"]) for r in listed] == [
        (
            "Halden Robotics (Company) — office in — Berlin (City).",
            ["supported", "supported", "not_found"],
        ),
        (
            "Halden Robotics (Company) — founded — 2012.",
            ["not_found", "not_found", "not_found"],
        ),
    ]
    assert report.run.models == {"ground": "anthropic/claude-haiku-4-5-20251001"}
    assert report.run.prompts == ("ground.span@1",)
    assert report.notes[-1].startswith("adjudication: 2 prediction(s) the gold lacks")
    assert check_report(report.model_dump(mode="json")) == []
    text = report.render()
    assert "gold adjudication  (each prediction the gold lacks grounded 3 times" in text
    assert f"the strict precision never changes; the list, with every verdict: {audit}" in text


def test_the_validator_row_is_adjudicated_on_the_same_answers(tmp_path: Path) -> None:
    report = evaluate_pipeline(
        predictions=TRIPLES / "triples.jsonl",
        validator=True,
        config=TRIPLES / "odke.yaml",
        adjudicate=tmp_path / "a.jsonl",
        **_labels(),
    )
    section = report.adjudication
    assert section is not None
    assert [r.name for r in section.rows] == ["pipeline", "+ validator"]
    # The example's recorded grounder says not_found to Berlin every time: nothing listed.
    assert [r.possibly_missing for r in section.rows] == [0, 0]
    for row, adjudicated in zip(report.rows, section.rows, strict=True):
        assert adjudicated.strict == adjudicated.adjudicated == row.performance.precision


def test_adjudication_needs_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--adjudicate needs --labels"):
        evaluate_pipeline(
            predictions=TRIPLES / "triples.jsonl", bench=tmp_path, adjudicate=tmp_path / "a"
        )


def test_odke_eval_pipeline_adjudicate_from_the_shell(judged: Path, tmp_path: Path) -> None:
    audit, report = tmp_path / "list.jsonl", tmp_path / "report.json"
    args = ["eval", "pipeline", "--predictions", str(TRIPLES / "triples.jsonl")]
    args += ["--labels", str(TRIPLES / "gold.jsonl"), "--documents", str(TRIPLES / "texts")]
    args += ["--config", str(judged), "--adjudicate", str(audit), "--report", str(report)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "gold adjudication" in result.output
    written = read_report(report)
    assert written.adjudication is not None
    assert written.adjudication.rows[0].possibly_missing == 1
    assert len(audit.read_text().splitlines()) == 2
    wrong = runner.invoke(app, ["eval", "extract", "--labels", "x", "--adjudicate", "y"])
    assert wrong.exit_code == 2 and "--adjudicate: these are for pipeline" in wrong.output


def test_a_report_with_the_section_reads_back(judged: Path, tmp_path: Path) -> None:
    report = evaluate_pipeline(
        predictions=TRIPLES / "triples.jsonl",
        config=judged,
        adjudicate=tmp_path / "a.jsonl",
        **_labels(),
    )
    assert read_report(report.write(tmp_path / "r.json")) == report
    assert isinstance(report, EvalReport)
