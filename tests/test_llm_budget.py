"""Budgets (#157): a run stops cleanly at its budget, keeping what it has.

No model is called. The clients held to a budget here are scripted, recorded or
counting, and a counting client says how many calls went out: once a budget is
exceeded, none more may.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import (
    Chunk,
    Document,
    Entity,
    Evidence,
    Fact,
    GroundingVerdict,
    Ontology,
    Pipeline,
    Span,
)
from openodke.cli.main import EXIT_BUDGET, app
from openodke.extract import LLMExtractor
from openodke.ground import LLMGrounder, RetryPolicy, is_transient
from openodke.llm import (
    Budget,
    BudgetExceeded,
    CachedClient,
    Completion,
    Ledger,
    Message,
    ModelRoles,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
    RecordedClient,
)
from openodke.llm.budget import estimate_tokens
from openodke.run import execute, load_config
from openodke.sinks.jsonl import JsonlSink
from test_run import EXTRACT, NOTES, PEOPLE, _config, _write
from test_run import GROUND as VERDICTS

runner = CliRunner()
SPEC = ModelSpec(model="test/model", max_tokens=64)
MESSAGES = [Message(role="system", content="Judge."), Message(content="Claim: x. Passage: y.")]
ROLES = ModelRoles.single("test/model")
REPO = Path(__file__).parent.parent


class _Counting:
    """Answers every call, priced or not, and counts the calls that reached it."""

    def __init__(
        self, *, cost: float | None = 0.0003, fail: bool = False, gate: Any = None
    ) -> None:
        self.calls = 0
        self.cost = cost
        self.fail = fail
        self.gate = gate
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        with self._lock:
            self.calls += 1
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise ProviderError("503 overloaded")
        return Completion(
            text='{"verdict": "supported"}',
            parsed={"verdict": "supported"},
            model=spec.model,
            prompt_tokens=100,
            completion_tokens=5,
            cost_usd=self.cost,
        )


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_the_call_that_would_pass_the_limit_is_refused_before_it_is_made() -> None:
    inner = _Counting()
    ledger = Ledger(Budget(calls=2))
    client = ledger.client(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(MESSAGES, spec=SPEC)
    with pytest.raises(BudgetExceeded) as stopped:
        client.complete(MESSAGES, spec=SPEC)
    assert inner.calls == 2
    assert stopped.value.limit == "calls"
    assert "stopped at budget: calls 2 of 2" in str(stopped.value)
    # The stop is the run's: every call after it is refused too, whatever its size.
    with pytest.raises(BudgetExceeded):
        client.complete(MESSAGES[:1], spec=SPEC.model_copy(update={"max_tokens": 1}))
    assert inner.calls == 2 and ledger.stopped is not None


def test_input_tokens_are_estimated_as_characters_over_four_before_the_call() -> None:
    chars = sum(len(m.content) for m in MESSAGES)
    assert estimate_tokens(MESSAGES) == math.ceil(chars / 4) + 4 * len(MESSAGES)
    schema = {"type": "object"}
    assert estimate_tokens(MESSAGES, schema) > estimate_tokens(MESSAGES)

    inner = _Counting()
    room = estimate_tokens(MESSAGES) - 1
    with pytest.raises(BudgetExceeded) as stopped:
        Ledger(Budget(input_tokens=room)).client(inner).complete(MESSAGES, spec=SPEC)
    assert (stopped.value.limit, inner.calls) == ("input_tokens", 0)


def test_output_tokens_are_reserved_at_max_tokens_and_counted_as_returned() -> None:
    inner = _Counting()
    ledger = Ledger(Budget(output_tokens=100))
    client = ledger.client(inner)
    # 64 fits in 100; then 5 + 64; then 10 + 64; then 15 + 64; 20 + 64; 25 + 64; 30 + 64; and
    # 35 + 64 = 99; then 40 + 64 does not.
    for _ in range(8):
        client.complete(MESSAGES, spec=SPEC)
    with pytest.raises(BudgetExceeded) as stopped:
        client.complete(MESSAGES, spec=SPEC)
    assert (stopped.value.limit, inner.calls, ledger.spent.output_tokens) == (
        "output_tokens",
        8,
        40,
    )


def test_usd_is_estimated_from_the_run_s_own_priced_calls() -> None:
    inner = _Counting(cost=0.0003)
    ledger = Ledger(Budget(usd=0.001))
    client = ledger.client(inner)
    # The first call has no rate to estimate with. After it, 0.0003 per 105 tokens,
    # times this call's 15 input and 64 output: 0.000226 a call. 0.0009 + 0.000226 > 0.001.
    for _ in range(3):
        client.complete(MESSAGES, spec=SPEC)
    with pytest.raises(BudgetExceeded) as stopped:
        client.complete(MESSAGES, spec=SPEC)
    assert (stopped.value.limit, inner.calls) == ("usd", 3)
    assert ledger.spent.usd == pytest.approx(0.0009)


def test_with_nothing_to_estimate_usd_is_checked_after_the_call() -> None:
    # Priced, but with no tokens reported, so no rate: the limit is met by the
    # second call, and the third is refused.
    class _Untokened(_Counting):
        def complete(self, messages: Any, *, spec: ModelSpec, schema: Any = None) -> Completion:
            made = super().complete(messages, spec=spec, schema=schema)
            return made.model_copy(update={"prompt_tokens": 0, "completion_tokens": 0})

    inner = _Untokened(cost=0.25)
    client = Ledger(Budget(usd=0.5)).client(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(MESSAGES, spec=SPEC)
    with pytest.raises(BudgetExceeded, match=r"usd \$0\.5000 of \$0\.50"):
        client.complete(MESSAGES, spec=SPEC)
    assert inner.calls == 2


def test_a_call_that_raised_is_not_counted_and_an_unpriced_one_leaves_usd_unknown() -> None:
    ledger = Ledger(Budget(calls=1))
    with pytest.raises(ProviderError):
        ledger.client(_Counting(fail=True)).complete(MESSAGES, spec=SPEC)
    assert ledger.spent.calls == 0
    ledger.client(_Counting(cost=None)).complete(MESSAGES, spec=SPEC)
    spent = ledger.spent
    assert (spent.calls, spent.unpriced_calls, spent.usd) == (1, 1, None)


def test_a_budget_stop_is_never_retried() -> None:
    stop = BudgetExceeded("calls", budget=Budget(calls=0), spent=Ledger().spent)
    assert is_transient(stop) is False
    assert not isinstance(stop, ProviderError)


def test_a_call_waits_for_the_calls_in_flight_rather_than_stopping_on_their_reservation() -> None:
    # Each call holds 64 output tokens of 100 while in flight and spends 5. A second
    # call does not fit beside the first's reservation, but fits once it settles.
    gate = threading.Event()
    inner = _Counting(gate=gate)
    ledger = Ledger(Budget(output_tokens=100))
    client = ledger.client(inner)
    first = threading.Thread(target=client.complete, args=(MESSAGES,), kwargs={"spec": SPEC})
    first.start()
    assert inner.entered.wait(5)
    second = threading.Thread(target=client.complete, args=(MESSAGES,), kwargs={"spec": SPEC})
    second.start()
    gate.set()
    first.join(5)
    second.join(5)
    assert inner.calls == 2 and ledger.stopped is None
    assert ledger.spent.output_tokens == 10


def test_threads_drawing_on_one_ledger_never_make_a_call_past_the_limit() -> None:
    inner = _Counting()
    ledger = Ledger(Budget(calls=50))
    client = ledger.client(inner)
    refused: list[BudgetExceeded] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(10):
            try:
                client.complete(MESSAGES, spec=SPEC)
            except BudgetExceeded as exc:
                with lock:
                    refused.append(exc)

    threads = [threading.Thread(target=work) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert inner.calls == ledger.spent.calls == 50
    assert len(refused) == 110
    # One exception per refused call: none shares another's traceback.
    assert len({id(exc) for exc in refused}) == 110


def test_a_cached_answer_costs_nothing_against_the_budget() -> None:
    inner = _Counting()
    ledger = Ledger(Budget(calls=1))
    client = CachedClient(ledger.client(inner))
    for _ in range(5):
        client.complete(MESSAGES, spec=SPEC)
    assert inner.calls == ledger.spent.calls == 1


# --------------------------------------------------------------------------- #
# The pipeline keeps what it has
# --------------------------------------------------------------------------- #


def _documents(n: int) -> list[Document]:
    return [Document(id=f"d{i}", text=f"Item {i} is numbered {i}.") for i in range(n)]


def _fact(doc: Document) -> Fact:
    span = Span(doc_id=doc.id, start=0, end=len(doc.text), quote=doc.text)
    return Fact(
        subject=Entity(key=f"item:{doc.id}", type="Thing", label=f"Item {doc.id}"),
        predicate="numbered",
        object_value=doc.id,
        evidence=(Evidence(doc_id=doc.id, span=span),),
    )


class _Given:
    """An extractor that hands back one fact per document, no model involved."""

    def extract(self, chunk: Chunk, ontology: Ontology) -> list[Fact]:
        return [_fact(Document(id=chunk.doc_id, text=chunk.text))]


def test_a_stop_mid_grounding_keeps_every_verdict_and_counts_the_rest(tmp_path: Path) -> None:
    docs = _documents(40)
    inner = _Counting()
    grounder = LLMGrounder(ROLES, client=Ledger(Budget(calls=17)).client(inner), max_workers=8)
    kg = Pipeline(Ontology(), _Given(), grounder=grounder, sinks=[JsonlSink(tmp_path)]).run(docs)

    # Thread pool or not, exactly the budget's calls went out.
    assert inner.calls == 17
    verdicts = [f.verdict for f in kg.facts]
    assert verdicts.count(GroundingVerdict.SUPPORTED) == 17
    assert verdicts.count(GroundingVerdict.UNCHECKED) == 23
    stopped = kg.stats["stopped"]
    assert (stopped["reason"], stopped["limit"], stopped["stage"]) == ("budget", "calls", "ground")
    assert (stopped["unchecked"], stopped["unextracted"]) == (23, 0)
    assert stopped["budget"] == {"calls": 17} and stopped["spent"]["calls"] == 17
    assert grounder.stats["unasked"] == 23 and grounder.stats["failed"] == 0

    # The sink wrote what was kept, and it reads back.
    lines = (tmp_path / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([json.loads(line) for line in lines]) == 40
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["stopped"]["unchecked"] == 23


def test_a_stop_mid_extraction_keeps_the_chunks_done_and_grounds_nothing_more(
    people: Ontology,
) -> None:
    docs = [Document(id=f"d{i}", text=f"Person {i} was born in {1800 + i}.") for i in range(6)]
    reply = {
        "entities": [
            {
                "type": "Person",
                "name": "Someone",
                "facts": [{"predicate": "birth_date", "value": "1800", "quote": "was born in"}],
            }
        ]
    }
    inner = RecordedClient(
        [
            {"match": "Claim:", "response": {"verdict": "supported"}},
            {"match": "born", "response": reply},
        ]
    )
    ledger = Ledger(Budget(calls=2))
    extractor = LLMExtractor(
        client=ledger.client(inner), spec=SPEC, types=["Person"], max_workers=1
    )
    grounder = LLMGrounder(ROLES, client=ledger.client(inner))
    kg = Pipeline(people, extractor, grounder=grounder).run(docs)

    stopped = kg.stats["stopped"]
    assert (stopped["stage"], stopped["unextracted"], stopped["unchecked"]) == ("extract", 4, 2)
    assert len(inner.calls) == 2
    assert [f.verdict for f in kg.facts] == [GroundingVerdict.UNCHECKED] * 2


def test_the_extractor_s_pool_stops_at_the_budget_and_keeps_the_chunks_it_finished(
    people: Ontology,
) -> None:
    chunks = [
        Chunk(doc_id=f"d{i}", index=0, text=f"Person {i} was born in 18{i:02}.", start=0, end=27)
        for i in range(20)
    ]
    inner = RecordedClient([{"match": "born", "response": {"entities": []}}])
    extractor = LLMExtractor(
        client=Ledger(Budget(calls=7)).client(inner), spec=SPEC, types=["Person"], max_workers=8
    )
    with pytest.raises(BudgetExceeded) as stopped:
        extractor.extract_many(chunks, people)
    partial = stopped.value.partial
    assert len(inner.calls) == len(extractor.calls) == 7
    assert sum(row is not None for row in partial) == 7 and len(partial) == 20


def test_a_run_inside_its_budget_reports_no_stop() -> None:
    docs = _documents(3)
    grounder = LLMGrounder(ROLES, client=Ledger(Budget(calls=3)).client(_Counting()))
    kg = Pipeline(Ontology(), _Given(), grounder=grounder).run(docs)
    assert "stopped" not in kg.stats
    assert {f.verdict for f in kg.facts} == {GroundingVerdict.SUPPORTED}


def test_a_retry_policy_never_spends_a_stopped_budget() -> None:
    inner = _Counting()
    ledger = Ledger(Budget(calls=1))
    grounder = LLMGrounder(
        ROLES, client=ledger.client(inner), retry=RetryPolicy(attempts=5), sleep=lambda s: None
    )
    docs = _documents(3)
    kg = Pipeline(Ontology(), _Given(), grounder=grounder).run(docs)
    assert inner.calls == 1 and grounder.stats["retries"] == 0
    assert kg.stats["stopped"]["unchecked"] == 2


# --------------------------------------------------------------------------- #
# odke run, odke validate, odke ground
# --------------------------------------------------------------------------- #


@pytest.fixture
def project(tmp_path: Path, people: Ontology) -> Path:
    """`test_run`'s project: a note and a CSV row, an ontology and recorded responses."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.md").write_text(NOTES, encoding="utf-8")
    (corpus / "people.csv").write_text(PEOPLE, encoding="utf-8")
    (tmp_path / "ontology.json").write_text(people.model_dump_json(), encoding="utf-8")
    (tmp_path / "extract.json").write_text(json.dumps(EXTRACT), encoding="utf-8")
    (tmp_path / "ground.json").write_text(json.dumps(VERDICTS), encoding="utf-8")
    return tmp_path


def test_odke_run_stops_at_its_budget_writes_what_it_kept_and_exits_3(project: Path) -> None:
    # One extraction call and six grounding calls would be needed; three are allowed.
    config = _config(models__budget={"calls": 3})
    result = runner.invoke(app, ["run", str(_write(project, config))])
    assert result.exit_code == EXIT_BUDGET == 3, result.output

    lines = (project / "out" / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    facts = [json.loads(line) for line in lines]
    assert [f["verdict"] for f in facts].count("unchecked") == 4
    stats = json.loads((project / "out" / "manifest.json").read_text(encoding="utf-8"))["stats"]
    assert stats["stopped"]["limit"] == "calls" and stats["stopped"]["stage"] == "ground"
    assert stats["spent"]["calls"] == 3 and stats["budget"] == {"calls": 3}
    assert "stopped       at budget, calls 3 of 3, during ground: 4 facts left unchecked" in (
        result.output
    )
    assert "cost          3 model calls" in result.output
    assert "budget        calls 3 of 3" in result.output
    assert "wrote         jsonl →" in result.output


def test_a_stop_before_the_first_call_keeps_what_the_pattern_path_read(project: Path) -> None:
    result = execute(load_config(_write(project, _config(models__budget={"calls": 0}))))
    stopped = result.stats["stopped"]
    # The note's prose needed the model; the CSV row's two facts cost nothing.
    assert (stopped["stage"], stopped["unextracted"], stopped["unchecked"]) == ("extract", 1, 2)
    assert result.stats["graph"]["facts"] == 2 and result.stats["spent"]["calls"] == 0
    assert result.stats["stages"]["extractor"]["paths"]["pattern_facts"] == 2


def test_the_flags_set_the_budget_over_the_config(project: Path) -> None:
    config = _config(models__budget={"calls": 100})
    result = runner.invoke(
        app, ["run", str(_write(project, config)), "--budget-calls", "1", "--dry-run"]
    )
    assert result.exit_code == 3, result.output
    # The one call went to extraction, so not one of its six facts was grounded.
    assert "calls 1 of 1, during ground: 6 facts left unchecked" in result.output


def test_every_run_report_has_a_cost_line_metered_or_not(project: Path) -> None:
    config = _config(models={"extract": "anthropic/claude-sonnet-5", "replay": {
        "extract": "extract.json", "ground": "ground.json"}})  # fmt: skip
    result = runner.invoke(app, ["run", str(_write(project, config)), "--dry-run"])
    assert result.exit_code == 0, result.output
    # The grounding replies carry no price, so USD is unknown rather than a partial sum.
    assert "cost          7 model calls, 1423 tokens, USD unknown" in result.output
    assert "budget" not in result.output and "stopped" not in result.output


def test_without_a_cache_a_missing_adapter_stops_the_run_before_anything_is_spent(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(spec: ModelSpec) -> Any:
        raise ProviderNotInstalled(f"no adapter for {spec.model}")

    monkeypatch.setattr(sys.modules["openodke.run.build"], "resolve_client", missing)
    # Extraction is recorded, grounding is not: the build refuses before extracting.
    config = _config(models__replay={"extract": "extract.json"})
    result = runner.invoke(app, ["run", str(_write(project, config))])
    assert result.exit_code == 1 and "no adapter for anthropic/claude-sonnet-5" in result.output
    assert not (project / "out").exists()


def test_a_budget_in_a_config_is_checked_like_any_other_key(project: Path) -> None:
    bad = _config(models__budget={"usd": -1})
    result = runner.invoke(app, ["run", str(_write(project, bad))])
    assert result.exit_code == 2
    assert "models.budget.usd" in result.output


@pytest.fixture
def triples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import shutil

    shutil.copytree(REPO / "examples" / "triples", tmp_path / "triples")
    monkeypatch.chdir(tmp_path)
    return tmp_path


TRIPLES = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "-o", "out"]


def test_odke_ground_stops_at_its_budget_and_still_writes(triples: Path) -> None:
    result = runner.invoke(
        app, ["ground", *TRIPLES, "--config", "triples/odke.yaml", "--budget-calls", "2"]
    )
    assert result.exit_code == 3, result.output
    summary = json.loads((triples / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped"]["limit"] == "calls" and summary["calls"] == 2
    assert summary["verdicts"]["unchecked"] == 2
    assert len((triples / "out" / "facts.jsonl").read_text().splitlines()) == 5
    assert "stopped       at budget, calls 2 of 2, during ground" in result.output


def test_odke_validate_stops_at_its_budget_and_still_writes(triples: Path) -> None:
    result = runner.invoke(
        app, ["validate", *TRIPLES, "--config", "triples/odke.yaml", "--budget-calls", "2"]
    )
    assert result.exit_code == 3, result.output
    assert "stopped       at budget, calls 2 of 2, during ground" in result.output
    manifest = json.loads((triples / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"]["validation"]["stopped"]["unchecked"] == 2
