"""The response cache (#156): a rerun of an unchanged batch costs nothing.

No model is called. The clients wrapped here are scripted, recorded or replayed,
and a counting client says how many requests got past the cache.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Chunk, Ontology, prompts
from openodke.cli.main import app
from openodke.eval.cost import CostMeter
from openodke.extract import LLMExtractor
from openodke.ground import LLMGrounder
from openodke.ground.llm import GROUNDING_SCHEMA, build_messages
from openodke.llm import (
    CachedClient,
    Completion,
    DirectoryCache,
    MemoryCache,
    Message,
    ModelRoles,
    ModelSpec,
    ProviderError,
    ProviderNotInstalled,
    ScriptedClient,
)
from openodke.llm.cache import FORMAT, cache_key, prompt_keys, request
from openodke.run import build, load_config
from openodke.run.execute import run_built
from test_run import EXTRACT, NOTES, PEOPLE, _config, _write
from test_run import GROUND as VERDICTS

runner = CliRunner()
SPEC = ModelSpec(model="test/model", max_tokens=64)
MESSAGES = [Message(role="system", content="Judge."), Message(content="Claim: x. Passage: y.")]
REPO = Path(__file__).parent.parent


class _Counting:
    """Answers every request with its own count, and says how many it was asked."""

    def __init__(self, *, fail: int = 0) -> None:
        self.calls = 0
        self._fail = fail
        self._lock = threading.Lock()

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: dict[str, Any] | None = None
    ) -> Completion:
        with self._lock:
            self.calls += 1
            count = self.calls
        if count <= self._fail:
            raise ProviderError("429 too many requests")
        return Completion(
            text=json.dumps({"verdict": "supported", "n": count}),
            parsed={"verdict": "supported", "n": count},
            model=spec.model,
            prompt_tokens=100,
            completion_tokens=5,
            cost_usd=0.0003,
        )


# --------------------------------------------------------------------------- #
# The key
# --------------------------------------------------------------------------- #


def test_the_key_is_the_sha256_of_the_request_as_canonical_json() -> None:
    key = cache_key(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    assert len(key) == 64 and int(key, 16) >= 0
    assert key == cache_key(list(MESSAGES), spec=SPEC.model_copy(), schema=dict(GROUNDING_SCHEMA))
    body = request(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    assert set(body) == {"format", "model", "base_url", "messages", "schema", "sampling", "prompts"}


@pytest.mark.parametrize(
    ("messages", "spec", "schema"),
    [
        (MESSAGES, SPEC.model_copy(update={"model": "test/other"}), None),
        (MESSAGES, SPEC.model_copy(update={"base_url": "http://gpu-box:8000/v1"}), None),
        ([Message(role="system", content="Judge!"), MESSAGES[1]], SPEC, None),
        ([MESSAGES[0], Message(content="Claim: x. Passage: z.")], SPEC, None),
        ([Message(role="user", content="Judge."), MESSAGES[1]], SPEC, None),
        (MESSAGES[1:], SPEC, None),
        (MESSAGES, SPEC, GROUNDING_SCHEMA),
        (MESSAGES, SPEC.model_copy(update={"temperature": 0.0}), None),
        (MESSAGES, SPEC.model_copy(update={"max_tokens": 65}), None),
        (MESSAGES, SPEC.model_copy(update={"extra": {"reasoning_effort": "low"}}), None),
    ],
    ids=[
        "model",
        "base_url",
        "system",
        "user",
        "role",
        "turns",
        "schema",
        "temperature",
        "max_tokens",
        "extra",
    ],
)
def test_any_component_that_changes_the_answer_misses(
    messages: list[Message], spec: ModelSpec, schema: dict[str, Any] | None
) -> None:
    inner = _Counting()
    client = CachedClient(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(messages, spec=spec, schema=schema)
    assert inner.calls == 2
    assert client.stats == {"hits": 0, "misses": 2, "failed": 0}


def test_what_decides_only_whether_a_call_succeeds_is_not_in_the_key() -> None:
    inner = _Counting()
    client = CachedClient(inner)
    client.complete(MESSAGES, spec=SPEC)
    client.complete(MESSAGES, spec=SPEC.model_copy(update={"timeout": 5.0, "api_key_env": "K"}))
    assert inner.calls == 1


def test_the_registered_prompt_and_its_version_are_in_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = build_messages(_fact(), "Ada Lovelace was born in 1815.")
    assert prompt_keys(messages) == ["ground.span@1"]
    before = cache_key(messages, spec=SPEC)

    # The same text under a second version: what a key records is the version too.
    span = prompts.get("ground.span@1")
    again = prompts.Prompt(span.id, 2, span.text, span.source)
    known = prompts.registered
    monkeypatch.setattr(prompts, "registered", lambda: [*known(), again])
    assert prompt_keys(messages) == ["ground.span@1", "ground.span@2"]
    assert cache_key(messages, spec=SPEC) != before


# --------------------------------------------------------------------------- #
# Hits, misses and errors
# --------------------------------------------------------------------------- #


def test_a_second_identical_request_is_answered_from_the_store_for_nothing() -> None:
    inner = _Counting()
    client = CachedClient(inner)
    first = client.complete(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)
    second = client.complete(MESSAGES, spec=SPEC, schema=GROUNDING_SCHEMA)

    assert inner.calls == 1
    assert (second.text, second.parsed, second.model) == (first.text, first.parsed, first.model)
    assert not first.cached and first.cost_usd == pytest.approx(0.0003)
    # Nothing was sent, so nothing was spent: a measured zero, not an unknown.
    assert second.cached
    assert (second.prompt_tokens, second.completion_tokens, second.cost_usd) == (0, 0, 0.0)
    assert client.stats == {"hits": 1, "misses": 1, "failed": 0}


def test_an_error_is_never_stored() -> None:
    inner = _Counting(fail=1)
    store = MemoryCache()
    client = CachedClient(inner, store)
    with pytest.raises(ProviderError):
        client.complete(MESSAGES, spec=SPEC)
    assert len(store) == 0

    answered = client.complete(MESSAGES, spec=SPEC)
    assert inner.calls == 2 and not answered.cached and len(store) == 1
    assert client.stats == {"hits": 0, "misses": 1, "failed": 1}


def test_a_malformed_reply_is_stored_and_its_repair_has_a_key_of_its_own(
    people: Ontology,
) -> None:
    chunk = Chunk(doc_id="d1", index=0, text="Ada Lovelace was born in 1815.", start=0, end=30)
    good = {
        "entities": [
            {
                "type": "Person",
                "name": "Ada Lovelace",
                "facts": [{"predicate": "birth_date", "value": "1815", "quote": "born in 1815"}],
            }
        ]
    }
    store = MemoryCache()
    first = LLMExtractor(
        client=CachedClient(ScriptedClient(["Sure! Here it is.", good]), store),
        spec=SPEC,
        types=["Person"],
    )
    facts = first.extract(chunk, people)
    assert len(facts) == 1 and len(store) == 2
    assert [c.repair for c in first.calls] == [False, True]

    # The rerun replays the reply, the rejection of it, and the repair: no call at all.
    again = LLMExtractor(
        client=CachedClient(ScriptedClient([]), store), spec=SPEC, types=["Person"]
    )
    replayed = again.extract(chunk, people)
    assert [f.signature for f in replayed] == [f.signature for f in facts]
    assert [(c.repair, c.cached) for c in again.calls] == [(False, True), (True, True)]
    assert [c.prompt for c in again.calls] == ["extract@1", "extract.repair@1"]


# --------------------------------------------------------------------------- #
# The directory store
# --------------------------------------------------------------------------- #


def test_the_directory_store_survives_the_process_and_shards_by_key(tmp_path: Path) -> None:
    inner = _Counting()
    CachedClient(inner, tmp_path / "cache").complete(MESSAGES, spec=SPEC)
    key = cache_key(MESSAGES, spec=SPEC)
    path = tmp_path / "cache" / key[:2] / f"{key}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    assert (entry["format"], entry["key"], entry["model"]) == (FORMAT, key, "test/model")
    # The answer's own usage is kept; only a hit reports zero.
    assert entry["completion"]["prompt_tokens"] == 100
    assert "raw" not in entry["completion"]

    later = CachedClient(_Counting(), str(tmp_path / "cache"))
    assert later.complete(MESSAGES, spec=SPEC).cached


def test_an_unreadable_or_foreign_entry_is_a_miss(tmp_path: Path) -> None:
    store = DirectoryCache(tmp_path)
    key = cache_key(MESSAGES, spec=SPEC)
    store.path(key).parent.mkdir(parents=True)
    store.path(key).write_text('{"format": 1, "completion": {"te', encoding="utf-8")
    inner = _Counting()
    assert not CachedClient(inner, store).complete(MESSAGES, spec=SPEC).cached
    store.put(key, {"format": FORMAT + 1, "completion": {"text": "old layout"}})
    assert not CachedClient(inner, store).complete(MESSAGES, spec=SPEC).cached
    assert inner.calls == 2


def test_concurrent_writers_never_leave_a_torn_entry(tmp_path: Path) -> None:
    store = DirectoryCache(tmp_path)
    keys = [f"{i:02x}" * 32 for i in range(8)]
    errors: list[BaseException] = []
    seen: list[dict[str, Any] | None] = []

    def write(worker: int) -> None:
        try:
            for round in range(40):
                key = keys[(worker + round) % len(keys)]
                store.put(key, {"format": FORMAT, "worker": worker, "pad": "x" * 20_000})
                seen.append(store.get(keys[round % len(keys)]))
        except BaseException as exc:  # pragma: no cover - the failure this test exists for
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    # Every read was nothing yet or a whole entry, never half of one.
    assert all(entry is None or len(entry["pad"]) == 20_000 for entry in seen)
    files = sorted(tmp_path.rglob("*"))
    assert [f for f in files if f.is_file() and not f.name.endswith(".json")] == []
    assert len(store) == len(keys)
    assert all(json.loads(store.path(k).read_text())["pad"] for k in keys)


def test_concurrent_grounding_through_one_cache_matches_a_serial_run(tmp_path: Path) -> None:
    docs = [_doc(i) for i in range(12)]
    batches = [([_fact(i)], doc) for i, doc in enumerate(docs)]
    roles = ModelRoles.single("test/model")

    inner = _Counting()
    first = LLMGrounder(roles, client=CachedClient(inner, tmp_path), max_workers=6)
    grounded = first.ground_documents(batches)
    assert inner.calls == 12

    rerun = _Counting()
    second = LLMGrounder(roles, client=CachedClient(rerun, tmp_path), max_workers=6)
    assert second.ground_documents(batches) == grounded
    assert rerun.calls == 0
    assert (second.stats["calls"], second.stats["cached"], second.stats["prompt_tokens"]) == (
        12,
        12,
        0,
    )
    assert "cached" not in first.stats


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


def test_the_meter_records_a_hit_as_a_cached_call_that_cost_nothing() -> None:
    meter = CostMeter()
    client = meter.client(CachedClient(_Counting()), stage="ground")
    client.complete(MESSAGES, spec=SPEC)
    client.complete(MESSAGES, spec=SPEC)

    first, second = meter.records
    assert (first.cached, first.cost_usd) == (False, pytest.approx(0.0003))
    assert (second.cached, second.cost_usd, second.prompt_tokens) == (True, 0.0, 0)
    report = meter.report(documents=1)
    assert report.total.cached_calls == 1 and report.total.calls == 2
    assert report.total.cost_usd == pytest.approx(0.0003)
    stage = report.as_stage_report()
    assert stage.metrics["cached_calls"] == 1
    assert stage.breakdown["ground"]["cached_calls"] == 1


# --------------------------------------------------------------------------- #
# odke run, odke ground, odke validate
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


def test_a_second_identical_run_makes_no_model_call(project: Path) -> None:
    config = load_config(_write(project, _config(models__cache="cache")))
    first = build(config)
    run_built(first, dry_run=True)
    asked = {role: len(client.calls) for role, client in first.context._replays.items()}
    assert asked == {"extract": 1, "ground": 6}

    again = build(config)
    result = run_built(again, dry_run=True)
    assert {role: len(c.calls) for role, c in again.context._replays.items()} == {
        "extract": 0,
        "ground": 0,
    }
    cost = result.stats["cost"]["metrics"]
    assert (cost["calls"], cost["cached_calls"], cost["cost_usd"]) == (7, 7, 0.0)
    assert result.stats["cache"] == {
        "directory": str(project / "cache"),
        "hits": 7,
        "misses": 0,
        "failed": 0,
    }
    assert result.stats["stages"]["grounder"]["cached"] == 6
    assert result.stats["stages"]["extractor"]["cached_calls"] == 1
    rendered = result.render()
    assert "cost          7 model calls (7 from the cache), 0 tokens, $0.0000" in rendered
    assert f"cache         7 answered from {project / 'cache'}, 0 asked and stored" in rendered


def test_a_rerun_from_the_cache_needs_neither_a_recording_nor_a_provider(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `--cache` is read against the working directory, as any path on the command line is.
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["run", str(_write(project, _config())), "--cache", "c"])
    assert result.exit_code == 0, result.output
    shutil.rmtree(project / "out")

    def missing(spec: ModelSpec) -> Any:
        # As on an install without the [llm] extra: no adapter for the provider.
        raise ProviderNotInstalled(f"no adapter for {spec.model}")

    # `openodke.run.build` the attribute is the function; the module is in sys.modules.
    monkeypatch.setattr(sys.modules["openodke.run.build"], "resolve_client", missing)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # No replay files: anything not in the cache would have to reach a provider.
    config = _config(models={"extract": "anthropic/claude-sonnet-5", "meter": True})
    again = runner.invoke(app, ["run", str(_write(project, config)), "--cache", "c"])
    assert again.exit_code == 0, again.output
    assert "7 model calls (7 from the cache)" in again.output
    assert (project / "c").is_dir()
    assert len((project / "out" / "facts.jsonl").read_text(encoding="utf-8").splitlines()) == 5


def test_a_cache_directory_that_cannot_be_one_is_a_config_error(project: Path) -> None:
    (project / "taken").write_text("a file", encoding="utf-8")
    result = runner.invoke(app, ["run", str(_write(project, _config(models__cache="taken")))])
    assert result.exit_code == 2
    assert "models.cache: cannot use" in result.output


@pytest.fixture
def triples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(REPO / "examples" / "triples", tmp_path / "triples")
    (tmp_path / "models.yaml").write_text(
        "models:\n  ground: anthropic/claude-haiku-4-5-20251001\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


TRIPLES = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "-o", "out"]


def test_odke_ground_answers_a_rerun_from_the_cache(triples: Path) -> None:
    first = runner.invoke(
        app, ["ground", *TRIPLES, "--config", "triples/odke.yaml", "--cache", "c"]
    )
    assert first.exit_code == 0, first.output
    # Without the recording, and with no key: every answer comes from the cache.
    again = runner.invoke(app, ["ground", *TRIPLES, "--config", "models.yaml", "--cache", "c"])
    assert again.exit_code == 0, again.output
    summary = json.loads((triples / "out" / "summary.json").read_text(encoding="utf-8"))
    assert (summary["calls"], summary["cached"], summary["tokens"]) == (4, 4, 0)
    assert "4 calls (4 from the cache)" in again.output


def test_odke_validate_reads_the_cache_from_the_models_block(triples: Path) -> None:
    (triples / "cached.yaml").write_text(
        "models:\n  replay: {ground: triples/recorded/ground.json}\n  cache: c\n",
        encoding="utf-8",
    )
    first = runner.invoke(app, ["validate", *TRIPLES, "--config", "cached.yaml"])
    assert first.exit_code == 0, first.output
    (triples / "rerun.yaml").write_text("models:\n  cache: c\n", encoding="utf-8")
    again = runner.invoke(app, ["validate", *TRIPLES, "--config", "rerun.yaml"])
    assert again.exit_code == 0, again.output
    assert "4 model calls (4 from the cache), 0 tokens, $0.0000" in again.output


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _doc(i: int) -> Any:
    from openodke import Document

    return Document(id=f"d{i}", text=f"Person {i} was born in {1800 + i}.")


def _fact(i: int = 0) -> Any:
    from openodke import Entity, Evidence, Fact, Span

    text = f"Person {i} was born in {1800 + i}."
    return Fact(
        subject=Entity(key=f"p{i}", type="Person", label=f"Person {i}"),
        predicate="birth_date",
        object_value=str(1800 + i),
        evidence=(
            Evidence(doc_id=f"d{i}", span=Span(doc_id=f"d{i}", start=0, end=len(text), quote=text)),
        ),
    )
