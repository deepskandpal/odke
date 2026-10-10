"""The run manifest (#160): what a run was asked, what answered it, and what it made.

Every run here answers from recorded responses: the e2e example's for
`odke run`, the triples example's for the Validator, `odke validate` and
`odke ground`. The done-when is the replay: two runs of one manifest are the
same run, and `odke run --from-manifest` is how the second is made.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import openodke
from openodke import Document, Ontology, prompts
from openodke.cli.main import app
from openodke.llm import ModelSpec
from openodke.llm.base import Completion, Message
from openodke.llm.roles import DEFAULT_EXTRACT
from openodke.llm.testing import RecordedClient
from openodke.loaders import DirectoryLoader
from openodke.manifest import (
    REDACTED,
    SINK_KEYS,
    ModelUse,
    Served,
    canonical,
    digest,
    read_manifest,
)
from openodke.run import load_config, parse_config
from openodke.sinks.jsonl import JsonlSink
from openodke.validator import Validator

runner = CliRunner()
REPO = Path(__file__).parent.parent
TRIPLES = REPO / "examples" / "triples"
TIMES = ("started_at", "ended_at", "created_at")


def _read(path: Path) -> dict[str, Any]:
    found: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _timeless(value: Any) -> Any:
    """A manifest without its clocks: the two times, the graph's, and the meter's latencies."""
    if isinstance(value, dict):
        return {k: _timeless(v) for k, v in value.items() if k not in TIMES and "latency" not in k}
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def _facts(directory: Path) -> set[tuple[str, str, float]]:
    """What a JSONL store holds, by claim: the ids are minted per run, the claims are not."""
    lines = (directory / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    out = set()
    for line in lines:
        fact = json.loads(line)
        claim = json.dumps(
            [fact["subject"]["key"], fact["predicate"], fact["object_value"], fact["polarity"]]
        )
        out.add((claim, fact["verdict"], round(fact["confidence"], 9)))
    return out


def _run(*args: str) -> Any:
    return runner.invoke(app, ["run", *args])


# --------------------------------------------------------------------------- #
# What a manifest holds
# --------------------------------------------------------------------------- #


def test_every_run_writes_its_manifest_beside_the_fields_older_readers_know(
    example: Path,
) -> None:
    result = _run(str(example / "odke.yaml"))
    assert result.exit_code == 0, result.output
    assert f"manifest      {example / 'out' / 'manifest.json'}" in result.output

    document = _read(example / "out" / "manifest.json")
    # What the JSONL sink wrote before 1.0 is still there, and means what it meant.
    assert set(SINK_KEYS) <= set(document)
    assert (document["ontology"], document["facts"]) == ("companies", 25)
    assert document["stats"]["graph"]["facts"] == 25

    manifest = read_manifest(example / "out")
    config = load_config(example / "odke.yaml")
    assert (manifest.command, manifest.dry_run, manifest.manifest_version) == ("run", False, 1)
    assert (manifest.config_file, manifest.base_dir) == ("odke.yaml", str(example))
    assert manifest.config == config.canonical()
    assert manifest.config_hash == digest(manifest.config)
    assert manifest.models == {
        "extract": ModelUse(
            model="anthropic/claude-sonnet-5", served=("anthropic/claude-sonnet-5",)
        ),
        "ground": ModelUse(
            model="anthropic/claude-haiku-4-5-20251001",
            served=("anthropic/claude-haiku-4-5-20251001",),
        ),
    }
    assert manifest.prompts == {
        key: prompts.get(key).sha256 for key in ("extract@1", "ground.span@1")
    }
    ontology = Ontology.from_json(example / "ontology.json")
    assert (manifest.ontology_version, manifest.ontology_hash) == (
        ontology.version,
        ontology.fingerprint,
    )
    assert manifest.package["openodke"] == openodke.__version__
    notes = example / "corpus" / "notes" / "halden-robotics.md"
    assert (
        manifest.inputs.documents["corpus/notes/halden-robotics.md"]
        == hashlib.sha256(notes.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    )
    assert len(manifest.inputs.documents) == manifest.counts["documents"] == 8
    assert manifest.inputs.facts is None
    assert (manifest.cache, manifest.budget) == (None, None)
    assert manifest.started_at <= manifest.ended_at
    assert manifest.counts["facts"] == document["facts"] and manifest.counts["refused"] == 1
    assert manifest.spent["calls"] == document["stats"]["spent"]["calls"]
    assert (manifest.stopped, manifest.failed) == (None, {})


def test_a_run_with_no_jsonl_sink_writes_it_beside_its_config_and_where_named(
    example: Path,
) -> None:
    config = yaml.safe_load((example / "odke.yaml").read_text(encoding="utf-8"))
    config["stages"].pop("sink")
    (example / "nosink.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = _run(str(example / "nosink.yaml"))
    assert result.exit_code == 0, result.output
    beside = _read(example / "nosink.manifest.json")
    # The graph's fields too, as a JSONL sink would have written them.
    assert (beside["command"], beside["facts"], beside["config_file"]) == ("run", 25, "nosink.yaml")

    config["stages"]["sink"] = {"use": "jsonl", "directory": "out"}
    config["manifest"] = "runs/last.json"
    (example / "named.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    assert _run(str(example / "named.yaml")).exit_code == 0
    named, into = _read(example / "runs" / "last.json"), _read(example / "out" / "manifest.json")
    assert named["config_hash"] == into["config_hash"] != beside["config_hash"]
    assert named["facts"] == into["facts"] == 25


def test_a_dry_run_has_a_manifest_and_writes_none(example: Path) -> None:
    from openodke.run import execute

    result = execute(load_config(example / "odke.yaml"), dry_run=True)
    assert result.manifest is not None and result.manifest.dry_run
    assert result.manifests == []
    assert not (example / "out").exists() and not (example / "odke.manifest.json").exists()


# --------------------------------------------------------------------------- #
# The done-when: two runs of one manifest
# --------------------------------------------------------------------------- #


def test_two_runs_of_one_config_write_the_same_manifest_but_for_the_times(example: Path) -> None:
    assert _run(str(example / "odke.yaml")).exit_code == 0
    first = _read(example / "out" / "manifest.json")
    assert _run(str(example / "odke.yaml")).exit_code == 0
    second = _read(example / "out" / "manifest.json")
    assert first["started_at"] != second["started_at"]
    assert _timeless(first) == _timeless(second)


def test_a_manifest_runs_again_as_the_run_it_recorded(example: Path) -> None:
    assert _run(str(example / "odke.yaml")).exit_code == 0
    first, facts = _read(example / "out" / "manifest.json"), _facts(example / "out")
    shutil.rmtree(example / "out")

    result = _run("--from-manifest", str(example / "odke.manifest.json"))
    assert result.exit_code == 2 and "no such file" in result.output.lower()
    (example / "out").mkdir()
    (example / "out" / "manifest.json").write_text(json.dumps(first), encoding="utf-8")

    result = _run("--from-manifest", str(example / "out"))
    assert result.exit_code == 0, result.output
    assert f"replaying {example / 'out'}: config {first['config_hash'][:12]}" in result.output
    second = _read(example / "out" / "manifest.json")
    assert _timeless(second) == _timeless(first)
    assert _facts(example / "out") == facts


def test_a_replay_is_refused_when_its_inputs_or_its_ontology_changed(example: Path) -> None:
    assert _run(str(example / "odke.yaml")).exit_code == 0
    recorded = (example / "out" / "manifest.json").read_text(encoding="utf-8")
    note = example / "corpus" / "notes" / "corvid-analytics.md"
    text = note.read_text(encoding="utf-8")

    note.write_text(text + "\nCorvid Analytics moved.\n", encoding="utf-8")
    result = _run("--from-manifest", str(example / "out"))
    assert result.exit_code == 2
    assert "documents changed: corpus/notes/corvid-analytics.md" in result.output
    # Refused before anything was written.
    assert (example / "out" / "manifest.json").read_text(encoding="utf-8") == recorded

    note.write_text(text, encoding="utf-8")
    schema = _read(example / "ontology.json")
    schema["predicates"]["motto"] = {"domain": ["Company"]}
    (example / "ontology.json").write_text(json.dumps(schema), encoding="utf-8")
    result = _run("--from-manifest", str(example / "out"))
    assert result.exit_code == 2 and "the ontology changed" in result.output


def test_a_replay_takes_no_override_and_no_other_commands_manifest(
    example: Path, tmp_path: Path
) -> None:
    assert _run(str(example / "odke.yaml")).exit_code == 0
    out = str(example / "out")
    for args in (
        ["--from-manifest", out, "--model", "openai/gpt-5.5"],
        ["--from-manifest", out, "--budget-calls", "3"],
        [str(example / "odke.yaml"), "--from-manifest", out],
        [],
    ):
        result = _run(*args)
        assert result.exit_code == 2, args

    written = _read(example / "out" / "manifest.json")
    (tmp_path / "validate.json").write_text(
        json.dumps({**written, "command": "validate"}), encoding="utf-8"
    )
    result = _run("--from-manifest", str(tmp_path / "validate.json"))
    assert result.exit_code == 2 and "records odke validate, not odke run" in result.output

    old = {key: written[key] for key in SINK_KEYS}
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    result = _run("--from-manifest", str(tmp_path / "old.json"))
    assert result.exit_code == 2 and "not a run manifest" in result.output


# --------------------------------------------------------------------------- #
# The config hash, the secrets, the models
# --------------------------------------------------------------------------- #


def test_the_config_hash_is_of_the_config_as_resolved(example: Path) -> None:
    config = load_config(example / "odke.yaml")
    data = yaml.safe_load((example / "odke.yaml").read_text(encoding="utf-8"))
    as_json = parse_config(json.loads(json.dumps(data)), base_dir=example)
    assert digest(as_json.canonical()) == digest(config.canonical())
    # A key given its default is the key left out: the same run.
    defaulted = {**data, "coverage": True, "models": {**data["models"], "budget": None}}
    assert digest(parse_config(defaulted, base_dir=example).canonical()) == digest(
        config.canonical()
    )
    # One option changed is another run.
    changed = {**data, "stages": {**data["stages"], "chunker": {"use": "sentence", "max_words": 9}}}
    assert digest(parse_config(changed, base_dir=example).canonical()) != digest(config.canonical())
    # And the canonical form reads back as itself.
    again = parse_config(config.canonical(), base_dir=example)
    assert again.canonical() == config.canonical()


def test_a_role_left_to_its_default_and_one_naming_it_are_one_config(tmp_path: Path) -> None:
    base = {"ontology": "o.json", "inputs": ["c"], "stages": {"extractor": "llm"}}
    left = parse_config(base, base_dir=tmp_path).canonical()
    named = parse_config(
        {**base, "models": {"extract": DEFAULT_EXTRACT}}, base_dir=tmp_path
    ).canonical()
    assert left["models"]["extract"] == named["models"]["extract"]
    assert left["models"]["extract"]["model"] == DEFAULT_EXTRACT


def test_no_secret_reaches_the_manifest_or_its_hash() -> None:
    def config(key: str) -> dict[str, Any]:
        return {
            "models": {
                "extract": {
                    "model": "gateway/large",
                    "base_url": f"https://ops:{key}@gateway.internal/v1",
                    "api_key_env": "GATEWAY_KEY",
                    "max_tokens": 512,
                    "extra": {"api_key": key, "Authorization": f"Bearer {key}", "region": "eu"},
                }
            },
            "store": {"uri": f"bolt://neo4j:{key}@db:7687", "password_env": "NEO4J_PASSWORD"},
            "token": key,
        }

    out = canonical(config("hunter2"))
    assert "hunter2" not in json.dumps(out)
    extract = out["models"]["extract"]
    assert extract["base_url"] == f"https://ops:{REDACTED}@gateway.internal/v1"
    assert extract["extra"] == {"api_key": REDACTED, "Authorization": REDACTED, "region": "eu"}
    # What only names where a secret is read is kept, and so is anything else.
    assert (extract["api_key_env"], extract["max_tokens"]) == ("GATEWAY_KEY", 512)
    assert out["store"] == {
        "uri": f"bolt://neo4j:{REDACTED}@db:7687",
        "password_env": "NEO4J_PASSWORD",
    }
    assert out["token"] == REDACTED
    assert digest(canonical(config("hunter2"))) == digest(canonical(config("s3cret")))


class _Dated:
    """A provider that answers an alias with the dated model it resolved it to."""

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
    ) -> Completion:
        return Completion(text='{"verdict": "supported"}', model="claude-large-20260501")


def test_the_pinned_id_that_answered_is_recorded_beside_the_alias_asked_for() -> None:
    served = Served()
    alias = ModelSpec(model="anthropic/claude-large")
    client = served.client("ground", alias, _Dated())
    for _ in range(2):
        client.complete([Message(content="Claim: …")], spec=alias)
    served.note("extract", ModelSpec(model="gpt-5.5"))
    assert served.models() == {
        "extract": ModelUse(model="openai/gpt-5.5"),
        "ground": ModelUse(
            model="anthropic/claude-large", served=("anthropic/claude-large-20260501",)
        ),
    }


# --------------------------------------------------------------------------- #
# The Validator, odke validate and odke ground
# --------------------------------------------------------------------------- #


def _texts() -> list[Document]:
    return list(DirectoryLoader().load(TRIPLES / "texts"))


def test_the_validator_writes_its_manifest_into_each_jsonl_sink_and_where_named(
    tmp_path: Path,
) -> None:
    client = RecordedClient.from_fixture(TRIPLES / "recorded" / "ground.json")
    ontology = Ontology.from_json(TRIPLES / "ontology.json")
    validator = Validator(
        ontology, client=client, sinks=[JsonlSink(tmp_path / "out")], manifest=tmp_path / "m.json"
    )
    kg, report = validator.validate(TRIPLES / "triples.jsonl", _texts(), extractor="hand-written")

    manifest = report.manifest
    assert manifest is not None and manifest.command == "validate"
    assert read_manifest(tmp_path / "out") == manifest == read_manifest(tmp_path / "m.json")
    assert _read(tmp_path / "out" / "manifest.json")["facts"] == 5
    assert _read(tmp_path / "m.json")["facts"] == 5
    described = manifest.config["validator"]
    assert described["grounder"] == "LLMGrounder"
    assert described["gate"] is None and described["sinks"] == ["openodke.sinks.jsonl.JsonlSink"]
    assert (manifest.config["extractor"], manifest.config["update"]) == ("hand-written", False)
    assert manifest.models["ground"].model == "anthropic/claude-haiku-4-5-20251001"
    assert manifest.prompts == {"ground.span@1": prompts.get("ground.span@1").sha256}
    assert manifest.ontology_hash == ontology.fingerprint
    assert manifest.inputs.facts is not None and manifest.inputs.facts.rows == 5
    assert manifest.counts == {
        "documents": 1,
        "facts_in": 5,
        "unmatched": 0,
        "refused": 0,
        "merged": 0,
        "restated": 0,
        "linked": 0,
        "derived": 0,
        "facts_out": 5,
        "edges": 4,
        "properties": 1,
        "entities": 5,
        "failed": 0,
    }
    assert manifest.spent["calls"] == report.calls == 4
    # The report is what it was: its JSON and its equality leave the manifest out.
    assert "manifest" not in kg.stats["validation"]
    assert report == report.model_copy(update={"manifest": None})

    _, dry = Validator(ontology).validate(TRIPLES / "triples.jsonl", _texts(), dry_run=True)
    assert dry.manifest is not None and dry.manifest.dry_run
    assert dry.manifest.ontology_hash == ontology.fingerprint and dry.manifest.models == {}


def test_a_merging_store_keeps_its_own_counts_beside_the_jobs(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in (TRIPLES / "triples.jsonl").read_text().splitlines()]
    client = RecordedClient.from_fixture(TRIPLES / "recorded" / "ground.json")
    validator = Validator(client=client, sinks=[JsonlSink(tmp_path, merge=True)])
    validator.validate(rows[:3], _texts())
    _, report = validator.validate(rows[3:], _texts())
    document = _read(tmp_path / "manifest.json")
    assert report.manifest is not None and report.manifest.counts["facts_in"] == len(rows) - 3
    assert document["counts"]["facts_out"] == report.facts_out < document["facts"]
    assert document["facts"] == len((tmp_path / "facts.jsonl").read_text().splitlines())


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory holding examples/triples, recorded answers and a models file."""
    shutil.copytree(TRIPLES, tmp_path / "triples", ignore=shutil.ignore_patterns("out"))
    recorded = [
        {"match": "— Berlin (City)", "response": {"verdict": "not_found"}},
        {"match": "Claim:", "response": {"verdict": "supported"}},
    ]
    (tmp_path / "recorded.json").write_text(json.dumps(recorded), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(
        "models:\n  replay:\n    ground: recorded.json\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_odke_validate_records_what_a_second_run_needs_to_be_the_same_run(here: Path) -> None:
    """The recipe for a manifest of odke validate: its options, and its models block as a file."""
    args = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts"]
    result = runner.invoke(app, ["validate", *args, "--config", "models.yaml", "-o", "out"])
    assert result.exit_code == 0, result.output
    assert "manifest      out/manifest.json" in result.output
    first = read_manifest(here / "out")
    assert first.command == "validate"
    assert first.config["facts"] == "triples/triples.jsonl"
    assert first.config["models"]["replay"] == {"ground": "recorded.json"}
    assert first.models["ground"].served == ("anthropic/claude-haiku-4-5-20251001",)

    # The recipe: the options as flags, and `models` written to a file of its own.
    (here / "again.json").write_text(json.dumps({"models": first.config["models"]}))
    options = first.config
    again = [
        "--facts", options["facts"], "--texts", options["texts"], "--adapter", options["adapter"],
        "--config", "again.json", "-o", "again",
    ]  # fmt: skip
    assert runner.invoke(app, ["validate", *again]).exit_code == 0
    second = read_manifest(here / "again")
    assert {**second.config, "out": "out"} == first.config
    assert (second.inputs, second.counts, second.models) == (
        first.inputs,
        first.counts,
        first.models,
    )

    # A run config's manifest keeps the run config, and where its paths resolve.
    result = runner.invoke(app, ["validate", "--config", "triples/odke.yaml"])
    assert result.exit_code == 0, result.output
    from_config = read_manifest(here / "triples" / "out")
    assert from_config.config["run"] == load_config(here / "triples" / "odke.yaml").canonical()
    assert (from_config.config_file, from_config.base_dir) == ("odke.yaml", str(here / "triples"))


def test_odke_ground_writes_its_manifest_beside_its_summary(here: Path) -> None:
    args = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "-o", "out"]
    result = runner.invoke(app, ["ground", *args, "--config", "models.yaml"])
    assert result.exit_code == 0, result.output
    assert "out/manifest.json" in result.output
    manifest, summary = read_manifest(here / "out"), _read(here / "out" / "summary.json")
    assert (manifest.command, manifest.config["adapter"]) == ("ground", "triples")
    assert manifest.counts["rows"] == summary["rows"] == 5
    assert manifest.counts["supported"] == summary["verdicts"]["supported"]
    assert manifest.spent["calls"] == summary["calls"]
    assert manifest.inputs.facts is not None and manifest.inputs.facts.rows == 5

    result = runner.invoke(app, ["ground", *args, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert read_manifest(here / "out").dry_run
