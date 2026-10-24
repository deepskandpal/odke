"""R4. A recorded answer never passes for a real one.

`models.replay` answers a role from a hand-written file, for runs with no key.
The response cache stores answers by request. If a replayed answer is stored
under the live model's key, a later run with the replay lines deleted is
answered from the recording, with no key and exit 0, and its manifest says the
real model served it (release audit, finding 1).

The check: for every command that puts a response cache in front of a model,
run once on the recording with the cache on, then again with the recording
removed. Every request of the second run must reach the live model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from _support import BENCH, EXAMPLES, SRC, LiveModel, answer_every_model_with, copy_example
from typer.testing import CliRunner

from openodke.cli.main import app

runner = CliRunner()

# The grounding calls `examples/triples` makes: one per answer in its recording.
RECORDED = len(json.loads((EXAMPLES / "triples" / "recorded" / "ground.json").read_text("utf-8")))
MODEL = "anthropic/claude-haiku-4-5-20251001"
TRIPLES = ["--facts", "triples/triples.jsonl", "--texts", "triples/texts", "-o", "out"]
FIX = (
    "{n} of {total} answers to `{command}` came from the recording, served from the cache "
    "as if {model} had given them. The cache key must say whether an answer came from a "
    "recording or a live model: keep replayed answers in a cache of their own "
    "(for example `<cache>/replay/<hash of the replay file>`), in every place that builds "
    "a cached client."
)


def _without_replay(project: Path) -> Path:
    """The example's config with its `replay` block deleted, as its comment says to."""
    config = yaml.safe_load((project / "triples" / "odke.yaml").read_text(encoding="utf-8"))
    del config["models"]["replay"]
    path = project / "triples" / "live.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _labels(project: Path) -> None:
    """The example's triples as grounding labels, for `odke calibrate`: the same four questions."""
    from openodke import Ontology
    from openodke.eval.formats import GroundingLabel
    from openodke.interop import to_fact
    from openodke.interop.triples import TripleRow
    from openodke.loaders import DirectoryLoader

    example = project / "triples"
    (doc,) = DirectoryLoader().load(example / "texts")
    ontology = Ontology.from_json(example / "ontology.json")
    rows = [
        json.loads(line) for line in (example / "triples.jsonl").read_text("utf-8").splitlines()
    ]
    labels = [
        GroundingLabel(
            text=doc.text,
            fact=to_fact(TripleRow.model_validate(row), doc, ontology),
            doc_id=doc.id,
            verdict="supported",
        )
        for row in rows
    ]
    (example / "labels.jsonl").write_text(
        "".join(label.model_dump_json() + "\n" for label in labels), encoding="utf-8"
    )


# What an entry point needs beside the copied example before its first run.
PREPARE = {"odke calibrate": _labels}
CALIBRATE = ["calibrate", "triples/labels.jsonl", "--cache", "c", "-o", "cards"]

# Each entry: the command, its first run (on the recording), its second (live).
ENTRY_POINTS = {
    "odke run": (
        ["run", "triples/odke.yaml", "--cache", "c"],
        ["run", "triples/live.yaml", "--cache", "c"],
    ),
    "odke ground": (
        ["ground", *TRIPLES, "--config", "triples/odke.yaml", "--cache", "c"],
        ["ground", *TRIPLES, "--config", "triples/live.yaml", "--cache", "c"],
    ),
    "odke validate": (
        ["validate", *TRIPLES, "--config", "triples/odke.yaml", "--cache", "c"],
        ["validate", *TRIPLES, "--config", "triples/live.yaml", "--cache", "c"],
    ),
    "odke calibrate": (
        [*CALIBRATE, "--config", "triples/odke.yaml"],
        [*CALIBRATE, "--config", "triples/live.yaml"],
    ),
}


@pytest.mark.parametrize("command", list(ENTRY_POINTS))
def test_a_rerun_without_the_recording_asks_the_live_model(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy_example("triples", tmp_path)
    _without_replay(tmp_path)
    if command in PREPARE:
        PREPARE[command](tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    recorded, live_args = ENTRY_POINTS[command]

    first = runner.invoke(app, recorded)
    assert first.exit_code == 0, first.output

    live = LiveModel()
    answer_every_model_with(monkeypatch, live)
    second = runner.invoke(app, live_args)
    # Exit 2 is a config the command refused before any model was asked.
    assert second.exit_code != 2, second.output

    served = RECORDED - live.calls
    assert served == 0, (
        FIX.format(n=served, total=RECORDED, command=command, model=MODEL)
        + f"\n\nThe second run printed:\n{second.output}"
    )


def test_every_place_that_caches_a_model_is_an_entry_point_of_this_rule() -> None:
    """A new `CachedClient(...)` is a new way for a recording to pass for a model: list it here."""
    known = {
        "src/openodke/run/build.py",  # odke run
        "src/openodke/cli/main.py",  # odke ground, odke validate, odke calibrate
        # The bench's grounding and label set G's claim-format check: a provider or the
        # cache, never a replay file.
        "bench/layer.py",
        "bench/labels/claim_format.py",
        # The calibration runner: a provider behind the cache, or with --replay recorded
        # answers cached apart, in `openodke.llm.cache.replay_directory` (test_bench_calibrate).
        "bench/calibrate.py",
    }
    found = set()
    for path in [*SRC.rglob("*.py"), *BENCH.rglob("*.py")]:
        if re.search(r"\bCachedClient\(", path.read_text(encoding="utf-8")):
            found.add(path.relative_to(SRC.parents[1]).as_posix())
    found.discard("src/openodke/llm/cache.py")
    assert found <= known, (
        f"{sorted(found - known)} wrap a client in the response cache. Add the command that "
        "reaches it to ENTRY_POINTS in tests/rules/test_r4_recorded_answers.py, so a recorded "
        "answer cannot pass through it as a live one."
    )
