"""`odke calibrate` (#135): a calibration card from your own grounding labels.

Every model call is answered from recorded responses named in a config's
`models.replay`, as a user tries a command with no key.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke.calibration import load_card
from openodke.cli.main import app

runner = CliRunner()
REPO = Path(__file__).parent.parent
G = REPO / "bench" / "labels" / "G"
HAIKU = "anthropic/claude-haiku-4-5-20251001"
# The Re-DocRED documents that name Lorde are not found; every other claim is supported.
RECORDED = [
    {"match": "Lorde", "response": {"verdict": "not_found"}},
    {"match": "## Statement", "response": "yes"},
    {"match": "Claim", "response": {"verdict": "supported"}},
]


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = G.joinpath("labels.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "labels.jsonl").write_text("\n".join(rows[:12]) + "\n", encoding="utf-8")
    planted = [json.loads(line) for line in G.joinpath("items.jsonl").read_text().splitlines()]
    private = [
        json.loads(line) for line in G.joinpath("items.private.jsonl").read_text().splitlines()
    ]
    fakes = [
        {**row, "verdict": "not_found"}
        for row, about in zip(planted, private, strict=True)
        if about["planted"]
    ][:4]
    (tmp_path / "planted.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in fakes), encoding="utf-8"
    )
    (tmp_path / "recorded.json").write_text(json.dumps(RECORDED), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(
        "models:\n  replay:\n    ground: recorded.json\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _calibrate(*args: str) -> Any:
    return runner.invoke(app, ["calibrate", *args])


def test_a_card_from_your_labels_is_printed_and_written(here: Path) -> None:
    result = _calibrate(
        "labels.jsonl", "--config", "models.yaml", "--runs", "2", "--planted", "planted.jsonl",
        "--label-set", "mine", "--split", "gate", "-o", "cards",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    name = f"ground.span@1__{HAIKU.replace('/', '--')}__inline.json"
    card = load_card(here / "cards" / name)
    assert (card.prompt, card.model, card.claim_format) == ("ground.span@1", HAIKU, "inline")
    assert (card.labels.name, card.labels.split, card.labels.n) == ("mine", "gate", 12)
    assert card.grounding is not None and card.grounding.runs == 2
    assert card.grounding.flips is not None and card.grounding.flips.k == 0
    assert card.grounding.planted is not None and card.grounding.planted.n == 4
    assert card.self_agreement is None and card.bar.note is not None
    assert "calibration card  ground.span@1__" in result.output
    assert f"wrote cards/{name}" in result.output


def test_a_challenger_prompt_makes_its_own_card(here: Path) -> None:
    result = _calibrate(
        "labels.jsonl", "--config", "models.yaml", "--prompt", "ground.onyx", "-o", "onyx.json"
    )
    assert result.exit_code == 0, result.output
    card = load_card(here / "onyx.json")
    assert (card.prompt, card.claim_format) == ("ground.onyx@1", "sentence")
    # The one recorded answer that is JSON, not a yes or no, is unreadable to Onyx: an abstention.
    assert card.grounding is not None and card.grounding.abstain.k == 1


def test_a_card_in_odke_cards_is_what_odke_validate_prints(
    here: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _calibrate("labels.jsonl", "--config", "models.yaml", "-o", "cards").exit_code == 0
    shutil.copytree(
        REPO / "examples" / "triples", here / "triples", ignore=shutil.ignore_patterns("out")
    )
    monkeypatch.setenv("ODKE_CARDS", str(here / "cards"))
    result = runner.invoke(
        app,
        [
            "validate", "--facts", "triples/triples.jsonl", "--texts", "triples/texts",
            "--config", "triples/odke.yaml",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (line,) = [line for line in result.output.splitlines() if line.startswith("calibration")]
    assert f"ground.span@1 on {HAIKU} (inline): κ" in line and "label set labels all, n 12" in line


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["missing.jsonl"], "missing.jsonl"),
        (["labels.jsonl", "--claim-format", "boxed"], "inline or labelled"),
        (["labels.jsonl", "--prompt", "ground.nothing"], "no prompt"),
        (["empty.jsonl"], "no labels"),
    ],
)
def test_input_it_cannot_use_is_exit_2(here: Path, args: list[str], message: str) -> None:
    (here / "empty.jsonl").write_text("", encoding="utf-8")
    result = _calibrate(*args, "--config", "models.yaml")
    assert result.exit_code == 2, result.output
    assert message in result.output
