"""`bench/labels/make_r.py`: label set R drawn from tiny made-up Re-DocRED documents.

Each document names one person twice under two names, a second person whose
name shares a word with the first, a third who shares nothing, a town and a
year, in Re-DocRED's own shape: token lists, and clusters of mentions with
their types and token positions.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke.eval import PairLabel, load_jsonl
from openodke.eval.sheets import PairItem, make_sheets

SCRIPT = Path(__file__).parent.parent / "bench" / "labels" / "make_r.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("odke_bench_make_r", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look their module up while it loads.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r = _load()
PLAN = r.Plan(same=6, hard=6, other=3, dev=5, audit_same=3, audit_hard=2, audit_other=1)
SURNAMES = ["Vale", "Moor", "Quill", "Rook", "Sable", "Tarn", "Umber", "Wren", "Yarrow", "Zeal"]


def _mention(name: str, sent: int, start: int, kind: str) -> dict[str, Any]:
    return {"name": name, "sent_id": sent, "pos": [start, start + len(name.split())], "type": kind}


def _doc(i: int) -> dict[str, Any]:
    surname = SURNAMES[i % len(SURNAMES)] + str(i)
    sents = [
        f"Ada {surname} wrote a book in 1901 .".split(),
        f"{surname} met Ada Brook{i} in Town{i} .".split(),
        f"Zed Quill{i} did not .".split(),
    ]
    return {
        "title": f"Doc {i}",
        "sents": sents,
        "vertexSet": [
            # One person under two names: "same".
            [_mention(f"Ada {surname}", 0, 0, "PER"), _mention(surname, 1, 0, "PER")],
            # Shares "Ada" with the first: a hard negative.
            [_mention(f"Ada Brook{i}", 1, 2, "PER")],
            [_mention(f"Zed Quill{i}", 2, 0, "PER")],
            [_mention(f"Town{i}", 1, 5, "LOC")],
            # A value, never in a pair.
            [_mention("1901", 0, 5, "TIME")],
        ],
        "labels": [],
    }


RAW = [_doc(i) for i in range(12)]


@pytest.fixture(scope="module")
def made() -> Any:
    return r.make(r.load(RAW), PLAN)


@pytest.fixture(scope="module")
def written(made: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("R")
    r.write(made, out)
    return out


def test_each_stratum_and_split_gets_its_count(made: Any) -> None:
    counts = Counter((p["stratum"], p["split"]) for p in made.private)
    assert counts == {
        ("same", "dev"): 2,
        ("same", "gate"): 4,
        ("hard", "dev"): 2,
        ("hard", "gate"): 4,
        ("other", "dev"): 1,
        ("other", "gate"): 2,
    }
    assert [p["id"] for p in made.private] == [f"R-{n:04d}" for n in range(1, 16)]
    # The audit: half same, hard negatives over their share of R.
    audited = Counter(p["stratum"] for p in made.private if p["audit"])
    assert audited == {"same": 3, "hard": 2, "other": 1}
    assert sorted(p["audit"] for p in made.private if p["audit"]) == [
        f"P-{n:04d}" for n in range(1, 7)
    ]


def test_each_label_is_the_datasets_and_no_pair_is_one_name(made: Any) -> None:
    for row, label, private in zip(made.rows, made.labels, made.private, strict=True):
        assert PairLabel.model_validate(label).same is (private["stratum"] == "same")
        assert label["same"] is (private["clusters"][0] == private["clusters"][1])
        assert (label["a"], label["b"]) == (row["a"]["key"], row["b"]["key"])
        assert row["a"]["type"] == row["b"]["type"]
        a, b = private["names"]
        assert a.casefold() != b.casefold()
        assert private["scope"] == "within"
    hard = [p for p in made.private if p["stratum"] == "hard"]
    assert all("Ada" in " ".join(p["names"]) for p in hard)
    # The year is a value: no pair names it.
    assert not any("1901" in name for p in made.private for name in p["names"])


def test_no_document_is_in_both_splits(made: Any, written: Path) -> None:
    by_split: dict[str, set[str]] = {"dev": set(), "gate": set()}
    for private in made.private:
        by_split[private["split"]].add(private["doc"])
    assert by_split["dev"] and by_split["gate"]
    assert not by_split["dev"] & by_split["gate"]
    # The gate's passages, for the prompt-leakage test: every gate document, whole.
    gate = [json.loads(line) for line in (written / "gate.jsonl").read_text().splitlines()]
    assert {g["doc"] for g in gate} >= by_split["gate"]
    assert not {g["doc"] for g in gate} & by_split["dev"]
    assert all(isinstance(g["text"], str) and g["text"] for g in gate)


def test_a_mention_shows_its_sentence_and_one_either_side(made: Any) -> None:
    row = next(row for row in made.rows if row["a"]["key"].split(":")[1] == "s1")
    assert row["a"]["context"].count(".") == 3
    first = next(row for row in made.rows if row["a"]["key"].split(":")[1] == "s0")
    assert first["a"]["context"].count(".") == 2


def test_the_sheet_reveals_no_gold(made: Any, written: Path, tmp_path: Path) -> None:
    sheets = make_sheets("pair", written / "audit.jsonl", tmp_path / "sheets", per_sheet=100)
    (sheet,) = sheets.sheets
    shown = sheet.read_text(encoding="utf-8")
    sidecar = sheet.with_suffix(".items.jsonl").read_text(encoding="utf-8")
    for row in map(json.loads, sidecar.splitlines()):
        item = PairItem.model_validate(row["row"])
        # Opaque keys: a mention's place, never its cluster.
        for mention in (item.a, item.b):
            assert re.fullmatch(r"test_\d{4}:s\d+:\d+-\d+", mention.key)
        assert item.a.type == item.b.type
        assert set(row["row"]) == {"a", "b"}
    for word in ("stratum", "hard", "cluster", '"same"', "gold", "R-0"):
        assert word not in shown and word not in sidecar
    # Shuffled: the sheet's order is not the order the strata were drawn in.
    audit = sorted((p["audit"], p["stratum"]) for p in made.private if p["audit"])
    drawn = [stratum for stratum in r.STRATA for _ in range(PLAN.audit(stratum))]
    assert [stratum for _, stratum in audit] != drawn


def test_the_same_seed_writes_the_same_bytes(written: Path, tmp_path: Path) -> None:
    again = tmp_path / "again"
    r.write(r.make(r.load(RAW), PLAN), again)
    for name in ("items.jsonl", "labels.jsonl", "items.private.jsonl", "gate.jsonl", "audit.jsonl"):
        assert (again / name).read_bytes() == (written / name).read_bytes(), name
    other = r.make(r.load(RAW), PLAN, seed=7)
    assert other.labels != r.make(r.load(RAW), PLAN).labels


def test_pairs_across_documents_wait_for_ids(made: Any) -> None:
    with pytest.raises(ValueError, match="T-REx"):
        r.make(r.load(RAW), r.Plan(across=10))
    too_many = r.Plan(same=40, hard=6, other=3, dev=5)
    with pytest.raises(ValueError, match="pairs, the documents have"):
        r.make(r.load(RAW), too_many)


def test_the_labels_file_is_what_odke_eval_resolve_reads(written: Path) -> None:
    labels = load_jsonl(written / "labels.jsonl", PairLabel)
    assert len(labels) == 15 and sum(label.same for label in labels) == 6
