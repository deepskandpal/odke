"""The comparison scripts in `bench/`, whose numbers get published: no model, no key.

`bench/` is not part of the package and its own tests need the competitor
libraries, so these load the scripts by path and test what runs without them.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

BENCH = Path(__file__).parent.parent / "bench"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


competitors = _load("competitors")
inverses = _load("inverses")
tables = _load("tables")


# --------------------------------------------------------------------------- #
# competitors.py
# --------------------------------------------------------------------------- #


def test_an_env_file_loses_its_quotes_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`KEY="sk-…"` kept its quotes, so every call 401'd and the table showed 0%."""
    names = [f"ODKE_BENCH_TEST_{c}" for c in "ABCDE"]
    for name in names:  # set, then unset: monkeypatch removes them again afterwards
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("ODKE_BENCH_TEST_E", "already set")
    env = tmp_path / "bench.env"
    env.write_text(
        "# a comment\n"
        'export ODKE_BENCH_TEST_A="double quoted"\n'
        "ODKE_BENCH_TEST_B='single quoted'\n"
        "ODKE_BENCH_TEST_C=plain\n"
        'ODKE_BENCH_TEST_D="unbalanced\n'
        "ODKE_BENCH_TEST_E=from the file\n"
    )
    competitors.load_env(str(env))
    assert [os.environ[name] for name in names] == [
        "double quoted",
        "single quoted",
        "plain",
        '"unbalanced',
        "already set",
    ]


def _prepared(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "d1.txt").write_text("Marie Curie was born in Warsaw.")
    (tmp_path / "docs" / "d2.txt").write_text("Nothing here.")
    ontology = {
        "types": {"Person": {}, "City": {}},
        "predicates": {"born_in": {"domain": ["Person"], "range": "City"}},
    }
    (tmp_path / "ontology.json").write_text(json.dumps(ontology))
    (tmp_path / "dataset.json").write_text("{}")
    (tmp_path / "gold.jsonl").write_text('{"id": "d1"}\n{"id": "d2"}\n')
    config = {"stages": {"extractor": {"use": "llm"}}, "models": {"extract": "openai/gpt-5"}}
    (tmp_path / "odke.json").write_text(json.dumps(config))
    return tmp_path


def test_a_competitor_run_that_lost_documents_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document that errored is an empty answer, and an empty answer scores as a result."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-printed")
    for key in competitors.USAGE:
        monkeypatch.setitem(competitors.USAGE, key, 0)

    async def every_call_fails(docs: list[tuple[str, str]], *args: Any) -> list[dict]:
        competitors.USAGE["failed_documents"] += len(docs)
        return []

    monkeypatch.setitem(competitors.SYSTEMS, "lgt", every_call_fails)
    with pytest.raises(SystemExit) as stopped:
        competitors.extract("lgt", _prepared(tmp_path))
    message = str(stopped.value.code)
    assert "2 of 2 documents failed" in message
    assert "sk-never-printed" not in message
    usage = json.loads((tmp_path / "competitors" / "lgt" / "usage.json").read_text())
    assert usage["failed_documents"] == 2


def test_usage_records_the_competitor_libraries_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published number names the library versions that made it; one not installed is None."""
    for key in competitors.USAGE:
        monkeypatch.setitem(competitors.USAGE, key, 0)

    async def nothing_found(docs: list[tuple[str, str]], *args: Any) -> list[dict]:
        return []

    monkeypatch.setitem(competitors.SYSTEMS, "lgt", nothing_found)
    monkeypatch.setattr(competitors, "LIBRARIES", ("litellm", "odke-no-such-library"))
    out = competitors.extract("lgt", _prepared(tmp_path))
    usage = json.loads((out / "usage.json").read_text())
    assert usage["versions"] == {
        "litellm": importlib.metadata.version("litellm"),
        "odke-no-such-library": None,
    }


# --------------------------------------------------------------------------- #
# tables.py
# --------------------------------------------------------------------------- #


def _report(precision: float, facts: tuple[int, int, int], verdicts: str) -> dict[str, Any]:
    rows = {
        name: {
            "precision": precision,
            "triples": n,
            "facts": n,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        for name, n in zip(tables.ROWS, facts, strict=True)
    }
    return {"breakdown": rows, "notes": [f"grounder verdicts on the candidates: {verdicts}"]}


def _write(folder: Path, name: str, data: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(data))


def test_a_competitor_set_with_failed_documents_is_dropped_not_averaged(tmp_path: Path) -> None:
    verdicts = "supported 4, not_found 0, contradicted 0, unchecked 0"
    for name in ("ont_1_movie", "ont_2_music"):
        folder = tmp_path / name
        _write(folder, "report.json", _report(0.5, (4, 4, 4), verdicts))
        _write(folder / "competitors" / "lgt", "report.json", _report(0.8, (4, 4, 4), verdicts))
        _write(folder / "competitors" / "lgt", "usage.json", {"failed_documents": 0})
    # Every call failed in the second set: nothing extracted, a precision of 0.
    lost = tmp_path / "ont_2_music" / "competitors" / "lgt"
    _write(lost, "report.json", _report(0.0, (0, 0, 0), verdicts))
    _write(lost, "usage.json", {"failed_documents": 20})

    lines = tables.table("t", sorted(tmp_path.glob("ont_*")), ["precision"], ["triples"])
    lgt = lines[lines.index(next(ln for ln in lines if "LLMGraphTransformer" in ln)) :]
    assert lgt[0].startswith("| **LLMGraphTransformer** | extraction alone | 80.0% |")
    assert "(1 of 2 sets dropped: documents failed to extract)" in lgt[3]


def test_rejected_by_grounder_is_what_the_gate_removed(tmp_path: Path) -> None:
    """Outside paper mode the gate keeps not_found and refuses contradicted.

    Ten candidates, eight after the gate: two rejected, whatever the verdict
    note says about not_found.
    """
    verdicts = "supported 5, not_found 3, contradicted 2, unchecked 0"
    _write(tmp_path / "ont_1_movie", "report.json", _report(0.5, (10, 8, 6), verdicts))
    lines = tables.table("t", [tmp_path / "ont_1_movie"], ["precision"], ["triples"])
    rows = [ln for ln in lines if " | + grounding | " in ln or " | + corroboration | " in ln]
    assert rows[0].startswith("|  | + grounding |") and rows[0].endswith("| 8 | 2 |")
    assert rows[1].endswith("| 6 | 2 |")


# --------------------------------------------------------------------------- #
# inverses.py
# --------------------------------------------------------------------------- #


def test_inverse_partners_are_scored_with_the_datasets_own_scorer(tmp_path: Path) -> None:
    """Saved triples, no model: the partner the gold wants is recovered, and said to be."""
    from openodke.eval.datasets import redocred
    from openodke.eval.datasets._common import snake

    located, contains = (redocred.RELATIONS[p] for p in ("P131", "P150"))
    ontology = {
        "types": {"Location": {}},
        "predicates": {
            snake(label): {"domain": ["Location"], "range": "Location"}
            for label in redocred.RELATIONS.values()
        },
    }
    labels = {snake(label): label for label in redocred.RELATIONS.values()}
    _write(tmp_path, "ontology.json", ontology)
    _write(tmp_path, "dataset.json", {"relation_labels": labels})
    gold = {
        "id": "test_0000",
        "text": "Rennes is in Brittany. Brittany is in France.",
        "entities": [["Rennes"], ["Brittany"], ["France"]],
        "facts": [[1, located, 2], [2, contains, 1], [0, located, 1]],
    }
    (tmp_path / "gold.jsonl").write_text(json.dumps(gold) + "\n")
    saved = {
        "id": "test_0000",
        "triples": [["Brittany", located, "France"], ["Rennes", located, "Paris"]],
    }
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "extraction-alone.jsonl").write_text(json.dumps(saved) + "\n")

    measured = inverses.measure_redocred(tmp_path)
    first = measured["results"][0]
    assert (first["pairs"], first["system"], first["row"]) == (
        "#106's six",
        "openodke",
        "extraction alone",
    )
    assert (first["recall_before"], first["recall_after"]) == (1 / 3, 2 / 3)
    assert (first["precision_before"], first["precision_after"]) == (1 / 2, 2 / 4)
    assert first["partners"] == {"in gold": 1, "unlabelled": 0, "from a wrong fact": 1}
    # The copy the pairs were added to loads strictly, and declares them.
    issue = measured["ontologies"]["#106's six"]["predicates"]
    assert issue[snake(located)]["inverse_of"] == snake(contains)
    assert issue["spouse"]["symmetric"] is True
