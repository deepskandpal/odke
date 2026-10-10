"""`bench/volume.py`: the volume test's slice, simulated model and tables, with no network."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from openodke import GroundingVerdict, Ontology
from openodke.extract import LLMExtractor
from openodke.llm import Message, ModelSpec, register, set_limit, unregister
from openodke.types import Chunk

BENCH = Path(__file__).parent.parent / "bench"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_bench_{name}", BENCH / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# volume.py imports its sibling as `conflicts`, as it does when run from bench/.
sys.modules.setdefault("conflicts", _load("conflicts"))
volume = _load("volume")

TEXT = (
    "Weekly sync notes. Priya Shah moved the Ingest Gateway rollout to Atlas Team. "
    "Nothing else changed here. Then Marco Diaz said Helios depends on Ingest Gateway! "
    "Only one Name here."
)


def _doc(n: int, source: str) -> dict[str, Any]:
    return {
        "title_field_name": "title",
        "content_field_names": ["body"],
        "title": f"Doc {n}",
        "body": f"Ana Ruiz told Kite Labs about Orbit {n}.",
        "updated_at": "2026-03-12",
        "dataset_doc_uuid": f"dsid_{n:04d}",
    }


def _archive(per_type: dict[str, int]) -> bytes:
    """A gzipped tar shaped like the repository's archive: a top folder, sources and more."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:

        def add(name: str, body: bytes) -> None:
            info = tarfile.TarInfo(f"EnterpriseRAG-Bench-sha/{name}")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))

        add("README.md", b"# not a document")
        add("generated_data/uuid_index.json", b"{}")
        n = 0
        for source, count in per_type.items():
            for _ in range(count):
                body = json.dumps(_doc(n, source)).encode()
                add(f"generated_data/sources/{source}/team/doc-{n}.json", body)
                n += 1
    return buffer.getvalue()


def _opener(asked: list[str], body: bytes) -> Any:
    def open_url(url: str) -> io.BytesIO:
        asked.append(url)
        return io.BytesIO(body)

    return open_url


def test_fetch_keeps_a_seeded_sample_of_the_sources_and_the_canary(tmp_path: Path) -> None:
    asked: list[str] = []
    body = _archive({"slack": 20, "jira": 6, "confluence": 4})
    root = volume.fetch(tmp_path / "a", documents=10, seed=3, opener=_opener(asked, body))
    assert asked == [volume.TARBALL] and volume.conflicts.SHA in asked[0]
    rows = volume.read_slice(root)
    assert len(rows) == 10 and len({r["id"] for r in rows}) == 10
    assert all(r["canary"] == volume.conflicts.CANARY for r in rows)
    first = rows[0]
    assert first["text"].startswith("Doc ") and first["time"] == "2026-03-12T00:00:00+00:00"
    about = json.loads((root / "slice.json").read_text())
    assert about["canary"] == volume.conflicts.CANARY and about["commit"] == volume.conflicts.SHA
    # The whole corpus is counted, the README and the index are not documents.
    assert about["corpus"] == {"confluence": 4, "jira": 6, "slack": 20}
    assert sum(about["slice"].values()) == 10
    again = volume.fetch(tmp_path / "b", documents=10, seed=3, opener=_opener([], body))
    assert [r["id"] for r in volume.read_slice(again)] == [r["id"] for r in rows]
    everything = volume.fetch(tmp_path / "c", documents=100, opener=_opener([], body))
    assert len(volume.read_slice(everything)) == 30


def test_a_claim_quotes_a_sentence_naming_two_names_where_it_is() -> None:
    found = volume.claims(TEXT)
    # A run of capitals that opens the sentence is not a name: "Priya Shah" here.
    assert [(c["subject"], c["object"]) for c in found] == [
        ("Ingest Gateway", "Atlas Team"),
        ("Helios", "Ingest Gateway"),
    ]
    for claim in found:
        assert TEXT[claim["start"] : claim["start"] + len(claim["quote"])] == claim["quote"]
        spec = volume.ONTOLOGY["predicates"][claim["predicate"]]
        assert claim["subject_type"] in spec["domain"] and claim["object_type"] == spec["range"]
    assert volume.claims(TEXT, limit=1) == found[:1]
    assert volume.claims(TEXT) == found  # by checksum, not by chance


def test_the_simulated_model_passes_the_extractors_own_checks() -> None:
    ontology = Ontology.from_dict(volume.ONTOLOGY)
    client = volume.Simulated({"extract": 0.0})
    extractor = LLMExtractor(client=client, spec=ModelSpec(model="sim/extract"), structured=False)
    chunk = Chunk(doc_id="d", start=0, end=len(TEXT), text=TEXT, index=0)
    facts = extractor.extract(chunk, ontology)
    assert len(facts) == 2 and not extractor.rejections
    assert {f.object_entity.label for f in facts if f.object_entity} == {
        "Atlas Team",
        "Ingest Gateway",
    }
    ground = ModelSpec(model="sim/ground")
    verdicts = [
        client.complete([Message(content=f"Claim: {n}")], spec=ground).parsed["verdict"]
        for n in range(400)
    ]
    shares = {v: verdicts.count(v) / 400 for v in set(verdicts)}
    assert set(shares) == {v.value for v in GroundingVerdict} - {"unchecked"}
    assert 0.8 < shares["supported"] < 0.9


def test_a_new_version_drops_every_second_claimed_sentence() -> None:
    new = volume.revise(TEXT)
    assert "Priya Shah" in new and "Marco Diaz" not in new
    assert [c["subject"] for c in volume.claims(new)] == ["Ingest Gateway"]


def test_the_cost_slice_takes_each_type_in_turn() -> None:
    docs = [{"id": f"{t}-{n}", "source_type": t} for t in ("slack", "jira") for n in range(5)]
    chosen = volume.select(docs, per_type=2, seed=0)
    assert [d["source_type"] for d in chosen] == ["jira", "slack", "jira", "slack"]
    assert chosen == volume.select(list(reversed(docs)), per_type=2, seed=0)


def test_a_run_folder_runs_on_the_simulated_model(tmp_path: Path) -> None:
    from openodke.run.build import build
    from openodke.run.config import parse_config
    from openodke.run.execute import run_built

    docs = [
        {"id": "dsid_a", "source_type": "slack", "text": TEXT},
        {"id": "dsid_b", "source_type": "confluence", "text": TEXT.replace("Helios", "Vega")},
    ]
    config = volume.run_config(["slack", "confluence"], volume.simulated(2), workers=2)
    volume.layout(tmp_path, docs, config, facts=5)
    assert json.loads((tmp_path / "CANARY.json").read_text())["canary"] == volume.conflicts.CANARY
    rows = (tmp_path / "triples.jsonl").read_text().splitlines()
    assert len(rows) == 4 and json.loads(rows[0])["doc"] == "dsid_a"
    register("sim", lambda spec: volume.Simulated({}))
    try:
        built = build(parse_config(config, base_dir=tmp_path))
        stats = run_built(built).stats
    finally:
        # Both are process-wide: the provider, and the limit the config set for it.
        unregister("sim")
        set_limit("sim", None)
    assert stats["documents"] == 2 and stats["spent"]["calls"] == 2 + 4
    curated = {f["evidence"][0]["tier"] for f in _facts(tmp_path / "out")}
    assert curated <= {"curated", "unverified"} and "curated" in curated


def _facts(folder: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (folder / "facts.jsonl").read_text().splitlines()]


def test_the_cost_per_thousand_is_a_ratio_estimate_with_a_range() -> None:
    rows = [
        {"source_type": "slack", "run": True, "chars": 1000, "usd": 0.01, "facts": 4},
        {"source_type": "slack", "run": True, "chars": 3000, "usd": 0.03, "facts": 8},
        {"source_type": "jira", "run": True, "chars": 2000, "usd": 0.05, "facts": 6},
        {"source_type": "jira", "run": False},
    ]
    found = volume.per_thousand(rows, {"slack": 2000, "jira": 4000}, {"slack": 0.75, "jira": 0.25})
    # 1,000 × the type's mean length × USD per character.
    assert found["slack"]["usd_per_1000"] == 20.0 and found["jira"]["usd_per_1000"] == 100.0
    assert found["corpus"]["usd_per_1000"] == 40.0
    assert found["slack"]["range"] == [20.0, 20.0]  # every draw has the same ratio
    assert found["slack"]["facts_per_document"] == 6.0 and found["jira"]["documents"] == 1


def test_a_retraction_is_read_off_either_report() -> None:
    reconciled = (
        "odke reconcile\nretracted     a, b\ncited         99 facts\n"
        "lost support  9 facts, still backed\nretired       90 facts left with no source, kept"
    )
    assert volume._retracted(reconciled) == {"cited": 99, "lost": 9, "retired": 90}
    updated = (
        "update        the old versions retracted first: 157 facts cited them, "
        "2 kept a source, 155 left with none"
    )
    assert volume._retracted(updated) == {"cited": 157, "lost": 2, "retired": 155}


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="VmHWM is Linux's")
def test_the_peak_is_this_process_own() -> None:
    peak = volume.peak_rss_mb()
    assert peak is not None and peak > 1


def test_the_tables_show_each_measurement(tmp_path: Path) -> None:
    row = {
        "command": "odke run",
        "concurrency": 4,
        "batch_size": None,
        "documents": 100,
        "facts": 900,
        "calls": {"extract": 100, "ground": 950},
        "seconds": 480.2,
        "bound_s": 467.5,
        "documents_per_s": 0.21,
        "calls_per_s": 2.2,
        "peak_rss_mb": 120.0,
        "load": 1.0,
    }
    streamed = {**row, "command": "odke validate", "batch_size": 1000}
    volume._record(tmp_path / "throughput" / "throughput.json", [row, streamed])
    volume._record(tmp_path / "memory" / "memory.json", [row])
    lines = volume.table(tmp_path).splitlines()
    assert "| odke run | 4 | no | 100 | 1,050 | 480.2 | 467.5 | 0.21 | 2.2 |" in lines
    assert "| odke validate | 4 | 1,000 rows | 100 | 1,050 | 480.2 | 467.5 | 0.21 | 2.2 |" in lines
    assert "| odke run | 100 | no | 900 | 120.0 | 480.2 | 0.21 |" in lines
    # A second pass replaces its own rows and keeps the others.
    volume._record(tmp_path / "memory" / "memory.json", [{**row, "peak_rss_mb": 99.0}])
    assert "| odke run | 100 | no | 900 | 99.0 | 480.2 | 0.21 |" in volume.table(tmp_path)


def test_each_stage_time_is_summed_from_the_json_log(tmp_path: Path) -> None:
    def stage(name: str, seconds: float) -> str:
        return json.dumps({"event": "stage", "stage": name, "latency_s": seconds})

    log = tmp_path / "events.jsonl"
    lines = [stage("ground", 1.25), "warning: a line as text", stage("ground", 0.5)]
    lines += [stage("write", 0.2), json.dumps({"event": "job.end", "latency_s": 9.0})]
    log.write_text("\n".join(lines) + "\n")
    # Streamed, a stage logs once a micro-batch.
    assert volume.stage_seconds(log) == {"ground": 1.75, "write": 0.2}
