"""The versioned formats' JSON Schemas: shipped, and held to the models (DECISIONS #48).

The eval report has its own (`test_eval_report.py`). These are the other two
formats a reader outside Python depends on: the triples input format, which
any extractor writes, and the run report, `ValidationReport`, which the
Validator returns and the graph's `stats["validation"]` keeps. Each schema is
the contract and each model is the reader, so the two cannot drift: the same
fields, the same required ones, and where `jsonschema` is installed, the same
answer on good rows and bad ones.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from openodke import Ontology, Validator
from openodke.interop.triples import SCHEMA_PATH, SCHEMA_VERSION, TripleRow, read_triples
from openodke.llm.testing import RecordedClient
from openodke.loaders import DirectoryLoader
from openodke.manifest import FORMAT, FactsIn, Inputs, ModelUse, RunManifest, read_manifest
from openodke.manifest import SCHEMA_PATH as MANIFEST_SCHEMA
from openodke.sinks import JsonlSink
from openodke.validator import REPORT_SCHEMA_PATH, REPORT_VERSION, ValidationReport

TRIPLES = Path(__file__).resolve().parents[1] / "examples" / "triples"
ROW = {"doc": "halden", "subject": "Halden Robotics", "predicate": "office_in", "object": "Lyon"}


def _schema(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _validator(path: Path) -> Any:
    jsonschema = pytest.importorskip("jsonschema")
    schema = _schema(path)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# --------------------------------------------------------------------------- #
# The triples input format
# --------------------------------------------------------------------------- #


def test_the_triples_schema_ships_inside_the_package_and_names_its_version() -> None:
    assert SCHEMA_PATH.parent.name == "interop" and SCHEMA_PATH.is_file()
    assert SCHEMA_VERSION == "1.0"
    version = _schema(SCHEMA_PATH)["properties"]["schema_version"]
    assert version == {"type": "string", "pattern": r"^1\.[0-9]+$"}


def test_the_triples_schema_names_every_field_of_the_row() -> None:
    schema = _schema(SCHEMA_PATH)
    fields = TripleRow.model_fields
    assert set(schema["properties"]) == set(fields)
    assert set(schema["required"]) == {name for name, f in fields.items() if f.is_required()}
    assert schema["additionalProperties"] is False


def test_a_row_names_its_version_or_is_read_as_this_one(tmp_path: Path) -> None:
    assert TripleRow.model_validate(ROW).schema_version == SCHEMA_VERSION
    assert TripleRow.model_validate({**ROW, "schema_version": "1.7"}).schema_version == "1.7"
    path = tmp_path / "triples.jsonl"
    path.write_text(
        json.dumps(ROW) + "\n" + json.dumps({**ROW, "schema_version": "2.0"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r":2: schema_version: .*'2\.0': this openodke reads 1\.0"):
        list(read_triples(path))
    with pytest.raises(ValidationError, match="not major.minor"):
        TripleRow.model_validate({**ROW, "schema_version": "1"})


def test_a_later_minors_row_is_read_for_what_this_version_knows(tmp_path: Path) -> None:
    """Call F (DECISIONS #52): a later 1.x row's new fields are left; this version's are not."""
    newer = {**ROW, "schema_version": "1.9", "added_in_1_9": {"x": 1}}
    row = TripleRow.model_validate(newer)
    assert row == TripleRow.model_validate({**ROW, "schema_version": "1.9"})
    path = tmp_path / "triples.jsonl"
    path.write_text(json.dumps(newer) + "\n", encoding="utf-8")
    assert [r.subject for r in read_triples(path)] == ["Halden Robotics"]
    for version in ({}, {"schema_version": SCHEMA_VERSION}):
        with pytest.raises(ValidationError, match="added_in_1_9"):
            TripleRow.model_validate({**ROW, **version, "added_in_1_9": 1})
    # Another major is still refused by name, whatever it holds.
    with pytest.raises(ValidationError, match="this openodke reads 1.0"):
        TripleRow.model_validate({**newer, "schema_version": "2.1"})


BAD_ROWS = [
    {**ROW, "objet": "Lyon"},
    {k: v for k, v in ROW.items() if k != "doc"},
    {**ROW, "doc": ""},
    {**ROW, "start": 3},
    {**ROW, "start": -1, "end": 4},
    {**ROW, "quote": ""},
    {**ROW, "polarity": "maybe"},
    {**ROW, "confidence": 1.5},
    {**ROW, "object": None},
    {**ROW, "schema_version": "2.0"},
]


def test_the_schema_and_the_reader_agree_on_every_row() -> None:
    """The example's rows and a full row pass both; each bad row fails both."""
    check = _validator(SCHEMA_PATH)
    rows = [json.loads(line) for line in (TRIPLES / "triples.jsonl").read_text().splitlines()]
    full = TripleRow.model_validate(
        {**ROW, "start": 0, "end": 5, "quote": "Halde", "qualifiers": {"since": 2019}}
    )
    for row in [*rows, full.model_dump(mode="json")]:
        check.validate(row)
        TripleRow.model_validate(row)
    for row in BAD_ROWS:
        assert not check.is_valid(row), row
        with pytest.raises(ValidationError):
            TripleRow.model_validate(row)


# --------------------------------------------------------------------------- #
# The run report
# --------------------------------------------------------------------------- #


def test_the_run_report_schema_ships_inside_the_package_and_names_its_version() -> None:
    assert REPORT_SCHEMA_PATH.parent.name == "openodke" and REPORT_SCHEMA_PATH.is_file()
    assert REPORT_VERSION == "1.1" and ValidationReport().schema_version == REPORT_VERSION
    version = _schema(REPORT_SCHEMA_PATH)["properties"]["schema_version"]
    assert version == {"type": "string", "pattern": r"^1\.[0-9]+$"}


# Fields a minor version added, which a report from before it does not have.
ADDED_IN = {"marked": "1.1", "calibration": "1.1"}


def test_the_run_report_schema_names_every_field_of_the_report() -> None:
    """Same fields, all of 1.0's required: what the report's JSON holds. The manifest is its own."""
    schema = _schema(REPORT_SCHEMA_PATH)
    fields = {name for name, f in ValidationReport.model_fields.items() if not f.exclude}
    assert set(schema["properties"]) == fields
    assert set(schema["required"]) == fields - set(ADDED_IN)
    for name, version in ADDED_IN.items():
        assert f"Added in {version}, so not required" in schema["properties"][name]["description"]
    assert "manifest" not in fields
    assert schema["additionalProperties"] is False


def test_a_job_s_report_fits_the_schema_in_the_report_and_in_the_graph(tmp_path: Path) -> None:
    check = _validator(REPORT_SCHEMA_PATH)
    client = RecordedClient.from_fixture(TRIPLES / "recorded" / "ground.json")
    ontology = Ontology.from_json(TRIPLES / "ontology.json")
    texts = list(DirectoryLoader().load(TRIPLES / "texts"))
    validator = Validator(ontology, client=client, sinks=[JsonlSink(tmp_path / "out")])
    kg, report = validator.validate(TRIPLES / "triples.jsonl", texts, batch_size=2)
    data = report.model_dump(mode="json")
    check.validate(data)
    assert kg.stats["validation"] == data and data["schema_version"] == REPORT_VERSION
    _, dry = Validator(ontology).validate(TRIPLES / "triples.jsonl", texts, dry_run=True)
    check.validate(dry.model_dump(mode="json"))
    data["refused"], data["surprise"] = -1, 0
    assert not check.is_valid(data)


# --------------------------------------------------------------------------- #
# The run manifest
# --------------------------------------------------------------------------- #


def test_the_manifest_schema_ships_beside_the_reader_and_names_its_version() -> None:
    """Call C, question 6: the manifest's format ships its schema, as the others do."""
    assert MANIFEST_SCHEMA.parent.name == "openodke" and MANIFEST_SCHEMA.is_file()
    assert _schema(MANIFEST_SCHEMA)["properties"]["manifest_version"] == {"const": FORMAT}


def test_the_manifest_schema_names_every_field_of_the_manifest_and_its_parts() -> None:
    schema = _schema(MANIFEST_SCHEMA)
    parts = {"model_use": ModelUse, "facts_in": FactsIn, "inputs": Inputs}
    for properties, required, model in [
        (schema["properties"], schema["required"], RunManifest),
        *(
            (schema["$defs"][k]["properties"], schema["$defs"][k]["required"], m)
            for k, m in parts.items()
        ),
    ]:
        fields = model.model_fields
        assert set(properties) == set(fields), model.__name__
        must = {name for name, f in fields.items() if f.is_required()}
        # A manifest names its version: the reader refuses one that does not.
        assert set(required) == must | ({"manifest_version"} if model is RunManifest else set())
    # The reader keeps what it knows (DECISIONS #52): a sink's keys share the file.
    assert "additionalProperties" not in schema


def test_a_runs_manifest_fits_the_schema_with_the_sinks_keys_beside_it(tmp_path: Path) -> None:
    from openodke.run import execute, load_config

    check = _validator(MANIFEST_SCHEMA)
    project = tmp_path / "triples"
    shutil.copytree(TRIPLES, project, ignore=shutil.ignore_patterns("out"))
    result = execute(load_config(project / "odke.yaml"))
    written = json.loads((project / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert result.manifest is not None and "entities" in written  # the sink's own keys
    check.validate(written)
    assert read_manifest(project / "out") == result.manifest
    written["command"] = "replay"
    assert not check.is_valid(written)
