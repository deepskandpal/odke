"""Structured records become documents whose every cell has a faithful span."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from odke import Document, Loader, SourceTier, Span
from odke.loaders import (
    CsvLoader,
    DirectoryLoader,
    JsonlLoader,
    JsonLoader,
    ParquetLoader,
    RecordsLoader,
    TsvLoader,
)
from odke.loaders.records import (
    format_path,
    parse_path,
    record_document,
    render_value,
    resolve_path,
)

BOM = "\N{ZERO WIDTH NO-BREAK SPACE}"


def _cells_are_faithful(doc: Document) -> None:
    """Every rendered value's span resolves to exactly the rendering of that value."""
    row = doc.metadata["row"]
    for path, (start, end) in doc.metadata["fields"].items():
        ((_, value),) = resolve_path(row, parse_path(path))
        span = Span(doc_id=doc.id, start=start, end=end, quote=render_value(value))
        assert span.is_faithful(doc), path


def test_csv_rows_are_structured_documents_with_cells_that_resolve(tmp_path: Path) -> None:
    raw = (
        BOM + "name,born,note\r\n"
        'Ada Lovelace,1815-12-10,"wrote, notes"\r\n'
        "\r\n"
        'Charles Babbage,1791-12-26,"two\r\nlines"\r\n'
        "Alan Turing,1912-06-23\r\n"
    )
    path = tmp_path / "people.csv"
    path.write_bytes(raw.encode("utf-8"))
    docs = CsvLoader(tier=SourceTier.CURATED).load(path)

    assert len(docs) == 3
    ada, babbage, turing = docs
    # The byte-order mark does not rename the first column.
    assert ada.metadata["row"] == {
        "name": "Ada Lovelace",
        "born": "1815-12-10",
        "note": "wrote, notes",
    }
    assert ada.text == "name: Ada Lovelace\nborn: 1815-12-10\nnote: wrote, notes"
    assert (ada.modality, ada.tier, ada.uri) == (
        "structured",
        SourceTier.CURATED,
        path.resolve().as_uri(),
    )
    # A multi-line cell renders on one line, as a JSON string.
    assert babbage.text.endswith('note: "two\\r\\nlines"')
    # A short row lacks the missing key rather than inventing an empty value.
    assert "note" not in turing.metadata["row"]
    # Lines are physical lines in the file; the blank line and the wrapped cell count.
    assert [d.metadata["line"] for d in docs] == [2, 4, 6]
    assert [d.metadata["row_index"] for d in docs] == [0, 1, 2]
    for doc in docs:
        _cells_are_faithful(doc)
    start, end = ada.metadata["fields"]["born"]
    assert ada.text[start:end] == "1815-12-10"


def test_rendering_is_deterministic() -> None:
    row = {"name": "Ada", "born": "1815-12-10"}
    first, second = record_document(row), record_document(dict(row))
    assert (first.text, first.metadata["fields"]) == (second.text, second.metadata["fields"])
    assert first.id != second.id


def test_csv_refuses_rows_that_would_lose_a_value() -> None:
    with pytest.raises(ValueError, match="line 2: 3 cells under 2 columns"):
        CsvLoader().load(b"a,b\r\n1,2,3\r\n")
    with pytest.raises(ValueError, match="duplicate column names"):
        CsvLoader().load(b"a,a\n1,2\n")
    assert CsvLoader().load(b"") == []
    headerless = CsvLoader(fieldnames=["x", "y"]).load(b"1,2\n")
    assert headerless[0].metadata["row"] == {"x": "1", "y": "2"}
    assert headerless[0].metadata["line"] == 1


def test_tsv_is_csv_with_tabs() -> None:
    (doc,) = TsvLoader().load(b"name\tnote\nAda\thas, commas\n")
    assert doc.metadata["row"] == {"name": "Ada", "note": "has, commas"}


def test_nested_json_flattens_to_paths_that_round_trip() -> None:
    record = {
        "name": "Ada",
        "address": {"city": "London", "zip": None},
        "tags": ["math", "poetry"],
        "a.b": 1,
        "empty": [],
        "flag": True,
        "pad": " x",
        "blank": "",
    }
    doc = record_document(record)
    assert doc.text == "\n".join(
        [
            "name: Ada",
            "address.city: London",
            "address.zip: null",
            "tags[0]: math",
            "tags[1]: poetry",
            '["a.b"]: 1',
            "empty: []",
            "flag: true",
            'pad: " x"',
            'blank: ""',
        ]
    )
    for path in doc.metadata["fields"]:
        assert format_path(parse_path(path)) == path
    _cells_are_faithful(doc)


def test_record_paths_parse_and_expand() -> None:
    assert parse_path("$.tags[*]") == ["tags", None]
    assert parse_path('a["x.y"][2]') == ["a", "x.y", 2]
    assert parse_path("Birth date") == ["Birth date"]
    for bad in ("", "$", "a[", "a[x]", ".a", "a..b", '["open'):
        with pytest.raises(ValueError, match="not a record path"):
            parse_path(bad)
    record: dict[str, Any] = {"tags": ["math", "poetry"], "people": [{"n": "Ada"}, {"n": "Alan"}]}
    assert list(resolve_path(record, parse_path("tags[*]"))) == [
        (["tags", 0], "math"),
        (["tags", 1], "poetry"),
    ]
    assert [v for _, v in resolve_path(record, parse_path("people[*].n"))] == ["Ada", "Alan"]
    assert list(resolve_path(record, parse_path("tags[5]"))) == []


def test_json_arrays_objects_and_wrapped_records(tmp_path: Path) -> None:
    array = JsonLoader(tier=SourceTier.COMMUNITY).load(b'[{"id": 1}, 7]')
    assert [d.text for d in array] == ["id: 1", "value: 7"]
    assert {d.tier for d in array} == {SourceTier.COMMUNITY}
    (single,) = JsonLoader().load((BOM + '{"id": 1}').encode())
    assert single.metadata["row"] == {"id": 1}
    wrapped = b'{"data": {"items": [{"id": 1}, {"id": 2}]}, "meta": {}}'
    assert [d.metadata["row"]["id"] for d in JsonLoader(records="data.items").load(wrapped)] == [
        1,
        2,
    ]
    with pytest.raises(ValueError, match="does not name one array"):
        JsonLoader(records="meta").load(wrapped)


def test_jsonl_skips_blank_lines_and_names_the_bad_one() -> None:
    docs = JsonlLoader().load(b'{"id": 1}\n\n{"id": 2}\r\n')
    assert [(d.metadata["line"], d.metadata["row_index"]) for d in docs] == [(1, 0), (3, 1)]
    with pytest.raises(ValueError, match="line 2"):
        JsonlLoader().load(b'{"id": 1}\n{oops}\n')


def test_records_already_in_memory() -> None:
    docs = RecordsLoader(uri="db://people", tier=SourceTier.AUTHORITATIVE).load(
        iter([{"id": 1}, {"id": 2}])
    )
    assert [d.uri for d in docs] == ["db://people", "db://people"]
    assert docs[1].metadata["row_index"] == 1


def test_parquet_rows_become_structured_documents(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "people.parquet"
    pq.write_table(
        pa.table({"name": ["Ada", "Alan"], "born": [date(1815, 12, 10), date(1912, 6, 23)]}),
        path,
    )
    docs = ParquetLoader(tier=SourceTier.CURATED).load(path)
    assert docs[0].metadata["row"] == {"name": "Ada", "born": date(1815, 12, 10)}
    assert docs[0].text == "name: Ada\nborn: 1815-12-10"
    assert docs[1].uri == path.resolve().as_uri()
    assert [d.metadata["row"]["name"] for d in ParquetLoader().load(path.read_bytes())] == [
        "Ada",
        "Alan",
    ]
    assert list(ParquetLoader(columns=["name"]).load(path)[0].metadata["row"]) == ["name"]


def test_parquet_without_pyarrow_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(ImportError, match=r"odke\[parquet\]"):
        ParquetLoader().load(b"PAR1")


def test_a_mixed_directory_sets_modality_per_file(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("name\nAda\nAlan\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"name": "Grace"}\n', encoding="utf-8")
    (tmp_path / "c.md").write_text("# Notes\n\nProse.\n", encoding="utf-8")
    (tmp_path / "d.tsv").write_text("name\nKatherine\n", encoding="utf-8")
    (tmp_path / "e.json").write_text('[{"name": "Dorothy"}]', encoding="utf-8")
    docs = list(DirectoryLoader().load(tmp_path))
    assert [(d.modality, d.metadata.get("row", {}).get("name")) for d in docs] == [
        ("structured", "Ada"),
        ("structured", "Alan"),
        ("structured", "Grace"),
        ("unstructured", None),
        ("structured", "Katherine"),
        ("structured", "Dorothy"),
    ]


def test_every_structured_loader_satisfies_the_protocol() -> None:
    loaders = (
        CsvLoader(),
        TsvLoader(),
        JsonLoader(),
        JsonlLoader(),
        RecordsLoader(),
        ParquetLoader(),
    )
    for loader in loaders:
        assert isinstance(loader, Loader)
