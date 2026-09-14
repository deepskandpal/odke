"""JSON, JSONL, CSV, TSV and Parquet: one document per record.

The structured half of the corpus, and the half that should never cost a model
call. Each record becomes a `structured` document (see `odke.loaders.records`),
so the hybrid extractor sends it to the pattern extractor and past the model.

Standard library only, except Parquet, which needs pyarrow and says so. These
loaders decode with `utf-8-sig` by default: spreadsheet exports start with a
byte-order mark, and left in place it silently renames the first column. The
offsets a structured fact cites are into the rendering, not the file, so
stripping it costs nothing.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from odke._text import iter_lines
from odke.loaders.base import read_text
from odke.loaders.records import parse_path, record_document, resolve_path
from odke.stages import Source
from odke.types import Document, SourceTier


def _as_record(item: Any) -> Mapping[str, Any]:
    # A bare scalar in an array still becomes a document rather than vanishing.
    return item if isinstance(item, Mapping) else {"value": item}


class RecordsLoader:
    """Mappings already in memory — a database cursor, `DataFrame.to_dict("records")`."""

    def __init__(self, *, tier: SourceTier = SourceTier.UNVERIFIED, uri: str | None = None) -> None:
        self.tier = tier
        self.uri = uri

    def load(self, source: Iterable[Any]) -> list[Document]:
        return [
            record_document(_as_record(item), uri=self.uri, tier=self.tier, row_index=i)
            for i, item in enumerate(source)
        ]


class JsonLoader:
    """A JSON file. An array is one document per element, an object is one document.

    `records` names the array inside a wrapper object that holds the rows —
    `"data.items"` for `{"data": {"items": [...]}}`.
    """

    def __init__(
        self,
        *,
        records: str | None = None,
        tier: SourceTier = SourceTier.UNVERIFIED,
        encoding: str = "utf-8-sig",
    ) -> None:
        self.records = records
        self.tier = tier
        self.encoding = encoding

    def load(self, source: Source) -> list[Document]:
        text, uri = read_text(source, self.encoding)
        data = json.loads(text)
        if self.records is not None:
            found = list(resolve_path(data, parse_path(self.records)))
            if len(found) != 1 or not isinstance(found[0][1], list):
                where = uri or "the source"
                raise ValueError(f"{self.records!r} does not name one array in {where}")
            data = found[0][1]
        items = data if isinstance(data, list) else [data]
        return [
            record_document(_as_record(item), uri=uri, tier=self.tier, row_index=i)
            for i, item in enumerate(items)
        ]


class JsonlLoader:
    """JSON Lines: one document per non-blank line, which it remembers in `metadata["line"]`."""

    def __init__(
        self, *, tier: SourceTier = SourceTier.UNVERIFIED, encoding: str = "utf-8-sig"
    ) -> None:
        self.tier = tier
        self.encoding = encoding

    def load(self, source: Source) -> list[Document]:
        text, uri = read_text(source, self.encoding)
        docs: list[Document] = []
        for number, (start, end) in enumerate(iter_lines(text), 1):
            line = text[start:end]
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{uri or '<bytes>'} line {number}: {exc.msg}") from exc
            docs.append(
                record_document(
                    _as_record(item), uri=uri, tier=self.tier, row_index=len(docs), line=number
                )
            )
        return docs


class CsvLoader:
    """Delimited text with a header row: one document per data row.

    `metadata["line"]` is the physical line the row starts on, so a fact can be
    traced to the file even though its span points into the rendering. A
    duplicate column name or a row with more cells than the header is refused:
    both would lose a value silently. A short row simply lacks the missing keys.
    """

    delimiter = ","

    def __init__(
        self,
        *,
        delimiter: str | None = None,
        fieldnames: Sequence[str] | None = None,
        tier: SourceTier = SourceTier.UNVERIFIED,
        encoding: str = "utf-8-sig",
    ) -> None:
        if delimiter is not None:
            self.delimiter = delimiter
        self.fieldnames = fieldnames
        self.tier = tier
        self.encoding = encoding

    def load(self, source: Source) -> list[Document]:
        text, uri = read_text(source, self.encoding)
        where = uri or "<bytes>"
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=self.delimiter)
        header = list(self.fieldnames) if self.fieldnames is not None else next(reader, None)
        if header is None:
            return []
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValueError(f"{where}: duplicate column names {duplicates}")

        docs: list[Document] = []
        next_line = reader.line_num + 1
        for cells in reader:
            line, next_line = next_line, reader.line_num + 1
            if not any(cell.strip() for cell in cells):
                continue
            if len(cells) > len(header):
                raise ValueError(
                    f"{where} line {line}: {len(cells)} cells under {len(header)} columns"
                )
            record = dict(zip(header, cells, strict=False))
            docs.append(
                record_document(record, uri=uri, tier=self.tier, row_index=len(docs), line=line)
            )
        return docs


class TsvLoader(CsvLoader):
    """Tab-separated values."""

    delimiter = "\t"


class ParquetLoader:
    """Parquet, one document per row. Needs `pip install "odke[parquet]"`.

    pyarrow is imported when a file is read, not when this module is, so the
    base install can name the class and a directory walk only fails if it
    actually meets a Parquet file.
    """

    def __init__(
        self, *, tier: SourceTier = SourceTier.UNVERIFIED, columns: Sequence[str] | None = None
    ) -> None:
        self.tier = tier
        self.columns = list(columns) if columns is not None else None

    def load(self, source: Source) -> list[Document]:
        try:
            parquet: Any = importlib.import_module("pyarrow.parquet")
        except ImportError as exc:
            raise ImportError(
                'reading Parquet needs pyarrow. Run: pip install "odke[parquet]"'
            ) from exc
        if isinstance(source, bytes | bytearray | memoryview):
            table, uri = parquet.read_table(io.BytesIO(source), columns=self.columns), None
        elif isinstance(source, str | os.PathLike):
            path = Path(source)
            table, uri = parquet.read_table(path, columns=self.columns), path.resolve().as_uri()
        else:
            raise TypeError(f"expected a path or bytes, got {type(source).__name__}")
        return [
            record_document(row, uri=uri, tier=self.tier, row_index=i)
            for i, row in enumerate(table.to_pylist())
        ]


__all__ = [
    "CsvLoader",
    "JsonLoader",
    "JsonlLoader",
    "ParquetLoader",
    "RecordsLoader",
    "TsvLoader",
]
