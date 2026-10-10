# Loading documents

A **loader** reads files into `Document`s and a **chunker** cuts each document
into `Chunk`s. Every span later in the pipeline is a pair of character offsets
into `Document.text` ([DECISIONS #3](decisions.md#3)). Both live in
`openodke.loaders` and `openodke.chunking`.

## Loaders

`load(source)` takes a path or bytes and returns documents. `JsonlLoader`
returns an iterator, reading a file a line at a time as its documents are
asked for, so a file of any size [streams](run.md#streaming).

| Loader | Suffix | One document per | `modality` | Extra |
|---|---|---|---|---|
| `TextLoader` | `.txt`, `.text` | file | `unstructured` | — |
| `MarkdownLoader` | `.md`, `.markdown` | file | `unstructured` | — |
| `HtmlLoader` | `.html`, `.htm` | file | `semi_structured` if it has a table or definition list, else `unstructured` | — |
| `PdfLoader` | `.pdf` | file, or page | `unstructured` | `pdf` |
| `DocxLoader` | `.docx` | file | `semi_structured` if it has a table, else `unstructured` | `docx` |
| `CsvLoader`, `TsvLoader` | `.csv`, `.tsv` | row | `structured` | — |
| `JsonLoader` | `.json` | array element | `structured` | — |
| `JsonlLoader` | `.jsonl`, `.ndjson` | line | `structured` | — |
| `ParquetLoader` | `.parquet` | row | `structured` | `parquet` |
| `RecordsLoader` | — | mapping in memory | `structured` | — |

| Option | Taken by | Default |
|---|---|---|
| `tier` | every loader | `unverified` |
| `encoding` | text, Markdown, HTML | `utf-8` |
| `encoding` | CSV, TSV, JSON, JSONL | `utf-8-sig`, which drops a byte-order mark |
| `modality` | Markdown, HTML, PDF, DOCX | as above |
| `records` | `JsonLoader` | the top level; `"data.items"` names a nested array |
| `strip_boilerplate` | `HtmlLoader` | `False`; `True` drops `nav` and `footer` |
| `per_page` | `PdfLoader` | `False` |

### Text and Markdown

The file is decoded from bytes, so CRLF stays and a span offset is a file
offset. `MarkdownLoader` records the headings in `metadata["headings"]`, and
`heading_path(doc, offset)` names the section an offset is in.

```python
from openodke.loaders import MarkdownLoader, heading_path

(guide,) = MarkdownLoader().load(b"# Guide\r\n\r\n## Install\r\n\r\nRun pip.\r\n")

assert guide.title == "Guide"
assert guide.text.index("Run pip.") == 25  # CRLF kept: a file offset
assert heading_path(guide, guide.text.index("Run pip.")) == ("Guide", "Install")
```

### Records

A record's `text` is one `path: value` line per leaf. `metadata["row"]` holds
the record, `metadata["fields"]` the offset of each value, and
`metadata["line"]` the line it starts on. A duplicate column name, or a row with
more cells than the header, is refused. `record_document(mapping)` turns any
mapping, such as a database row, into a document.

```python
from openodke.loaders import CsvLoader

(ada,) = CsvLoader(tier="curated").load(b"\xef\xbb\xbfname,born\nAda Lovelace,1815-12-10\n")

print(ada.text)
# name: Ada Lovelace
# born: 1815-12-10
assert ada.metadata["fields"] == {"name": [6, 18], "born": [25, 35]}
assert (ada.modality, ada.tier.value, ada.metadata["line"]) == ("structured", "curated", 2)
```

### HTML, PDF and Word

`Document.text` is the extracted text, shaped like Markdown: blank lines
between blocks, `#` headings, pipe tables. Spans index that text.
`source_locations(doc, start, end)` maps a range back to the HTML element, PDF
page or Word paragraph it came from. PDFs get no OCR.

```python
from openodke.loaders import HtmlLoader, source_locations

markup = b"""<html><head><title>Acme</title></head><body>
<h1>Acme &amp; Sons</h1>
<p>Acme was   founded in 1999.</p>
<table><tr><th>Founded</th><td>1999</td></tr><tr><th>HQ</th><td>Leeds</td></tr></table>
</body></html>"""
(page,) = HtmlLoader(tier="authoritative").load(markup)

print(page.text)
# # Acme & Sons
#
# Acme was founded in 1999.
#
# Founded: 1999
# HQ: Leeds
assert (page.title, page.modality) == ("Acme", "semi_structured")

start = page.text.index("founded in 1999")
(where,) = source_locations(page, start, start + len("founded in 1999"))
assert where["path"] == "/html[1]/body[1]/p[1]"
assert markup.decode()[where["source_start"] : where["source_end"]] == "founded in 1999"
```

### Directories

`DirectoryLoader(pattern="**/*")` walks a directory, or takes one file, and
hands each file to the loader for its suffix, in sorted order.

- A file no loader claims is skipped.
- A file whose extra is missing is skipped with a `MissingExtraWarning` naming
  the install command. `missing_extras="raise"` raises instead.
- `loaders={".suffix": loader}` replaces the suffix table.

## Chunking

`SentenceChunker(max_words=200, *, overlap=0)` packs whole sentences into
chunks of at most `max_words` words. Without a chunker, a document is one chunk.

- A chunk ends at a paragraph break, or a sentence break if that would leave it
  less than half full. It never splits a sentence
  ([DECISIONS #19](decisions.md#19)).
- `chunk.text == doc.text[chunk.start:chunk.end]`, always.
- `overlap` repeats that many sentences at the start of the next chunk.

```python
from openodke import Document, SentenceChunker

doc = Document(
    id="d1",
    text="Dr. Ada Lovelace was born in 1815. She wrote notes.\r\n\r\n"
    "Charles Babbage designed engines. He was born in 1791.",
)
chunks = list(SentenceChunker(max_words=12).chunk(doc))

for chunk in chunks:
    print(chunk.start, chunk.end, chunk.text)
# 0 51 Dr. Ada Lovelace was born in 1815. She wrote notes.
# 55 109 Charles Babbage designed engines. He was born in 1791.
assert all(doc.text[c.start : c.end] == c.text for c in chunks)
```

## Routing

A `Router` sees each chunk before extraction and returns a `RouteVerdict`. A
chunk routed `skip` or `defer` is counted in `kg.stats` and never extracted.

| Field | Values |
|---|---|
| `action` | `extract`, `skip` or `defer` |
| `label` | Your own word for the chunk's kind; the package ships no taxonomy |
| `scope` | `chunk` (the default), or `document`: a `skip` or `defer` then stops the rest of the document |
| `reason` | Optional text saying why |

## In a run config

In an [`odke run`](run.md) config, a loader is its lower-case format name and
the chunker is `sentence`. A bare path uses `stages.loader`, else the directory
loader.

```yaml
inputs:
  - path: corpus/register.csv
    loader: {use: csv, tier: curated}
  - corpus/notes
stages:
  chunker: {use: sentence, max_words: 120}
```
