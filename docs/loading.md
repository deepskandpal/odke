# Loading documents

A **loader** (`openodke.loaders`) reads files into `Document`s, and a
**chunker** (`openodke.chunking`) cuts each into `Chunk`s. Every span is a pair
of character offsets into `Document.text` ([DECISIONS #3](decisions.md#3)).

## Loaders

`load(source)` takes a path or bytes and returns documents. `JsonlLoader`
returns an iterator, so a file of any size [streams](run.md#streaming).

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
| `delimiter`, `fieldnames` | `CsvLoader`, `TsvLoader` | `,` or tab; the header row |
| `columns` | `ParquetLoader` | every column |
| `strip_boilerplate` | `HtmlLoader` | `False`; `True` drops `nav` and `footer` |
| `per_page`, `page_separator` | `PdfLoader` | `False`, `"\n\n"` |
| `pattern`, `missing_extras`, `encoding`, `relative_to` | `DirectoryLoader` | `"**/*"`, `"warn"`, each loader's own, the current directory |

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
more cells than the header, is refused. `record_document(mapping)` makes a
document of any mapping, such as a database row.

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
- `loaders={".suffix": loader}` replaces the suffix table (Python only).
- Each document's id is its file's path relative to `relative_to` (the current
  directory when left out), as `odke validate --texts` names it: `texts/a.txt`,
  or `corpus/register.csv#L2` for a record. A rerun over the same files cites
  the same documents, so a merging store counts each source once.

## Chunking

`SentenceChunker(max_words=200, *, overlap=0)` packs whole sentences into
chunks of at most `max_words` words; without a chunker, a document is one chunk.

- A chunk ends at a paragraph break, or a sentence break if that would leave it
  under half full, and never splits a sentence ([DECISIONS #19](decisions.md#19)).
- `chunk.text == doc.text[chunk.start:chunk.end]`, always.
- `overlap` repeats that many sentences at the next chunk's start.

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

A `Router` returns a `RouteVerdict` for each chunk before extraction; a chunk
routed `skip` or `defer` is counted in `kg.stats` and not extracted.

| Field | Values |
|---|---|
| `action` | `extract`, `skip` or `defer` |
| `label` | Your own name for the chunk's kind |
| `scope` | `chunk` (default), or `document`: a `skip` or `defer` stops the rest of the document |
| `reason` | Optional: why |

## In a run config

In an [`odke run`](run.md#inputs) config, a loader is its lower-case format
name with options as extra keys, and the chunker is `sentence`.

```yaml
inputs:
  - path: corpus/register.csv
    loader: {use: csv, tier: curated}
  - corpus/notes
stages:
  chunker: {use: sentence, max_words: 120}
```
