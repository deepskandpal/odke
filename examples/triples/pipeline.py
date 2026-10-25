"""A stand-in for your extraction pipeline: what `odke eval pipeline` points at.

It "extracts" by looking up the hand-written rows in `triples.jsonl` for each
text it is given, so it needs no model. Yours calls whatever it calls; the
harness only sees what goes in and what comes out.

    python pipeline.py IN_DIR OUT_FILE   # the command mode: --cmd "... {in} {out}"
    extract(documents)                   # the callable mode: --run pipeline.py:extract
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROWS = Path(__file__).with_name("triples.jsonl")


def rows_for(names: Iterable[str]) -> list[dict[str, Any]]:
    """The rows whose `doc` is one of `names`: a text's file name without its suffix."""
    wanted = set(names)
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row["doc"] in wanted]


def extract(documents: Iterable[Any]) -> list[dict[str, Any]]:
    """The callable mode: `Document`s in, triples rows out."""
    return rows_for(Path(doc.id).stem for doc in documents)


def main(inbox: str, out: str) -> None:
    """The command mode: a folder of texts in, a triples file out."""
    rows = rows_for(path.stem for path in Path(inbox).glob("*.txt"))
    Path(out).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
