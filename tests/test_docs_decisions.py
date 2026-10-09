"""The site's decisions page is an index of DECISIONS.md, one row per entry.

Each row carries the entry's number as its anchor, so `decisions.md#24` lands on
24, and links to the heading GitHub renders for that entry. A decision added to
the log without a row, or a heading reworded under a row, fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = "https://github.com/deepskandpal/odke/blob/main/DECISIONS.md"
HEADING = re.compile(r"^### (\w+)\. (.+)$", re.MULTILINE)
ROW = re.compile(
    r"^\| (\w+) \{#(\w+)\} \| \[[^\]]+\]\(" + re.escape(LOG) + r"#([^)\s]+)\)", re.MULTILINE
)


def github_anchor(number: str, title: str) -> str:
    """The id GitHub gives `### N. Title`: lower case, punctuation dropped, spaces to hyphens."""
    text = f"{number}. {title}".replace("`", "").lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def test_the_decisions_index_has_one_row_per_entry_linked_to_it() -> None:
    log = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    entries = {number: github_anchor(number, title) for number, title in HEADING.findall(log)}
    rows = ROW.findall(index)

    assert [shown for shown, anchor, _ in rows if shown != anchor] == []
    assert len(rows) == len({anchor for _, anchor, _ in rows})
    assert {anchor: link for _, anchor, link in rows} == entries
