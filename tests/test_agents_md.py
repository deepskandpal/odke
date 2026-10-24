"""AGENTS.md stays a short map, and everything it points at exists (DECISIONS #50).

A map that names a file that moved sends an agent to the wrong place, and one
that grows into a manual stops being read. So it stays under 120 lines, every
relative link resolves, and every path it names in backticks exists.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "AGENTS.md"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
CODE = re.compile(r"`([^`\n]+)`")
# A backticked word is a path when it has a slash, or a file name's suffix.
SUFFIXES = (".md", ".py", ".sh", ".toml", ".yml", ".yaml", ".cfg")


def _paths(text: str) -> list[str]:
    found = []
    for word in CODE.findall(text):
        if " " in word or any(c in word for c in "<>$={}@#"):
            continue
        if "/" in word or word.endswith(SUFFIXES):
            found.append(word)
    return found


def test_agents_md_is_a_short_map() -> None:
    lines = MAP.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 120, f"AGENTS.md has {len(lines)} lines: keep the map short"


def test_every_link_in_agents_md_resolves() -> None:
    text = MAP.read_text(encoding="utf-8")
    links = [link for link in LINK.findall(text) if "://" not in link]
    assert links, "AGENTS.md links nothing"
    missing = [link for link in links if not (ROOT / link.split("#")[0]).exists()]
    assert not missing, f"AGENTS.md links to what is not there: {missing}"


def test_every_path_agents_md_names_exists() -> None:
    named = _paths(MAP.read_text(encoding="utf-8"))
    assert "scripts/verify.sh" in named
    missing = [p for p in named if not any(ROOT.glob(p.rstrip("/")))]
    assert not missing, f"AGENTS.md names paths that do not exist: {missing}"


def test_agents_md_lists_each_rule_file() -> None:
    named = set(_paths(MAP.read_text(encoding="utf-8")))
    rules = {p.relative_to(ROOT).as_posix() for p in (ROOT / "tests" / "rules").glob("test_r*.py")}
    assert len(rules) == 6 and rules <= named, f"rules AGENTS.md does not list: {rules - named}"


def test_claude_md_is_the_map() -> None:
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8") == "@AGENTS.md\n"


def test_human_md_is_the_escalation_list() -> None:
    text = (ROOT / "HUMAN.md").read_text(encoding="utf-8")
    assert "## Open" in text and "## Done" in text
