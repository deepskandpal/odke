"""Every Python example in the documentation runs, against the code it documents.

The pages under `docs/` are read the way a reader reads them: top to bottom,
each page's ```python blocks in order and in one namespace, so a later block can
use what an earlier one built. A block directly under a `<!-- docs: no-run -->`
line is shown and not run, because it needs a live database or a model
provider. Everything else runs here, offline, on every push, so an example that
stops matching the API fails the build instead of misleading a reader.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
NO_RUN = "<!-- docs: no-run -->"
FENCE = re.compile(r"^```python[ \t]*\n(?P<code>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def blocks(page: Path) -> list[tuple[int, str, bool]]:
    """`(line, code, runs)` for every ```python block on a page, in order."""
    text = page.read_text(encoding="utf-8")
    found = []
    for match in FENCE.finditer(text):
        previous = text[: match.start()].rstrip("\n").rsplit("\n", 1)[-1]
        line = text.count("\n", 0, match.start("code")) + 1
        found.append((line, match.group("code"), previous.strip() != NO_RUN))
    return found


PAGES = sorted(page for page in DOCS.glob("*.md") if blocks(page))


def test_the_documentation_has_examples_to_run() -> None:
    # An empty parametrisation passes silently; this is what would notice.
    assert sum(runs for page in PAGES for _, _, runs in blocks(page)) >= 20


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_python_example_on_the_page_runs(page: Path) -> None:
    namespace: dict[str, object] = {"__name__": f"docs_{page.stem}"}
    for line, code, runs in blocks(page):
        if not runs:
            continue
        # dont_inherit: this module's `from __future__ import annotations` must
        # not turn the examples' annotations into strings a reader never wrote.
        compiled = compile(code, f"docs/{page.name}:{line}", "exec", dont_inherit=True)
        try:
            exec(compiled, namespace)
        except Exception as exc:
            pytest.fail(f"docs/{page.name}, example at line {line}: {type(exc).__name__}: {exc}")
