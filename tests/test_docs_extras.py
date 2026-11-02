"""The site's Markdown copies and llms.txt (scripts/docs_extras.py).

A coding agent reads openodke.dev through these files, so every page in the
navigation must be there, at the URL the page's "Copy page" menu asks for.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "odke_script_docs_extras", ROOT / "scripts" / "docs_extras.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extras = _load()


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site")
    extras.main(out)
    return out


def _nav() -> list[tuple[str, str, str]]:
    import yaml

    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=extras._Config)
    return list(extras.pages(config["nav"]))


def test_every_page_in_the_navigation_is_published_as_markdown(site: Path) -> None:
    pages = _nav()
    assert len(pages) >= 20
    for _, _, source in pages:
        copy = site / extras.markdown_path(source)
        assert copy.read_text(encoding="utf-8") == (ROOT / "docs" / source).read_text(
            encoding="utf-8"
        ), source


def test_every_markdown_page_is_in_the_navigation() -> None:
    # A page left out of the navigation would be left out of llms.txt too.
    listed = {source for _, _, source in _nav()}
    on_disk = {page.name for page in (ROOT / "docs").glob("*.md")}
    assert on_disk == listed


def test_llms_txt_lists_every_page_once_at_its_published_url(site: Path) -> None:
    text = (site / "llms.txt").read_text(encoding="utf-8")
    assert text.startswith("# openodke\n\n> ")
    links = re.findall(r"^- \[[^\]]+\]\((\S+)\)", text, flags=re.MULTILINE)
    expected = [f"https://openodke.dev/{extras.markdown_path(s)}" for _, _, s in _nav()]
    assert links == expected
    sections = re.findall(r"^## (.+)$", text, flags=re.MULTILINE)
    assert sections == list(dict.fromkeys(section for section, _, _ in _nav()))


def test_llms_full_txt_holds_every_page(site: Path) -> None:
    full = (site / "llms-full.txt").read_text(encoding="utf-8")
    for _, _, source in _nav():
        first_heading = next(
            line
            for line in (ROOT / "docs" / source).read_text(encoding="utf-8").splitlines()
            if line.startswith("# ")
        )
        assert first_heading in full, source


@pytest.mark.parametrize(
    ("source", "published"),
    [("index.md", "index.md"), ("quickstart.md", "quickstart.md")],
)
def test_the_markdown_path_matches_the_page_menu(source: str, published: str) -> None:
    # docs/assets/openodke.js asks for "<page path>.md", and "index.md" at the root.
    assert extras.markdown_path(source) == published
    script = (ROOT / "docs" / "assets" / "openodke.js").read_text(encoding="utf-8")
    assert '|| "index") + ".md"' in script


def test_a_summary_is_the_first_sentence_of_prose() -> None:
    page = "# Title\n\n!!! note\n    skipped\n\nSee [the guide](a.md) for `x`. More here.\n"
    assert extras.summary(page) == "See the guide for x."
