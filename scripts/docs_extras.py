"""What the docs site gives AI tools, written into the built site after `zensical build`.

- every page in the navigation as Markdown, at its path plus `.md`
  (`/quickstart/` is also `/quickstart.md`; the home page is `/index.md`);
- `/llms.txt`: the site's index in the llms.txt format (https://llmstxt.org);
- `/llms-full.txt`: every page in navigation order, in one file.

Usage: python scripts/docs_extras.py [site-dir]   (default: site)
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class _Config(yaml.SafeLoader):
    """mkdocs.yml with its `!!python/name:` tags read as nothing."""


_Config.add_multi_constructor("", lambda loader, suffix, node: None)
_Config.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def pages(nav: list, section: str = "") -> Iterator[tuple[str, str, str]]:
    """`(section, title, source file)` for every page, in navigation order."""
    for item in nav:
        for title, value in item.items():
            if isinstance(value, list):
                yield from pages(value, title)
            else:
                yield section, title, value


def summary(markdown: str) -> str:
    """The page's first sentence of prose, without link targets or code marks."""
    body = markdown.split("\n# ", 1)[-1] if not markdown.startswith("# ") else markdown
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block or block[0] in "#!<|`-*>:=" or block[0].isdigit():
            continue
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", " ".join(block.split()))
        text = text.replace("`", "").replace("**", "")
        sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        return sentence if len(sentence) <= 200 else sentence[:197].rstrip() + "..."
    return ""


def markdown_path(source: str) -> str:
    return source if source == "index.md" else source[: -len(".md")] + ".md"


def main(site: Path) -> None:
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=_Config)
    base = config["site_url"].rstrip("/")
    entries = list(pages(config["nav"]))

    description = " ".join(config["site_description"].split())
    index = [f"# {config['site_name']}", "", f"> {description}"]
    full = []
    current = None
    for section, title, source in entries:
        text = (DOCS / source).read_text(encoding="utf-8")
        target = site / markdown_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        if section != current:
            index += ["", f"## {section}", ""]
            current = section
        line = f"- [{title}]({base}/{markdown_path(source)})"
        about = summary(text)
        index.append(f"{line}: {about}" if about else line)
        page_url = base + "/" if source == "index.md" else f"{base}/{source[: -len('.md')]}/"
        full.append(f"<!-- {page_url} -->\n\n{text.strip()}\n")

    (site / "llms.txt").write_text("\n".join(index) + "\n", encoding="utf-8")
    (site / "llms-full.txt").write_text("\n---\n\n".join(full), encoding="utf-8")
    print(f"wrote {len(entries)} Markdown pages, llms.txt and llms-full.txt into {site}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "site"))
