"""The README's quickstart runs, as written, from a clone."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"


def _section(title: str) -> str:
    text = README.read_text(encoding="utf-8")
    return text.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def test_the_python_quickstart_runs_offline_in_twenty_lines(
    example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (code,) = re.findall(r"```python\n(.*?)```", _section("Quickstart"), re.DOTALL)
    assert len(code.strip().splitlines()) <= 20
    monkeypatch.chdir(example.parent.parent)
    namespace: dict[str, object] = {}
    exec(compile(code, "README.md:quickstart", "exec", dont_inherit=True), namespace)
    manifest = json.loads(Path("out/manifest.json").read_text(encoding="utf-8"))
    assert manifest["facts"] > 0
    assert manifest["stats"]["refused"] == 1


def test_the_shell_quickstart_is_the_example_config_and_install_is_from_github() -> None:
    quickstart = _section("Quickstart")
    assert "odke run examples/e2e/odke.yaml" in quickstart
    assert (README.parent / "examples" / "e2e" / "odke.yaml").is_file()
    install = _section("Install")
    assert "pip install openodke\n" not in install
    assert '"openodke @ git+https://github.com/deepskandpal/odke"' in install
    assert "git+https://github.com/deepskandpal/odke" in install
    assert "pip install openodke" in install


def test_the_readme_says_the_numbers_are_not_a_benchmark() -> None:
    assert "not a benchmark" in _section("Does it work?")
