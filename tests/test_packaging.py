"""What ships: the version a wheel reports is the version it was built as."""

from __future__ import annotations

import tomllib
from pathlib import Path

import openodke

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_reports_the_version_in_pyproject() -> None:
    # Two hand-written copies drift: bump one, forget the other, and PyPI gets a
    # wheel whose `odke --version` names a release it is not.
    with (ROOT / "pyproject.toml").open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert openodke.__version__ == declared
