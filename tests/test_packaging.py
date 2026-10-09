"""What ships: the version a wheel reports, and the workflow that uploads it."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

import openodke

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_reports_the_version_in_pyproject() -> None:
    # Two hand-written copies drift: bump one, forget the other, and PyPI gets a
    # wheel whose `odke --version` names a release it is not.
    with (ROOT / "pyproject.toml").open("rb") as f:
        declared = tomllib.load(f)["project"]["version"]
    assert openodke.__version__ == declared


def test_every_action_in_the_release_workflow_is_pinned_to_a_commit() -> None:
    # The publish jobs can mint a PyPI token. A tag such as `@v4` can be moved to
    # new code by whoever controls the action; a commit SHA cannot.
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    used = [
        step["uses"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "uses" in step
    ]
    assert used
    unpinned = [u for u in used if not re.fullmatch(r"[^@]+@[0-9a-f]{40}", u)]
    assert unpinned == []
