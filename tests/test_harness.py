"""The harness around the rules (DECISIONS #50): markers, recordings, evidence, the edit hook.

Each rule's known breaks are strict xfails that name what fixes them; a
recording is an asciicast v2 file and exits as its command did; the evidence
bundle is in the shape the review desk reads; and the edit hook acts on files
under `src/` alone.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"odke_script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# The markers
# --------------------------------------------------------------------------- #


def _xfails() -> tuple[list[tuple[str, ast.Call]], list[str]]:
    """Every `xfail(...)` call under tests/rules, and every bare `xfail` that is not called."""
    calls, bare = [], []
    for path in sorted((ROOT / "tests" / "rules").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = set()
        for node in ast.walk(tree):
            func = node.func if isinstance(node, ast.Call) else None
            if isinstance(func, ast.Attribute) and func.attr == "xfail":
                calls.append((f"{path.name}:{node.lineno}", node))
                called.add(id(func))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "xfail" and id(node) not in called:
                bare.append(f"{path.name}:{node.lineno}")
    return calls, bare


def test_a_rule_waits_only_on_a_strict_xfail_that_names_its_fix() -> None:
    """Fix PR 5 of the 1.0 audit removed the last marker; a later one is held to this too."""
    marks, bare = _xfails()
    assert not bare, f"xfail markers with no strict=True and no reason: {bare}"
    loose = []
    for where, call in marks:
        keywords = {k.arg: k.value for k in call.keywords}
        strict = keywords.get("strict")
        reason = keywords.get("reason")
        text = ast.unparse(reason) if reason is not None else ""
        if not (isinstance(strict, ast.Constant) and strict.value is True) or "fix PR" not in text:
            loose.append(where)
    assert not loose, f"xfail markers that are not strict or name no fix PR: {loose}"


def test_xfail_is_strict_for_the_whole_suite() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["pytest"]["ini_options"]["xfail_strict"] is True


# --------------------------------------------------------------------------- #
# record.py
# --------------------------------------------------------------------------- #


def test_a_recording_is_asciicast_v2_and_exits_as_its_command(tmp_path: Path) -> None:
    out = tmp_path / "run.cast"
    program = "import sys, time; print('first'); time.sleep(0.05); print('second'); sys.exit(3)"
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "record.py"), "-o", str(out), "--title", "t", "--quiet",
         "--width", "80", "--height", "20", "--", sys.executable, "-c", program],
        capture_output=True, text=True,
    )  # fmt: skip
    assert done.returncode == 3 and done.stdout == ""
    header, *lines = out.read_text(encoding="utf-8").splitlines()
    meta = json.loads(header)
    assert (meta["version"], meta["width"], meta["height"], meta["title"]) == (2, 80, 20, "t")
    assert sys.executable in meta["command"]
    events = [json.loads(line) for line in lines]
    assert all(kind == "o" for _, kind, _ in events)
    times = [at for at, _, _ in events]
    assert times == sorted(times)
    text = "".join(chunk for _, _, chunk in events)
    assert "first" in text and "second" in text


def test_a_command_that_cannot_run_exits_as_a_shell_would(tmp_path: Path) -> None:
    record = _load("record")
    assert record.main(["-o", str(tmp_path / "x.cast"), "--", "odke-no-such-program"]) == 127
    script = tmp_path / "not-executable.sh"
    script.write_text("echo hi\n")
    assert record.main(["-o", str(tmp_path / "y.cast"), "--", str(script)]) == 126


# --------------------------------------------------------------------------- #
# evidence.py
# --------------------------------------------------------------------------- #


def test_the_evidence_bundle_has_the_desks_shape(tmp_path: Path) -> None:
    evidence = _load("evidence")
    out = tmp_path / "desk"
    common = ["--out", str(out), "--branch", "fix/example"]
    summary = "Recorded answers get their own cache. A rerun asks the model."
    what = ["--title", "t", "--summary", summary, "--finding", "release 1", "--rule", "R4"]
    assert evidence.main([*common, "init", *what, "--risk", "numbers"]) == 0
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"review": [{"finding": "x", "outcome": "fixed", "commit": "a"}]}))
    assert evidence.main([*common, "set", 'base.verify="10/10"', "--merge", str(review)]) == 0
    data = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    assert set(data) == set(evidence.skeleton())
    assert (data["branch"], data["rules"], data["base"]["verify"]) == (
        "fix/example",
        ["R4"],
        "10/10",
    )
    assert data["review"][0]["outcome"] == "fixed"
    assert evidence.problems(data, out) == []
    assert evidence.main([*common, "check"]) == 0

    bad = {**data, "summary": "One. Two. Three. Four. Five. Six.", "risk_tags": ["speed"]}
    bad["review"] = [{"finding": "y", "outcome": "ignored"}]
    found = evidence.problems(bad)
    assert len(found) == 3 and any("five or fewer" in p for p in found)


def test_a_base_whose_recordings_all_pass_shows_no_failure() -> None:
    evidence = _load("evidence")
    data = evidence.skeleton()
    data["base"]["recordings"] = [{"title": "t", "command": "c", "file": "f", "exit": 0}]
    assert any("none shows the failure" in p for p in evidence.problems(data))
    data["base"]["recordings"][0]["exit"] = 1
    assert evidence.problems(data) == []
    data["base"]["commit"] = data["head"]["commit"] = "abc"
    assert any("is the head's" in p for p in evidence.problems(data))


def test_a_base_recording_refuses_the_head(tmp_path: Path) -> None:
    evidence = _load("evidence")
    common = ["--out", str(tmp_path), "--branch", "fix/example"]
    assert evidence.main([*common, "init", "--head", "HEAD"]) == 0
    assert evidence.main([*common, "record", "--side", "base", "--run", "true"]) == 2


def test_a_recording_lands_in_the_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _load("evidence")
    # `--here` refuses a checkout with uncommitted changes; this test runs in one too.
    git = evidence.git
    monkeypatch.setattr(evidence, "git", lambda *a, **k: "" if a[0] == "status" else git(*a, **k))
    out = tmp_path / "desk"
    run = f"says hello::{sys.executable} -c \"print('hello')\""
    assert (
        evidence.main(["--out", str(out), "record", "--side", "head", "--here", "--run", run]) == 0
    )
    data = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    (recording,) = data["head"]["recordings"]
    assert (recording["title"], recording["file"], recording["exit"]) == (
        "says hello",
        "head-1.cast",
        0,
    )
    assert "hello" in (out / "head-1.cast").read_text(encoding="utf-8")
    assert evidence.problems(data, out) == []


def test_the_bundle_goes_where_the_desk_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    evidence = _load("evidence")
    monkeypatch.setenv("ODKE_DESK", str(tmp_path))
    assert evidence.bundle_dir(None, "fix/a b") == tmp_path / "fix-a-b"
    assert evidence.bundle_dir(tmp_path / "x", "fix/a") == tmp_path / "x"


# --------------------------------------------------------------------------- #
# The edit hook
# --------------------------------------------------------------------------- #


def test_the_edit_hook_runs_for_a_python_file_under_src_alone() -> None:
    hook = _load("edit_hook")
    under = {"tool_input": {"file_path": str(ROOT / "src" / "openodke" / "tenants.py")}}
    assert hook.edited(under) == Path("src/openodke/tenants.py")
    for elsewhere in (str(ROOT / "README.md"), str(ROOT / "tests" / "test_cli.py"), "/etc/hosts"):
        assert hook.edited({"tool_input": {"file_path": elsewhere}}) is None
    assert hook.edited({"tool_input": {}}) is None


def test_the_edit_hook_is_wired_after_edits_and_writes() -> None:
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    (entry,) = settings["hooks"]["PostToolUse"]
    assert {"Edit", "Write"} <= set(entry["matcher"].split("|"))
    (command,) = entry["hooks"]
    assert "scripts/edit_hook.py" in command["command"]
