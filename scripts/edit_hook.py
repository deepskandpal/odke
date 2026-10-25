#!/usr/bin/env python3
"""After an agent edits a file under `src/`: ruff on that file, then the six rules.

Claude Code runs this after each Edit or Write (`.claude/settings.json`), with
the tool call as JSON on standard input. It is silent when both pass. When
either fails it prints the failures on standard error and exits 2, which hands
them to the agent while it is still in the file: a rule's failure message is
its fix instruction (DECISIONS #50). CI still runs everything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHOWN = 120


def edited(event: dict[str, object]) -> Path | None:
    """The file the tool call edited, relative to the repository, when it is under `src/`."""
    tool_input = event.get("tool_input")
    raw = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        relative = Path(raw).resolve().relative_to(REPO)
    except ValueError:
        return None
    return relative if relative.parts[:1] == ("src",) and relative.suffix == ".py" else None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    path = edited(event) if isinstance(event, dict) else None
    if path is None:
        return 0
    checks = [
        ["uv", "run", "--quiet", "ruff", "check", str(path)],
        ["uv", "run", "--quiet", "pytest", "tests/rules", "--tb=short"],
    ]
    failed = []
    for argv in checks:
        try:
            done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
        except FileNotFoundError:
            print("edit hook: uv is not on PATH; ruff and the rules did not run", file=sys.stderr)
            return 1
        if done.returncode != 0:
            lines = (done.stdout + done.stderr).rstrip().splitlines()
            failed.append(f"$ {' '.join(argv[3:])}\n" + "\n".join(lines[-SHOWN:]))
    if not failed:
        return 0
    print(f"After editing {path}:\n\n" + "\n\n".join(failed), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
