#!/usr/bin/env python3
"""Assemble a pull request's evidence bundle: `evidence.json` and its recordings.

Every pull request into a release branch carries its evidence (DECISIONS #50):
what it fixes and why, the check on the base and on the head, the recordings
of the failure and of the fix, the change file by file, the review trail, the
build failures and what needs the maintainer. The maintainer's review desk
reads the bundle; nothing here posts anywhere.

    evidence.py init --title "..." --summary "..." --finding "release 1" --rule R4 \\
        --decision "#50" --risk numbers --base 311c54a --head HEAD
    evidence.py record --side base --commit 311c54a \\
        --run "the rule fails::uv run pytest tests/rules/test_r4_recorded_answers.py -q"
    evidence.py record --side head --here --run "the rule passes::uv run pytest tests/rules -q"
    evidence.py set base.verify='"10/10 passed"' --merge review.json
    evidence.py ci --pr 230
    evidence.py check

The bundle goes to `--out`, else `$ODKE_DESK/<branch>`, else
`$TMPDIR/desk/<branch>` (the branch's `/` becomes `-`). `set` and `--merge`
deep-merge JSON into the file: objects merge, lists and values replace. A
recording runs `scripts/record.py` in a temporary worktree of the commit it
records, unless `--here`; the commands are the same ones on base and head.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
FILE = "evidence.json"
RISK_TAGS = ("numbers", "tenants", "secrets", "statistics", "api", "docs")
OUTCOMES = ("fixed", "dropped as unproven")
SIDES = ("base", "head")


def skeleton() -> dict[str, Any]:
    """An empty bundle, in the shape the desk reads."""
    return {
        "branch": "",
        "pr": None,
        "title": "",
        "summary": "",
        "findings": [],
        "rules": [],
        "decisions": [],
        "risk_tags": [],
        "base": {"commit": "", "verify": "", "recordings": []},
        "head": {"commit": "", "verify": "", "recordings": []},
        "changes": [],
        "review": [],
        "build_failures": [],
        "needs_you": [],
        "ci": {},
        "merged": None,
    }


# --------------------------------------------------------------------------- #
# The file
# --------------------------------------------------------------------------- #


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def slug(branch: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "detached"


def bundle_dir(out: Path | None, branch: str) -> Path:
    if out is not None:
        return out
    root = os.environ.get("ODKE_DESK") or str(Path(tempfile.gettempdir()) / "desk")
    return Path(root) / slug(branch)


def load(directory: Path) -> dict[str, Any]:
    path = directory / FILE
    found = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return merge(skeleton(), found)


def save(directory: Path, data: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILE
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def merge(into: Mapping[str, Any], fragment: Mapping[str, Any]) -> dict[str, Any]:
    """`fragment` over `into`: objects merge key by key; lists and values replace."""
    out = dict(into)
    for key, value in fragment.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def assignment(text: str) -> dict[str, Any]:
    """`base.verify="10/10"` as `{"base": {"verify": "10/10"}}`; a value not JSON is text."""
    path, sep, raw = text.partition("=")
    if not sep or not path:
        raise ValueError(f"{text!r}: give KEY=VALUE, the key dotted, the value JSON")
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    for part in reversed(path.split(".")):
        value = {part: value}
    return value


# --------------------------------------------------------------------------- #
# The checks the desk relies on
# --------------------------------------------------------------------------- #


def problems(data: Mapping[str, Any], directory: Path | None = None) -> list[str]:
    """What is wrong with a bundle, in words; empty when the desk can read it."""
    found = []
    for key, empty in skeleton().items():
        if key not in data:
            found.append(f"{key}: missing")
        elif empty is not None and not isinstance(data[key], type(empty)):
            found.append(f"{key}: a {type(empty).__name__}, not {type(data[key]).__name__}")
    if found:
        return found
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", data["summary"].strip()) if s]
    if len(sentences) > 5:
        found.append(f"summary: {len(sentences)} sentences; five or fewer")
    found += [f"risk_tags: {t!r} is not one of {', '.join(RISK_TAGS)}"
              for t in data["risk_tags"] if t not in RISK_TAGS]  # fmt: skip
    for change in data["changes"]:
        found += [f"changes: {change.get('file')}: tag {t!r} is not a risk tag"
                  for t in change.get("tags", []) if t not in RISK_TAGS]  # fmt: skip
    for item in data["review"]:
        if item.get("outcome") not in OUTCOMES:
            found.append(f"review: {item.get('finding')!r}: outcome is one of {OUTCOMES}")
    for side in SIDES:
        for at in data[side]["recordings"]:
            missing = {"title", "command", "file", "exit"} - set(at)
            if missing:
                found.append(f"{side}.recordings: {at.get('file')}: missing {sorted(missing)}")
            elif directory is not None and not (directory / at["file"]).is_file():
                found.append(f"{side}.recordings: {at['file']} is not in {directory}")
    return found


# --------------------------------------------------------------------------- #
# Recordings
# --------------------------------------------------------------------------- #


def parse_run(text: str) -> tuple[str, str]:
    """`title::command`, or a command alone, which is then its own title."""
    title, sep, command = text.partition("::")
    return (title.strip(), command.strip()) if sep else (text.strip(), text.strip())


def record_side(
    directory: Path,
    side: str,
    runs: Sequence[str],
    *,
    commit: str | None,
    here: bool,
    quiet: bool = True,
) -> list[dict[str, Any]]:
    """Each run recorded on `commit`'s tree; appended to the side's recordings."""
    data = load(directory)
    resolved = git("rev-parse", commit or "HEAD")
    worktree: Path | None = None
    if here:
        if git("rev-parse", "HEAD") != resolved:
            raise ValueError(f"--here records the checkout, which is not at {commit}")
        tree = REPO
    else:
        worktree = Path(tempfile.mkdtemp(prefix=f"evidence-{side}-"))
        git("worktree", "add", "--detach", str(worktree), resolved)
        tree = worktree
    made = []
    try:
        start = len(data[side]["recordings"])
        for n, run in enumerate(runs, start=start + 1):
            title, command = parse_run(run)
            name = f"{side}-{n}.cast"
            argv = [sys.executable, str(REPO / "scripts" / "record.py"), "-o",
                    str(directory / name), "--title", title, "--cwd", str(tree)]  # fmt: skip
            if quiet:
                argv.append("--quiet")
            status = subprocess.run([*argv, "--", *shlex.split(command)]).returncode
            made.append({"title": title, "command": command, "file": name, "exit": status})
    finally:
        if worktree is not None:
            subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=REPO)
            shutil.rmtree(worktree, ignore_errors=True)
    data[side]["commit"] = resolved
    data[side]["recordings"] = [*data[side]["recordings"], *made]
    save(directory, data)
    return made


def ci_of(pr: int) -> dict[str, str]:
    """Each check's state on the pull request, from `gh pr checks`."""
    raw = subprocess.run(
        ["gh", "pr", "checks", str(pr), "--json", "name,state"],
        cwd=REPO, capture_output=True, text=True, check=False,
    ).stdout  # fmt: skip
    return {row["name"]: row["state"] for row in json.loads(raw or "[]")}


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, help="the bundle's directory")
    parser.add_argument("--branch", help="default: the checkout's branch")
    sub = parser.add_subparsers(dest="mode", required=True)

    init = sub.add_parser("init", help="start or update the bundle's what and why")
    init.add_argument("--title", default="")
    init.add_argument("--summary", default="", help="what and why, in five sentences or fewer")
    init.add_argument("--pr", type=int)
    init.add_argument("--finding", action="append", default=[], dest="findings")
    init.add_argument("--rule", action="append", default=[], dest="rules")
    init.add_argument("--decision", action="append", default=[], dest="decisions")
    init.add_argument("--risk", action="append", default=[], dest="risk_tags", choices=RISK_TAGS)
    init.add_argument("--base", help="the base commit")
    init.add_argument("--head", help="the head commit")

    put = sub.add_parser("set", help="merge KEY=JSON assignments and JSON fragment files")
    put.add_argument("assignments", nargs="*")
    put.add_argument("--merge", type=Path, action="append", default=[], dest="fragments")

    rec = sub.add_parser("record", help="record commands on the base or the head")
    rec.add_argument("--side", choices=SIDES, required=True)
    rec.add_argument("--commit", help="default: HEAD")
    rec.add_argument("--here", action="store_true", help="record in this checkout")
    rec.add_argument("--run", action="append", required=True, help='"title::command"')
    rec.add_argument("--live", action="store_true", help="show the output as it records")

    ci = sub.add_parser("ci", help="each check's state, from gh pr checks")
    ci.add_argument("--pr", type=int, help="default: the bundle's pr")

    sub.add_parser("check", help="say what the desk could not read")
    sub.add_parser("show", help="print the bundle")

    args = parser.parse_args(argv)
    branch = args.branch or git("rev-parse", "--abbrev-ref", "HEAD")
    directory = bundle_dir(args.out, branch)
    data = load(directory)
    data["branch"] = data["branch"] or branch

    try:
        if args.mode == "init":
            update: dict[str, Any] = {
                k: v for k, v in vars(args).items()
                if k in ("title", "summary", "pr", "findings", "rules", "decisions", "risk_tags")
                and v not in ("", [], None)
            }  # fmt: skip
            for side in SIDES:
                if getattr(args, side):
                    update[side] = {"commit": git("rev-parse", getattr(args, side))}
            data = merge(data, update)
        elif args.mode == "set":
            for path in args.fragments:
                data = merge(data, json.loads(path.read_text(encoding="utf-8")))
            for text in args.assignments:
                data = merge(data, assignment(text))
        elif args.mode == "record":
            save(directory, data)
            made = record_side(
                directory, args.side, args.run, commit=args.commit, here=args.here,
                quiet=not args.live,
            )  # fmt: skip
            for at in made:
                print(f"{at['file']}: exit {at['exit']}  {at['title']}")
            return 0
        elif args.mode == "ci":
            pr = args.pr or data["pr"]
            if pr is None:
                raise ValueError("no pull request: give --pr, or init --pr first")
            data["ci"] = ci_of(pr)
        elif args.mode == "check":
            found = problems(data, directory)
            for problem in found:
                print(problem)
            print(f"{directory / FILE}: " + (f"{len(found)} problem(s)" if found else "readable"))
            return 1 if found else 0
        elif args.mode == "show":
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return 0
    except (ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"evidence.py: {exc}", file=sys.stderr)
        return 2
    print(save(directory, data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
