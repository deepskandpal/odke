#!/usr/bin/env python3
"""Record a command in a pseudo-terminal, as an asciicast v2 file.

    python scripts/record.py -o out.cast -- uv run pytest tests/rules/test_r4_recorded_answers.py -q

openodke has no screen, so the video of a failure is a terminal recording
(DECISIONS #50): what the command printed and when, a few KB, which
asciinema-player replays and lets a reviewer pause, scrub and copy from. The
command runs in a pseudo-terminal, so it colours and wraps as it would for a
person. Its output is shown live too, unless `--quiet`.

The file is asciicast v2: one JSON header line, then one `[seconds, "o", text]`
line per chunk of output. The script exits with the command's exit status, so
a recording of a failing test fails the same way.

Standard library only, and POSIX only: `pty` has no Windows implementation.
"""

from __future__ import annotations

import argparse
import codecs
import fcntl
import json
import os
import pty
import select
import shlex
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path


def record(
    command: list[str],
    out: Path,
    *,
    title: str | None = None,
    width: int = 120,
    height: int = 32,
    quiet: bool = False,
    cwd: Path | None = None,
) -> int:
    """Run `command` in a pseudo-terminal of `width` × `height`, write `out`, return its status."""
    env = {**os.environ, "TERM": os.environ.get("TERM", "xterm-256color")}
    env.update(COLUMNS=str(width), LINES=str(height))
    main, child = pty.openpty()
    started = time.time()
    try:
        fcntl.ioctl(child, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        process = subprocess.Popen(
            command, stdin=child, stdout=child, stderr=child, cwd=cwd, env=env, close_fds=True
        )
    except OSError:
        os.close(main)
        os.close(child)
        raise
    os.close(child)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    events: list[tuple[float, str]] = []
    clock = time.monotonic()
    while True:
        ready, _, _ = select.select([main], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(main, 65536)
            except OSError:  # EIO: the child closed its end
                chunk = b""
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                events.append((round(time.monotonic() - clock, 6), text))
                if not quiet:
                    sys.stdout.write(text)
                    sys.stdout.flush()
        elif process.poll() is not None:
            # Exited, and nothing left to read.
            break
    tail = decoder.decode(b"", final=True)
    if tail:
        events.append((round(time.monotonic() - clock, 6), tail))
    os.close(main)
    status = process.wait()
    header = {
        "version": 2,
        "width": width,
        "height": height,
        "timestamp": int(started),
        "command": shlex.join(command),
        "env": {"SHELL": os.environ.get("SHELL", "/bin/sh"), "TERM": env["TERM"]},
    }
    if title:
        header["title"] = title
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
        for at, text in events:
            fh.write(json.dumps([at, "o", text], ensure_ascii=False) + "\n")
    # A signal is the shell's convention: 128 + its number.
    return status if status >= 0 else 128 - status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--out", type=Path, required=True, help="the .cast file to write")
    parser.add_argument("--title", help="the recording's title, shown by the player")
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--cwd", type=Path, help="run the command here")
    parser.add_argument("--quiet", action="store_true", help="record without showing it live")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    try:
        return record(
            command,
            args.out,
            title=args.title,
            width=args.width,
            height=args.height,
            quiet=args.quiet,
            cwd=args.cwd,
        )
    except FileNotFoundError as exc:
        print(f"record.py: cannot run {command[0]}: {exc}", file=sys.stderr)
        return 127
    except PermissionError as exc:
        print(f"record.py: cannot run {command[0]}: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    sys.exit(main())
