"""Writing a file whole: a reader sees the old file or the new one, never part of either."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


def write_whole(target: Path, text: str) -> Path:
    """`text` in `target`, whole or not at all: written beside it, then renamed over it.

    `os.replace` is atomic within a directory, so a run killed while it writes
    leaves the file that was there rather than half of one (release audit S8),
    and a second writer or a reader at the same moment finds a whole file
    (S15). The file is opened as any other, so it takes the umask's
    permissions, as a file written in place would.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex[:12]}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
