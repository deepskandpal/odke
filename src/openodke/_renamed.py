"""Old names that still work, and say so, until 1.0.0 (DECISIONS #24, #26).

Nothing public is removed before 1.0, so a rename keeps the old name working and
warns wherever it is used. A module serves its old names through
`module_getattr`, so the warning fires where an old name is imported or read,
not on every `import openodke`. The old name is the same object as the new one:
`isinstance`, subclassing and a pickle made under the old name keep working.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

REMOVED = "The old name is removed in 1.0.0 (DECISIONS #26)."


class Renamed(NamedTuple):
    """An old name: what it is now, what to write instead, and when it goes."""

    obj: Any
    use: str
    note: str = REMOVED


def deprecated(old: str, use: str, *, note: str = REMOVED, stacklevel: int = 2) -> None:
    """One `DeprecationWarning` for an old spelling, pointed at the caller's line."""
    warnings.warn(
        f"{old} is deprecated: use {use}. {note}", DeprecationWarning, stacklevel=stacklevel + 1
    )


def module_getattr(module: str, renamed: Mapping[str, Renamed]) -> Callable[[str], Any]:
    """A module-level `__getattr__` that serves each old name with a warning."""

    def __getattr__(name: str) -> Any:
        found = renamed.get(name)
        if found is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        # 2: past this function, to the line that imported or read the old name.
        deprecated(f"{module}.{name}", found.use, note=found.note, stacklevel=2)
        return found.obj

    return __getattr__
