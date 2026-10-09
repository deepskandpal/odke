"""The 0.2 home of the gate, kept so old imports still work until 1.0.0.

`VerdictValidator` is now `openodke.gate.VerdictGate`: "Validator" names the
whole verification layer, and the gate is one stage of it (DECISIONS #26).
Reading the old name here warns and returns the new class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openodke._renamed import Renamed, module_getattr
from openodke.gate import VerdictGate

if TYPE_CHECKING:
    # What a type checker sees; at run time the name comes from `__getattr__`.
    VerdictValidator = VerdictGate

__getattr__ = module_getattr(
    __name__, {"VerdictValidator": Renamed(VerdictGate, "openodke.gate.VerdictGate")}
)

__all__ = ["VerdictValidator"]
