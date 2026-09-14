"""The reserved keys the M3 stages record their work under.

`Fact` has no free-form metadata slot, and adding one to a frozen type is the
migration DECISIONS #4 warns about. `Fact.qualifiers` is already an open mapping
and never enters `signature` unless a key is named in `identity_keys`, which a
namespaced `odke.` key never is. So each stage writes what it did under one
known key: nothing is overwritten, and a caller asking "why does this value look
like that?" queries a key rather than reading code.
"""

from __future__ import annotations

# On Fact.qualifiers: {field: (original spelling, ...)} for every value the
# normaliser rewrote. Union-merged by the corroborator, so all spellings survive.
SOURCE_FORM = "odke.source_form"

# On Entity.attributes: the comparison key for the entity's name — casefolded,
# accents and legal suffixes stripped. The label stays the display form.
NAME_KEY = "odke.name_key"

__all__ = ["NAME_KEY", "SOURCE_FORM"]
