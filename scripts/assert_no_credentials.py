"""Fail if a provider or database credential is visible to the test run.

The suite is meant to be free and offline. A key in the environment means a test
could quietly start spending money or writing to somebody's real graph, and the
run that discovers this should be the one that refuses to start.
"""

from __future__ import annotations

import os
import sys

SUSPECT = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "NEO4J_PASSWORD",
    "NEO4J_URI",
)

found = [name for name in SUSPECT if os.environ.get(name)]
if found:
    print(f"refusing to run with live credentials present: {', '.join(found)}", file=sys.stderr)
    print("unset them, or run in a clean shell.", file=sys.stderr)
    sys.exit(1)
