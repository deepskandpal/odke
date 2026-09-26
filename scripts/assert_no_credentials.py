"""Fail if a provider or database credential is visible to the test run.

The suite is meant to be free and offline. A key in the environment means a test
could quietly start spending money or writing to somebody's real graph, and the
run that discovers this should be the one that refuses to start.
"""

from __future__ import annotations

import os
import sys

# The provider variables `openodke.llm.providers` reads, plus the database's. Not
# AWS_*, HF_TOKEN or GOOGLE_APPLICATION_CREDENTIALS: those are a developer's
# ambient credentials for other work, and refusing to run because they exist
# would be this script overreaching.
SUSPECT = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "XAI_API_KEY",
    "PERPLEXITYAI_API_KEY",
    "FIREWORKS_AI_API_KEY",
    "CEREBRAS_API_KEY",
    "NEO4J_PASSWORD",
    "NEO4J_URI",
)

found = [name for name in SUSPECT if os.environ.get(name)]
if found:
    print(f"refusing to run with live credentials present: {', '.join(found)}", file=sys.stderr)
    print("unset them, or run in a clean shell.", file=sys.stderr)
    sys.exit(1)
