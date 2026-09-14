"""Where a finished graph goes.

Only the JSONL sink is importable from the base install. The others each need a
driver, so they are imported from their own module behind an extra —
`openodke.sinks.neo4j` needs `pip install openodke[neo4j]` — rather than being re-exported
here, which would make `import openodke.sinks` fail for anyone who installed the
base package.
"""

from openodke.sinks.jsonl import JsonlSink

__all__ = ["JsonlSink"]
