"""Where a finished graph goes.

Only the JSONL sink is importable from the base install. The others each need a
driver, so they are imported from their own module behind an extra —
`odke.sinks.neo4j` needs `pip install odke[neo4j]` — rather than being re-exported
here, which would make `import odke.sinks` fail for anyone who installed the
base package.
"""

from odke.sinks.jsonl import JsonlSink

__all__ = ["JsonlSink"]
