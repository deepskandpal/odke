"""The run manifest (#160): what a run was asked, what answered it, and what it made.

Every run here answers from recorded responses: the e2e example's for
`odke run`, the triples example's for the Validator, `odke validate` and
`odke ground`. The done-when is the replay: two runs of one manifest are the
same run, and `odke run --from-manifest` is how the second is made.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openodke.llm import ModelSpec
from openodke.llm.base import Completion, Message
from openodke.manifest import (
    REDACTED,
    ModelUse,
    Served,
    canonical,
    digest,
)


def test_no_secret_reaches_the_manifest_or_its_hash() -> None:
    def config(key: str) -> dict[str, Any]:
        return {
            "models": {
                "extract": {
                    "model": "gateway/large",
                    "base_url": f"https://ops:{key}@gateway.internal/v1",
                    "api_key_env": "GATEWAY_KEY",
                    "max_tokens": 512,
                    "extra": {"api_key": key, "Authorization": f"Bearer {key}", "region": "eu"},
                }
            },
            "store": {"uri": f"bolt://neo4j:{key}@db:7687", "password_env": "NEO4J_PASSWORD"},
            "token": key,
        }

    out = canonical(config("hunter2"))
    assert "hunter2" not in json.dumps(out)
    extract = out["models"]["extract"]
    assert extract["base_url"] == f"https://ops:{REDACTED}@gateway.internal/v1"
    assert extract["extra"] == {"api_key": REDACTED, "Authorization": REDACTED, "region": "eu"}
    # What only names where a secret is read is kept, and so is anything else.
    assert (extract["api_key_env"], extract["max_tokens"]) == ("GATEWAY_KEY", 512)
    assert out["store"] == {
        "uri": f"bolt://neo4j:{REDACTED}@db:7687",
        "password_env": "NEO4J_PASSWORD",
    }
    assert out["token"] == REDACTED
    assert digest(canonical(config("hunter2"))) == digest(canonical(config("s3cret")))


class _Dated:
    """A provider that answers an alias with the dated model it resolved it to."""

    def complete(
        self, messages: Sequence[Message], *, spec: ModelSpec, schema: Any = None
    ) -> Completion:
        return Completion(text='{"verdict": "supported"}', model="claude-large-20260501")


def test_the_pinned_id_that_answered_is_recorded_beside_the_alias_asked_for() -> None:
    served = Served()
    alias = ModelSpec(model="anthropic/claude-large")
    client = served.client("ground", alias, _Dated())
    for _ in range(2):
        client.complete([Message(content="Claim: …")], spec=alias)
    served.note("extract", ModelSpec(model="gpt-5.5"))
    assert served.models() == {
        "extract": ModelUse(model="openai/gpt-5.5"),
        "ground": ModelUse(
            model="anthropic/claude-large", served=("anthropic/claude-large-20260501",)
        ),
    }
