"""`LLMExtractor(structured=False)`: the prompt alone holds the model to the JSON shape."""

from __future__ import annotations

from openodke import Chunk, Ontology
from openodke.extract.llm import LLMExtractor
from openodke.llm import ScriptedClient

ONTOLOGY = Ontology.model_validate(
    {
        "name": "t",
        "types": {"Film": {}, "Person": {}},
        "predicates": {"director": {"domain": ["Film"], "range": "Person"}},
    }
)
TEXT = "Bleach was directed by Noriyuki Abe."
REPLY = (
    '{"entities": [{"type": "Film", "name": "Bleach", "facts": [{"predicate": "director", '
    '"value": "Noriyuki Abe", "quote": "Bleach was directed by Noriyuki Abe.", "start": 0, '
    '"mention": "", "polarity": "asserted"}]}]}'
)


def test_no_schema_is_sent_and_the_reply_is_still_parsed() -> None:
    client = ScriptedClient([REPLY])
    extractor = LLMExtractor(client=client, structured=False)
    facts = extractor.extract(
        Chunk(doc_id="d1", index=0, start=0, end=len(TEXT), text=TEXT), ONTOLOGY
    )
    assert [(f.subject.label, f.predicate) for f in facts] == [("Bleach", "director")]
    assert client.calls[0][2] is None  # the schema argument


def test_the_default_still_sends_the_typed_schema() -> None:
    client = ScriptedClient([REPLY])
    LLMExtractor(client=client).extract(
        Chunk(doc_id="d1", index=0, start=0, end=len(TEXT), text=TEXT), ONTOLOGY
    )
    assert client.calls[0][2] is not None
