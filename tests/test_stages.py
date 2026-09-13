"""The thirteen-Protocol surface: every default is a no-op, and says so."""

from __future__ import annotations

from odke import (
    Chunk,
    Chunker,
    Constrainer,
    Corroborator,
    Document,
    Entity,
    Extractor,
    Fact,
    Grounder,
    GroundingVerdict,
    Inferrer,
    Loader,
    Normalizer,
    Ontology,
    Resolver,
    Router,
    Scorer,
    Sink,
    Validator,
    stages,
)

ONTOLOGY = Ontology(name="demo")
DOC = Document(id="d1", text="Ada Lovelace wrote the first algorithm.")
FACT = Fact(subject=Entity(key="p1", type="Person"), predicate="name", object_value="Ada")

THIRTEEN = (
    Loader,
    Chunker,
    Router,
    Extractor,
    Grounder,
    Normalizer,
    Resolver,
    Corroborator,
    Scorer,
    Validator,
    Sink,
    Constrainer,
    Inferrer,
)

# The extractor has no identity function, and a sink's no-op is "no sinks".
DEFAULTS = {
    Loader: stages.PassThroughLoader,
    Chunker: stages.PassThroughChunker,
    Router: stages.PassThroughRouter,
    Grounder: stages.PassThroughGrounder,
    Normalizer: stages.PassThroughNormalizer,
    Resolver: stages.PassThroughResolver,
    Corroborator: stages.PassThroughCorroborator,
    Scorer: stages.PassThroughScorer,
    Validator: stages.PassThroughValidator,
    Constrainer: stages.PassThroughConstrainer,
    Inferrer: stages.PassThroughInferrer,
}


def test_the_surface_is_thirteen_protocols() -> None:
    assert len(THIRTEEN) == 13
    assert len({p.__name__ for p in THIRTEEN}) == 13


def test_every_default_satisfies_its_protocol() -> None:
    for protocol, default in DEFAULTS.items():
        assert isinstance(default(), protocol), protocol.__name__


def test_the_loader_default_hands_documents_on_and_reads_nothing() -> None:
    loader = stages.PassThroughLoader()
    assert list(loader.load(DOC)) == [DOC]
    assert list(loader.load([DOC, DOC])) == [DOC, DOC]
    (from_text,) = loader.load("plain text")
    assert from_text.text == "plain text"


def test_the_chunker_default_is_one_chunk_per_document() -> None:
    """A chunker configured not to split is what document-level routing is."""
    (chunk,) = stages.PassThroughChunker().chunk(DOC)
    assert chunk == Chunk(doc_id="d1", start=0, end=len(DOC.text), text=DOC.text, index=0)
    assert DOC.text[chunk.start : chunk.end] == chunk.text


def test_the_fact_to_fact_defaults_return_the_same_object() -> None:
    assert stages.PassThroughGrounder().ground(FACT, DOC) is FACT
    assert FACT.verdict is GroundingVerdict.UNCHECKED
    assert stages.PassThroughNormalizer().normalize(FACT) is FACT
    assert stages.PassThroughScorer().score(FACT) is FACT


def test_the_batch_defaults_return_the_facts_untouched() -> None:
    facts = [FACT, FACT.model_copy(update={"object_value": "Ada L."})]
    resolved, links = stages.PassThroughResolver().resolve(facts, {})
    assert list(resolved) == facts
    assert list(links) == []
    assert list(stages.PassThroughCorroborator().corroborate(facts)) == facts
    assert all(f.support == 1 for f in facts)


def test_the_validator_default_accepts_everything() -> None:
    assert stages.PassThroughValidator().validate(FACT, ONTOLOGY).action == "accept"


def test_the_constrainer_default_emits_no_ddl() -> None:
    assert list(stages.PassThroughConstrainer().constrain(ONTOLOGY)) == []


def test_the_inferrer_default_proposes_nothing() -> None:
    proposed = stages.PassThroughInferrer().infer([DOC])
    assert proposed.types == {} and proposed.predicates == {}
    assert not proposed.inferred
