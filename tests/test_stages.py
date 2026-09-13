"""The thirteen-Protocol surface: every default is a no-op, and says so."""

from __future__ import annotations

from odke import (
    Chunk,
    Chunker,
    Constrainer,
    Corroborator,
    Delegated,
    Document,
    Entity,
    Extractor,
    Fact,
    Grounder,
    GroundingVerdict,
    Inferrer,
    KnowledgeGraph,
    Loader,
    Normalizer,
    Ontology,
    PlatformProfile,
    Resolution,
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


def test_delegated_satisfies_every_stage_protocol() -> None:
    """One marker slots in wherever the platform already does the work."""
    marker = Delegated(to="neo4j-graphrag:FuzzyMatchResolver")
    for protocol in THIRTEEN:
        assert isinstance(marker, protocol), protocol.__name__
    assert repr(marker) == "Delegated(to='neo4j-graphrag:FuzzyMatchResolver')"


def test_delegated_is_a_pass_through_everywhere() -> None:
    marker = Delegated(to="platform")
    chunk = Chunk(doc_id="d1", start=0, end=4, text="Ada.", index=0)
    assert list(marker.load(DOC)) == [DOC]
    assert list(marker.chunk(DOC)) == list(stages.PassThroughChunker().chunk(DOC))
    assert marker.route(chunk).action == "extract"
    assert list(marker.extract(chunk, ONTOLOGY)) == []
    assert marker.ground(FACT, DOC) is FACT
    assert marker.normalize(FACT) is FACT
    assert list(marker.corroborate([FACT])) == [FACT]
    assert marker.score(FACT) is FACT
    assert marker.validate(FACT, ONTOLOGY).action == "accept"
    assert marker.write(KnowledgeGraph()) is None
    assert list(marker.constrain(ONTOLOGY)) == []
    assert marker.infer([DOC]).predicates == {}


def test_a_delegated_resolver_stamps_who_decided_the_key() -> None:
    """The stamp is what lets a platform's merge be read back and scored like ours."""
    to = "neo4j-graphrag:FuzzyMatchResolver"
    already = Entity(key="Q95", type="Company", resolution=Resolution(method="external_id"))
    fact = Fact(subject=Entity(key="acme", type="Company"), predicate="owns", object_entity=already)
    (resolved,), links = Delegated(to=to).resolve([fact], {})
    assert resolved.subject.resolution == Resolution(method="linker", linker=to)
    # An identity settled upstream keeps its own provenance.
    assert resolved.object_entity is not None
    assert resolved.object_entity.resolution == Resolution(method="external_id")
    assert list(links) == []


def test_delegated_verdicts_name_the_platform() -> None:
    marker = Delegated(to="platform")
    chunk = Chunk(doc_id="d1", start=0, end=4, text="Ada.", index=0)
    assert marker.route(chunk).reason == "delegated to platform"
    assert marker.validate(FACT, ONTOLOGY).reason == "delegated to platform"


def test_a_platform_profile_covers_nothing_unless_told() -> None:
    profile = PlatformProfile(name="neo4j-graphrag")
    assert (profile.resolves, profile.constrains, profile.prunes) == (False, False, False)
    assert PlatformProfile(name="x", resolves=True).resolves
