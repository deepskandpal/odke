"""The ontology compiled into Neo4j's constraints, and checks for what it cannot enforce."""

from __future__ import annotations

import os
import re
import uuid
import warnings
from collections.abc import Iterable

import pytest

from odke import (
    Chunk,
    Constrainer,
    Document,
    DoubleStageWarning,
    Entity,
    EntityType,
    Evidence,
    Fact,
    KnowledgeGraph,
    Ontology,
    Pipeline,
    Predicate,
    Qualifier,
)
from odke.sinks.neo4j import (
    CHECK_MARKER,
    Neo4jConstrainer,
    Neo4jSink,
    cardinality_scope,
    check_target,
    is_check,
)
from test_neo4j_sink import FakeDriver


def _ontology() -> Ontology:
    return Ontology(
        name="demo",
        types={"Person": EntityType(name="Person"), "Company": EntityType(name="Company")},
        predicates={
            "employer": Predicate(name="employer", domain=("Person",), range="Company"),
            "uptime": Predicate(
                name="uptime",
                domain=("Company",),
                qualifiers={"percentile": Qualifier(identity=True), "start_time": Qualifier()},
            ),
            "product": Predicate(name="product", domain=("Company",), cardinality="multi"),
            "name": Predicate(name="name"),
        },
    )


def _schema() -> list[str]:
    return Neo4jConstrainer().schema(_ontology())


def _check(predicate: str, ontology: Ontology | None = None) -> str:
    checks = Neo4jConstrainer().checks(ontology or _ontology())
    return next(c for c in checks if check_target(c) == predicate)


# --------------------------------------------------------------------------- #
# What Neo4j enforces
# --------------------------------------------------------------------------- #


def test_every_entity_type_gets_a_uniqueness_constraint_on_key() -> None:
    schema = _schema()
    for name in ("Person", "Company"):
        assert (
            f"CREATE CONSTRAINT odke_key_{name} IF NOT EXISTS "
            f"FOR (n:`{name}`) REQUIRE n.key IS UNIQUE"
        ) in schema


def test_external_id_and_names_are_indexed_per_type() -> None:
    """Full-text is the index that reaches inside a list of aliases."""
    schema = _schema()
    assert (
        "CREATE INDEX odke_external_id_Person IF NOT EXISTS FOR (n:`Person`) ON (n.external_id)"
    ) in schema
    assert (
        "CREATE FULLTEXT INDEX odke_names_Person IF NOT EXISTS "
        "FOR (n:`Person`) ON EACH [n.label, n.aliases]"
    ) in schema


def test_facts_are_unique_per_signature_so_the_sinks_merges_are_safe() -> None:
    schema = _schema()
    assert (
        "CREATE CONSTRAINT odke_signature_employer IF NOT EXISTS "
        "FOR ()-[r:`employer`]-() REQUIRE r.signature IS UNIQUE"
    ) in schema
    assert (
        "CREATE CONSTRAINT odke_claim_signature IF NOT EXISTS "
        "FOR (c:`Claim`) REQUIRE c.signature IS UNIQUE"
    ) in schema
    assert "CREATE INDEX odke_entity_key IF NOT EXISTS FOR (n:`Entity`) ON (n.key)" in schema


def test_the_ddl_is_idempotent_stable_and_enterprise_free() -> None:
    schema = _schema()
    assert schema == _schema()
    assert len(schema) == 2 + 3 * 2 + 4
    for statement in schema:
        assert statement.startswith("CREATE ") and " IF NOT EXISTS " in statement
        # Existence, type and node-key constraints are Enterprise-only.
        assert "IS NOT NULL" not in statement
        assert "IS NODE KEY" not in statement and "IS KEY" not in statement
        assert "IS ::" not in statement and "IS TYPED" not in statement


def test_an_empty_ontology_compiles_only_what_the_sink_itself_owns() -> None:
    """The store enforces nothing it was not told about."""
    ddl = Neo4jConstrainer().constrain(Ontology())
    assert [s.split(" IF NOT EXISTS")[0] for s in ddl] == [
        "CREATE INDEX odke_entity_key",
        "CREATE CONSTRAINT odke_claim_signature",
    ]


def test_names_are_quoted_and_schema_names_stay_distinct() -> None:
    ontology = Ontology(
        types={"A-B": EntityType(name="A-B"), "A_B": EntityType(name="A_B")},
        predicates={"x`) DROP": Predicate(name="x`) DROP", cardinality="multi")},
    )
    schema = Neo4jConstrainer().schema(ontology)
    names = [re.search(r"(?:CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s) for s in schema]
    identifiers = [m.group(1) for m in names if m]
    assert len(identifiers) == len(schema) == len(set(identifiers))
    assert all(re.fullmatch(r"odke_\w+", n, flags=re.ASCII) for n in identifiers)
    assert any("FOR (n:`A-B`)" in s for s in schema)
    assert any("[r:`x``) DROP`]" in s for s in schema)


# --------------------------------------------------------------------------- #
# What it cannot — checks
# --------------------------------------------------------------------------- #


def test_single_cardinality_predicates_get_a_check_and_multi_ones_do_not() -> None:
    checks = Neo4jConstrainer().checks(_ontology())
    assert [check_target(c) for c in checks] == ["employer", "name", "uptime"]
    assert all(is_check(c) for c in checks)
    assert not any(is_check(s) for s in _schema())


def test_a_check_returns_violators_and_changes_nothing() -> None:
    check = _check("employer")
    assert check.startswith(f"{CHECK_MARKER}employer\nMATCH (s)-[r:`employer`]->(o)")
    assert "WHERE size(objects) > 1" in check
    assert "RETURN labels(s) AS labels, s.key AS subject, objects" in check
    for verb in ("CREATE", "MERGE", "SET ", "DELETE", "REMOVE"):
        assert verb not in check


def test_denials_and_expired_facts_are_not_violations() -> None:
    """'Not employed by X' is not a second employer, and a job that ended is history."""
    assert "WHERE r.polarity = 'asserted' AND r.valid_to IS NULL" in _check("employer")


def test_identity_qualifiers_scope_the_check() -> None:
    """Uptime at p50 and at p95 are two claims, so holding both is not a violation."""
    uptime = _check("uptime")
    assert "WITH s, [r.`percentile`] AS scope, collect(DISTINCT" in uptime
    assert "start_time" not in uptime
    assert "scope" not in _check("employer")


class _ScopedPredicate(Predicate):
    """What M1 may add. The constrainer reads it by name, so this stands in for it."""

    cardinality_scope: tuple[str, ...] = ()


def test_m1s_cardinality_scope_is_read_when_it_exists() -> None:
    predicate = _ScopedPredicate(
        name="ceo", qualifiers={"tier": Qualifier(identity=True)}, cardinality_scope=("region",)
    )
    assert cardinality_scope(predicate) == ("region", "tier")
    assert cardinality_scope(Predicate(name="plain")) == ()
    ontology = Ontology(predicates={"ceo": predicate})
    assert "[r.`region`, r.`tier`] AS scope" in _check("ceo", ontology)


def test_a_qualifier_named_like_provenance_is_checked_under_its_prefix() -> None:
    ontology = Ontology(
        predicates={"p": Predicate(name="p", qualifiers={"support": Qualifier(identity=True)})}
    )
    assert "[r.`qualifier_support`] AS scope" in _check("p", ontology)


def test_the_constrainer_is_a_constrainer() -> None:
    constrainer = Neo4jConstrainer()
    assert isinstance(constrainer, Constrainer)
    ddl = list(constrainer.constrain(_ontology()))
    assert ddl == _schema() + constrainer.checks(_ontology())


# --------------------------------------------------------------------------- #
# The sink applies it
# --------------------------------------------------------------------------- #


def test_bootstrap_runs_the_ddl_and_not_the_checks() -> None:
    driver = FakeDriver()
    sink = Neo4jSink(driver=driver)
    applied = sink.bootstrap(_ontology())
    assert applied == _schema()
    assert [cypher for _, cypher, _ in driver.calls] == _schema()
    assert sink.ontology == _ontology()


def test_bootstrap_twice_is_harmless() -> None:
    driver = FakeDriver()
    sink = Neo4jSink(driver=driver)
    sink.bootstrap(_ontology())
    sink.bootstrap(_ontology())
    sent = [cypher for _, cypher, _ in driver.calls]
    assert sent == _schema() * 2
    assert all(" IF NOT EXISTS " in s for s in sent)


def test_a_dry_run_returns_the_ddl_for_a_dba_and_runs_nothing() -> None:
    driver = FakeDriver()
    assert Neo4jSink(driver=driver).bootstrap(_ontology(), dry_run=True) == _schema()
    assert driver.calls == [] and driver.sessions == []


def test_check_returns_the_violators_by_predicate_from_read_transactions() -> None:
    violator = {"labels": ["Person", "Entity"], "subject": "p:ada", "objects": ["c:1", "c:2"]}
    driver = FakeDriver(answers={"[r:`employer`]": [violator]})
    found = Neo4jSink(driver=driver).check(_ontology())
    assert found == {"employer": [violator]}
    assert {mode for mode, _, _ in driver.calls} == {"read"}
    assert len(driver.calls) == len(Neo4jConstrainer().checks(_ontology()))


def test_the_bootstrap_accepts_any_constrainer() -> None:
    class _One:
        def constrain(self, ontology: Ontology) -> list[str]:
            return ["CREATE INDEX mine IF NOT EXISTS FOR (n:X) ON (n.y)", f"{CHECK_MARKER}p\nx"]

    driver = FakeDriver()
    assert Neo4jSink(driver=driver).bootstrap(Ontology(), constrainer=_One()) == [
        "CREATE INDEX mine IF NOT EXISTS FOR (n:X) ON (n.y)"
    ]


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


class _Extractor:
    def extract(self, chunk: Chunk, ontology: Ontology) -> Iterable[Fact]:
        return [Fact(subject=Entity(key="p:1", type="Person"), predicate="name", object_value="A")]


def test_the_pipeline_returns_the_neo4j_ddl_without_a_double_stage_warning() -> None:
    """The constrainer is the platform's other half, so the profile must not call it a rerun."""
    sink = Neo4jSink(driver=FakeDriver())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipeline = Pipeline(_ontology(), _Extractor(), constrainer=Neo4jConstrainer(), sinks=[sink])
        pipeline.run([Document(id="d1", text="A.")])
    assert list(pipeline.constraints()) == list(Neo4jConstrainer().constrain(_ontology()))


def test_a_constrainer_for_another_platform_still_warns() -> None:
    class _Shacl:
        platform = "rdf"

        def constrain(self, ontology: Ontology) -> list[str]:
            return []

    with pytest.warns(DoubleStageWarning, match="constrainer"):
        Pipeline(
            _ontology(), _Extractor(), constrainer=_Shacl(), sinks=[Neo4jSink(driver=FakeDriver())]
        )


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_bootstrap_and_check_against_a_live_neo4j() -> None:
    """A first run constrains an empty schema; the check finds a second employer."""
    pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company, employer = f"Person_{suffix}", f"Company_{suffix}", f"employer_{suffix}"
    ontology = Ontology(
        types={person: EntityType(name=person), company: EntityType(name=company)},
        predicates={employer: Predicate(name=employer, domain=(person,), range=company)},
    )
    ada = Entity(key=f"p:ada:{suffix}", type=person)
    graph = KnowledgeGraph(
        facts=tuple(
            Fact(
                subject=ada,
                predicate=employer,
                object_entity=Entity(key=f"c:{name}:{suffix}", type=company),
                evidence=(Evidence(doc_id="d1"),),
            )
            for name in ("acme", "globex")
        )
    )
    created = [
        re.search(r"CREATE (?:FULLTEXT )?(CONSTRAINT|INDEX) (\S+) IF NOT EXISTS", s)
        for s in Neo4jConstrainer().schema(ontology)
    ]
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    with Neo4jSink(os.environ["NEO4J_URI"], auth) as sink:
        driver = sink._driver
        try:
            assert sink.bootstrap(ontology) == sink.bootstrap(ontology)
            shown = {r["name"] for r in driver.execute_query("SHOW CONSTRAINTS YIELD name").records}
            assert {f"odke_key_{person}", f"odke_signature_{employer}"} <= shown
            sink.write(graph)
            sink.write(graph)
            (violation,) = sink.check(ontology)[employer]
            assert violation["subject"] == ada.key
            assert sorted(violation["objects"]) == [f"c:acme:{suffix}", f"c:globex:{suffix}"]
        finally:
            driver.execute_query(f"MATCH (n) WHERE n:`{person}` OR n:`{company}` DETACH DELETE n")
            for match in created:
                if match and suffix in match.group(2):
                    driver.execute_query(f"DROP {match.group(1)} {match.group(2)} IF EXISTS")
