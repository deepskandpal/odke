"""The bulk sinks: `Neo4jSink`'s plan as a replayable script and as neo4j-admin CSV.

No database and no neo4j-admin run here. What both sinks own is that they lay
out exactly the rows `plan()` produces, under the same keys, in a form the
offline tool accepts — and that is checkable from the files. The live test
replays the script against a real server when `NEO4J_URI` is set.
"""

from __future__ import annotations

import csv
import importlib
import io
import os
import sys
import uuid
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest

from openodke import (
    Entity,
    EntityLink,
    EntityType,
    Fact,
    KnowledgeGraph,
    LinkKind,
    Ontology,
    Polarity,
    Predicate,
    Sink,
)
from openodke.eval.sinks import assert_idempotent
from openodke.sinks.bulk import CypherFileSink, Neo4jAdminCsvSink, Table, cypher_literal
from openodke.sinks.neo4j import Neo4jConstrainer, Neo4jSink, plan, signature_of
from test_neo4j_sink import _graph, live_graph

# --------------------------------------------------------------------------- #
# Cypher literals
# --------------------------------------------------------------------------- #


def test_strings_are_escaped_so_no_value_can_end_a_statement_or_a_literal() -> None:
    assert cypher_literal("it's") == "'it\\'s'"
    assert cypher_literal("a\\b") == "'a\\\\b'"
    assert cypher_literal("x;\n\ny") == "'x;\\n\\ny'"
    assert cypher_literal("\x00\t\r") == "'\\u0000\\t\\r'"
    assert "\n" not in cypher_literal("line\nbreak ")


def test_values_are_typed_the_way_the_driver_would_send_them() -> None:
    assert cypher_literal(None) == "null"
    assert cypher_literal(True) == "true"
    assert cypher_literal(3) == "3"
    assert cypher_literal(2.5) == "2.5"
    assert cypher_literal(1e20) == "1e20"
    assert cypher_literal(float("nan")) == "0.0 / 0.0"
    assert cypher_literal(float("-inf")) == "-1.0 / 0.0"
    aware = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
    assert cypher_literal(aware) == "datetime('2026-09-01T12:30:00+00:00')"
    assert cypher_literal(datetime(2026, 9, 1)) == "localdatetime('2026-09-01T00:00:00')"
    assert cypher_literal(date(2026, 9, 1)) == "date('2026-09-01')"
    assert cypher_literal(time(9, 5)) == "localtime('09:05:00')"
    ist = timezone(timedelta(hours=5, minutes=30))
    assert cypher_literal(time(9, 5, tzinfo=ist)) == "time('09:05:00+05:30')"
    assert cypher_literal({"key": "k", "odd name": [1, 2], "a`b": None}) == (
        "{key: 'k', `odd name`: [1, 2], `a``b`: null}"
    )
    with pytest.raises(TypeError, match="no Cypher literal"):
        cypher_literal(object())


# --------------------------------------------------------------------------- #
# CypherFileSink
# --------------------------------------------------------------------------- #


def _script_statements(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    code = "\n".join(line for line in text.splitlines() if not line.startswith("//"))
    chunks = [c.strip() for c in code.split(";\n\n")]
    return [c.removesuffix(";") for c in chunks if c]


def test_the_script_is_the_plan_with_each_batch_inlined(tmp_path: Path) -> None:
    graph = _graph()
    path = tmp_path / "out" / "graph.cypher"
    CypherFileSink(path).write(graph)
    statements = _script_statements(path)
    planned = plan(graph)
    assert len(statements) == len(planned)
    for text, st in zip(statements, planned, strict=True):
        head, body = text.split("\n] AS row\n", 1)
        # Everything after the UNWIND is the driver's statement, byte for byte.
        assert body == st.cypher.removeprefix("UNWIND $rows AS row\n")
        assert head.startswith("UNWIND [\n")
        assert [line.strip().removesuffix(",") for line in head.splitlines()[1:]] == [
            cypher_literal(row) for row in st.rows
        ]
        assert "$" not in head.splitlines()[0]


def test_the_script_batches_like_the_driver(tmp_path: Path) -> None:
    acme = Entity(key="c:acme", type="Company")
    facts = tuple(Fact(subject=acme, predicate="product", object_value=f"p{i}") for i in range(5))
    path = tmp_path / "g.cypher"
    CypherFileSink(path, batch_size=2).write(KnowledgeGraph(facts=facts))
    claims = [s for s in _script_statements(path) if "MERGE (c:`Claim`" in s]
    assert [s.count("predicate: 'product'") for s in claims] == [2, 2, 1]
    with pytest.raises(ValueError, match="batch_size"):
        CypherFileSink(path, batch_size=0)


def test_with_an_ontology_the_script_opens_with_the_ddl_and_never_runs_checks(
    tmp_path: Path,
) -> None:
    ontology = Ontology(
        types={"Person": EntityType(name="Person"), "Company": EntityType(name="Company")},
        predicates={"employer": Predicate(name="employer", domain=("Person",), range="Company")},
    )
    path = tmp_path / "g.cypher"
    CypherFileSink(path, ontology=ontology).write(_graph())
    statements = _script_statements(path)
    ddl = Neo4jConstrainer().schema(ontology)
    assert statements[: len(ddl)] == ddl
    assert all(s.startswith("UNWIND [") for s in statements[len(ddl) :])
    assert "odke:check" not in path.read_text(encoding="utf-8")


def test_a_value_with_semicolons_and_newlines_never_splits_a_statement(tmp_path: Path) -> None:
    ada = Entity(key="p:ada", type="Person", label="x;\n\nMATCH (n) DETACH DELETE n;\n\n")
    path = tmp_path / "g.cypher"
    CypherFileSink(path).write(KnowledgeGraph(entities=(ada,)))
    (statement,) = _script_statements(path)
    assert "DETACH DELETE n;\\n\\n" in statement


def test_the_cypher_file_sink_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "g.cypher"
    sink = CypherFileSink(path)
    assert isinstance(sink, Sink)

    def counts() -> dict[str, int]:
        text = path.read_text(encoding="utf-8")
        return {
            "statements": len(_script_statements(path)),
            "lines": text.count("\n"),
            "bytes": len(text.encode("utf-8")),
        }

    graph = _graph()
    assert_idempotent(sink, graph, counts)
    before = path.read_bytes()
    sink.write(graph)
    assert path.read_bytes() == before


def test_the_bulk_sinks_declare_the_neo4j_platform() -> None:
    assert CypherFileSink("x.cypher").profile == Neo4jSink.profile
    assert Neo4jAdminCsvSink("x").profile == Neo4jSink.profile


def test_the_bulk_sinks_need_no_driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Both are on the base install: the driver is never imported."""
    monkeypatch.setitem(sys.modules, "neo4j", None)
    monkeypatch.delitem(sys.modules, "openodke.sinks.bulk")
    module = importlib.import_module("openodke.sinks.bulk")
    module.CypherFileSink(tmp_path / "g.cypher").write(_graph())
    module.Neo4jAdminCsvSink(tmp_path / "csv").write(_graph())
    assert (tmp_path / "csv" / "import.args").is_file()


# --------------------------------------------------------------------------- #
# Neo4jAdminCsvSink
# --------------------------------------------------------------------------- #


def _read(directory: Path) -> dict[str, list[dict[str, str]]]:
    """Every CSV under `directory` as rows keyed by header cell, the way neo4j-admin splits them."""
    out = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            out[path.name] = list(csv.DictReader(fh))
    return out


def _headers(directory: Path) -> dict[str, list[str]]:
    out = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            out[path.name] = next(csv.reader(fh))
    return out


def _written_csv(tmp_path: Path, graph: KnowledgeGraph | None = None, **kw: object) -> Path:
    directory = tmp_path / "import"
    # `is None`, not `or`: a graph with entities and no facts has len() 0.
    graph = _graph() if graph is None else graph
    Neo4jAdminCsvSink(directory, **kw).write(graph)  # type: ignore[arg-type]
    return directory


def test_the_layout_is_one_file_per_plan_group(tmp_path: Path) -> None:
    tables = Neo4jAdminCsvSink(tmp_path).tables(_graph())
    # A hashed suffix keeps `A-B` and `A_B` apart; the readable part is what matters here.
    assert [(t.kind, t.file.rsplit("_", 1)[0], len(t.rows)) for t in tables] == [
        ("nodes", "odke_nodes", 2),
        ("nodes", "odke_nodes", 1),
        ("relationships", "odke_edges_Person_employer_Company", 1),
        ("nodes", "odke_claims_Company_sells", 1),
        ("relationships", "odke_claim_edges_Company_sells", 1),
        ("nodes", "odke_claims_Company_uptime", 2),
        ("relationships", "odke_claim_edges_Company_uptime", 2),
        ("nodes", "odke_claims_Person_name", 1),
        ("relationships", "odke_claim_edges_Person_name", 1),
        ("relationships", "odke_links_DIFFERENT_Company_Company", 1),
        ("relationships", "odke_links_SAME_AS_Person_Person", 1),
        ("relationships", "odke_links_SIMILAR_Company_Company", 1),
    ]
    assert [t.file for t in tables[:2]] == ["odke_nodes_Company.csv", "odke_nodes_Person.csv"]
    assert all(isinstance(t, Table) for t in tables)
    # Distinct names for every group, and plain file names neo4j-admin accepts.
    assert len({t.file for t in tables}) == len(tables)
    assert all(t.file.replace(".csv", "").replace("_", "").isalnum() for t in tables)


def test_nodes_carry_id_label_entity_fields_attributes_and_projections(tmp_path: Path) -> None:
    directory = _written_csv(tmp_path)
    assert _headers(directory)["odke_nodes_Person.csv"] == [
        "key:ID(odke_key_Person)",
        ":LABEL",
        "label:string",
        "aliases:string[]",
        "external_id:string",
        "resolution_method:string",
        "resolution_score:string",
        "resolution_linker:string",
        "born:long",
        "tags:string",
        "name:string",
    ]
    (ada,) = _read(directory)["odke_nodes_Person.csv"]
    assert ada["key:ID(odke_key_Person)"] == "p:ada"
    assert ada[":LABEL"] == "Person;Entity"
    assert ada["aliases:string[]"] == "Ada;Countess of Lovelace"
    assert (ada["born:long"], ada["tags:string"], ada["name:string"]) == ("1815", '{"a": 1}', "Ada")
    # An unset field is an empty, unquoted cell: null, not the empty string.
    raw = (directory / "odke_nodes_Person.csv").read_text(encoding="utf-8").splitlines()[1]
    assert raw == (
        '"p:ada","Person;Entity","Ada Lovelace","Ada;Countess of Lovelace","Q7259",'
        '"external_id",,,1815,"{""a"": 1}","Ada"'
    )


def test_edges_carry_every_provenance_property_typed(tmp_path: Path) -> None:
    directory = _written_csv(tmp_path)
    graph = _graph()
    (edge_file,) = [f for f in _headers(directory) if f.startswith("odke_edges_Person_employer")]
    header = _headers(directory)[edge_file]
    assert header[:3] == [":START_ID(odke_key_Person)", ":END_ID(odke_key_Company)", ":TYPE"]
    for column in (
        "signature:string",
        "polarity:string",
        "confidence:double",
        "support:long",
        "valid_from:datetime",
        "valid_to:string",
        "retrieved_at:datetime",
        "extracted_at:datetime",
        "evidence_doc_ids:string[]",
        "evidence_uris:string[]",
        "evidence_starts:long[]",
        "evidence_ends:long[]",
        "evidence_retrieved_at:datetime[]",
        "start_time:string",
        "qualifier_signature:string",
    ):
        assert column in header
    (row,) = _read(directory)[edge_file]
    assert (row[":START_ID(odke_key_Person)"], row[":END_ID(odke_key_Company)"]) == (
        "p:ada",
        "c:acme",
    )
    assert row[":TYPE"] == "employer"
    assert row["signature:string"] == signature_of(graph.facts[0])
    assert row["evidence_starts:long[]"] == "0;-1"
    assert row["valid_from:datetime"] == "2019-01-01T00:00:00+00:00"
    assert row["valid_to:string"] == ""


def test_claims_are_nodes_in_one_space_and_a_denial_keeps_its_polarity(tmp_path: Path) -> None:
    directory = _written_csv(tmp_path)
    rows = _read(directory)
    claims = [r for f, rs in rows.items() if f.startswith("odke_claims_") for r in rs]
    assert sorted((c["predicate:string"], c["value:string"]) for c in claims) == [
        ("name", "Ada"),
        ("sells", "customer_data"),
        ("uptime", "99.9%"),
        ("uptime", "99.9%"),
    ]
    assert {c[":LABEL"] for c in claims} == {"Claim"}
    (sells,) = next(rs for f, rs in rows.items() if f.startswith("odke_claim_edges_Company_sells"))
    assert sells["polarity:string"] == "denied"
    assert sells[":END_ID(odke_claim_signature)"] == signature_of(_graph().facts[4])


def test_rows_the_driver_would_merge_into_one_are_one_row(tmp_path: Path) -> None:
    acme = Entity(key="c:acme", type="Company")
    ada = Entity(key="p:ada", type="Person")
    once = Fact(subject=ada, predicate="employer", object_entity=acme, confidence=0.4)
    again = once.model_copy(update={"id": "second", "confidence": 0.8})
    claim = Fact(subject=acme, predicate="hq", object_value="Berlin")
    graph = KnowledgeGraph(
        facts=(once, again, claim, claim.model_copy(update={"id": "c2"})),
        links=(
            EntityLink(source_key="c:acme", target_key="p:ada", kind=LinkKind.SIMILAR, score=0.1),
            EntityLink(source_key="c:acme", target_key="p:ada", kind=LinkKind.SIMILAR, score=0.2),
        ),
    )
    rows = _read(_written_csv(tmp_path, graph))
    (edge,) = [r for f, rs in rows.items() if f.startswith("odke_edges_") for r in rs]
    # The later SET wins, as `SET r += row.props` over two rows would leave it.
    assert (edge["fact_id:string"], edge["confidence:double"]) == ("second", "0.8")
    assert len([r for f, rs in rows.items() if f.startswith("odke_claims_") for r in rs]) == 1
    assert len([r for f, rs in rows.items() if f.startswith("odke_claim_edges_") for r in rs]) == 1
    (link,) = [r for f, rs in rows.items() if f.startswith("odke_links_") for r in rs]
    assert link["score:double"] == "0.2"


def test_links_join_every_node_holding_each_key_and_skip_keys_nobody_holds(
    tmp_path: Path,
) -> None:
    directory = _written_csv(tmp_path)
    rows = _read(directory)
    links = {f: rs for f, rs in rows.items() if f.startswith("odke_links_")}
    assert sorted(f.split("_")[2] for f in links) == ["DIFFERENT", "SAME", "SIMILAR"]
    (different,) = next(rs for f, rs in links.items() if "DIFFERENT" in f)
    assert different[":TYPE"] == "DIFFERENT"
    assert different["reason:string"] == "external_id mismatch: DE-114322 vs GB-889401"

    ghost = EntityLink(source_key="p:ada", target_key="nobody", kind=LinkKind.SAME_AS)
    graph = _graph().model_copy(update={"links": (ghost,)})
    assert not [t for t in Neo4jAdminCsvSink(tmp_path).tables(graph) if "links" in t.file]


def test_every_relationship_end_is_a_node_id_in_its_declared_space(tmp_path: Path) -> None:
    """The check neo4j-admin fails an import on: a START_ID or END_ID with no node."""
    directory = _written_csv(tmp_path)
    ids: dict[str, set[str]] = {}
    for path in directory.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if ":LABEL" not in header:
                continue
            space = header[0].split("(", 1)[1].rstrip(")")
            ids.setdefault(space, set()).update(row[0] for row in reader)
    checked = 0
    for path in directory.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if ":TYPE" not in header:
                continue
            start = header[0].split("(", 1)[1].rstrip(")")
            end = header[1].split("(", 1)[1].rstrip(")")
            for row in reader:
                assert row[0] in ids[start] and row[1] in ids[end]
                checked += 1
    assert checked == 8
    # And node ids are unique within each space.
    assert sum(len(v) for v in ids.values()) == 3 + 4


def test_import_args_carry_every_file_and_the_parsing_options(tmp_path: Path) -> None:
    directory = _written_csv(tmp_path)
    lines = (directory / "import.args").read_text(encoding="utf-8").splitlines()
    options = [line for line in lines if not line.startswith("#")]
    files = sorted(p.name for p in directory.glob("*.csv"))
    listed = [o.split("=", 1)[1] for o in options if o.startswith(("--nodes", "--rel"))]
    assert sorted(listed) == files
    assert options[-4:] == [
        "--delimiter=,",
        "--array-delimiter=;",
        "--multiline-fields=true",
        "--id-type=string",
    ]
    sink = Neo4jAdminCsvSink(directory, delimiter="\t", array_delimiter="|")
    args = sink.arguments(sink.tables(_graph()), root="/import")
    assert args[0] == "--nodes=/import/odke_nodes_Company.csv"
    assert "--delimiter=TAB" in args and "--array-delimiter=|" in args


def test_values_with_quotes_delimiters_and_newlines_survive_the_csv(tmp_path: Path) -> None:
    ada = Entity(key='p:"ada",\nlovelace', type="Person", label='say "hi", then\nleave')
    directory = _written_csv(tmp_path, KnowledgeGraph(entities=(ada,)))
    (row,) = _read(directory)["odke_nodes_Person.csv"]
    assert row["key:ID(odke_key_Person)"] == ada.key
    assert row["label:string"] == ada.label


def test_a_column_whose_values_disagree_on_type_is_text(tmp_path: Path) -> None:
    graph = KnowledgeGraph(
        entities=(
            Entity(key="a", type="T", attributes={"size": 3, "ratio": 1, "when": date(2026, 1, 1)}),
            Entity(key="b", type="T", attributes={"size": "big", "ratio": 1.5, "when": None}),
        )
    )
    directory = _written_csv(tmp_path, graph)
    header = _headers(directory)["odke_nodes_T.csv"]
    assert {"size:string", "ratio:double", "when:date"} <= set(header)
    a, b = _read(directory)["odke_nodes_T.csv"]
    assert (a["size:string"], b["size:string"], a["ratio:double"]) == ("3", "big", "1.0")


def test_what_neo4j_admin_would_misread_is_refused_by_name(tmp_path: Path) -> None:
    semi = Entity(key="a", type="T", aliases=("x;y",))
    with pytest.raises(ValueError, match=r"aliases.*array delimiter"):
        _written_csv(tmp_path, KnowledgeGraph(entities=(semi,)))
    _written_csv(tmp_path, KnowledgeGraph(entities=(semi,)), array_delimiter="|")
    colon = Entity(key="a", type="T", attributes={"schema:name": "x"})
    with pytest.raises(ValueError, match="schema:name"):
        _written_csv(tmp_path, KnowledgeGraph(entities=(colon,)))
    with pytest.raises(ValueError, match="differ"):
        Neo4jAdminCsvSink(tmp_path, delimiter=";", array_delimiter=";")
    with pytest.raises(ValueError, match="one of"):
        Neo4jAdminCsvSink(tmp_path, delimiter=":")


def test_the_csv_sink_is_idempotent(tmp_path: Path) -> None:
    directory = tmp_path / "import"
    sink = Neo4jAdminCsvSink(directory)
    assert isinstance(sink, Sink)

    def counts() -> dict[str, int]:
        found = {"files": len(list(directory.iterdir()))}
        for path in directory.iterdir():
            found[path.name] = path.read_text(encoding="utf-8").count("\n")
        return found

    graph = _graph()
    assert_idempotent(sink, graph, counts)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    sink.write(graph)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_render_is_what_write_puts_on_disk(tmp_path: Path) -> None:
    sink = Neo4jAdminCsvSink(tmp_path)
    graph = _graph()
    sink.write(graph)
    for table in sink.tables(graph):
        text = sink.render(table)
        assert (tmp_path / table.file).read_text(encoding="utf-8") == text
        assert len(list(csv.reader(io.StringIO(text)))) == len(table.rows) + 1


def test_the_same_projection_rule_as_the_driver_decides_node_properties(tmp_path: Path) -> None:
    acme = Entity(key="c:acme", type="Company")
    facts = tuple(
        Fact(subject=acme, predicate="product", object_value=v, support=s)
        for v, s in (("anvils", 1), ("rockets", 2))
    ) + (Fact(subject=acme, predicate="sells", object_value="x", polarity=Polarity.DENIED),)
    ontology = Ontology(predicates={"product": Predicate(name="product", cardinality="multi")})
    directory = _written_csv(tmp_path, KnowledgeGraph(facts=facts), ontology=ontology)
    (row,) = _read(directory)["odke_nodes_Company.csv"]
    assert row["product:string[]"] == "rockets;anvils"
    assert "sells:string" not in row


# --------------------------------------------------------------------------- #
# A real server, when there is one
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="NEO4J_URI is not set")
def test_replaying_the_script_against_a_live_neo4j(tmp_path: Path) -> None:
    """Replay twice, and over what the driver wrote: one graph each time."""
    neo4j = pytest.importorskip("neo4j")
    suffix = uuid.uuid4().hex[:8]
    person, company = f"Person_{suffix}", f"Company_{suffix}"
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    mine = (
        f"MATCH (n) WHERE n:`{person}` OR n:`{company}` OR "
        f"(n:Claim AND n.subject_type IN ['{person}', '{company}']) "
    )
    count = mine + (
        "OPTIONAL MATCH (n)-[r]-() RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS rels"
    )
    driver = neo4j.GraphDatabase.driver(os.environ["NEO4J_URI"], auth=auth)
    path = tmp_path / "g.cypher"
    try:
        CypherFileSink(path).write(live_graph(suffix))
        for statement in _script_statements(path):
            driver.execute_query(statement)
        first = driver.execute_query(count).records[0].data()
        assert (first["nodes"], first["rels"]) == (7, 8)
        for statement in _script_statements(path):
            driver.execute_query(statement)
        Neo4jSink(driver=driver).write(live_graph(suffix))
        assert driver.execute_query(count).records[0].data() == first
    finally:
        driver.execute_query(mine + "DETACH DELETE n")
        driver.close()
