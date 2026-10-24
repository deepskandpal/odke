"""R2. The tenant goes everywhere.

One store holds many tenants, and two tenants who state the same claim own two
facts (DECISIONS #44): every key a store merges on is `<tenant>/<key>`, every
fact signature is salted with the tenant, and every read is the reader's
tenant's. The Cypher-file and Neo4j-CSV sinks ignored the tenant, so two
tenants' files merge when loaded (release audit, finding 3), and
`checked_under` read every tenant's facts (finding 20).

The check, in two halves:

- write one graph as `acme` and as `globex` through every sink a store loads,
  from both commands that take `--tenant`: no key and no signature is shared;
- read as `acme` through every read path, from a store that holds the same
  fact for `acme`, for `globex` and for no tenant: nothing comes back but
  `acme`'s.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from _support import copy_example
from typer.testing import CliRunner

from openodke import Entity, Evidence, Fact, KnowledgeGraph, Ontology, Span
from openodke.cli.main import app
from openodke.interop import read_neo4j
from openodke.sinks import neo4j as neo4j_module
from openodke.sinks.neo4j import (
    ONTOLOGY,
    SHOW_INDEXES,
    Neo4jSink,
    is_check,
    provenance_of,
)
from openodke.tenants import scoped
from test_neo4j_sink import FakeDriver

runner = CliRunner()

TENANTS = ("acme", "globex")
RELEASE_3 = pytest.mark.xfail(strict=True, reason="release audit finding 3; fixed in fix PR 1")
RELEASE_20 = pytest.mark.xfail(strict=True, reason="release audit finding 20; fixed in fix PR 1")

# --------------------------------------------------------------------------- #
# Writes: one graph as two tenants, through every sink a store is loaded from
# --------------------------------------------------------------------------- #

# The MERGE keys of the Neo4j shape: what makes two rows one node or one fact.
_IDENTITY = ("key", "signature", "subject_key", "object_key", "source_key", "target_key")
_CYPHER_IDENTITY = re.compile(r"\b(?:" + "|".join(_IDENTITY) + r"): '((?:[^'\\]|\\.)*)'")


def _driver_identities(driver: FakeDriver) -> set[str]:
    found = set()
    for _, params in driver.writes:
        for row in params.get("rows", []):
            found |= {str(row[k]) for k in _IDENTITY if row.get(k) is not None}
    return found


def _csv_identities(directory: Path) -> set[str]:
    """Every cell of an `:ID`, `:START_ID`, `:END_ID` or `signature` column."""
    found = set()
    for path in sorted(directory.glob("*.csv")):
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        columns = [
            i for i, name in enumerate(header)
            if ":ID" in name or name.endswith((":START_ID", ":END_ID")) or name.split(":")[0]
            == "signature"
        ]  # fmt: skip
        found |= {row[i] for row in body for i in columns if i < len(row) and row[i]}
    return found


SINKS: dict[str, dict[str, Any]] = {
    "neo4j": {
        "use": "neo4j",
        "uri": "bolt://store.invalid:7687",
        "password_env": "RULE2_STORE_PASSWORD",
    },
    "cypher_file": {"use": "cypher_file", "path": "graph.cypher"},
    "neo4j_admin_csv": {"use": "neo4j_admin_csv", "directory": "import"},
    # An entity's and a fact's IRI are its key and its signature: a triple store merges on them.
    "rdf": {"use": "rdf", "path": "graph.ttl"},
}
_IRI = re.compile(r"<([^<>\s]+/(?:entity|fact)/[^<>\s]+)>")
RDF_IGNORES_THE_TENANT = pytest.mark.xfail(
    strict=True,
    reason="found by R2 in PR 0, not in the release audit: the RDF sink writes two tenants' "
    "identical IRIs; proposed for fix PR 1, the maintainer's call",
)
MARKS = {"neo4j": (), "cypher_file": RELEASE_3, "neo4j_admin_csv": RELEASE_3}


def _identities(sink: str, out: Path, driver: FakeDriver) -> set[str]:
    if sink == "neo4j":
        return _driver_identities(driver)
    if sink == "cypher_file":
        return set(_CYPHER_IDENTITY.findall((out / "graph.cypher").read_text(encoding="utf-8")))
    if sink == "rdf":
        return set(_IRI.findall((out / "graph.ttl").read_text(encoding="utf-8")))
    return _csv_identities(out / "import")


def _commands(config: str) -> dict[str, list[str]]:
    return {"odke run": ["run", config], "odke validate": ["validate", "--config", config]}


WRITES = [
    pytest.param(
        command, sink, marks=MARKS.get(sink, RDF_IGNORES_THE_TENANT), id=f"{command}-{sink}"
    )
    for command in ("odke run", "odke validate")
    for sink in SINKS
]


@pytest.mark.parametrize(("command", "sink"), WRITES)
def test_two_tenants_writing_one_graph_share_no_key_and_no_signature(
    command: str, sink: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RULE2_STORE_PASSWORD", "not-a-real-password")
    written: dict[str, set[str]] = {}
    for tenant in TENANTS:
        project = copy_example("triples", tmp_path / tenant)
        config = yaml.safe_load((project / "odke.yaml").read_text(encoding="utf-8"))
        config["stages"]["sink"] = SINKS[sink]
        (project / "odke.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        driver = FakeDriver()
        monkeypatch.setattr(neo4j_module, "_connect", lambda uri, auth, d=driver: d)
        monkeypatch.chdir(project)
        args = [*_commands("odke.yaml")[command], "--tenant", tenant]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
        written[tenant] = _identities(sink, project, driver)
        assert written[tenant], f"{sink} wrote no key or signature: {result.output}"

    shared = written["acme"] & written["globex"]
    assert not shared, (
        f"`{command} --tenant` wrote {len(shared)} identical keys or signatures for acme and "
        f"globex through the {sink} sink (e.g. {sorted(shared)[0]!r}), so loaded into one store "
        "their facts become one. Every sink must key a tenant's writes apart: give it a "
        "`tenant` option and a `scoped(tenant)` that the run's tenant reaches, and build its "
        "keys and signatures with it (`openodke.sinks.neo4j.plan(tenant=...)`, `scoped`, "
        "`signature_of`)."
    )


def test_a_jsonl_directory_holds_one_tenant(tmp_path: Path) -> None:
    """JSONL keys nothing by tenant; it refuses a second tenant's write instead (#44)."""
    from openodke.sinks import JsonlSink

    JsonlSink(tmp_path, merge=True, tenant="acme").write(_graph())
    for reader in (
        JsonlSink(tmp_path, merge=True, tenant="globex"),
        JsonlSink(tmp_path, merge=True),
    ):
        with pytest.raises(ValueError, match="holds tenant"):
            reader.write(_graph())
        with pytest.raises(ValueError, match="holds tenant"):
            reader.stored(_graph().facts)


def test_every_sink_a_run_can_write_is_covered_by_this_rule() -> None:
    from openodke.run.build import SINKS as RUN_SINKS

    # A graph in memory, or one run's node-link file that networkx reads back whole: no
    # store merges two of them, so a tenant has no key to scope.
    single_graph = {"networkx"}
    covered = {*SINKS, "jsonl"}
    assert set(RUN_SINKS) == covered | single_graph, (
        f"{sorted(set(RUN_SINKS) - covered - single_graph)} is a sink this rule does not "
        "check. Add it to SINKS in tests/rules/test_r2_tenants.py, so two tenants' writes "
        "through it are shown never to share a key."
    )


# --------------------------------------------------------------------------- #
# Reads: as acme, from a store holding acme's, globex's and nobody's copy
# --------------------------------------------------------------------------- #

WHEN = datetime(2026, 10, 1, tzinfo=UTC)
TEXT = "Halden Robotics is run by Tomas Ferrand."
HALDEN = Entity(key="c:halden", type="Company", label="Halden Robotics")
TOMAS = Entity(key="p:tomas", type="Person", label="Tomas Ferrand")
ONTOLOGY_DOC = Ontology.model_validate_json(
    (Path(__file__).parents[2] / "examples" / "triples" / "ontology.json").read_text("utf-8")
)
FINGERPRINT = ONTOLOGY_DOC.fingerprint
# Who holds each copy of the fact in the store. Nobody's is a store written with no tenant.
OWNERS = (*TENANTS, None)


def _name(owner: str | None) -> str:
    return owner or "nobody"


def _fact(owner: str | None = "acme") -> Fact:
    """The one claim. Its owner shows only where a read cannot drop it: the evidence's URI."""
    span = Span(doc_id="halden", start=0, end=len(TEXT), quote=TEXT)
    uri = f"https://{_name(owner)}.example/halden"
    evidence = (Evidence(doc_id="halden", span=span, uri=uri, retrieved_at=WHEN),)
    return Fact(subject=HALDEN, predicate="chief_executive", object_entity=TOMAS, evidence=evidence)


def _graph() -> KnowledgeGraph:
    return KnowledgeGraph(entities=(HALDEN, TOMAS), facts=(_fact(),))


def _record(owner: str | None) -> dict[str, Any]:
    """The fact as a store holds it for `owner`: every column any read path returns."""
    props = {**provenance_of(_fact(owner), WHEN, owner), ONTOLOGY: FINGERPRINT}
    subject, obj = scoped(HALDEN.key, owner), scoped(TOMAS.key, owner)
    return {
        "owner": owner,
        "id": f"rel-{_name(owner)}",
        "signature": props["signature"],
        "type": "chief_executive",
        "predicate": "chief_executive",
        "props": props,
        "properties": props,
        "subject_key": subject,
        "subject_labels": ["Company", "Entity"],
        "subject_label": f"{HALDEN.label} ({_name(owner)})",
        "subject_name": None,
        "subject_id": f"node-{subject}",
        "object_key": obj,
        "object_labels": ["Person", "Entity"],
        "object_label": f"{TOMAS.label} ({_name(owner)})",
        "object_name": None,
        "object_id": f"node-{obj}",
        "value": None,
    }


_INDEXES = [
    {"name": f"odke_{what}_chief_executive", "type": kind, "entityType": "RELATIONSHIP",
     "labelsOrTypes": ["chief_executive"], "properties": [prop]}
    for what, kind, prop in (
        ("signature", "RANGE", "signature"),
        ("ontology", "RANGE", ONTOLOGY),
        ("evidence", "FULLTEXT", "evidence_doc_ids"),
    )
] + [
    {"name": f"odke_{what}_{label}", "type": kind, "entityType": "NODE",
     "labelsOrTypes": [label], "properties": props}
    for label in ("Company", "Person")
    for what, kind, props in (
        ("key", "RANGE", ["key"]),
        ("names", "FULLTEXT", ["label", "aliases"]),
    )
]  # fmt: skip
_SIDES = {"Company": "subject", "Person": "object"}

_IS_TENANT = re.compile(r"\w+\.`?tenant`? = \$tenant\b")
_NO_TENANT = re.compile(r"\w+\.`?tenant`? IS NULL")


class TwoTenantStore(FakeDriver):
    """A store holding the fact for acme, for globex and for no tenant.

    It answers a read as Neo4j would, for the filters this codebase writes: a
    tenant condition (`r.tenant = $tenant`, `n.tenant IS NULL`, or either), a MERGE key
    looked up by `row.signature` or `row.value`, and an ontology fingerprint.
    A read with no filter gets every owner's rows, as an unfiltered query would.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records = [_record(owner) for owner in OWNERS]

    def owner_of(self, signature: str) -> str:
        return next(_name(r["owner"]) for r in self.records if r["signature"] == signature)

    def answer(self, cypher: str) -> list[dict[str, Any]]:
        _, _, params = self.calls[-1]
        if cypher.startswith(SHOW_INDEXES):
            return _INDEXES
        if "db.relationshipTypes" in cypher:
            return [{"name": "chief_executive"}]
        if "AS unread" in cypher:
            return [{"unread": 0}]
        records = self.records
        # One condition filters; both (`r.tenant = $tenant OR r.tenant IS NULL`) let in both.
        allowed: set[str | None] = set()
        if _IS_TENANT.search(cypher):
            allowed.add(params.get("tenant"))
        if _NO_TENANT.search(cypher):
            allowed.add(None)
        if allowed:
            records = [r for r in records if r["owner"] in allowed]
        if "$fingerprint" in cypher:
            records = [r for r in records if r["props"][ONTOLOGY] == params["fingerprint"]]
        if is_check(cypher):
            deputy = "p:{}-deputy"
            return [
                {"labels": ["Company", "Entity"], "subject": r["subject_key"],
                 "objects": [r["object_key"], scoped(deputy.format(_name(r["owner"])), r["owner"])]}
                for r in records
            ]  # fmt: skip
        if "row.signature" in cypher:
            asked = {row["signature"] for row in params["rows"]}
            return [r for r in records if r["signature"] in asked]
        if "row.value" in cypher:
            return [
                {"block": row["block"], "node": self._node(r, side)}
                for row in params["rows"]
                for r in records
                for side in ("subject", "object")
                if r[f"{side}_key"] == row["value"]
            ]
        if "queryNodes" in cypher:
            # A name matches every owner's node of the index's type: the tenant filter decides.
            side = _SIDES[params["index"].rsplit("_", 1)[-1]]
            return [
                {"block": row["block"], "node": self._node(r, side)}
                for row in params["rows"]
                for r in records
            ]
        return records

    @staticmethod
    def _node(record: dict[str, Any], side: str) -> dict[str, Any]:
        node = {"key": record[f"{side}_key"], "label": record[f"{side}_label"]}
        if record["owner"] is not None:
            node["tenant"] = record["owner"]
        return node


def _owner_in(text: str | None) -> str:
    """Whose copy a returned value came from: the owner's name it carries."""
    found = [_name(o) for o in OWNERS if text and re.search(rf"\b{_name(o)}\b", text)]
    return found[0] if found else f"unknown ({text!r})"


def _read_checked_under(store: TwoTenantStore, tenant: str | None) -> list[str]:
    facts = Neo4jSink(driver=store, tenant=tenant).checked_under(FINGERPRINT)
    return [_owner_in(e.uri) for f in facts for e in f.evidence]


def _read_stored(store: TwoTenantStore, tenant: str | None) -> list[str]:
    found = Neo4jSink(driver=store, tenant=tenant).stored([_fact()])
    return [_owner_in(e.uri) for f in found.values() for e in f.evidence]


def _read_retract(store: TwoTenantStore, tenant: str | None) -> list[str]:
    Neo4jSink(driver=store, tenant=tenant).retract(["halden"], at=WHEN)
    return [store.owner_of(row["signature"])
            for _, params in store.writes for row in params.get("rows", [])
            if "signature" in row]  # fmt: skip


def _read_check(store: TwoTenantStore, tenant: str | None) -> list[str]:
    rows = Neo4jSink(driver=store, tenant=tenant).check(ONTOLOGY_DOC)
    return [_owner_in(" ".join(map(str, row["objects"]))) for found in rows.values()
            for row in found]  # fmt: skip


def _read_lookup(store: TwoTenantStore, tenant: str | None) -> list[str]:
    found = Neo4jSink(driver=store, tenant=tenant).lookup().candidates([HALDEN, TOMAS])
    return [_owner_in(e.label) for entities in found.values() for e in entities]


def _read_neo4j(store: TwoTenantStore, tenant: str | None) -> list[str]:
    rows, _ = read_neo4j(store, tenant=tenant)
    return [_owner_in(str(row.id)) for row in rows]


# Each read as acme; and, where a reader with no tenant reads only what no tenant wrote
# (#44), as no tenant. `checked_under`, `check()` and `read_neo4j` read the whole store
# with no tenant, as documented, so they are read as acme only.
READS = [
    pytest.param(_read_checked_under, "acme", marks=RELEASE_20, id="Neo4jSink.checked_under"),
    pytest.param(_read_stored, "acme", id="Neo4jSink.stored"),
    pytest.param(_read_stored, None, id="Neo4jSink.stored-no tenant"),
    pytest.param(_read_retract, "acme", id="Neo4jSink.retract"),
    pytest.param(_read_retract, None, id="Neo4jSink.retract-no tenant"),
    pytest.param(_read_check, "acme", id="Neo4jSink.check"),
    pytest.param(_read_lookup, "acme", id="Neo4jLookup.candidates"),
    pytest.param(_read_lookup, None, id="Neo4jLookup.candidates-no tenant"),
    pytest.param(_read_neo4j, "acme", id="read_neo4j"),
]


@pytest.mark.parametrize(("read", "tenant"), READS)
def test_reading_as_one_tenant_returns_nothing_of_anothers(read: Any, tenant: str | None) -> None:
    store = TwoTenantStore()
    owners = read(store, tenant)
    reader = _name(tenant)
    assert owners, f"{read.__name__} read nothing: the store holds {reader}'s fact, so it should"
    others = sorted(set(owners) - {reader})
    assert not others, (
        f"{read.__name__.removeprefix('_read_')} read as {reader} returned {others}'s facts. "
        "Every read path must filter on `tenant`: `r.tenant = $tenant` for a tenant, "
        "`r.tenant IS NULL` for none, or look up by the tenant's own keys and signatures."
    )


@RELEASE_20
def test_odke_ontology_diff_store_reads_one_tenant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command over `checked_under` takes the tenant it reads, as every store command does."""
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(ONTOLOGY_DOC.model_dump_json(), encoding="utf-8")
    changed = ONTOLOGY_DOC.model_copy(update={"version": "2"})
    new.write_text(changed.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(neo4j_module, "_connect", lambda uri, auth: TwoTenantStore())
    monkeypatch.setenv("NEO4J_PASSWORD", "not-a-real-password")
    args = ["ontology", "diff", str(old), str(new), "--store", "bolt://store.invalid:7687"]
    result = runner.invoke(app, [*args, "--tenant", "acme"])
    assert result.exit_code == 0 and "1 stored fact checked under it" in result.output, (
        "`odke ontology diff --store --tenant acme` must list acme's facts alone; it printed:\n"
        f"{result.output}\nGive the command --tenant, and read through "
        "`Neo4jSink(tenant=...).checked_under`, which must filter on `r.tenant = $tenant`."
    )
