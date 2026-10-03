"""Competitor extractors on an `odke bench prepare` directory, then openodke's verification.

    python bench/competitors.py extract lgt   runs/movie     # LangChain LLMGraphTransformer
    python bench/competitors.py extract neo4j runs/movie     # neo4j-graphrag's ER extractor
    odke bench run text2kgbench runs/movie/competitors/lgt   # raw, + grounding, + corroboration

`extract` gives each competitor what openodke gets: the same documents, whole,
the same schema (the prepared ontology's types and its domain-range patterns),
and the same extraction model, with the same 16,000-token room to think. Its
triples go to `competitors/<system>/facts.jsonl`, and the directory is made
runnable: the prepared config with the extractor swapped for `replay:Replayed`,
which hands those triples to openodke's grounder and corroborator unchanged.
Both competitors are held to the schema the way LLMGraphTransformer's
strict mode does it: a triple whose (head type, relation, tail type) is not a
pattern of the ontology is dropped.

Two shims, both about reading the model's reply, never about what it is asked:
Claude Sonnet 5.5 answers with a thinking block before its text. neo4j-graphrag's
AnthropicLLM reads only the first block and raises on it; LLMGraphTransformer
without tool calling hands the whole block list to its JSON parser. Each gets
the reply's text. (Tool calling is not an option on this model: forced
`tool_choice` is refused, and neo4j-graphrag's structured output and
LLMGraphTransformer's tool path both force it.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

EXTRACT_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 16000
CONCURRENCY = 3
# What the shims saw: tokens billed and documents that failed after the clients' retries.
USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "failed_documents": 0}
LITERAL_NODES = {"date": "Date", "number": "Number", "integer": "Number", "string": "Text"}


# --------------------------------------------------------------------------- #
# the shared schema
# --------------------------------------------------------------------------- #


def load_env(path: str = "/root/.config/odke-bench.env") -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key:
            os.environ.setdefault(key, value)


def schema(prepared: Path) -> tuple[list[str], list[tuple[str, str, str]], dict[str, Any]]:
    """Node types, (head type, RELATION, tail type) patterns, and the ontology."""
    ontology = json.loads((prepared / "ontology.json").read_text())
    types = set(ontology["types"])
    nodes = set(types)
    patterns: list[tuple[str, str, str]] = []
    for name, predicate in ontology["predicates"].items():
        target = predicate.get("range", "string")
        tail = target if target in types else LITERAL_NODES.get(target, "Text")
        nodes.add(tail)
        for head in predicate.get("domain") or sorted(types):
            patterns.append((head, name.upper(), tail))
    return sorted(nodes), patterns, ontology


def documents(prepared: Path) -> list[tuple[str, str]]:
    return [(p.stem, p.read_text()) for p in sorted((prepared / "docs").glob("*.txt"))]


def keep(rows: Iterable[dict[str, Any]], patterns: list[tuple[str, str, str]]) -> list[dict]:
    allowed = {(h.lower(), r.lower(), t.lower()) for h, r, t in patterns}
    out = []
    for row in rows:
        key = (row["subject_type"].lower(), row["predicate"].lower(), row["object_type"].lower())
        if key in allowed and row["subject"] and row["object"]:
            out.append(row)
    return out


# --------------------------------------------------------------------------- #
# LangChain LLMGraphTransformer
# --------------------------------------------------------------------------- #


async def run_lgt(docs: list[tuple[str, str]], nodes: list[str], patterns: list) -> list[dict]:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    def text_of(message: Any) -> str:  # the shim: the reply's text, not its block list
        usage = getattr(message, "usage_metadata", None) or {}
        USAGE["input_tokens"] += usage.get("input_tokens", 0)
        USAGE["output_tokens"] += usage.get("output_tokens", 0)
        USAGE["calls"] += 1
        content = message.content
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))

    llm = ChatAnthropic(model=EXTRACT_MODEL, max_tokens=MAX_TOKENS, max_retries=6) | RunnableLambda(
        text_of
    )
    transformer = LLMGraphTransformer(
        llm=llm,  # type: ignore[arg-type]
        allowed_nodes=nodes,
        allowed_relationships=patterns,
        ignore_tool_usage=True,
    )
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(doc_id: str, text: str) -> list[dict]:
        async with gate:
            try:
                (graph,) = await transformer.aconvert_to_graph_documents(
                    [Document(page_content=text)]
                )
            except Exception as exc:  # one failed document must not lose the run
                print(f"lgt {doc_id}: {exc}")
                USAGE["failed_documents"] += 1
                return []
        return [
            {
                "doc": doc_id,
                "subject": str(rel.source.id),
                "subject_type": rel.source.type,
                "predicate": rel.type,
                "object": str(rel.target.id),
                "object_type": rel.target.type,
            }
            for rel in graph.relationships
        ]

    results = await asyncio.gather(*(one(d, t) for d, t in docs))
    return [row for rows in results for row in rows]


# --------------------------------------------------------------------------- #
# neo4j-graphrag LLMEntityRelationExtractor
# --------------------------------------------------------------------------- #


async def run_neo4j(docs: list[tuple[str, str]], nodes: list[str], patterns: list) -> list[dict]:
    from neo4j_graphrag.components.entity_relation_extractor import LLMEntityRelationExtractor
    from neo4j_graphrag.components.schema import (
        GraphSchema,
        NodeType,
        PropertyType,
        RelationshipType,
    )
    from neo4j_graphrag.components.types import TextChunk, TextChunks
    from neo4j_graphrag.llm import AnthropicLLM

    class TextAnthropicLLM(AnthropicLLM):  # the shim: skip the thinking block
        @staticmethod
        def _extract_text(response: Any) -> str:
            usage = getattr(response, "usage", None)
            USAGE["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            USAGE["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
            USAGE["calls"] += 1
            return "".join(getattr(b, "text", "") for b in response.content if b.type == "text")

    llm = TextAnthropicLLM(
        model_name=EXTRACT_MODEL, model_params={"max_tokens": MAX_TOKENS}, max_retries=6
    )
    graph_schema = GraphSchema(
        node_types=[
            NodeType(label=n, properties=[PropertyType(name="name", type="STRING")]) for n in nodes
        ],
        relationship_types=[RelationshipType(label=r) for r in sorted({p[1] for p in patterns})],
        patterns=patterns,
        additional_node_types=False,
        additional_relationship_types=False,
        additional_patterns=False,
    )
    extractor = LLMEntityRelationExtractor(llm=llm, create_lexical_graph=False)
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(doc_id: str, text: str) -> list[dict]:
        async with gate:
            try:
                graph = await extractor.run(
                    TextChunks(chunks=[TextChunk(text=text, index=0)]), schema=graph_schema
                )
            except Exception as exc:
                print(f"neo4j {doc_id}: {exc}")
                USAGE["failed_documents"] += 1
                return []
        by_id = {n.id: n for n in graph.nodes}
        rows = []
        for rel in graph.relationships:
            head, tail = by_id.get(rel.start_node_id), by_id.get(rel.end_node_id)
            if head is None or tail is None:
                continue
            rows.append(
                {
                    "doc": doc_id,
                    "subject": str(head.properties.get("name") or head.id),
                    "subject_type": head.label,
                    "predicate": rel.type,
                    "object": str(tail.properties.get("name") or tail.id),
                    "object_type": tail.label,
                }
            )
        return rows

    results = await asyncio.gather(*(one(d, t) for d, t in docs))
    return [row for rows in results for row in rows]


SYSTEMS = {"lgt": run_lgt, "neo4j": run_neo4j}


# --------------------------------------------------------------------------- #
# a runnable directory for openodke's verification
# --------------------------------------------------------------------------- #


def extract(system: str, prepared: Path) -> Path:
    load_env()
    nodes, patterns, _ = schema(prepared)
    docs = documents(prepared)
    raw = asyncio.run(SYSTEMS[system](docs, nodes, patterns))
    rows = keep(raw, patterns)
    out = prepared / "competitors" / system
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "facts.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    usage = {
        **USAGE,
        "model": EXTRACT_MODEL,
        "triples_raw": len(raw),
        "triples_in_schema": len(rows),
    }
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    for name in ("docs",):
        (out / name).symlink_to((prepared / name).resolve())
    for name in ("ontology.json", "gold.jsonl", "dataset.json"):
        shutil.copy(prepared / name, out / name)
    config = json.loads((prepared / "odke.json").read_text())
    # Absolute: openodke resolves its own stages' paths against the config, not a plugin's.
    facts = str((out / "facts.jsonl").resolve())
    config["stages"]["extractor"] = {"use": "replay:Replayed", "path": facts, "system": system}
    config["pythonpath"] = [str(Path(__file__).resolve().parent)]
    (out / "odke.json").write_text(json.dumps(config, indent=2))
    print(
        f"{system}: {len(raw)} triples, {len(rows)} inside the schema, {len(docs)} documents,"
        f" {USAGE['failed_documents']} failed -> {out}"
    )
    return out


def stem(uri: str | None) -> str | None:
    return Path(unquote(urlparse(uri).path)).stem if uri else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["extract"])
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    extract(args.system, args.prepared)


if __name__ == "__main__":
    main()
