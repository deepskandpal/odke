"""Competitor extractors on an `odke bench prepare` directory, then openodke's verification.

    python bench/competitors.py extract lgt   runs/movie     # LangChain LLMGraphTransformer
    python bench/competitors.py extract neo4j runs/movie     # neo4j-graphrag's ER extractor
    odke bench run text2kgbench runs/movie/competitors/lgt   # raw, + grounding, + corroboration

`extract` gives each competitor what openodke gets: the same documents, whole,
the same schema (the prepared ontology's types and its domain-range patterns),
and the same extraction model with the same output room. The model is the
prepared config's `models.extract`, a LiteLLM model string (openai/..., anthropic/...,
ollama/..., gemini/...); `--model` overrides it. Keys come from the environment, as
LiteLLM reads them (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...); `--env-file` loads a file
of KEY=value lines first.

Both competitors reach the model through LiteLLM, as openodke does, so one model
string runs all three and a reasoning model's thinking never reaches either
library's parser: LiteLLM returns the reply's text, and each library gets that.
Nothing about what the model is asked changes. (The libraries' structured paths
force a tool choice, which some reasoning models refuse; both run their
prompt-and-parse paths here.)

Triples go to `competitors/<system>/facts.jsonl` in openodke's triples format
(docs/triples.md), and the directory is made runnable: the prepared config with
the extractor swapped for `triples`, which hands those triples to openodke's
grounder and corroborator unchanged.
Both competitors are held to the schema the way LLMGraphTransformer's strict
mode does it: a triple whose (head type, relation, tail type) is not a pattern
of the ontology is dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CONCURRENCY = 3
RETRIES = 6
DEFAULT_MAX_TOKENS = 16000
# What the calls cost in tokens, and documents that failed after LiteLLM's retries.
USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "failed_documents": 0}
LITERAL_NODES = {"date": "Date", "number": "Number", "integer": "Number", "string": "Text"}
ROLES = {"system": "system", "human": "user", "user": "user", "ai": "assistant"}


# --------------------------------------------------------------------------- #
# the model: one LiteLLM call, text out
# --------------------------------------------------------------------------- #


def load_env(path: str | None) -> None:
    """KEY=value lines into the environment, never overriding what is already set."""
    if not path or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key and not key.startswith("#"):
            os.environ.setdefault(key, value)


def extract_model(prepared: Path, override: str | None) -> tuple[str, int]:
    """The model string and output room openodke's extractor uses in this set."""
    models = json.loads((prepared / "odke.json").read_text()).get("models", {})
    spec = models.get("extract")
    if isinstance(spec, dict):
        model, max_tokens = spec.get("model"), spec.get("max_tokens", DEFAULT_MAX_TOKENS)
    else:
        model, max_tokens = spec, DEFAULT_MAX_TOKENS
    model = override or model
    if not model:
        raise SystemExit(
            "no extraction model: prepare with --extract-model, or pass --model to extract"
        )
    return model, int(max_tokens)


def _record(response: Any) -> str:
    usage = getattr(response, "usage", None)
    USAGE["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    USAGE["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    USAGE["calls"] += 1
    return response.choices[0].message.content or ""


async def complete(messages: Sequence[dict[str, str]], model: str, max_tokens: int) -> str:
    """The reply's text. LiteLLM keeps a reasoning model's thinking out of `content`."""
    import litellm

    response = await litellm.acompletion(
        model=model, messages=list(messages), max_tokens=max_tokens, num_retries=RETRIES
    )
    return _record(response)


def complete_sync(messages: Sequence[dict[str, str]], model: str, max_tokens: int) -> str:
    import litellm

    response = litellm.completion(
        model=model, messages=list(messages), max_tokens=max_tokens, num_retries=RETRIES
    )
    return _record(response)


def as_messages(prompt: Any) -> list[dict[str, str]]:
    """A LangChain prompt value as LiteLLM chat messages."""
    out = []
    for message in prompt.to_messages():
        content = message.content
        if not isinstance(content, str):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        out.append({"role": ROLES.get(message.type, "user"), "content": content})
    return out


# --------------------------------------------------------------------------- #
# the shared schema
# --------------------------------------------------------------------------- #


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


def documents(prepared: Path, limit: int | None = None) -> list[tuple[str, str]]:
    docs = [(p.stem, p.read_text()) for p in sorted((prepared / "docs").glob("*.txt"))]
    return docs[:limit] if limit else docs


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


async def run_lgt(
    docs: list[tuple[str, str]], nodes: list[str], patterns: list, model: str, max_tokens: int
) -> list[dict]:
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    # Without tool calling the transformer runs `prompt | llm` and parses the
    # text it gets back, so the model can be any runnable that returns text.
    async def call(prompt: Any) -> str:
        return await complete(as_messages(prompt), model, max_tokens)

    llm = RunnableLambda(lambda p: complete_sync(as_messages(p), model, max_tokens), afunc=call)
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


def litellm_for_neo4j(model: str, max_tokens: int) -> Any:
    """A neo4j-graphrag LLM that calls LiteLLM; the extractor's text path uses `ainvoke(str)`."""
    from neo4j_graphrag.llm.base import LLMInterface
    from neo4j_graphrag.llm.types import LLMResponse

    class LiteLLM(LLMInterface):
        def _messages(self, input: str, history: Any, system: str | None) -> list[dict]:
            messages = [{"role": "system", "content": system}] if system else []
            for m in history or []:
                messages.append({"role": m["role"], "content": m["content"]})
            return messages + [{"role": "user", "content": input}]

        def invoke(self, input, message_history=None, system_instruction=None):  # type: ignore[no-untyped-def]
            text = complete_sync(
                self._messages(input, message_history, system_instruction), model, max_tokens
            )
            return LLMResponse(content=text)

        async def ainvoke(self, input, message_history=None, system_instruction=None):  # type: ignore[no-untyped-def]
            text = await complete(
                self._messages(input, message_history, system_instruction), model, max_tokens
            )
            return LLMResponse(content=text)

    return LiteLLM(model_name=model)


async def run_neo4j(
    docs: list[tuple[str, str]], nodes: list[str], patterns: list, model: str, max_tokens: int
) -> list[dict]:
    from neo4j_graphrag.components.entity_relation_extractor import LLMEntityRelationExtractor
    from neo4j_graphrag.components.schema import (
        GraphSchema,
        NodeType,
        PropertyType,
        RelationshipType,
    )
    from neo4j_graphrag.components.types import TextChunk, TextChunks

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
    extractor = LLMEntityRelationExtractor(
        llm=litellm_for_neo4j(model, max_tokens), create_lexical_graph=False
    )
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


def extract(
    system: str, prepared: Path, model: str | None = None, limit: int | None = None
) -> Path:
    model, max_tokens = extract_model(prepared, model)
    nodes, patterns, _ = schema(prepared)
    docs = documents(prepared, limit)
    raw = asyncio.run(SYSTEMS[system](docs, nodes, patterns, model, max_tokens))
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
        "model": model,
        "documents": len(docs),
        "triples_raw": len(raw),
        "triples_in_schema": len(rows),
    }
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    (out / "docs").mkdir()
    for doc_id, _ in docs:  # only the documents extracted, so a --limit run scores fairly
        (out / "docs" / f"{doc_id}.txt").symlink_to((prepared / "docs" / f"{doc_id}.txt").resolve())
    for name in ("ontology.json", "dataset.json"):
        shutil.copy(prepared / name, out / name)
    # The gold for the documents extracted, so a --limit run's recall is not diluted.
    kept = {doc_id for doc_id, _ in docs}
    gold = [line for line in (prepared / "gold.jsonl").read_text().splitlines() if line.strip()]
    (out / "gold.jsonl").write_text(
        "".join(line + "\n" for line in gold if json.loads(line).get("id") in kept)
    )
    config = json.loads((prepared / "odke.json").read_text())
    # Absolute: openodke resolves its own stages' paths against the config, not a plugin's.
    facts = str((out / "facts.jsonl").resolve())
    config["stages"]["extractor"] = {"use": "triples", "path": facts, "extractor": system}
    (out / "odke.json").write_text(json.dumps(config, indent=2))
    print(
        f"{system} ({model}): {len(raw)} triples, {len(rows)} inside the schema,"
        f" {len(docs)} documents, {USAGE['failed_documents']} failed -> {out}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["extract"])
    parser.add_argument("system", choices=sorted(SYSTEMS))
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--model", help="A LiteLLM model string; default: the set's extractor.")
    parser.add_argument("--limit", type=int, help="Only the first N documents (a smoke test).")
    parser.add_argument(
        "--env-file", default=os.environ.get("ODKE_ENV_FILE"), help="KEY=value lines to load."
    )
    args = parser.parse_args()
    load_env(args.env_file)
    extract(args.system, args.prepared, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
