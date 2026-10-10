"""openodke at volume, on a slice of EnterpriseRAG-Bench's corpus (#164).

    python bench/volume.py fetch data/erb-volume --documents 30000
    python bench/volume.py cost data/erb-volume --out runs/volume/cost --per-type 5 \\
        --budget-usd 1.20 --extract-model anthropic/claude-sonnet-5-5 \\
        --ground-model anthropic/claude-haiku-4-5-20251001 --cache .odke-cache --estimate
    python bench/volume.py throughput data/erb-volume --out runs/volume/throughput
    python bench/volume.py reconcile data/erb-volume --out runs/volume/reconcile
    python bench/volume.py table runs/volume

The corpus is Onyx's EnterpriseRAG-Bench (github.com/onyx-dot-app/EnterpriseRAG-Bench;
code MIT, dataset card `license: mit`): about 500,000 documents of a fictional
company across nine source types, shaped like company-internal data. `fetch`
streams the repository's archive at the commit `conflicts.py` pins, keeps a
uniform random sample (a seeded reservoir), and writes it as `slice.jsonl`,
each document rendered as `conflicts.text_of` renders it. The archive is read
once and never lands on disk. Every file stored carries the benchmark's canary.

Each run reads a folder `layout` writes: the documents as text files under
`texts/<source type>/`, each type at its tier (`conflicts.TIERS`), the
ontology (`ONTOLOGY`, five types and twelve predicates for a company's
people, teams, customers, projects and services) and `odke.json`.

- `cost`: the real models on `--per-type` documents of each source type, each
  document its own `odke run`, under one USD budget across them (the
  package's own, exit 3) and the response cache. `--estimate` prices them
  first and calls nothing; `--limit` runs the first N, a pilot. Run again with
  the same cache, it costs nothing, and the cache's hits say so.
- `throughput`: `odke run`, and `odke validate` on triples, at several
  concurrency limits, against `Simulated`: a model that answers from the prompt
  after a fixed wait per call, so it costs nothing.
- `reconcile`: on `Simulated` too, a store `odke validate` builds, a share of
  its documents replaced by new versions (`odke validate --update`) and a
  share deleted (`odke reconcile --delete`).

Every measured command runs in a fresh process (`job`), so its peak RSS is its
own: `VmHWM`, not `ru_maxrss`, which a forked child inherits from its parent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import conflicts

from openodke.llm.base import Completion, Message, ModelSpec

TARBALL = f"https://codeload.github.com/{conflicts.REPO}/tar.gz/{conflicts.SHA}"
SOURCES = "generated_data/sources/"

ONTOLOGY: dict[str, Any] = {
    "name": "erb-volume",
    "version": "1",
    "types": {
        "Person": {"description": "A person: an employee, or a customer's or partner's staff"},
        "Team": {"description": "A team, group or function inside the company"},
        "Customer": {"description": "A customer, prospect or partner organisation"},
        "Project": {"description": "A project, initiative, epic, migration or launch"},
        "Service": {"description": "A product, service, system, model, library or tool"},
    },
    "predicates": {
        "member_of": {
            "description": "A team the person is on",
            "domain": ["Person"],
            "range": "Team",
            "cardinality": "multi",
        },
        "role": {"description": "The person's job title or role", "domain": ["Person"]},
        "works_on": {
            "description": "A project the person or team works on",
            "domain": ["Person", "Team"],
            "range": "Project",
            "cardinality": "multi",
        },
        "owns": {
            "description": "A service the person or team is responsible for",
            "domain": ["Person", "Team"],
            "range": "Service",
            "cardinality": "multi",
        },
        "status": {"description": "The project's current status", "domain": ["Project"]},
        "due_date": {
            "description": "When the project is due",
            "domain": ["Project"],
            "range": "date",
        },
        "depends_on": {
            "description": "A service the project or service depends on",
            "domain": ["Project", "Service"],
            "range": "Service",
            "cardinality": "multi",
        },
        "uses": {
            "description": "A service the customer uses or is evaluating",
            "domain": ["Customer"],
            "range": "Service",
            "cardinality": "multi",
        },
        "account_owner": {
            "description": "The person at the company responsible for the customer",
            "domain": ["Customer"],
            "range": "Person",
        },
        "stage": {"description": "The customer's deal or lifecycle stage", "domain": ["Customer"]},
        "region": {"description": "The customer's region", "domain": ["Customer"]},
        "version": {"description": "The service's current version", "domain": ["Service"]},
    },
}
# The predicates between two entities: all `Simulated` ever states.
EDGES = sorted(
    name for name, p in ONTOLOGY["predicates"].items() if p.get("range") in ONTOLOGY["types"]
)
# The simulated model's wait per call, in seconds, by role, unless told otherwise:
# the mean per call of the T-REx pilot's real calls on this machine (#117),
# Sonnet 5.5 extracting (5 calls) and Haiku 4.5 grounding (23).
LATENCY = {"extract": 7.3, "ground": 1.2}
# The most claims `Simulated` states in one passage, unless told otherwise.
FACTS = 12

Opener = Callable[[str], Any]


def _open(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": conflicts.USER_AGENT})
    return urllib.request.urlopen(request)


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the slice
# --------------------------------------------------------------------------- #


def fetch(
    dest: str | Path, *, documents: int = 30000, seed: int = 0, opener: Opener | None = None
) -> Path:
    """A uniform sample of `documents` of the corpus, as `slice.jsonl` under `dest`.

    Algorithm R over the archive's source documents in archive order, seeded,
    so the same commit and seed give the same slice. Only a kept document is
    read out of the stream. `slice.json` counts the corpus and the slice by
    source type.
    """
    root = Path(dest)
    rng = random.Random(seed)
    kept: list[tuple[str, bytes]] = []
    corpus: Counter[str] = Counter()
    open_url = opener if opener is not None else _open
    with open_url(TARBALL) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            path = member.name.split("/", 1)[-1]  # the archive's own top folder goes
            if not (member.isfile() and path.startswith(SOURCES) and path.endswith(".json")):
                continue
            path = path[len(SOURCES) :]
            seen = sum(corpus.values())
            corpus[path.split("/", 1)[0]] += 1
            slot = seen if seen < documents else rng.randrange(seen + 1)
            if slot >= documents:
                continue
            body = archive.extractfile(member)
            assert body is not None  # a regular file always has a body
            if slot == len(kept):
                kept.append((path, body.read()))
            else:
                kept[slot] = (path, body.read())
    records = sorted((record(path, raw) for path, raw in kept), key=lambda r: r["id"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "slice.jsonl").open("w", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    chars: Counter[str] = Counter()
    for row in records:
        chars[row["source_type"]] += len(row["text"])
    taken = Counter(row["source_type"] for row in records)
    _write(
        root / "slice.json",
        {
            "canary": conflicts.CANARY,
            "commit": conflicts.SHA,
            "seed": seed,
            "documents": len(records),
            "corpus": dict(sorted(corpus.items())),
            "slice": dict(sorted(taken.items())),
            "mean_chars": {t: round(chars[t] / n) for t, n in sorted(taken.items())},
        },
    )
    return root


def record(path: str, raw: bytes) -> dict[str, Any]:
    """One stored document: its id, source type, path, time and text, and the canary."""
    doc = json.loads(raw.decode("utf-8"))
    moment = conflicts.time_of(doc)
    return {
        "canary": conflicts.CANARY,
        "id": str(doc.get("dataset_doc_uuid") or Path(path).stem),
        "source_type": path.split("/", 1)[0],
        "path": path,
        "time": moment.isoformat() if moment is not None else None,
        "text": conflicts.text_of(doc),
    }


def read_slice(root: str | Path) -> list[dict[str, Any]]:
    with (Path(root) / "slice.jsonl").open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def shuffled(docs: Sequence[Mapping[str, Any]], seed: int = 0) -> list[Mapping[str, Any]]:
    """The slice in a seeded order: its first N are a uniform sample of N, nested."""
    out = list(docs)
    random.Random(seed).shuffle(out)
    return out


def select(
    docs: Sequence[Mapping[str, Any]], per_type: int, seed: int = 0
) -> list[Mapping[str, Any]]:
    """`per_type` documents of each source type, seeded, taken a type at a time.

    The first document of every type comes first, then the second of every
    type, so a pilot of the first few spans types.
    """
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for doc in sorted(docs, key=lambda d: d["id"]):
        by_type[doc["source_type"]].append(doc)
    picks = {
        t: random.Random(f"{seed}:{t}").sample(group, min(per_type, len(group)))
        for t, group in sorted(by_type.items())
    }
    return [picks[t][i] for i in range(per_type) for t in picks if i < len(picks[t])]


# --------------------------------------------------------------------------- #
# the simulated model
# --------------------------------------------------------------------------- #

_SENTENCE = re.compile(r"[^\n.!?]+[.!?]?")
_NAME = re.compile(r"\b[A-Z][\w&'-]*(?: [A-Z][\w&'-]*)*")


def claims(text: str, limit: int = FACTS) -> list[dict[str, Any]]:
    """What `Simulated` states in `text`: one claim per sentence that names two names.

    A name is a capitalised run of words, unless it opens the sentence. The
    subject is the first name, the object the second, and the predicate one of
    `EDGES`, picked by the sentence's checksum. The quote is the sentence, at
    its offset in `text`. At most `limit`, in reading order.
    """
    out: list[dict[str, Any]] = []
    for match in _SENTENCE.finditer(text):
        sentence = match.group().strip()
        names = list(dict.fromkeys(n.group() for n in _NAME.finditer(sentence) if n.start()))
        if len(names) < 2:
            continue
        key = zlib.crc32(sentence.encode("utf-8"))
        predicate = EDGES[key % len(EDGES)]
        spec = ONTOLOGY["predicates"][predicate]
        out.append(
            {
                "subject": names[0],
                "subject_type": spec["domain"][key % len(spec["domain"])],
                "predicate": predicate,
                "object": names[1],
                "object_type": spec["range"],
                "quote": sentence,
                "start": match.start() + len(match.group()) - len(match.group().lstrip()),
            }
        )
        if len(out) >= limit:
            break
    return out


def _extraction(found: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`claims` as the extractor's reply: entities, each with its facts."""
    entities: dict[tuple[str, str], dict[str, Any]] = {}
    for claim in found:
        key = (claim["subject_type"], claim["subject"])
        entity = entities.setdefault(key, {"type": key[0], "name": key[1], "facts": []})
        entity["facts"].append(
            {
                "predicate": claim["predicate"],
                "value": claim["object"],
                "quote": claim["quote"],
                "start": claim["start"],
                "mention": "",
                "polarity": "asserted",
            }
        )
    return {"entities": list(entities.values())}


class Simulated:
    """A model that answers from its prompt after a fixed wait: no network, no cost.

    Registered as the provider `sim`, so `sim/extract` extracts and
    `sim/ground` grounds. Extraction states `claims` of the passage. Grounding
    supports a claim, but for one in ten `not_found` and one in twenty
    `contradicted`, by the prompt's checksum. `latency` is the wait per call
    by role, in seconds; time.sleep releases the GIL, as a network call does.
    The cost is unknown (`None`), as a provider that does not say.
    """

    calls: Counter[str] = Counter()
    _lock = threading.Lock()

    def __init__(self, latency: Mapping[str, float] | None = None, facts: int = FACTS) -> None:
        self.latency = dict(LATENCY if latency is None else latency)
        self.facts = facts

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        role = spec.model.split("/", 1)[-1]
        time.sleep(self.latency.get(role, 0.0))
        prompt = messages[-1].content
        if role == "extract":
            body = _extraction(claims(prompt, self.facts))
        else:
            draw = zlib.crc32(prompt.encode("utf-8")) % 20
            verdict = "contradicted" if draw == 0 else "not_found" if draw < 3 else "supported"
            body = {"verdict": verdict}
        text = json.dumps(body, ensure_ascii=False)
        with self._lock:
            self.calls[role] += 1
        return Completion(
            text=text,
            parsed=body,
            model=spec.model,
            prompt_tokens=sum(len(m.content) for m in messages) // 4,
            completion_tokens=len(text) // 4,
        )


def revise(text: str, limit: int = FACTS) -> str:
    """A new version of `text`: every second sentence `claims` quotes is gone."""
    for claim in reversed(claims(text, limit)[1::2]):
        start = claim["start"]
        text = text[:start] + text[start + len(claim["quote"]) :]
    return text


# --------------------------------------------------------------------------- #
# a run's folder, and one measured command
# --------------------------------------------------------------------------- #


def run_config(
    types: Sequence[str],
    models: Mapping[str, Any],
    *,
    workers: int = 8,
    triples: bool = False,
    sink: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The run config: each type's texts at its tier, every stage, a JSONL sink.

    `triples` reads `triples.jsonl` in place of the model extractor, which is
    what `odke validate --config` runs. `workers` is each model stage's calls
    in flight.
    """
    extractor = (
        {"use": "triples", "path": "triples.jsonl"}
        if triples
        else {"use": "llm", "structured": False, "max_workers": workers}
    )
    return {
        "ontology": "ontology.json",
        "inputs": [
            {"path": f"texts/{t}", "loader": {"use": "directory", "tier": conflicts.TIERS[t].value}}
            for t in sorted(set(types))
        ],
        "models": {"meter": True, **models},
        "stages": {
            "extractor": extractor,
            "grounder": {"use": "llm", "max_workers": workers},
            "resolver": "native",
            "corroborator": "signature",
            "scorer": "evidence",
            "gate": "verdict",
            "sink": dict(sink or {"use": "jsonl", "directory": "out"}),
        },
    }


def simulated(workers: int) -> dict[str, Any]:
    """The models block for `Simulated`, `workers` calls in flight to it."""
    return {"extract": "sim/extract", "ground": "sim/ground", "limits": {"sim": workers}}


def layout(
    folder: Path,
    docs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    facts: int | None = None,
) -> Path:
    """The documents as texts by source type, the ontology, the canary and `odke.json`.

    With `facts`, also `triples.jsonl`: `claims` of each text, at most `facts`
    each, as another extractor's triples.
    """
    for doc in docs:
        path = folder / "texts" / doc["source_type"] / f"{doc['id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc["text"], encoding="utf-8")
    if facts is not None:
        with (folder / "triples.jsonl").open("w", encoding="utf-8") as rows:
            for doc in docs:
                for claim in claims(doc["text"], facts):
                    row = {k: v for k, v in claim.items() if k != "start"}
                    rows.write(json.dumps({"doc": doc["id"], **row}, ensure_ascii=False) + "\n")
    _write(folder / "ontology.json", ONTOLOGY)
    _write(folder / "CANARY.json", {"canary": conflicts.CANARY})
    _write(folder / "odke.json", dict(config))
    return folder / "odke.json"


def peak_rss_mb() -> float | None:
    """This process's own peak resident set in MB, from /proc; None off Linux."""
    try:
        status = Path("/proc/self/status").read_text()
    except OSError:
        return None
    found = re.search(r"^VmHWM:\s+(\d+) kB", status, re.MULTILINE)
    return round(int(found.group(1)) / 1024, 1) if found else None


def job(
    result: Path,
    cwd: Path,
    args: Sequence[str],
    *,
    latency: Mapping[str, float],
    facts: int,
) -> dict[str, Any]:
    """One `odke` command, in this process, on `Simulated`; what it took, written to `result`."""
    from openodke.cli.main import app
    from openodke.llm import register

    register("sim", lambda spec: Simulated(latency, facts))
    os.chdir(cwd)
    start = time.perf_counter()
    try:
        code = app(list(args), standalone_mode=False)
    except SystemExit as exc:
        code = exc.code
    took = {
        "exit": int(code or 0),
        "seconds": round(time.perf_counter() - start, 2),
        "peak_rss_mb": peak_rss_mb(),
        "calls": dict(Simulated.calls),
        # The machine's one-minute load as the job ended: another job's CPU is in the seconds.
        "load": round(os.getloadavg()[0], 2),
    }
    _write(result, took)
    return took


def spawn(
    folder: Path, args: Sequence[str], *, latency: Mapping[str, float], facts: int
) -> dict[str, Any]:
    """`job` in a fresh interpreter: its report in `folder/report.txt`, its log in `events.jsonl`.

    `odke run` and `odke validate` log as JSON (`--log-format json`), and what
    each stage took is summed from their `stage` events into `stages_s`.
    """
    result = folder / "job.json"
    if args[0] in ("run", "validate"):
        args = [*args, "--log-format", "json"]
    timing = _latency_text(latency)
    command = [sys.executable, str(Path(__file__).resolve()), "job", "--result", str(result)]
    command += ["--cwd", str(folder), "--latency", timing, "--facts", str(facts), "--", *args]
    with (
        (folder / "report.txt").open("w", encoding="utf-8") as report,
        (folder / "events.jsonl").open("w", encoding="utf-8") as events,
    ):
        subprocess.run(command, stdout=report, stderr=events, check=False)
    took: dict[str, Any] = _read(result) if result.exists() else {"exit": None}
    if took["exit"] not in (0, 3):
        raise RuntimeError(f"{' '.join(args)} exited {took['exit']}: see {folder / 'events.jsonl'}")
    return {**took, "stages_s": stage_seconds(folder / "events.jsonl")}


def stage_seconds(log: Path) -> dict[str, float]:
    """Each stage's seconds, summed over a job's `stage` events (once a micro-batch, streamed)."""
    took: Counter[str] = Counter()
    with log.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.startswith("{"):
                continue  # a line printed as text, a warning say
            event = json.loads(line)
            if event.get("event") == "stage":
                took[event["stage"]] += event.get("latency_s") or 0.0
    return {stage: round(seconds, 2) for stage, seconds in took.items()}


def _tidy(folder: Path) -> None:
    """Drop what the slice can make again: the texts, triples and facts; keep the manifests."""
    shutil.rmtree(folder / "texts", ignore_errors=True)
    for path in [folder / "triples.jsonl", *folder.glob("**/*.jsonl")]:
        path.unlink(missing_ok=True)


def _latency_text(latency: Mapping[str, float]) -> str:
    return ",".join(f"{role}={wait}" for role, wait in latency.items())


def _parse_latency(text: str) -> dict[str, float]:
    pairs = (item.split("=", 1) for item in text.split(",") if item)
    return {role: float(wait) for role, wait in pairs}


# --------------------------------------------------------------------------- #
# throughput
# --------------------------------------------------------------------------- #


def measure(
    folder: Path,
    docs: Sequence[Mapping[str, Any]],
    command: str,
    *,
    level: int = 8,
    batch_size: int | None = None,
    latency: Mapping[str, float] = LATENCY,
    facts: int = FACTS,
) -> dict[str, Any]:
    """`odke run` (the documents in) or `odke validate` (their triples in), on `Simulated`.

    `level` is each model stage's calls in flight and the provider's limit;
    `batch_size` streams the run. `bound_s` is the time the model alone needs:
    every call's wait over the limit.
    """
    shutil.rmtree(folder, ignore_errors=True)
    validate = command == "validate"
    types = [d["source_type"] for d in docs]
    config = run_config(types, simulated(level), workers=level, triples=validate)
    layout(folder, docs, config, facts=facts if validate else None)
    args = ["validate", "--config", "odke.json"] if validate else ["run", "odke.json"]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    took = spawn(folder, args, latency=latency, facts=facts)
    calls = took["calls"]
    row = {
        "command": f"odke {command}",
        "concurrency": level,
        "batch_size": batch_size,
        "documents": len(docs),
        "facts": _read(folder / "out" / "manifest.json").get("facts"),
        "calls": calls,
        "seconds": took["seconds"],
        "bound_s": round(sum(n * latency.get(r, 0.0) for r, n in calls.items()) / level, 1),
        "documents_per_s": round(len(docs) / took["seconds"], 2),
        "calls_per_s": round(sum(calls.values()) / took["seconds"], 1),
        "peak_rss_mb": took["peak_rss_mb"],
        "stages_s": took["stages_s"],
        "load": took["load"],
    }
    print(json.dumps(row), flush=True)
    _tidy(folder)
    return row


def throughput(
    root: str | Path,
    out: str | Path,
    *,
    levels: Sequence[int] = (4, 16, 64),
    per_call: int = 25,
    batch_size: int | None = None,
    latency: Mapping[str, float] = LATENCY,
    facts: int = FACTS,
    commands: Sequence[str] = ("run", "validate"),
    seed: int = 0,
) -> dict[str, Any]:
    """Each command at each concurrency limit, on `per_call` × the limit documents.

    Scaling the documents with the limit keeps the time the model alone needs
    the same at every level, so what grows is openodke's own share. The rows
    add to `throughput.json`, so a streamed pass sits beside a whole one.
    """
    docs = shuffled(read_slice(root), seed)
    target = Path(out)
    name = "whole" if batch_size is None else f"batch{batch_size}"
    rows = [
        measure(
            target / f"{command}-{level}-{name}",
            docs[: per_call * level],
            command,
            level=level,
            batch_size=batch_size,
            latency=latency,
            facts=facts,
        )
        for level in levels
        for command in commands
    ]
    return _record(target / "throughput.json", rows, latency_s=dict(latency), facts=facts)


def memory(
    root: str | Path,
    out: str | Path,
    *,
    sizes: Sequence[int] = (1000, 5000, 10000, 20000, 30000),
    batch_size: int | None = None,
    facts: int = FACTS,
    commands: Sequence[str] = ("run", "validate"),
    seed: int = 0,
) -> dict[str, Any]:
    """Peak RSS of each command as the slice grows, streamed in `batch_size` or whole.

    The first N documents of one seeded order, so each size holds the last.
    No wait per call: memory does not depend on it, and the runs are shorter.
    """
    docs = shuffled(read_slice(root), seed)
    target = Path(out)
    name = "whole" if batch_size is None else f"batch{batch_size}"
    quiet = {"extract": 0.0, "ground": 0.0}
    rows = [
        measure(
            target / f"{command}-{size}-{name}",
            docs[:size],
            command,
            batch_size=batch_size,
            latency=quiet,
            facts=facts,
        )
        for command in commands
        for size in sizes
    ]
    return _record(target / "memory.json", rows, facts=facts)


def _record(path: Path, rows: Sequence[Mapping[str, Any]], **about: Any) -> dict[str, Any]:
    """`rows` added to the results at `path`, replacing any for the same command and setting."""
    keys = ("command", "concurrency", "batch_size", "documents")
    old = _read(path)["rows"] if path.exists() else []
    new = {tuple(r.get(k) for k in keys) for r in rows}
    kept = [r for r in old if tuple(r.get(k) for k in keys) not in new]
    found = {
        "canary": conflicts.CANARY,
        "commit": conflicts.SHA,
        **about,
        "rows": kept + list(rows),
    }
    _write(path, found)
    return found


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #

# What `estimate` assumes: output tokens per input token of an extraction call
# (T-REx's run gave 0.15), output tokens per fact, and the tokens of one
# grounding call on a cited span.
OUTPUT_RATIO = 0.2
FACT_TOKENS = 100
GROUND_TOKENS = (400, 8)


def estimate(docs: Sequence[Mapping[str, Any]], extract_model: str, ground_model: str) -> dict:
    """What `cost` should spend on `docs`, priced before anything is called.

    The extraction input is exact: the messages the extractor sends, as the
    budget counts them (characters over four, and four a message). The rest is
    assumed: `OUTPUT_RATIO` of that out, a fact every `FACT_TOKENS` of output,
    and one grounding call of `GROUND_TOKENS` a fact. Prices are LiteLLM's.
    """
    from tables import price

    from openodke import Ontology
    from openodke.extract import LLMExtractor
    from openodke.types import Chunk

    ontology = Ontology.from_dict(ONTOLOGY)
    extractor = LLMExtractor(spec=ModelSpec(model=extract_model), structured=False)
    snippets = extractor.snippets(ontology)
    tokens_in = 0
    for doc in docs:
        chunk = Chunk(doc_id=doc["id"], start=0, end=len(doc["text"]), text=doc["text"], index=0)
        messages = extractor.messages(chunk, snippets)
        tokens_in += -(-sum(len(m.content) for m in messages) // 4) + 4 * len(messages)
    tokens_out = OUTPUT_RATIO * tokens_in
    grounding = tokens_out / FACT_TOKENS
    usd_extract = price(extract_model, tokens_in, tokens_out)
    usd_ground = price(ground_model, grounding * GROUND_TOKENS[0], grounding * GROUND_TOKENS[1])
    return {
        "documents": len(docs),
        "extract": {
            "calls": len(docs),
            "input_tokens": tokens_in,
            "output_tokens": round(tokens_out),
        },
        "ground": {"calls": round(grounding)},
        "usd": round((usd_extract or 0.0) + (usd_ground or 0.0), 4),
        "assumed": {
            "output_ratio": OUTPUT_RATIO,
            "fact_tokens": FACT_TOKENS,
            "ground_tokens": list(GROUND_TOKENS),
        },
    }


def _spent(records: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """The meter's calls by stage: calls, cached calls, tokens, USD and latencies."""
    out: dict[str, dict[str, Any]] = {}
    for call in records:
        row = out.setdefault(
            call.stage,
            {
                "model": call.model,
                "calls": 0,
                "cached": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "usd": 0.0,
                "latency_s": [],
            },
        )
        if call.cached:
            row["cached"] += 1
            continue
        row["calls"] += 1
        row["input_tokens"] += call.prompt_tokens
        row["output_tokens"] += call.completion_tokens
        row["usd"] += call.cost_usd or 0.0
        row["latency_s"].append(round(call.latency_s, 2))
    return out


def cost(
    root: str | Path,
    out: str | Path,
    *,
    models: Mapping[str, Any],
    per_type: int,
    budget_usd: float,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Each chosen document its own run, on the real models, under one budget.

    Each run may spend what the earlier ones left. A run's first call is
    checked against the budget only after it returns, since nothing is priced
    yet (DECISIONS #32), so the last run can pass the budget by one call.
    """
    from openodke.run.build import build
    from openodke.run.config import parse_config
    from openodke.run.execute import run_built

    target = Path(out)
    chosen = select(read_slice(root), per_type, seed)[:limit]
    rows: list[dict[str, Any]] = []
    spent = 0.0
    for doc in chosen:
        left = round(budget_usd - spent, 6)
        if left <= 0:
            rows.append({"id": doc["id"], "source_type": doc["source_type"], "run": False})
            continue
        folder = target / "docs" / doc["id"]
        config = run_config([doc["source_type"]], {**models, "budget": {"usd": left}})
        layout(folder, [doc], config)
        built = build(parse_config(config, base_dir=folder))
        start = time.perf_counter()
        stats = run_built(built).stats
        meter = built.context.meter
        stages = _spent(meter.records if meter is not None else [])
        usd = sum(row["usd"] for row in stages.values())
        spent += usd
        rows.append(
            {
                "id": doc["id"],
                "source_type": doc["source_type"],
                "run": True,
                "chars": len(doc["text"]),
                "facts": stats["graph"]["facts"],
                "seconds": round(time.perf_counter() - start, 2),
                "usd": round(usd, 6),
                "stages": stages,
                "cache": stats.get("cache"),
                "stopped": stats.get("stopped"),
                "failed": stats.get("failed"),
            }
        )
        print(f"{doc['source_type']} {doc['id']}: {rows[-1]['facts']} facts, ${spent:.4f} so far")
    found = {
        "canary": conflicts.CANARY,
        "commit": conflicts.SHA,
        "models": dict(models),
        "budget_usd": budget_usd,
        "spent_usd": round(spent, 6),
        "documents": rows,
    }
    _write(target / "cost.json", found)
    return found


def per_thousand(
    rows: Sequence[Mapping[str, Any]],
    mean_chars: Mapping[str, float],
    shares: Mapping[str, float],
    *,
    draws: int = 2000,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """USD per 1,000 documents by source type, and for the corpus's own mix, with a 95% range.

    A type's cost per character on the documents run, times the type's mean
    length in the slice: a ratio estimate, so a sample of short documents is
    not read as a cheap type. The range resamples the documents run, each type
    apart (2,000 draws); `corpus` weighs the types by `shares`.
    """
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("run") and not row.get("stopped"):
            groups[row["source_type"]].append(row)

    def value(by_type: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
        out = {
            t: 1000 * mean_chars[t] * sum(r["usd"] for r in g) / sum(r["chars"] for r in g)
            for t, g in by_type.items()
        }
        weight = sum(shares[t] for t in out)
        out["corpus"] = sum(shares[t] * v for t, v in out.items()) / weight
        return out

    point = value(groups)
    rng = random.Random(seed)
    resampled = [
        value({t: rng.choices(g, k=len(g)) for t, g in groups.items()}) for _ in range(draws)
    ]
    found: dict[str, dict[str, Any]] = {}
    for t, v in point.items():
        spread = sorted(draw[t] for draw in resampled)
        group = groups.get(t, [r for g in groups.values() for r in g])
        found[t] = {
            "documents": len(group),
            "usd_per_1000": round(v, 2),
            "range": [
                round(spread[int(0.025 * draws)], 2),
                round(spread[int(0.975 * draws) - 1], 2),
            ],
            "facts_per_document": round(sum(r["facts"] for r in group) / len(group), 1),
        }
    return found


# --------------------------------------------------------------------------- #
# the reconciler
# --------------------------------------------------------------------------- #

_RECONCILED = re.compile(r"^(cited|lost support|retired|deleted)\s+(\d+)", re.MULTILINE)
_UPDATED = re.compile(r"(\d+) facts? cited them, (\d+) kept a source, (\d+) left with none")


def _retracted(report: str) -> dict[str, int]:
    """What a retraction printed it did: facts cited, those that kept a source, those left bare.

    `odke reconcile` prints a line each; `odke validate --update` one line,
    counted before the new versions are written.
    """
    found = _UPDATED.search(report)
    if found is not None:
        return dict(zip(("cited", "lost", "retired"), map(int, found.groups()), strict=True))
    return {name.replace(" support", ""): int(n) for name, n in _RECONCILED.findall(report)}


def _store(folder: Path) -> dict[str, int]:
    facts = retired = 0
    with (folder / "facts.jsonl").open(encoding="utf-8") as lines:
        for line in lines:
            facts += 1
            retired += json.loads(line).get("retired_at") is not None
    return {"facts": facts, "retired": retired}


def reconcile(
    root: str | Path,
    out: str | Path,
    *,
    documents: int = 2000,
    updated: float = 0.1,
    deleted: float = 0.02,
    facts: int = FACTS,
    seed: int = 0,
) -> dict[str, Any]:
    """A store of `documents`, then a share updated and a share deleted, each timed.

    `odke validate` writes the store from triples (`claims` of each text) into
    a JSONL sink that merges. The first `updated` share then gets a new
    version, `revise`d, with its own triples, through `odke validate --update`;
    the next `deleted` share is retracted by `odke reconcile --delete`. Each is
    a fresh process on `Simulated` with no wait.
    """
    docs = shuffled(read_slice(root), seed)[:documents]
    target = Path(out)
    shutil.rmtree(target, ignore_errors=True)
    types = [d["source_type"] for d in docs]
    sink = {"use": "jsonl", "directory": "../store", "merge": True}
    models = simulated(8)
    quiet = {"extract": 0.0, "ground": 0.0}
    steps: dict[str, Any] = {}

    def step(name: str, folder: Path, args: list[str]) -> None:
        took = spawn(folder, args, latency=quiet, facts=facts)
        steps[name] = {
            "seconds": took["seconds"],
            "peak_rss_mb": took["peak_rss_mb"],
            "stages_s": took["stages_s"],
            "load": took["load"],
            "calls": took["calls"],
            "store": _store(target / "store"),
        }
        print(name, json.dumps(steps[name]))

    first = target / "first"
    layout(first, docs, run_config(types, models, triples=True, sink=sink), facts=facts)
    step("validate", first, ["validate", "--config", "odke.json"])
    changed = docs[: round(updated * len(docs))]
    revised = [{**doc, "text": revise(doc["text"], facts)} for doc in changed]
    second = target / "update"
    types = [d["source_type"] for d in revised]
    layout(second, revised, run_config(types, models, triples=True, sink=sink), facts=facts)
    step("update", second, ["validate", "--config", "odke.json", "--update"])
    steps["update"]["documents"] = len(revised)
    steps["update"]["retracted"] = _retracted((second / "report.txt").read_text())
    gone = docs[len(changed) : len(changed) + round(deleted * len(docs))]
    ids = [f"texts/{d['source_type']}/{d['id']}.txt" for d in gone]
    step(
        "delete",
        target,
        ["reconcile", "--sink", "store", *(a for i in ids for a in ("--delete", i))],
    )
    steps["delete"]["documents"] = len(gone)
    steps["delete"]["retracted"] = _retracted((target / "report.txt").read_text())
    found = {
        "canary": conflicts.CANARY,
        "commit": conflicts.SHA,
        "documents": len(docs),
        "facts_per_passage": facts,
        "steps": steps,
    }
    _write(target / "reconcile.json", found)
    for folder in (first, second):
        _tidy(folder)
    return found


# --------------------------------------------------------------------------- #
# the tables
# --------------------------------------------------------------------------- #


def table(runs: str | Path) -> str:
    """Every measurement under `runs` that has results, as Markdown."""
    base = Path(runs)
    parts: list[str] = []
    if (found := base / "throughput" / "throughput.json").exists():
        parts.append(throughput_table(_read(found)))
    if (found := base / "memory" / "memory.json").exists():
        parts.append(memory_table(_read(found)))
    if done := sorted(base.glob("reconcile/*/reconcile.json")):
        parts.append(reconcile_table([_read(path) for path in done]))
    if (found := base / "cost" / "cost.json").exists():
        data = _read(found)
        parts.append(cost_table(data, _read(Path(data["slice"]) / "slice.json")))
    return "\n\n".join(parts) + "\n"


def _streamed(row: Mapping[str, Any]) -> str:
    size = row.get("batch_size")
    if size is None:
        return "no"
    return f"{size:,} {'rows' if row['command'] == 'odke validate' else 'documents'}"


def throughput_table(data: Mapping[str, Any]) -> str:
    lines = [
        "| Command | Concurrency | Streamed | Documents | Model calls | Seconds | Model-bound s"
        " | Documents/s | Calls/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(data["rows"], key=lambda r: (r["command"], r["concurrency"])):
        lines.append(
            f"| {r['command']} | {r['concurrency']} | {_streamed(r)} | {r['documents']:,} |"
            f" {sum(r['calls'].values()):,} | {r['seconds']} | {r['bound_s']} |"
            f" {r['documents_per_s']} | {r['calls_per_s']} |"
        )
    return "\n".join(lines)


def memory_table(data: Mapping[str, Any]) -> str:
    lines = [
        "| Command | Documents | Streamed | Facts | Peak RSS MB | Seconds | Documents/s |",
        "|---|---|---|---|---|---|---|",
    ]
    order = sorted(
        data["rows"], key=lambda r: (r["command"], r["batch_size"] is not None, r["documents"])
    )
    for r in order:
        lines.append(
            f"| {r['command']} | {r['documents']:,} | {_streamed(r)} | {r['facts']:,} |"
            f" {r['peak_rss_mb']} | {r['seconds']} | {r['documents_per_s']} |"
        )
    return "\n".join(lines)


def reconcile_table(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Store of | Step | Documents | Seconds | Peak RSS MB | Cited | Kept a source"
        " | Left with none | Facts in the store | Retired in the store |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for data in sorted(results, key=lambda d: d["documents"]):
        for name, s in data["steps"].items():
            done = s.get("retracted", {})
            lines.append(
                f"| {data['documents']:,} | {name} | {s.get('documents', data['documents']):,} |"
                f" {s['seconds']} | {s['peak_rss_mb']} | {done.get('cited', '—')} |"
                f" {done.get('lost', '—')} | {done.get('retired', '—')} |"
                f" {s['store']['facts']:,} | {s['store']['retired']:,} |"
            )
    return "\n".join(lines)


def cost_table(data: Mapping[str, Any], about: Mapping[str, Any]) -> str:
    """USD per 1,000 documents by source type, as `per_thousand` estimates it."""
    total = sum(about["corpus"].values())
    shares = {t: n / total for t, n in about["corpus"].items()}
    found = per_thousand(data["documents"], about["mean_chars"], shares)
    lines = [
        "| Source type | Share of corpus | Documents run | Facts/document | USD per 1,000"
        " | 95% range |",
        "|---|---|---|---|---|---|",
    ]
    for t, row in found.items():
        share = "100%" if t == "corpus" else f"{100 * shares[t]:.1f}%"
        low, high = row["range"]
        lines.append(
            f"| {t} | {share} | {row['documents']} | {row['facts_per_document']} |"
            f" {row['usd_per_1000']:.2f} | {low:.2f}–{high:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("fetch", help="a uniform sample of the corpus")
    get.add_argument("dest", type=Path)
    get.add_argument("--documents", type=int, default=30000)
    get.add_argument("--seed", type=int, default=0)
    pay = sub.add_parser("cost", help="the real models, a few documents of each type")
    pay.add_argument("root", type=Path)
    pay.add_argument("--out", type=Path, required=True)
    pay.add_argument("--extract-model", required=True)
    pay.add_argument("--ground-model", required=True)
    pay.add_argument("--per-type", type=int, default=5)
    pay.add_argument("--budget-usd", type=float, required=True)
    pay.add_argument("--cache", type=Path, help="the response cache's directory")
    pay.add_argument("--limit", type=int, help="the first N documents only (a pilot)")
    pay.add_argument("--estimate", action="store_true", help="price them; call nothing")
    pay.add_argument(
        "--env-file", default=os.environ.get("ODKE_ENV_FILE"), help="KEY=value lines to load."
    )
    speed = sub.add_parser("throughput", help="odke run and odke validate on Simulated")
    speed.add_argument("root", type=Path)
    speed.add_argument("--out", type=Path, required=True)
    speed.add_argument("--levels", default="4,16,64", help="concurrency limits")
    speed.add_argument("--per-call", type=int, default=25, help="documents per unit of limit")
    speed.add_argument("--latency", default=_latency_text(LATENCY), help="seconds per call")
    speed.add_argument("--batch-size", type=int, help="stream in micro-batches of this many")
    speed.add_argument("--facts", type=int, default=FACTS)
    speed.add_argument("--commands", default="run,validate")
    held = sub.add_parser("memory", help="peak RSS as the slice grows, on Simulated")
    held.add_argument("root", type=Path)
    held.add_argument("--out", type=Path, required=True)
    held.add_argument("--sizes", default="1000,5000,10000,20000,30000")
    held.add_argument("--batch-size", type=int, help="stream in micro-batches of this many")
    held.add_argument("--facts", type=int, default=FACTS)
    held.add_argument("--commands", default="run,validate")
    again = sub.add_parser("reconcile", help="a store, then updates and deletes")
    again.add_argument("root", type=Path)
    again.add_argument("--out", type=Path, required=True)
    again.add_argument("--documents", type=int, default=2000)
    again.add_argument("--updated", type=float, default=0.1)
    again.add_argument("--deleted", type=float, default=0.02)
    again.add_argument("--facts", type=int, default=FACTS)
    show = sub.add_parser("table", help="the results as Markdown")
    show.add_argument("runs", type=Path)
    one = sub.add_parser("job", help="one odke command on Simulated (used by the others)")
    one.add_argument("--result", type=Path, required=True)
    one.add_argument("--cwd", type=Path, required=True)
    one.add_argument("--latency", default="")
    one.add_argument("--facts", type=int, default=FACTS)
    one.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "fetch":
        print(f"fetched into {fetch(args.dest, documents=args.documents, seed=args.seed)}")
    elif args.command == "cost":
        chosen = select(read_slice(args.root), args.per_type)[: args.limit]
        priced = estimate(chosen, args.extract_model, args.ground_model)
        print(f"estimate: {json.dumps(priced)}")
        if args.estimate:
            return
        from competitors import load_env

        load_env(args.env_file)
        models: dict[str, Any] = {
            "extract": {"model": args.extract_model, "max_tokens": 8000},
            "ground": args.ground_model,
        }
        if args.cache is not None:
            models["cache"] = str(args.cache.resolve())
        found = cost(
            args.root,
            args.out,
            models=models,
            per_type=args.per_type,
            budget_usd=args.budget_usd,
            limit=args.limit,
        )
        _write(args.out / "cost.json", {**found, "estimate": priced, "slice": str(args.root)})
        print(f"spent ${found['spent_usd']:.4f} (estimated ${priced['usd']:.4f})")
    elif args.command == "throughput":
        throughput(
            args.root,
            args.out,
            levels=[int(n) for n in args.levels.split(",")],
            per_call=args.per_call,
            batch_size=args.batch_size,
            latency=_parse_latency(args.latency),
            facts=args.facts,
            commands=args.commands.split(","),
        )
    elif args.command == "memory":
        memory(
            args.root,
            args.out,
            sizes=[int(n) for n in args.sizes.split(",")],
            batch_size=args.batch_size,
            facts=args.facts,
            commands=args.commands.split(","),
        )
    elif args.command == "reconcile":
        reconcile(
            args.root,
            args.out,
            documents=args.documents,
            updated=args.updated,
            deleted=args.deleted,
            facts=args.facts,
        )
    elif args.command == "table":
        print(table(args.runs), end="")
    else:
        rest = args.args[1:] if args.args[:1] == ["--"] else args.args
        job(args.result, args.cwd, rest, latency=_parse_latency(args.latency), facts=args.facts)


if __name__ == "__main__":
    main()
