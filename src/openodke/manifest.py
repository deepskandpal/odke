"""The run manifest: what a run was asked, what answered it, and what it made (#160).

A graph is only as good as the run that made it, and "which model, which
prompt, which schema, which texts" is what a person asks first when a number
moves. So every run writes one manifest: `odke run`, `odke validate`,
`odke ground` and `Validator.validate`. It holds

- `config` and `config_hash`: the canonical resolved config, every default
  filled in and every secret taken out (`canonical`), and its SHA-256;
- `models`: each role's provider-qualified model id, and the ids its provider
  said answered (`served`), which is the pinned id when the config named an
  alias;
- `prompts`: each registered prompt key the run sent, with its SHA-256
  (DECISIONS #27);
- `ontology_version` and `ontology_hash`, the schema's label and its
  `Ontology.fingerprint`;
- `package`: openodke's version, Python's and the platform;
- `inputs`: each document's id with the SHA-256 of its text, the facts handed
  in (a triples file's rows, say) as a count and a hash, and `hash`, one over
  all of it;
- `cache` and `budget`, `started_at` and `ended_at`;
- `counts`, `spent`, and the `stopped` and `failed` summaries;
- `run`, the id every log event of the run carries, and `job`, the counts its
  `job.end` event logs (`openodke.observe`), so a manifest and a log stream
  can be joined, and say the same.

Two runs with the same inputs write the same manifest but for its two times
and its run id.

The manifest is `manifest.json`. A JSONL sink already writes one there, with
the graph's counts and stats; the run adds its fields beside them after the
sinks have written, so every key an older reader knows is still there and
means what it meant. A run with no JSONL sink writes the same document on its
own (`RunManifest.write`), the graph's fields included.

Nothing here reads a secret. `canonical` replaces any value under a key that
names one (`password`, `token`, `api_key`, `secret`...) and the password in a
URL, so the hash is of the config without them and a manifest can be shared.
A key that only names where a secret is read (`password_env`) is kept.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from openodke.llm.base import Completion, LLMClient, Message, ModelSpec
from openodke.types import Document, Fact, Frozen, KnowledgeGraph

if TYPE_CHECKING:
    from openodke.ontology import Ontology

# The format of the run's own fields; a reader checks it before trusting them.
FORMAT = 1
FILE = "manifest.json"
REDACTED = "<redacted>"

# What a `JsonlSink` writes into `manifest.json`. A run adds its own fields
# beside these and never replaces them: the sink's counts are the files'.
SINK_KEYS = ("ontology", "created_at", "entities", "facts", "edges", "properties", "links", "stats")

Command = Literal["run", "validate", "ground"]

# A key that names a secret, as a word of its own: `api_key`, `neo4j_password`,
# `token`, but not `max_tokens` or `password_env`, which says where one is read.
_SECRET = re.compile(
    r"(^|[_-])(password|passwd|secret|token|api[_-]?key|apikey|credentials?|authorization|auth)"
    r"($|[_-])",
    re.IGNORECASE,
)
_USERINFO = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^/@:\s]*):[^/@\s]*@")


def now() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Canonical form and hashes
# --------------------------------------------------------------------------- #


def canonical(value: Any) -> Any:
    """`value` as plain JSON types, with every secret replaced by `<redacted>`.

    A value under a key that names a secret is replaced whatever it is; a key
    ending in `_env` names an environment variable, not its value, and is kept.
    A URL's password is replaced and its user kept: `bolt://neo4j:<redacted>@db`.
    """
    plain = json.loads(json.dumps(value, default=str, ensure_ascii=False))
    return _redact(plain)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_secret(key) and item is not None else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _USERINFO.sub(rf"\g<scheme>\g<user>:{REDACTED}@", value)
    return value


def _is_secret(key: str) -> bool:
    return not key.lower().endswith("_env") and _SECRET.search(key) is not None


def digest(value: Any) -> str:
    """The SHA-256 of `value` as canonical JSON: keys sorted, no whitespace."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def qualified(model: str, provider: str) -> str:
    """`model` with its provider in front, unless it already starts with it."""
    return model if model.startswith(f"{provider}/") else f"{provider}/{model}"


def package() -> dict[str, str]:
    """openodke's version, Python's, and the platform the run ran on."""
    from openodke import __version__

    return {
        "openodke": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #


class ModelUse(Frozen):
    """One role's model: the id the run asked for, and the ids its provider said answered."""

    model: str
    served: tuple[str, ...] = ()


class FactsIn(Frozen):
    """The facts a run was handed rather than extracted: how many, and their hash."""

    rows: int = 0
    hash: str


class Inputs(Frozen):
    """What the run read: each document's text by hash, the facts handed in, and one hash."""

    documents: dict[str, str] = Field(default_factory=dict)
    facts: FactsIn | None = None
    hash: str


class RunManifest(Frozen):
    """One run, enough to run it again: see the module docstring for every field."""

    manifest_version: int = FORMAT
    command: Command
    dry_run: bool = False
    # The id every log event of this run carries (`openodke.observe`).
    run: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str
    # Where the config's relative paths resolve, and the file it was read from,
    # relative to that: what `odke run --from-manifest` needs to run it again.
    config_file: str | None = None
    base_dir: str | None = None
    models: dict[str, ModelUse] = Field(default_factory=dict)
    prompts: dict[str, str | None] = Field(default_factory=dict)
    ontology_version: str | None = None
    ontology_hash: str | None = None
    package: dict[str, str] = Field(default_factory=package)
    inputs: Inputs
    cache: str | None = None
    budget: dict[str, float | int] | None = None
    started_at: datetime
    ended_at: datetime
    counts: dict[str, int] = Field(default_factory=dict)
    # Facts in, out, refused, merged, linked and sent to review: what `job.end` logs.
    job: dict[str, int] = Field(default_factory=dict)
    spent: dict[str, Any] = Field(default_factory=dict)
    stopped: dict[str, Any] | None = None
    failed: dict[str, str] = Field(default_factory=dict)

    def document(self, kg: KnowledgeGraph | None = None) -> dict[str, Any]:
        """The manifest as `manifest.json` holds it; with `kg`, the graph's fields beside it."""
        own = self.model_dump(mode="json")
        if kg is None:
            return own
        from openodke.sinks.jsonl import manifest_of

        return {**own, **manifest_of(kg, kg.entities, kg.facts, kg.links)}

    def write(self, path: str | Path, kg: KnowledgeGraph | None = None) -> Path:
        """A manifest of its own at `path`: this run's fields, and the graph's beside them."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.document(kg), indent=2) + "\n", encoding="utf-8")
        return target

    def write_into(self, path: str | Path) -> Path:
        """This run's fields added to the `manifest.json` a JSONL sink has just written.

        The sink's own keys (`SINK_KEYS`) are kept as the sink wrote them: in a
        store that merges, its counts are the files', not this run's.
        """
        target = Path(path)
        held: dict[str, Any] = {}
        if target.is_file():
            found = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(found, dict):
                held = {key: found[key] for key in SINK_KEYS if key in found}
        document = {**self.model_dump(mode="json"), **held}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return target

    def differences(
        self, *, ontology: Ontology | None = None, inputs: Inputs | None = None
    ) -> list[str]:
        """What is not as this manifest recorded it: the ontology, and the inputs, by document."""
        problems: list[str] = []
        recorded = self.ontology_hash
        if ontology is not None and recorded is not None and ontology.fingerprint != recorded:
            problems.append(
                f"the ontology changed: fingerprint {ontology.fingerprint[:12]}, "
                f"recorded {recorded[:12]}"
            )
        if inputs is None or inputs.hash == self.inputs.hash:
            return problems
        was, found = self.inputs.documents, inputs.documents
        changed = sorted(d for d in was.keys() & found.keys() if was[d] != found[d])
        for label, ids in (
            ("changed", changed),
            ("gone", sorted(was.keys() - found.keys())),
            ("new", sorted(found.keys() - was.keys())),
        ):
            if ids:
                more = f" and {len(ids) - 5} more" if len(ids) > 5 else ""
                problems.append(f"documents {label}: {', '.join(ids[:5])}{more}")
        if self.inputs.facts != inputs.facts:
            problems.append("the facts handed in changed")
        return problems


def read_manifest(path: str | Path) -> RunManifest:
    """The run manifest in `path`, a `manifest.json` or a directory holding one.

    Raises ValueError for a file that holds none: a manifest a JSONL sink wrote
    before 1.0 has the graph's fields and no run.
    """
    source = Path(path)
    if source.is_dir():
        source = source / FILE
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "config_hash" not in data:
        raise ValueError(f"{source}: not a run manifest (no config_hash); it predates them")
    version = data.get("manifest_version")
    if version != FORMAT:
        raise ValueError(f"{source}: manifest_version {version!r}; this openodke reads {FORMAT}")
    return RunManifest.model_validate({k: v for k, v in data.items() if k not in SINK_KEYS})


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def inputs_of(documents: Iterable[Document], facts: Sequence[Any] | None = None) -> Inputs:
    """Each document's text by hash, the facts handed in by count and hash, and one hash."""
    hashes = {doc.id: text_hash(doc.text) for doc in documents}
    handed = None
    if facts is not None:
        rows = [_row(item) for item in facts]
        handed = FactsIn(rows=len(rows), hash=digest(rows))
    rollup = digest({"documents": hashes, "facts": handed.hash if handed else None})
    return Inputs(documents=hashes, facts=handed, hash=rollup)


def _row(item: Any) -> Any:
    """A fact handed in, as plain data, without what changes on every load.

    A `Fact` mints its id and stamps its evidence with the time it was made, so
    two loads of one file differ there and nowhere else.
    """
    if isinstance(item, Fact):
        data = item.model_dump(mode="json", exclude={"id"})
        for evidence in data.get("evidence", ()):
            evidence.pop("retrieved_at", None)
        for support in data.get("supported_by", ()):
            support.pop("retrieved_at", None)
        return data
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return canonical(item)


def prompt_hashes(keys: Iterable[str]) -> dict[str, str | None]:
    """Each registered prompt key with its SHA-256; None for a key the registry lacks."""
    from openodke import prompts

    out: dict[str, str | None] = {}
    for key in keys:
        try:
            out[str(key)] = prompts.get(str(key)).sha256
        except (LookupError, ValueError):
            out[str(key)] = None
    return out


# --------------------------------------------------------------------------- #
# Recording a run
# --------------------------------------------------------------------------- #


class Served:
    """The models a run asked for, by role, and the ids their providers said answered.

    `client(role, spec, inner)` wraps the client a stage calls, so each answer's
    `Completion.model` is noted under the role. A model whose answers are not
    seen, a grounder handed in say, is noted with `note` and has no `served`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._asked: dict[str, str] = {}
        self._served: dict[str, dict[str, None]] = {}

    def note(self, role: str, spec: ModelSpec) -> None:
        with self._lock:
            self._asked.setdefault(role, qualified(spec.model, spec.provider))
            self._served.setdefault(role, {})

    def client(self, role: str, spec: ModelSpec, inner: LLMClient) -> LLMClient:
        """`inner`, with every answer's model noted under `role`."""
        self.note(role, spec)
        return _ServedClient(inner, self, role)

    def answered(self, role: str, spec: ModelSpec, completion: Completion) -> None:
        if not completion.model:
            return
        with self._lock:
            self._asked.setdefault(role, qualified(spec.model, spec.provider))
            served = self._served.setdefault(role, {})
            served[qualified(completion.model, spec.provider)] = None

    def models(self) -> dict[str, ModelUse]:
        with self._lock:
            return {
                role: ModelUse(model=model, served=tuple(self._served.get(role, ())))
                for role, model in sorted(self._asked.items())
            }


class _ServedClient:
    def __init__(self, inner: LLMClient, served: Served, role: str) -> None:
        self.inner = inner
        self.served = served
        self.role = role

    def complete(
        self,
        messages: Sequence[Message],
        *,
        spec: ModelSpec,
        schema: dict[str, Any] | None = None,
    ) -> Completion:
        completion = self.inner.complete(messages, spec=spec, schema=schema)
        self.served.answered(self.role, spec, completion)
        return completion


class Recorder:
    """One run's manifest in the making: begun before the run, finished once it has written.

    `config` is what the run was asked to do, as a mapping, and may be added
    to until the run ends, as `odke validate` adds the models it resolves. It
    is made canonical when the manifest is finished, so a secret in it never
    reaches the manifest or its hash. `cache` and `budget` may be set the same
    way, by whoever builds the clients.
    """

    def __init__(
        self,
        command: Command,
        config: Mapping[str, Any],
        *,
        config_file: str | None = None,
        base_dir: str | Path | None = None,
        served: Served | None = None,
        cache: str | Path | None = None,
        budget: Mapping[str, float | int] | None = None,
    ) -> None:
        self.command: Command = command
        self.config: dict[str, Any] = dict(config)
        self.config_file = config_file
        self.base_dir = str(base_dir) if base_dir is not None else None
        self.served = served if served is not None else Served()
        self.cache = cache
        self.budget = budget
        self.started_at = now()

    def finish(
        self,
        *,
        inputs: Inputs,
        ontology: Ontology | None = None,
        prompts: Iterable[str] = (),
        counts: Mapping[str, int] | None = None,
        spent: Mapping[str, Any] | None = None,
        stopped: Mapping[str, Any] | None = None,
        failed: Mapping[str, str] | None = None,
        dry_run: bool = False,
        run: str | None = None,
        job: Mapping[str, int] | None = None,
    ) -> RunManifest:
        """The manifest, ended now. An ontology with no types and no predicates is none."""
        schema = (
            ontology if ontology is not None and (ontology.types or ontology.predicates) else None
        )
        config = canonical(self.config)
        return RunManifest(
            command=self.command,
            dry_run=dry_run,
            run=run,
            config=config,
            config_hash=digest(config),
            config_file=self.config_file,
            base_dir=self.base_dir,
            models=self.served.models(),
            prompts=prompt_hashes(prompts),
            ontology_version=schema.version if schema is not None else None,
            ontology_hash=schema.fingerprint if schema is not None else None,
            inputs=inputs,
            cache=str(self.cache) if self.cache is not None else None,
            budget=dict(self.budget) if self.budget else None,
            started_at=self.started_at,
            ended_at=now(),
            counts={str(k): int(v) for k, v in (counts or {}).items()},
            job={str(k): int(v) for k, v in (job or {}).items()},
            spent=canonical(dict(spent or {})),
            stopped=canonical(dict(stopped)) if stopped else None,
            failed={str(k): str(v) for k, v in (failed or {}).items()},
        )


def spent_of(stats: Mapping[str, Any]) -> dict[str, Any]:
    """A stage's model calls, tokens and cost, as a manifest's `spent` names them."""
    usd = stats.get("cost_usd", stats.get("usd"))
    return {
        "calls": int(stats.get("calls", 0)),
        "cached_calls": int(stats.get("cached", stats.get("cached_calls", 0))),
        "input_tokens": int(stats.get("prompt_tokens", stats.get("input_tokens", 0))),
        "output_tokens": int(stats.get("completion_tokens", stats.get("output_tokens", 0))),
        "usd": float(usd) if isinstance(usd, int | float) else None,
    }


__all__ = [
    "FILE",
    "FORMAT",
    "REDACTED",
    "SINK_KEYS",
    "FactsIn",
    "Inputs",
    "ModelUse",
    "Recorder",
    "RunManifest",
    "Served",
    "canonical",
    "digest",
    "inputs_of",
    "package",
    "prompt_hashes",
    "qualified",
    "read_manifest",
    "spent_of",
    "text_hash",
]
