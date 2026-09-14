"""An inferred ontology written out for a person to review, and written back once frozen.

The review is the step that makes inference safe (DECISIONS #8), so the file is
built for it. YAML, when PyYAML is installed (`odke[yaml]`), opens with a header
saying the schema is inferred and what to do about it, and every type and
predicate carries its evidence as comments directly above it: how many
documents support it, which proposers found it, and the first few quotes with
where they came from. Comments, so the file still loads as an ontology and the
evidence disappears the moment a reviewer deletes the entry it backs.

JSON has no comments. A JSON draft is the bare ontology, and the header and the
evidence live in the `<name>.evidence.json` file written beside it — which a
YAML draft gets too, with the complete evidence the comments abbreviate.

Entries are written without the fields they leave at their defaults, and
without a `name` that repeats its key, so a reviewer reads what was inferred
rather than forty lines of `required: false`. Loading the file gives back the
same ontology.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from odke.infer.build import EvidenceRef, Inference
from odke.ontology import Ontology

Format = Literal["yaml", "json"]
_QUOTES_SHOWN = 3
_QUOTE_WIDTH = 72


def format_for(path: Path) -> Format:
    """`yaml` for `.yaml`/`.yml`, `json` for `.json`; anything else is refused."""
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(f"{path}: write the ontology as .yaml, .yml or .json")


def evidence_path(path: Path) -> Path:
    """`ontology.yaml` → `ontology.evidence.json`, beside it."""
    return path.with_name(f"{path.stem}.evidence.json")


def ontology_data(ontology: Ontology) -> dict[str, Any]:
    """The ontology as plain data, defaults and repeated names left out."""
    data = ontology.model_dump(mode="json", exclude_defaults=True)
    for section in ("types", "predicates"):
        for key, body in (data.get(section) or {}).items():
            if body.get("name") == key:
                body.pop("name")
    return data


def render(
    ontology: Ontology,
    *,
    fmt: Format,
    header: Sequence[str] = (),
    notes: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """The file a reviewer opens. `notes` are comment lines keyed `types.X` / `predicates.y`."""
    data = ontology_data(ontology)
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            'writing YAML needs PyYAML. Run: pip install "odke[yaml]" — or write .json'
        ) from exc

    def dump(value: Any) -> list[str]:
        text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100)
        return text.rstrip("\n").splitlines()

    lines = [f"# {line}".rstrip() for line in header]
    scalars = {k: v for k, v in data.items() if k not in {"types", "predicates"}}
    if scalars:
        lines += dump(scalars)
    for section in ("types", "predicates"):
        entries = data.get(section) or {}
        if not entries:
            lines.append(f"{section}: {{}}")
            continue
        lines.append(f"{section}:")
        for key, body in entries.items():
            lines += [
                # Newlines out, indentation kept: a quote sits under its support line.
                f"  # {note.replace(chr(13), ' ').replace(chr(10), ' ')}".rstrip()
                for note in (notes or {}).get(f"{section}.{key}", ())
            ]
            lines += [f"  {line}" for line in dump({key: body})]
    return "\n".join(lines) + "\n"


def inferred_header(inference: Inference, *, evidence_file: str | None = None) -> list[str]:
    sample = inference.sample
    model = inference.settings.get("model") or "none — deterministic proposers only"
    lines = [
        "INFERRED ONTOLOGY — review it before anything is extracted against it.",
        "",
        "Proposed from a corpus by odke; no person has checked it yet (DECISIONS #8).",
        "Each entry's evidence is in the comments above it. Edit or delete what is",
        "wrong, run `odke ontology validate` on this file, then `odke ontology freeze`.",
        "",
        f"sample: {sample.words} words, {len(sample.chunks)} chunks from "
        f"{sample.documents_sampled} of {sample.documents_seen} documents, seed {sample.seed}",
        f"model: {model}",
    ]
    if evidence_file:
        lines.append(f"evidence: {evidence_file}")
    return lines


def frozen_header(ontology: Ontology) -> list[str]:
    when = ontology.frozen_at.isoformat() if ontology.frozen_at else "an unknown time"
    return [
        f"Reviewed and frozen by {ontology.frozen_by or 'an unknown reviewer'} at {when}.",
        "From here on an edit is a schema change: check it with `odke ontology diff`.",
    ]


def notes(inference: Inference) -> dict[str, list[str]]:
    """The comment lines above each entry: support, proposers, and the first quotes."""
    out: dict[str, list[str]] = {}
    for path, evidence in inference.evidence.items():
        plural = "" if evidence.support == 1 else "s"
        lines = [
            f"support: {evidence.support} document{plural}, "
            f"found by {', '.join(evidence.proposers)}"
        ]
        lines += [
            f'  "{_clip(ref.quote)}" — {where(ref)}' for ref in evidence.spans[:_QUOTES_SHOWN]
        ]
        out[path] = lines
    return out


def summary(inference: Inference) -> str:
    """What `odke ontology infer` prints: every proposal with what backs it, then the rest."""
    ontology = inference.ontology
    lines = [f"types ({len(ontology.types)})"]
    for name, entity in ontology.types.items():
        extra = f" is a {', '.join(entity.parents)}" if entity.parents else ""
        lines.append(f"  {name}{extra}  {_backing(inference, f'types.{name}')}")
    lines.append(f"predicates ({len(ontology.predicates)})")
    for name, predicate in ontology.predicates.items():
        domain = ", ".join(predicate.domain) or "any"
        aliases = f"  aka {', '.join(predicate.aliases)}" if predicate.aliases else ""
        lines.append(
            f"  {name}: {domain} -> {predicate.range}, {predicate.cardinality}{aliases}  "
            f"{_backing(inference, f'predicates.{name}')}"
        )
    for title, items in (
        ("merges", [str(d) for d in inference.decisions]),
        ("dropped", [str(d) for d in inference.dropped]),
        (
            "rejected from the model",
            [f"{r.kind} {r.name}: {r.reason}" for r in inference.rejections],
        ),
    ):
        if items:
            lines.append(f"{title} ({len(items)})")
            lines += [f"  {item}" for item in items]
    if inference.calls:
        known = [c.cost_usd for c in inference.calls if c.cost_usd is not None]
        cost = f"${sum(known):.4f}" if len(known) == len(inference.calls) else "cost unknown"
        tokens = sum(c.prompt_tokens + c.completion_tokens for c in inference.calls)
        lines.append(f"model calls: {len(inference.calls)}, {tokens} tokens, {cost}")
    return "\n".join(lines)


def where(ref: EvidenceRef) -> str:
    """`people.csv line 3`, `notes.md @120`: where a quote came from, short enough to scan."""
    if ref.source:
        parsed = urlparse(ref.source)
        name = PurePosixPath(unquote(parsed.path)).name or ref.source
    else:
        name = f"document {ref.doc_key}" if ref.doc_key else "document"
    return f"{name} line {ref.line}" if ref.line is not None else f"{name} @{ref.start}"


def _backing(inference: Inference, path: str) -> str:
    evidence = inference.evidence.get(path)
    if evidence is None:
        return ""
    first = evidence.spans[0] if evidence.spans else None
    quote = f' · "{_clip(first.quote)}" ({where(first)})' if first else ""
    return f"[support {evidence.support} · {', '.join(evidence.proposers)}{quote}]"


def _clip(text: str | None) -> str:
    flat = _one_line(text or "")
    return flat if len(flat) <= _QUOTE_WIDTH else flat[: _QUOTE_WIDTH - 1] + "…"


def _one_line(text: str) -> str:
    return " ".join(text.split())


__all__ = [
    "Format",
    "evidence_path",
    "format_for",
    "frozen_header",
    "inferred_header",
    "notes",
    "ontology_data",
    "render",
    "summary",
    "where",
]
