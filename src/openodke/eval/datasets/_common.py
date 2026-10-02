"""What every dataset adapter shares: downloading, names, the run config, the report."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from openodke.eval.cost import StageCost
from openodke.eval.report import Metric, StageReport
from openodke.types import Document, Fact

Triple = tuple[str, str, str]
Opener = Callable[[str], Any]

# Literal ranges a dataset may name, in openodke's vocabulary (ontology value types).
LITERALS = {
    "": "string",
    "string": "string",
    "langstring": "string",
    "text": "string",
    "date": "date",
    "time": "date",
    "datetime": "date",
    "year": "integer",
    "number": "number",
    "integer": "integer",
    "int": "integer",
    "float": "number",
    "double": "number",
    "quantity": "number",
    "boolean": "boolean",
}


def download(url: str, dest: Path, *, opener: Opener | None = None) -> Path:
    """`url` saved at `dest`, unless it is already there. Written atomically."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener if opener is not None else urllib.request.urlopen
    with (
        open_url(url) as response,
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
    Path(tmp.name).replace(dest)
    return dest


def pascal(label: str) -> str:
    """`film production company` -> `FilmProductionCompany`; `University` stays."""
    words = re.split(r"[^0-9A-Za-z]+", label.strip())
    return "".join(w[:1].upper() + w[1:] for w in words if w) or "Thing"


def snake(label: str) -> str:
    """`cast member` -> `cast_member`; `academicStaffSize` -> `academic_staff_size`."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label.strip())
    return "_".join(w.lower() for w in re.split(r"[^0-9A-Za-z]+", spaced) if w) or "related_to"


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return path


def write_documents(out: Path, docs: Iterable[tuple[str, str]]) -> int:
    """One `<id>.txt` per document under `out/docs`: the id survives as the file's stem."""
    folder = out / "docs"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    count = 0
    for doc_id, text in docs:
        (folder / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        count += 1
    return count


def run_config(
    *,
    extract_model: str | None,
    ground_model: str | None,
    paper: bool,
) -> dict[str, Any]:
    """The run config a prepared dataset ships: text in, nothing written out.

    No chunker, so each document reaches the extractor whole — the paper's unit
    is a page. `paper=True` is ODKE+'s own grounder and gate: the whole context,
    True or False, and only affirmed facts kept.
    """
    grounder: dict[str, Any] = {"use": "llm"}
    validator: dict[str, Any] = {"use": "verdict"}
    if paper:
        grounder |= {"context": "document", "verdicts": "binary"}
        validator |= {"refuse_not_found": True}
    config: dict[str, Any] = {
        "ontology": "ontology.json",
        "inputs": [{"path": "docs", "loader": "directory"}],
        "stages": {
            "extractor": "llm",
            "grounder": grounder,
            "normalizer": "value",
            "corroborator": "signature",
            "scorer": "evidence",
            "validator": validator,
        },
    }
    models = {k: v for k, v in (("extract", extract_model), ("ground", ground_model)) if v}
    if models:
        config["models"] = models
    return config


def doc_names(documents: Sequence[Document]) -> dict[str, str]:
    """Document id -> the dataset's own id, read back from the file's stem."""
    out: dict[str, str] = {}
    for doc in documents:
        if doc.uri:
            out[doc.id] = Path(unquote(urlparse(doc.uri).path)).stem
    return out


def triples_by_doc(
    facts: Iterable[Fact], names: Mapping[str, str], relation: Callable[[str], str]
) -> dict[str, list[Triple]]:
    """Each fact as `(subject, relation, object)` strings, under every document it cites.

    Labels, not keys: a benchmark's gold is written in surface strings. `relation`
    maps an openodke predicate name back to the dataset's own relation label.
    """
    out: dict[str, list[Triple]] = {}
    for fact in facts:
        triple = (
            fact.subject.label or fact.subject.key,
            relation(fact.predicate),
            (fact.object_entity.label or fact.object_entity.key)
            if fact.object_entity is not None
            else str(fact.object_value),
        )
        for doc_id in dict.fromkeys(e.doc_id for e in fact.evidence):
            name = names.get(doc_id)
            if name is not None and triple not in out.setdefault(name, []):
                out[name].append(triple)
    return out


def report(
    stage: str,
    n: int,
    rows: Sequence[tuple[str, Mapping[str, Metric], Sequence[Any], int]],
    notes: Sequence[str],
) -> StageReport:
    """The ablation table: one row per configuration, its metrics, calls and cost."""
    breakdown: dict[str, dict[str, Metric]] = {}
    for name, row, calls, facts in rows:
        cost = StageCost.of(name, list(calls))
        breakdown[name] = {
            **row,
            "facts": facts,
            "model_calls": cost.calls,
            "cost_usd": cost.cost_usd,
        }
    summary: dict[str, Metric] = {}
    for label, (name, _, _, _) in zip(
        ("extraction", "grounding", "corroboration"), rows, strict=True
    ):
        for key in ("precision", "recall", "f1"):
            summary[f"{key}_{label}"] = breakdown[name].get(key)
    return StageReport(stage=stage, n=n, metrics=summary, breakdown=breakdown, notes=tuple(notes))


def change(before: float, after: float) -> str:
    """`-35%`, `+4%`, or `n/a` when there was nothing to change."""
    if before == 0:
        return "n/a"
    return f"{(after - before) / before:+.0%}"
