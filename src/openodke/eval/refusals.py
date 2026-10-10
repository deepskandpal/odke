"""What the grounder threw away, for a person to judge (#113).

Nobody reads what a grounder refuses. In the published comparison it refused
5.5 to 10.6% of the facts, and on Text2KGBench it removed more facts the gold
called right than wrong; that gold is distantly supervised, so some of those
refusals may be right. The numbers cannot settle it, and a person reading a
sample can. So `odke eval refusals` draws a stratified random sample of the
refused facts in a run's output and writes it as `odke label` sheets. Each item
shows the claim as the grounder read it, the text it read, its verdict and the
reason, and whether the gold lists the fact, where there is gold. A person
ticks one of four answers:

- **refusal correct**: the text does not state the fact, so refusing it was right;
- **refusal wrong**: the text states it, and a true fact was lost;
- **gold wrong**: the gold lists the fact and the text does not state it: the
  refusal was right, and the gold is not;
- **unsure**: the text does not settle it.

Read back (`odke label read`), the labels give the **refusal precision**: the
right refusals, correct and gold wrong, over every judgement but unsure, with
its Wilson 95% interval beside it, overall and per dataset, extractor and
verdict (`report_refusals`).

What it reads (`read_refusals`), any of:

- **`odke ground` output**: `facts.jsonl`, with `summary.json` beside it when
  there is one. A fact the free checks stamped (`odke.check`), or whose
  verdict is `contradicted` or `not_found`, is a refusal. The texts come from
  `documents`.
- **`odke validate -o` output**: `refused.jsonl`, which holds each fact the
  gate refused and the gate's reason. The texts come from `documents`.
- **an `odke bench run` set**, or a directory of them: each extractor's
  triples before grounding (`predictions/extraction-alone.jsonl`) that are not
  in the row after it (`grounding.jsonl`), its own and those in
  `competitors/<name>/`. A bench run keeps triples, not verdicts, so the
  verdict is the one the set's config refuses: the paper's binary grounder
  answers only "False" for a refusal, which is `not_found`. The text is the
  document, which is what a grounder that reads the whole document (`--paper`)
  read, and the gold is the dataset's own scoring.

A long text is cut to the sentences around what is bold, so a sheet stays one
sitting. Bold is the cited span, or else where the claim's names are, as the
span locator finds them: a reading aid, not what the grounder was shown.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from openodke.chunking import sentences
from openodke.corroborate.provenance import CHECK
from openodke.eval.eval_report import Dataset, EvalReport, Run, from_stage
from openodke.eval.formats import Refusal, RefusalLabel
from openodke.eval.report import Metric, StageReport
from openodke.eval.sheets import Made, make_sheets
from openodke.eval.stats import wilson
from openodke.gate import REFUSED_FILE
from openodke.ground.llm import render_claim
from openodke.ground.locate import locate_span
from openodke.types import Document, Entity, Fact, GroundingVerdict, SpanOrigin

# What `--make-sheet` writes beside the sheets: the drawn rows, in sheet order.
SAMPLE_FILE = "refusals.jsonl"
SAMPLE = 100
SEED = 0
# A text longer than this is cut to the sentences around what is bold.
MAX_TEXT = 3000
# Sentences either side of the bold one a cut text keeps.
AROUND = 1

JUDGEMENTS = ("refusal_correct", "refusal_wrong", "gold_wrong", "unsure")
# Why a verdict is refused, as the gate says it (`openodke.gate.VerdictGate`).
BECAUSE = {
    "contradicted": "the passage says otherwise",
    "not_found": "the passage does not settle it",
}

DESCRIPTION = """\
odke eval refusals RUN... [--documents DOCS] --make-sheet DIR [--n 100] [--seed 0]
odke eval refusals --labels LABELS

What the grounder threw away, for a person to judge. RUN is any of: an
`odke ground` output (its directory, or facts.jsonl), an `odke validate -o`
directory (refused.jsonl), or an `odke bench run` set, or a directory of
them. --documents gives the texts a ground or validate run read.

--make-sheet DIR draws a random sample of the refusals, stratified by
dataset, extractor and verdict and spread over predicates, and writes it to
DIR as `odke label` sheets: the claim as the grounder read it, the text, the
verdict and the reason, and whether the gold lists the fact. Tick one box an
item: refusal correct, refusal wrong (the text states the fact), gold wrong
(the gold lists it, and the text does not state it) or unsure.

--labels reads the ticks back (`odke label read DIR -o LABELS`) and reports
the refusal precision, correct and gold wrong over everything but unsure,
with its Wilson 95% interval, overall and per dataset, extractor and verdict."""


# --------------------------------------------------------------------------- #
# Reading a run's refusals
# --------------------------------------------------------------------------- #


def read_refusals(path: str | Path, documents: Sequence[Document] | None = None) -> list[Refusal]:
    """Every refusal in one run's output, in the order it holds them.

    `path` is an `odke ground` output (its directory or `facts.jsonl`), an
    `odke validate -o` directory, or an `odke bench run` set or a directory
    of them. `documents` are the texts a ground or validate run read; a bench
    set has its own.
    """
    source = Path(path)
    if source.is_file():
        return _from_facts(source, documents)
    if not source.is_dir():
        raise ValueError(f"{source}: no such file or directory")
    if (source / REFUSED_FILE).is_file():
        return _from_validate(source / REFUSED_FILE, documents)
    sets = bench_sets(source)
    if sets:
        return [r for folder in sets for r in _from_bench(folder)]
    if (source / "facts.jsonl").is_file():
        return _from_facts(source / "facts.jsonl", documents)
    raise ValueError(
        f"{source}: no {REFUSED_FILE} (odke validate -o), facts.jsonl (odke ground) or "
        "bench set (dataset.json and predictions/) in it"
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def _texts(documents: Sequence[Document] | None, what: str) -> dict[str, Document]:
    """The documents by every name a fact may cite them by: id, file name, and stem when unique.

    A run config's directory loader names a text by its path under the config
    (`texts/halden.txt`), and `--documents texts/` by its name in the folder.
    """
    if documents is None:
        raise ValueError(f"{what} keeps the facts and not the texts: give --documents")
    out = {doc.id: doc for doc in documents}
    for pick in (lambda d: Path(d.id).name, lambda d: Path(d.id).stem):
        named: dict[str, list[Document]] = defaultdict(list)
        for doc in documents:
            named[pick(doc)].append(doc)
        for name, docs in named.items():
            if len(docs) == 1:
                out.setdefault(name, docs[0])
    return out


def _cited(doc_id: str | None, texts: Mapping[str, Document]) -> Document | None:
    if doc_id is None:
        return None
    for name in (doc_id, Path(doc_id).name, Path(doc_id).stem):
        if name in texts:
            return texts[name]
    return None


def _from_facts(path: Path, documents: Sequence[Document] | None) -> list[Refusal]:
    """`odke ground`'s facts: each one the free checks stamped, or not supported."""
    texts = _texts(documents, "odke ground")
    summary = path.with_name("summary.json")
    shapes: dict[str, str] = {}
    if summary.is_file():
        found = json.loads(summary.read_text(encoding="utf-8"))
        for key, why in (
            ("too_narrow", "the citation is too narrow for its claim"),
            ("unsupported", "the evidence does not support the fact"),
        ):
            for fact_id in found.get(key, {}).get("ids", ()):
                shapes[str(fact_id)] = why
    out = []
    for row in _rows(path):
        fact = Fact.model_validate(row)
        stamp = fact.qualifiers.get(CHECK)
        refused = fact.verdict in (GroundingVerdict.CONTRADICTED, GroundingVerdict.NOT_FOUND)
        if not refused and not isinstance(stamp, Mapping):
            continue
        if isinstance(stamp, Mapping):
            reason = f"free check, {stamp.get('check')}: {stamp.get('reason')}"
        elif fact.id in shapes:
            reason = shapes[fact.id]
        elif not any(e.span is not None for e in fact.evidence):
            reason = "no citation resolves in the text"
        else:
            reason = BECAUSE[fact.verdict.value]
        out.append(_refusal(fact, reason, texts, source=str(path)))
    return out


def _from_validate(path: Path, documents: Sequence[Document] | None) -> list[Refusal]:
    """`odke validate -o`'s refused.jsonl: each fact the gate refused, with its reason."""
    texts = _texts(documents, "odke validate")
    return [
        _refusal(
            Fact.model_validate(row["fact"]), str(row.get("reason") or ""), texts, source=str(path)
        )
        for row in _rows(path)
    ]


def _refusal(fact: Fact, reason: str, texts: Mapping[str, Document], *, source: str) -> Refusal:
    cited = next((e for e in fact.evidence if _cited(e.doc_id, texts) is not None), None)
    doc = _cited(cited.doc_id, texts) if cited is not None else None
    if cited is None or doc is None:
        raise ValueError(
            f"fact {fact.id} cites none of the documents given: a sheet shows the text it read"
        )
    span = cited.span
    bold = None
    if span is not None and cited.span_origin is not SpanOrigin.CONTEXT:
        bold = (span.start, span.end)
    elif (found := locate_span(fact, doc)) is not None:
        bold = (found.start, found.end)
    text, bold = cut(doc.text, bold)
    return Refusal(
        id=_id(source, doc.id, fact.id),
        claim=render_claim(fact),
        text=text,
        bold=bold,
        verdict=fact.verdict.value,
        reason=reason,
        predicate=fact.predicate,
        extractor=fact.extractor if fact.extractor != "unknown" else None,
        doc_id=doc.id,
        source=source,
    )


def bench_sets(root: Path) -> list[Path]:
    """Every `odke bench run` set at or under `root`, its competitors' included, in order."""
    found = [
        marker.parent
        for marker in sorted(root.rglob("dataset.json"))
        if (marker.parent / "predictions" / "extraction-alone.jsonl").is_file()
    ]
    return sorted(dict.fromkeys(found), key=str)


def _triples(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    if not path.is_file():
        return {}
    out: dict[str, list[tuple[str, str, str]]] = {}
    for row in _rows(path):
        out[str(row["id"])] = list(dict.fromkeys((s, r, o) for s, r, o in row["triples"]))
    return out


def _extractor(folder: Path, config: Mapping[str, Any]) -> str:
    """Who wrote a set's triples: a replayed system's name, or openodke's own extractor."""
    stage = config.get("stages", {}).get("extractor")
    if isinstance(stage, Mapping) and stage.get("system"):
        return str(stage["system"])
    if folder.parent.name == "competitors":
        return folder.name
    use = stage.get("use") if isinstance(stage, Mapping) else stage
    return "openodke" if use in (None, "llm") else str(use)


def _verdict(config: Mapping[str, Any]) -> tuple[str, str]:
    """What a set's grounder and gate refuse, and why, from its run config."""
    stages = config.get("stages", {})
    grounder = stages.get("grounder") if isinstance(stages.get("grounder"), Mapping) else {}
    gate = stages.get("gate", stages.get("validator"))
    strict = isinstance(gate, Mapping) and bool(gate.get("refuse_not_found"))
    if grounder.get("verdicts") == "binary":
        return "not_found", "the grounder answered False: the text does not state it"
    if not strict:
        return "contradicted", BECAUSE["contradicted"]
    return (
        "not_found or contradicted",
        "the gate refused it; the run kept the triple, not its verdict",
    )


def _listed(dataset: str, meta: Mapping[str, Any], row: Mapping[str, Any], triple: Any) -> bool:
    """Whether the dataset's own scoring matches `triple` to a gold fact of `row`."""
    from openodke.eval.datasets import redocred, text2kgbench

    predicted = {str(row["id"]): [tuple(triple)]}
    if dataset == "text2kgbench":
        found = text2kgbench.score([row], predicted, meta["ontology"], hallucination=False)
    else:
        found = redocred.score([row], predicted)
    return float(found.get("precision") or 0) >= 1


def _from_bench(folder: Path) -> list[Refusal]:
    """One bench set's refusals: triples before grounding that the gated row lacks."""
    meta = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    config_path = folder / "odke.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    dataset = str(meta.get("dataset", folder.name))
    extractor = _extractor(folder, config)
    verdict, reason = _verdict(config)
    before = _triples(folder / "predictions" / "extraction-alone.jsonl")
    after = _triples(folder / "predictions" / "grounding.jsonl")
    gold_path = folder / "gold.jsonl"
    gold = {str(r["id"]): r for r in _rows(gold_path)} if gold_path.is_file() else {}
    out = []
    for doc_id in before:
        kept = set(after.get(doc_id, ()))
        text_path = folder / "docs" / f"{doc_id}.txt"
        text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
        for triple in before[doc_id]:
            if triple in kept:
                continue
            subject, relation, obj = triple
            claim = Fact(
                subject=Entity(key=subject, type="Thing", label=subject),
                predicate=relation,
                object_value=obj,
            )
            found = locate_span(claim, Document(id=doc_id, text=text)) if text else None
            shown, bold = cut(text, (found.start, found.end) if found is not None else None)
            row = gold.get(doc_id)
            out.append(
                Refusal(
                    id=_id(dataset, str(folder.name), extractor, doc_id, *triple),
                    claim=f"{subject} — {relation} — {obj}.",
                    text=shown,
                    bold=bold,
                    verdict=verdict,
                    reason=reason,
                    gold=_listed(dataset, meta, row, triple) if row is not None else None,
                    predicate=relation,
                    extractor=extractor,
                    dataset=dataset,
                    doc_id=doc_id,
                    source=str(folder),
                )
            )
    return out


def cut(text: str, bold: tuple[int, int] | None) -> tuple[str, tuple[int, int] | None]:
    """`text` as a sheet shows it, and `bold` moved with it.

    Up to `MAX_TEXT` characters it is whole. A longer one keeps the sentences
    that hold `bold` and `AROUND` either side, with `…` where it was cut; with
    nothing bold, its first `MAX_TEXT` characters. A `bold` that is the whole
    text points at nothing, and is dropped.
    """
    if bold is not None and not text[: bold[0]].strip() and not text[bold[1] :].strip():
        bold = None  # all of it: nothing to point at
    if len(text) <= MAX_TEXT:
        return text, bold
    if bold is None:
        head = text[:MAX_TEXT]
        return head[: head.rfind(" ")] + " …" if " " in head else head, None
    bounds = sentences(text)
    hit = [k for k, (s, e) in enumerate(bounds) if s < bold[1] and bold[0] < e]
    if not hit:
        return cut(text, None)
    first, last = max(0, hit[0] - AROUND), min(len(bounds) - 1, hit[-1] + AROUND)
    start, end = bounds[first][0], bounds[last][1]
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(text) else ""
    shift = len(prefix) - start
    return prefix + text[start:end] + suffix, (bold[0] + shift, bold[1] + shift)


def _id(*parts: str) -> str:
    return hashlib.sha256(json.dumps(list(parts)).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The sample
# --------------------------------------------------------------------------- #


def _stratum(refusal: Refusal) -> tuple[str, str, str]:
    return (refusal.dataset or "", refusal.extractor or "", refusal.verdict)


def _quotas(sizes: Mapping[Any, int], n: int) -> dict[Any, int]:
    """`n` shared as evenly as the strata allow: an equal share each, capped at its size.

    What a small stratum cannot take goes to the others, a unit at a time, in
    stratum order, so the same sizes always give the same quotas.
    """
    quota = dict.fromkeys(sizes, 0)
    left = min(n, sum(sizes.values()))
    while left:
        open_ = [k for k in sorted(sizes) if quota[k] < sizes[k]]
        share = max(1, left // len(open_))
        for key in open_:
            give = min(share, sizes[key] - quota[key], left)
            quota[key] += give
            left -= give
            if not left:
                break
    return quota


def sample(refusals: Sequence[Refusal], n: int = SAMPLE, *, seed: int = SEED) -> list[Refusal]:
    """`n` refusals at random, stratified by dataset, extractor and verdict, spread over predicates.

    Each stratum gets an equal share, as its size allows (`_quotas`). Within
    one, the predicates take turns, in an order drawn by seed, each giving a
    refusal drawn at random, so no predicate takes the stratum. The sample is
    shuffled, so a sheet mixes strata. Asking for as many as there are draws
    them all.
    """
    if n < 1:
        raise ValueError(f"a sample needs at least one refusal, got n={n}")
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[Refusal]] = defaultdict(list)
    for refusal in sorted(refusals, key=lambda r: r.id):
        strata[_stratum(refusal)].append(refusal)
    quotas = _quotas({k: len(v) for k, v in strata.items()}, n)
    drawn: list[Refusal] = []
    for key in sorted(strata):
        by_predicate: dict[str, list[Refusal]] = defaultdict(list)
        for refusal in strata[key]:
            by_predicate[refusal.predicate].append(refusal)
        order = sorted(by_predicate)
        rng.shuffle(order)
        for name in order:
            rng.shuffle(by_predicate[name])
        taken = 0
        while taken < quotas[key]:
            for name in order:
                if taken < quotas[key] and by_predicate[name]:
                    drawn.append(by_predicate[name].pop())
                    taken += 1
    rng.shuffle(drawn)
    return drawn


def write_sample(
    refusals: Sequence[Refusal],
    out: str | Path,
    *,
    n: int = SAMPLE,
    seed: int = SEED,
    per_sheet: int = 50,
) -> tuple[list[Refusal], Made]:
    """Draw the sample and write it to `out`: `refusals.jsonl`, then the sheets.

    A directory that already holds a sample or sheets is refused: those may be
    hours of ticks.
    """
    target = Path(out)
    if target.is_dir() and ((target / SAMPLE_FILE).exists() or any(target.glob("sheet-*"))):
        raise ValueError(f"{target} already has a sample or sheets; write to an empty directory")
    if not refusals:
        raise ValueError("no refusals to draw from")
    drawn = sample(refusals, n, seed=seed)
    target.mkdir(parents=True, exist_ok=True)
    rows = target / SAMPLE_FILE
    rows.write_text(
        "".join(r.model_dump_json(exclude_none=False) + "\n" for r in drawn),
        encoding="utf-8",
        newline="\n",
    )
    return drawn, make_sheets("refusal", rows, target, per_sheet=per_sheet)


def strata(refusals: Iterable[Refusal]) -> list[tuple[tuple[str, str, str], int]]:
    """How many refusals each stratum holds, in order: what `--make-sheet` draws from."""
    return sorted(Counter(_stratum(r) for r in refusals).items())


# --------------------------------------------------------------------------- #
# Reading the labels back
# --------------------------------------------------------------------------- #


def precision(labels: Sequence[RefusalLabel]) -> dict[str, Metric]:
    """The refusal precision of `labels`, its Wilson 95% interval, and each judgement's count."""
    counts: Counter[str] = Counter(str(label.judgement) for label in labels)
    judged = len(labels) - counts["unsure"]
    right = counts["refusal_correct"] + counts["gold_wrong"]
    interval = wilson(right, judged)
    listed = [label for label in labels if label.gold]
    return {
        "precision": right / judged if judged else None,
        "low": interval[0] if interval else None,
        "high": interval[1] if interval else None,
        "judged": judged,
        **{name: counts[name] for name in JUDGEMENTS},
        # Of the refusals the gold lists, how many the person found the gold wrong on.
        "gold_listed": len(listed),
        "gold_wrong_of_listed": (counts["gold_wrong"] / len(listed) if listed else None),
    }


def report_refusals(labels: Sequence[RefusalLabel], *, path: str | None = None) -> EvalReport:
    """The refusal precision with its interval, overall and per dataset, extractor and verdict."""
    if not labels:
        raise ValueError("no labels: tick the sheets, and read them back with odke label read")
    overall = precision(labels)
    breakdown: dict[str, dict[str, Metric]] = {"all": overall}
    for field in ("dataset", "extractor", "verdict"):
        groups: dict[str, list[RefusalLabel]] = defaultdict(list)
        for label in labels:
            value = getattr(label, field)
            if value is not None:
                groups[str(value)].append(label)
        if len(groups) > 1:
            for value in sorted(groups):
                breakdown[f"{field}: {value}"] = precision(groups[value])
    notes = [_headline(overall)]
    stage = StageReport(
        stage="refusals",
        n=len(labels),
        metrics=overall,
        breakdown=breakdown,
        notes=tuple(notes),
    )
    dataset = Dataset(name=Path(path).name if path else "refusals", path=path, labels=len(labels))
    return from_stage(stage, run=Run(dataset=dataset))


def _headline(found: Mapping[str, Metric]) -> str:
    unsure = found["unsure"]
    if found["precision"] is None:
        return f"refusal precision: nothing judged ({unsure} unsure)"
    return (
        f"refusal precision {found['precision']:.3f} [{found['low']:.3f}, {found['high']:.3f}] "
        f"(Wilson 95%), {found['judged']} judged, {unsure} unsure set aside"
    )


__all__ = [
    "DESCRIPTION",
    "JUDGEMENTS",
    "REFUSED_FILE",
    "SAMPLE",
    "SAMPLE_FILE",
    "bench_sets",
    "cut",
    "precision",
    "read_refusals",
    "report_refusals",
    "sample",
    "strata",
    "write_sample",
]
