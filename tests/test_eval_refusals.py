"""`odke eval refusals` (#113): what the grounder threw away, sampled, ticked and scored.

The refusals come from the three outputs the command reads: an `odke ground`
run's facts, an `odke validate -o` directory's refused.jsonl, and a bench set
made by hand in `odke bench`'s shape. No model is called.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openodke import Document, Entity, Evidence, Fact, GroundingVerdict, Span
from openodke.cli.main import app
from openodke.corroborate.provenance import CHECK
from openodke.eval.formats import Refusal, RefusalLabel, load_jsonl
from openodke.eval.refusals import (
    MAX_TEXT,
    SAMPLE_FILE,
    _stratified,
    bench_sets,
    cut,
    precision,
    read_refusals,
    report_refusals,
    sample,
    write_sample,
)
from openodke.eval.sheets import read_sheets
from openodke.eval.stats import wilson
from openodke.gate import REFUSED_FILE, Kept, VerdictGate

REPO = Path(__file__).parent.parent
runner = CliRunner()

TEXT = "Ada Lovelace was born in London. She worked with Charles Babbage. She died in 1852."
DOC = Document(id="d1", text=TEXT)
ADA = Entity(key="person:ada", type="Person", label="Ada Lovelace")


def _fact(predicate: str, value: object, verdict: str, span: tuple[int, int] | None) -> Fact:
    evidence = Evidence(
        doc_id="d1", span=Span(doc_id="d1", start=span[0], end=span[1]) if span else None
    )
    return Fact(
        subject=ADA,
        predicate=predicate,
        object_value=value,
        evidence=(evidence,),
        verdict=GroundingVerdict(verdict),
        extractor="hand",
    )


def _jsonl(path: Path, rows: list[Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _tick(sheet: Path, item_id: str, box: str) -> None:
    head, tail = sheet.read_text(encoding="utf-8").split(f"### {item_id}\n", 1)
    tail = tail.replace(f"- [ ] {box}\n", f"- [x] {box}\n", 1)
    sheet.write_text(f"{head}### {item_id}\n{tail}", encoding="utf-8")


# --------------------------------------------------------------------------- #
# The Wilson interval
# --------------------------------------------------------------------------- #


def test_the_wilson_interval_is_the_textbook_one_and_never_leaves_0_to_1() -> None:
    low, high = wilson(81, 100) or (None, None)
    assert (round(low, 4), round(high, 4)) == (0.7222, 0.8749)
    assert wilson(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    assert wilson(10, 10) == (pytest.approx(0.7225, abs=1e-4), 1.0)
    assert wilson(0, 0) is None
    with pytest.raises(ValueError, match="between 0 and n"):
        wilson(3, 2)


# --------------------------------------------------------------------------- #
# Reading refusals
# --------------------------------------------------------------------------- #


def test_odke_ground_output_gives_each_fact_not_supported_with_why(tmp_path: Path) -> None:
    worked = (TEXT.index("She worked"), TEXT.index(" She died"))
    stamped = _fact("born_in", "Paris", "unchecked", (0, 31)).model_copy(
        update={"qualifiers": {CHECK: {"check": "range", "reason": "object outside the range"}}}
    )
    facts = [
        _fact("born_in", "London", "supported", (0, 31)),  # kept
        _fact("worked_with", "Charles Babbage", "not_found", worked),  # too narrow, says summary
        _fact("died", 1851, "contradicted", (TEXT.index("She died"), len(TEXT))),
        _fact("likes", "tea", "not_found", None),  # no citation resolved
        stamped,
    ]
    out = tmp_path / "ground"
    _jsonl(out / "facts.jsonl", [json.loads(f.model_dump_json()) for f in facts])
    summary = {"too_narrow": {"ids": [facts[1].id]}, "unsupported": {"ids": [facts[2].id]}}
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    found = read_refusals(out, [DOC])
    assert [r.reason for r in found] == [
        "the citation is too narrow for its claim",
        "the evidence does not support the fact",
        "no citation resolves in the text",
        "free check, range: object outside the range",
    ]
    assert [r.verdict for r in found] == ["not_found", "contradicted", "not_found", "unchecked"]
    first = found[0]
    assert first.claim == 'Ada Lovelace (Person) — worked with — "Charles Babbage".'
    assert first.text == TEXT and first.bold == worked
    assert (first.predicate, first.extractor, first.doc_id, first.gold) == (
        "worked_with",
        "hand",
        "d1",
        None,
    )
    # No documents, no text to show.
    with pytest.raises(ValueError, match="give --documents"):
        read_refusals(out / "facts.jsonl")


def test_a_gate_that_keeps_a_record_writes_what_validate_reads(tmp_path: Path) -> None:
    kept = Kept(VerdictGate(refuse_not_found=True))
    facts = [
        _fact("born_in", "London", "supported", (0, 31)),
        _fact("died", 1851, "not_found", (TEXT.index("She died"), len(TEXT))),
    ]
    from openodke import Ontology

    actions = [kept.validate(f, Ontology()).action for f in facts]
    assert actions == ["accept", "refuse"] and kept.stats == {
        "accepted": 1,
        "refused": {"not_found": 1},
    }
    kept.write(tmp_path / REFUSED_FILE)
    (refusal,) = read_refusals(tmp_path, [DOC])
    assert refusal.reason == "grounding verdict not_found: the passage does not settle it"
    assert refusal.bold == (TEXT.index("She died"), len(TEXT))


def test_a_text_is_found_by_the_name_a_run_config_gave_it(tmp_path: Path) -> None:
    """A config's directory loader cites `texts/d1.txt`; `--documents texts/` names it `d1.txt`."""
    kept = Kept(VerdictGate(refuse_not_found=True))
    fact = _fact("died", 1851, "not_found", (0, 31))
    cited = fact.model_copy(
        update={
            "evidence": (
                Evidence(doc_id="texts/d1.txt", span=Span(doc_id="texts/d1.txt", start=0, end=31)),
            )
        }
    )
    from openodke import Ontology

    kept.validate(cited, Ontology())
    kept.write(tmp_path / REFUSED_FILE)
    (refusal,) = read_refusals(tmp_path, [Document(id="d1.txt", text=TEXT)])
    assert refusal.doc_id == "d1.txt" and refusal.text == TEXT
    with pytest.raises(ValueError, match="cites none of the documents given"):
        read_refusals(tmp_path, [Document(id="other.txt", text=TEXT)])


# --------------------------------------------------------------------------- #
# A bench set, in `odke bench`'s shape
# --------------------------------------------------------------------------- #

DOCS = {
    "test_0000": "Mara Quill was born in Oakford. Oakford is a town in Veland.",
    "test_0001": "Ivo Rook was born in Pinebury. Pinebury is a town in Wexia.",
}


def _set(
    folder: Path,
    before: dict[str, list[list[str]]],
    after: dict[str, list[list[str]]],
    extractor: dict[str, Any],
) -> None:
    meta = {"dataset": "redocred", "relation_labels": {"place_of_birth": "place of birth"}}
    (folder / "predictions").mkdir(parents=True)
    (folder / "dataset.json").write_text(json.dumps(meta))
    stages = {
        "extractor": extractor,
        "grounder": {"use": "llm", "context": "document", "verdicts": "binary"},
        "validator": {"use": "verdict", "refuse_not_found": True},
    }
    (folder / "odke.json").write_text(json.dumps({"stages": stages}))
    gold = [
        {
            "id": "test_0000",
            "text": DOCS["test_0000"],
            "entities": [["Mara Quill"], ["Oakford"], ["Veland"]],
            "facts": [[0, "place of birth", 1], [1, "country", 2]],
        },
        {
            "id": "test_0001",
            "text": DOCS["test_0001"],
            "entities": [["Ivo Rook"], ["Pinebury"], ["Wexia"]],
            "facts": [[0, "place of birth", 1]],
        },
    ]
    _jsonl(folder / "gold.jsonl", gold)
    (folder / "docs").mkdir()
    for doc, text in DOCS.items():
        (folder / "docs" / f"{doc}.txt").write_text(text)
    rows = lambda found: [{"id": d, "triples": t} for d, t in found.items()]  # noqa: E731
    _jsonl(folder / "predictions" / "extraction-alone.jsonl", rows(before))
    _jsonl(folder / "predictions" / "grounding.jsonl", rows(after))


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    root = tmp_path / "cmp" / "redocred"
    _set(
        root,
        {
            "test_0000": [
                ["Mara Quill", "place of birth", "Oakford"],
                ["Oakford", "country", "Veland"],
            ],
            "test_0001": [["Ivo Rook", "place of birth", "Wexia"]],
        },
        {"test_0000": [["Mara Quill", "place of birth", "Oakford"]]},
        {"use": "llm"},
    )
    _set(
        root / "competitors" / "lgt",
        {
            "test_0001": [
                ["Ivo Rook", "place of birth", "Pinebury"],
                ["Pinebury", "country", "Wexia"],
            ]
        },
        {"test_0001": [["Pinebury", "country", "Wexia"]]},
        {"use": "replay:Replayed", "system": "lgt"},
    )
    return tmp_path / "cmp"


def test_a_bench_set_s_refusals_are_its_triples_the_gated_row_lacks(bench: Path) -> None:
    assert [p.relative_to(bench).as_posix() for p in bench_sets(bench)] == [
        "redocred",
        "redocred/competitors/lgt",
    ]
    found = read_refusals(bench)
    by_claim = {r.claim: r for r in found}
    assert set(by_claim) == {
        "Oakford — country — Veland.",
        "Ivo Rook — place of birth — Wexia.",
        "Ivo Rook — place of birth — Pinebury.",
    }
    country = by_claim["Oakford — country — Veland."]
    assert (country.extractor, country.dataset, country.verdict) == (
        "openodke",
        "redocred",
        "not_found",
    )
    assert country.reason == "the grounder answered False: the text does not state it"
    # The dataset's own scoring: the country fact and Pinebury are gold; Wexia is not.
    assert country.gold is True
    assert by_claim["Ivo Rook — place of birth — Wexia."].gold is False
    lgt = by_claim["Ivo Rook — place of birth — Pinebury."]
    assert (lgt.extractor, lgt.gold) == ("lgt", True)
    # The text is the document, with where both names are in bold.
    assert country.text == DOCS["test_0000"]
    assert country.text[slice(*country.bold)] == "Oakford is a town in Veland."  # type: ignore[misc]


def test_cut_keeps_the_sentences_around_what_is_bold() -> None:
    filler = "Nothing happened here at all. " * 150
    text = filler + "Ada Lovelace was born in London. " + filler
    start = text.index("Ada")
    shown, bold = cut(text, (start, start + 32))
    assert len(text) > MAX_TEXT and len(shown) < 200
    assert shown.startswith("… Nothing happened") and shown.endswith(" …")
    assert bold is not None and shown[slice(*bold)] == "Ada Lovelace was born in London."
    assert cut("Short.", (0, 6)) == ("Short.", None)  # the whole of it points at nothing
    head, none = cut(text, None)
    assert none is None and len(head) <= MAX_TEXT + 2 and head.endswith(" …")


# --------------------------------------------------------------------------- #
# The sample
# --------------------------------------------------------------------------- #


def _refusals() -> list[Refusal]:
    out = []
    plan = {("t2k", "openodke"): 3, ("t2k", "lgt"): 20, ("redocred", "neo4j"): 20}
    for (dataset, extractor), n in plan.items():
        for k in range(n):
            out.append(
                Refusal(
                    id=f"{dataset}-{extractor}-{k}",
                    claim=f"A{k} — p{k % 4} — B{k}.",
                    text="A passage.",
                    verdict="not_found",
                    reason="the passage does not settle it",
                    predicate=f"p{k % 4}",
                    extractor=extractor,
                    dataset=dataset,
                    gold=k % 2 == 0,
                )
            )
    return out


def test_a_sample_shares_out_by_stratum_as_sizes_allow_and_spreads_predicates() -> None:
    drawn = sample(_refusals(), 21, seed=118)
    counts = Counter((r.dataset, r.extractor) for r in drawn)
    # An equal share, seven each; the small stratum's four left over go to the others.
    assert counts == {("t2k", "openodke"): 3, ("t2k", "lgt"): 9, ("redocred", "neo4j"): 9}
    lgt = Counter(r.predicate for r in drawn if r.extractor == "lgt")
    assert max(lgt.values()) - min(lgt.values()) <= 1
    assert len({r.id for r in drawn}) == len(drawn)
    assert [r.id for r in sample(_refusals(), 21, seed=118)] == [r.id for r in drawn]
    assert [r.id for r in sample(_refusals(), 21, seed=1)] != [r.id for r in drawn]
    assert len(sample(_refusals(), 500)) == 43
    with pytest.raises(ValueError, match="at least one"):
        sample(_refusals(), 0)


def test_the_sheet_shows_claim_verdict_reason_gold_and_text_and_reads_back(tmp_path: Path) -> None:
    rows = _refusals()[:4]
    rows[0] = rows[0].model_copy(update={"text": "A0 is not B0 at all.", "bold": (0, 9)})
    drawn, made = write_sample(rows, tmp_path / "s", n=4, seed=3)
    (sheet,) = made.sheets
    text = sheet.read_text(encoding="utf-8")
    assert text.startswith("# Refusal sheet 01\n\nX-0001 to X-0004: was the grounder right")
    item = text.split(f"### X-000{[r.id for r in drawn].index(rows[0].id) + 1}\n", 1)[1]
    assert item.split("\n### ", 1)[0] == (
        "\n"
        "Claim: A0 — p0 — B0.\n"
        "\n"
        "Refused: not\\_found, the passage does not settle it\n"
        "\n"
        "Gold: lists this fact\n"
        "\n"
        "> **A0 is not** B0 at all.\n"
        "\n"
        "- [ ] refusal correct\n"
        "- [ ] refusal wrong\n"
        "- [ ] gold wrong\n"
        "- [ ] unsure\n"
        "\n"
        "note:\n"
    )
    assert load_jsonl(tmp_path / "s" / SAMPLE_FILE, Refusal) == drawn
    for n, box in enumerate(["refusal correct", "gold wrong", "refusal wrong", "unsure"], 1):
        _tick(sheet, f"X-{n:04d}", box)
    reading = read_sheets(tmp_path / "s")
    labels = [RefusalLabel.model_validate(row) for row in reading.labels]
    assert [label.judgement for label in labels] == [
        "refusal_correct",
        "gold_wrong",
        "refusal_wrong",
        "unsure",
    ]
    with pytest.raises(ValueError, match="already has a sample or sheets"):
        write_sample(rows, tmp_path / "s")


def _label(judgement: str, dataset: str = "t2k", gold: bool | None = None) -> RefusalLabel:
    return RefusalLabel(
        id=judgement,
        claim="c",
        text="t",
        verdict="not_found",
        reason="r",
        predicate="p",
        dataset=dataset,
        gold=gold,
        judgement=judgement,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# The stratified precision, simulated
# --------------------------------------------------------------------------- #


def _held(strata: list[tuple[str, int, float]], n: int, reps: int) -> float:
    """How often the stratified 95% range holds the true refusal precision, over seeded draws.

    `(dataset, size, share right)` per stratum. The refusals are made once;
    each rep draws which of them were right refusals, then the sample.
    """
    import random

    refusals = [
        Refusal.model_construct(
            id=f"{name}-{i:05d}", claim="c", text="t", verdict="not_found", reason="r",
            predicate="p", dataset=name, extractor=None, stratum_size=None,
        )
        for name, size, _ in strata
        for i in range(size)
    ]  # fmt: skip
    shares = [share for _, size, share in strata for _ in range(size)]
    held = 0
    for rep in range(reps):
        rng = random.Random(rep)
        right = {r.id: rng.random() < share for r, share in zip(refusals, shares, strict=True)}
        truth = sum(right.values()) / len(right)
        labels = [
            RefusalLabel.model_construct(
                **dict(drawn), judgement="refusal_correct" if right[drawn.id] else "refusal_wrong"
            )
            for drawn in sample(refusals, n, seed=rep)
        ]
        found = _stratified(labels)
        assert found["low"] is not None and found["high"] is not None
        held += found["low"] <= truth <= found["high"]  # type: ignore[operator]
    return held / reps


@pytest.mark.parametrize(
    ("strata", "n", "measured"),
    [
        # The release audit's case: pooled, it read 62% [52%, 71%] against 87%, and held 0 of 2,000.
        pytest.param(
            [("big", 1000, 0.9), ("small", 50, 0.3)], 100, 0.952, id="1000 at 90%, 50 at 30%"
        ),
        pytest.param(
            [("a", 500, 0.7), ("b", 300, 0.9), ("c", 100, 0.5)], 60, 0.956, id="three strata"
        ),
        # The shape of the 10 Oct refusal labels: six strata of 8 to 83, 100 drawn.
        pytest.param(
            [
                ("a", 83, 0.35),
                ("b", 51, 0.3),
                ("c", 20, 0.25),
                ("d", 35, 0.2),
                ("e", 28, 0.25),
                ("f", 8, 0.25),
            ],
            100,
            0.968,
            id="six small strata",
        ),  # fmt: skip
    ],
)
def test_the_stratified_range_holds_its_level(
    strata: list[tuple[str, int, float]], n: int, measured: float
) -> None:
    """Release audit 5: each stratum weighted by its size, Wilson on the effective sample size.

    `measured` is the coverage offline, 5,000 seeded populations each; here,
    200, so a draw may sit a few points from it. Never under 90%.
    """
    found = _held(strata, n, reps=200)
    assert found >= 0.90 and abs(found - measured) < 0.04, (found, measured)


def test_refusal_precision_counts_gold_wrong_as_right_and_sets_unsure_aside() -> None:
    labels = (
        [_label("refusal_correct")] * 6
        + [_label("gold_wrong", "redocred", gold=True)] * 2
        + [_label("refusal_wrong", "redocred", gold=True)] * 2
        + [_label("unsure")] * 3
    )
    found = precision(labels)
    assert (found["precision"], found["judged"], found["unsure"]) == (0.8, 10, 3)
    assert (found["low"], found["high"]) == wilson(8, 10)
    assert (found["gold_listed"], found["gold_wrong_of_listed"]) == (4, 0.5)
    # One stratum's labels are a simple random sample of it, so they pool, sizes or not.
    t2k = [label for label in labels if label.dataset == "t2k"]
    assert (
        "refusal precision 1.000 [0.610, 1.000] (Wilson 95%), 6 judged, 3 unsure set aside"
        in report_refusals(t2k).render()
    )
    # Two strata and no sizes: each on its own, never pooled (release audit 5).
    report = report_refusals(labels, path="labels.jsonl")
    (stage,) = report.stages
    assert set(stage.breakdown) == {
        "all",
        "stratum: redocred / not_found",
        "stratum: t2k / not_found",
    }
    assert stage.breakdown["all"]["precision"] is None
    assert stage.breakdown["stratum: t2k / not_found"]["precision"] == 1.0
    assert stage.breakdown["stratum: redocred / not_found"]["precision"] == 0.5
    assert "refusal precision: not pooled" in report.render()
    # The sizes given back: each stratum weighted by its own, not by how many were drawn.
    sizes = {("t2k", "", "not_found"): 90, ("redocred", "", "not_found"): 10}
    (stage,) = report_refusals(labels, sizes=sizes).stages
    assert stage.breakdown["all"]["precision"] == pytest.approx(0.9 * 1.0 + 0.1 * 0.5)
    assert set(stage.breakdown) == {
        "all", "dataset: redocred", "dataset: t2k",
        "stratum: redocred / not_found", "stratum: t2k / not_found",
    }  # fmt: skip
    with pytest.raises(ValueError, match="no labels"):
        report_refusals([])


# --------------------------------------------------------------------------- #
# odke eval refusals
# --------------------------------------------------------------------------- #


def test_an_explicit_n_is_told_apart_from_the_default(
    bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--n 150` draws 150, though 150 is the flag's default for precision (release audit S2)."""
    from openodke.eval import refusals as module

    asked: list[int] = []
    draw = module.write_sample

    def spy(*args: Any, n: int, **kwargs: Any) -> Any:
        asked.append(n)
        return draw(*args, n=n, **kwargs)

    monkeypatch.setattr(module, "write_sample", spy)
    for n, flag in ((module.SAMPLE, []), (150, ["--n", "150"])):
        out = tmp_path / f"sheets-{n}"
        made = runner.invoke(app, ["eval", "refusals", str(bench), "--make-sheet", str(out), *flag])
        assert made.exit_code == 0, made.output
    assert asked == [100, 150]


def test_odke_eval_refusals_draws_the_sheets_and_scores_the_ticks(
    bench: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sheets"
    made = runner.invoke(
        app, ["eval", "refusals", str(bench), "--make-sheet", str(out), "--n", "2", "--seed", "4"]
    )
    assert made.exit_code == 0, made.output
    assert "drew 2 of 3 refusals (seed 4)" in made.output
    assert "redocred / lgt / not_found: 1 of 1" in made.output
    sheet = out / "sheet-01.md"
    _tick(sheet, "X-0001", "refusal correct")
    _tick(sheet, "X-0002", "refusal wrong")
    labels = tmp_path / "labels.jsonl"
    read = runner.invoke(app, ["label", "read", str(out), "-o", str(labels)])
    assert read.exit_code == 0, read.output
    # Each row read back carries its stratum's size: 1 for redocred's, 2 for the other.
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    assert sorted(row["stratum_size"] for row in rows) == [1, 2]
    scored = runner.invoke(app, ["eval", "refusals", "--labels", str(labels)])
    assert scored.exit_code == 0, scored.output
    # The right refusal stands for a stratum of 1, the wrong one for a stratum of 2: 1/3,
    # where pooling the draws as one random sample said 0.500 (release audit 5).
    assert (
        "refusal precision 0.333 [0.045, 0.841] (each of 2 strata weighted by its size, "
        "3 refusals; Wilson 95% on an effective 2.0 labels), 2 judged, 0 unsure set aside"
    ) in " ".join(scored.output.split())
    as_json = runner.invoke(app, ["eval", "refusals", "--labels", str(labels), "--json"])
    assert json.loads(as_json.output)["stages"][0]["metrics"]["precision"] == pytest.approx(1 / 3)
    # Labels with their sizes taken off are reported per stratum, until the run gives them back.
    bare = tmp_path / "bare.jsonl"
    bare.write_text("".join(json.dumps({**row, "stratum_size": None}) + "\n" for row in rows))
    alone = runner.invoke(app, ["eval", "refusals", "--labels", str(bare), "--json"])
    assert json.loads(alone.output)["stages"][0]["metrics"]["precision"] is None
    again = runner.invoke(app, ["eval", "refusals", str(bench), "--labels", str(bare), "--json"])
    assert again.exit_code == 0, again.output
    assert json.loads(again.output)["stages"][0]["metrics"]["precision"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["eval", "refusals"], "needs a run to draw from"),
        (
            ["eval", "refusals", "x", "--make-sheet", "d", "--by-predicate"],
            "--by-predicate is for precision",
        ),
        (
            ["eval", "refusals", "x", "--labels", "l", "--make-sheet", "d"],
            "--make-sheet draws a sample and --labels reads one back",
        ),
        (["eval", "refusals", "x"], "writes its sample with --make-sheet DIR"),
        (["eval", "extract", "--make-sheet", "d"], "these are for precision and refusals"),
    ],
)
def test_refusals_mistakes_exit_2(args: list[str], message: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2 and message in result.output, result.output
