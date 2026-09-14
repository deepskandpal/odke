"""Resolution: pairwise P/R over labelled pairs, and B-cubed over three clusters.

Expected numbers are hand-computed from tests/fixtures/eval/resolve.*.jsonl —
not a benchmark.

Labels imply three gold clusters: {a, b, c}, {d, e}, {f}.
Links: a-b same_as, c-d same_as, d-e same_as (as an EntityLink), f-a different,
b-c different, e-f similar. Predicted clusters: {a, b}, {c, d, e}, {f}.

Pairwise over the six labelled pairs:
    a-b same  -> together  tp        c-d different -> together  fp
    b-c same  -> apart     fn        e-f different -> apart     tn
    d-e same  -> together  tp        a-f different -> apart     tn
P = R = F1 = 2/3.

B-cubed, per key (overlap / predicted size, overlap / gold size):
    a 2/2 2/3   b 2/2 2/3   c 1/3 1/3   d 2/3 2/2   e 2/3 2/2   f 1/1 1/1
precision = (1 + 1 + 1/3 + 2/3 + 2/3 + 1) / 6 = 7/9, recall = 14/3 / 6 = 7/9.

With `similar_as_same`, e-f merges too: predicted {a, b}, {c, d, e, f}.
    a 1 2/3   b 1 2/3   c 1/4 1/3   d 2/4 1   e 2/4 1   f 1/4 1
precision = 3.5 / 6 = 7/12, recall = 7/9, F1 = 2/3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openodke import EntityLink, LinkKind
from openodke.eval import LinkRow, PairLabel, load_jsonl
from openodke.eval.resolution import (
    as_triples,
    clusters,
    evaluate_resolution,
    links_from_clusters,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
LABELS = load_jsonl(FIXTURES / "resolve.labels.jsonl", PairLabel)
LINKS = load_jsonl(FIXTURES / "resolve.predictions.jsonl", LinkRow)


def test_pairwise_precision_and_recall() -> None:
    report = evaluate_resolution(LABELS, LINKS)
    m = report.metrics
    assert report.stage == "resolve" and report.n == 6
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["pairwise_precision"] == pytest.approx(2 / 3)
    assert m["pairwise_recall"] == pytest.approx(2 / 3)
    assert m["pairwise_accuracy"] == pytest.approx(4 / 6)
    assert report.confusion == {
        "same": {"same": 2, "different": 1},
        "different": {"same": 1, "different": 2},
    }


def test_bcubed_over_three_gold_clusters() -> None:
    report = evaluate_resolution(LABELS, LINKS)
    m = report.metrics
    assert (m["gold_clusters"], m["predicted_clusters"]) == (3, 3)
    assert m["bcubed_precision"] == pytest.approx(7 / 9)
    assert m["bcubed_recall"] == pytest.approx(7 / 9)
    assert m["bcubed_f1"] == pytest.approx(7 / 9)
    abc = report.breakdown["a, b, c"]
    assert abc["size"] == 3
    assert abc["bcubed_precision"] == pytest.approx(7 / 9)
    assert abc["bcubed_recall"] == pytest.approx(5 / 9)
    # The resolver split {a, b, c} across two clusters.
    assert abc["predicted_clusters"] == 2
    assert report.breakdown["d, e"]["bcubed_recall"] == 1.0
    assert report.breakdown["f"]["bcubed_precision"] == 1.0


def test_similar_as_same_makes_bcubed_asymmetric() -> None:
    """Merging e-f too costs precision and leaves recall where it was."""
    report = evaluate_resolution(LABELS, LINKS, similar_as_same=True)
    m = report.metrics
    assert m["bcubed_precision"] == pytest.approx(7 / 12)
    assert m["bcubed_recall"] == pytest.approx(7 / 9)
    assert m["bcubed_f1"] == pytest.approx(2 / 3)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 1, 1)
    assert m["pairwise_precision"] == 0.5


def test_link_counts_and_the_similar_note() -> None:
    report = evaluate_resolution(LABELS, LINKS)
    m = report.metrics
    assert (m["links_same_as"], m["links_similar"], m["links_different"]) == (3, 1, 2)
    assert "1 similar link(s) counted as no decision" in report.notes


def test_links_from_any_source_score_the_same() -> None:
    """EntityLinks, plain tuples and LinkRows are one triple."""
    as_rows = evaluate_resolution(LABELS, LINKS)
    as_tuples = evaluate_resolution(LABELS, [(r.a, r.b, r.kind.value) for r in LINKS])
    as_entity_links = evaluate_resolution(
        LABELS, [EntityLink(source_key=r.a, target_key=r.b, kind=r.kind) for r in LINKS]
    )
    assert as_rows.metrics == as_tuples.metrics == as_entity_links.metrics
    assert as_triples([("x", "y", "different")]) == [("x", "y", LinkKind.DIFFERENT)]


def test_a_platforms_merge_groups_score_like_links() -> None:
    """A store that replaced nodes is read back as groups, and measured the same way."""
    merged = links_from_clusters([["a", "b", "c"], ["d", "e"]])
    assert merged[0] == LinkRow(a="a", b="b", kind=LinkKind.SAME_AS)
    report = evaluate_resolution(LABELS, merged)
    m = report.metrics
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 0, 0, 3)
    assert m["bcubed_precision"] == 1.0 and m["bcubed_recall"] == 1.0


def test_same_as_chains_are_followed_through_unlabelled_keys() -> None:
    labels = [PairLabel(a="a", b="b", same=True)]
    report = evaluate_resolution(labels, [("a", "x", "same_as"), ("x", "b", "same_as")])
    assert report.metrics["tp"] == 1
    assert any("1 linked key(s) appear in no labelled pair" in n for n in report.notes)


def test_a_resolver_that_disagrees_with_itself_is_noted() -> None:
    links = [("a", "b", "same_as"), ("b", "c", "same_as"), ("a", "c", "different")]
    report = evaluate_resolution(LABELS, links)
    assert "1 different link(s) join keys a same_as chain merges anyway" in report.notes


def test_inconsistent_labels_are_noted() -> None:
    labels = [
        PairLabel(a="a", b="b", same=True),
        PairLabel(a="b", b="c", same=True),
        PairLabel(a="a", b="c", same=False),
    ]
    report = evaluate_resolution(labels, [])
    assert any("1 pair(s) labelled different are joined" in n for n in report.notes)
    # No links: nothing predicted same, so pairwise precision is undefined, not zero.
    assert report.metrics["pairwise_precision"] is None
    assert report.metrics["pairwise_recall"] == 0.0


def test_clusters_is_the_transitive_closure() -> None:
    groups = clusters(["a", "b", "c", "d"], [("a", "b"), ("c", "b")])
    assert groups["c"] == frozenset({"a", "b", "c"})
    assert groups["d"] == frozenset({"d"})
