"""Shared fixtures: the people ontology, and a copy of the examples to run."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from openodke import EntityType, Ontology, Predicate

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def every_eval_report_is_sound(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every `EvalReport` any test builds fits the schema, and its ranges hold its numbers (#139).

    Collected as they are made and checked when the test ends, so a builder
    that drifts from the schema fails whichever test reached it first.
    """
    from openodke.eval.eval_report import EvalReport, check_report

    made: list[EvalReport] = []
    build = EvalReport.__init__

    def init(self: EvalReport, /, **data: object) -> None:
        build(self, **data)
        made.append(self)

    monkeypatch.setattr(EvalReport, "__init__", init)
    yield
    for report in made:
        problems = check_report(report.model_dump(mode="json"))
        assert not problems, f"{report.title}: {problems}"
        for row in report.rows:
            for name in ("precision", "recall", "f1"):
                found = getattr(row.performance, name)
                if found.value is not None and found.low is not None:
                    assert found.high is not None
                    assert found.low <= found.value <= found.high, (row.name, name, found)


@pytest.fixture
def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """`examples/e2e/` inside a copy of `examples/`, so a run writes nowhere in the repository.

    The example's config puts its own directory on `sys.path` and imports
    `e2e_stages`; both are undone afterwards.
    """
    shutil.copytree(EXAMPLES, tmp_path / "examples", ignore=shutil.ignore_patterns("out"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop("e2e_stages", None)
    yield tmp_path / "examples" / "e2e"
    sys.modules.pop("e2e_stages", None)


@pytest.fixture
def people() -> Ontology:
    """People and companies: an edge, a reconcilable qualifier and an identity one."""
    return Ontology(
        name="people",
        version="1",
        types={
            "Person": EntityType(
                name="Person", description="An individual human being.", keys=("full_name",)
            ),
            "Company": EntityType(
                name="Company", description="An incorporated organisation.", keys=("legal_name",)
            ),
        },
        predicates={
            "full_name": Predicate(
                name="full_name",
                aliases=("name",),
                description="The person's full name as written.",
                domain=("Person",),
                importance=1.0,
            ),
            "birth_date": Predicate(
                name="birth_date",
                aliases=("born",),
                domain=("Person",),
                range="date",
                importance=0.9,
            ),
            "employer": Predicate(
                name="employer",
                description="An organisation the person works or worked for.",
                domain=("Person",),
                range="Company",
                cardinality="multi",
                qualifiers={"start_date": {}},
                importance=0.8,
            ),
            "legal_name": Predicate(
                name="legal_name", aliases=("company",), domain=("Company",), importance=1.0
            ),
            "uptime": Predicate(
                name="uptime",
                description="Measured service availability.",
                domain=("Company",),
                qualifiers={"percentile": {"identity": True}, "year": {}},
                importance=0.6,
            ),
        },
    )
