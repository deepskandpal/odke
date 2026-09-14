"""Shared fixtures for the extractor tests."""

from __future__ import annotations

import pytest

from odke import EntityType, Ontology, Predicate


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
