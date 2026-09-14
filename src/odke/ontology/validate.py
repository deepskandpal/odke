"""What is subtly wrong with a schema, found before a model is ever asked.

A schema that is subtly wrong produces extractions that are subtly wrong, and
the user blames the model. A range spelt `Compnay` quietly turns an edge into a
string property; a domain naming a type that does not exist means a predicate
no snippet will ever include. None of that fails loudly on its own, so it is
looked for here.

Two severities, and the line between them is whether extraction goes wrong.
An **error** is a schema that produces wrong or missing facts: strict loading
refuses it and `odke ontology validate` exits non-zero. A **warning** is a
schema that works but is probably not what was meant — an inheritance cycle
`lineage()` already tolerates, a domain that is half right.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from odke.ontology import EntityType, Ontology, Predicate

Severity = Literal["error", "warning"]


class Diagnostic(BaseModel):
    """One thing wrong with a schema, addressed by the exact key that is wrong.

    Structured rather than a sentence, so a CI step counts errors, a test
    asserts on `code` without matching prose, and an editor integration can
    jump to `path`. The path is dotted and indexes into lists —
    `types.Scientist.parents[0]` — so it names the value to edit, not just the
    entry it sits in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    path: str
    message: str
    severity: Severity

    def __str__(self) -> str:
        return f"{self.severity}: {self.path}: {self.message} [{self.code}]"


def diagnose(ontology: Ontology) -> list[Diagnostic]:
    """Every diagnostic for one ontology, types first, in declaration order."""
    found: list[Diagnostic] = []
    for key, entity_type in ontology.types.items():
        found += _type(ontology, key, entity_type)
    found += _cycles(ontology)
    for key, predicate in ontology.predicates.items():
        found += _predicate(ontology, key, predicate)
        found += _scope(key, predicate)
    found += _aliases("types", ((k, t.aliases) for k, t in ontology.types.items()), "type")
    found += _aliases(
        "predicates", ((k, p.aliases) for k, p in ontology.predicates.items()), "predicate"
    )
    return found


def _type(ontology: Ontology, key: str, entity_type: EntityType) -> Iterable[Diagnostic]:
    base = f"types.{key}"
    if entity_type.name != key:
        yield _error(
            "name-mismatch",
            f"{base}.name",
            f"is {entity_type.name!r} but the entry is keyed {key!r}; entities will carry "
            f"{entity_type.name!r} and the ontology looks them up by {key!r}",
        )
    for i, parent in enumerate(entity_type.parents):
        if parent not in ontology.types:
            yield _error(
                "unknown-parent",
                f"{base}.parents[{i}]",
                f"{parent!r} is not an entity type{_close(parent, ontology.types)}; "
                f"{key} inherits nothing from it",
            )
    lineage = ontology.lineage(key)
    for i, identity_key in enumerate(entity_type.keys):
        predicate = ontology.predicates.get(identity_key)
        if predicate is None:
            yield _error(
                "unknown-key",
                f"{base}.keys[{i}]",
                f"{identity_key!r} is not a predicate{_close(identity_key, ontology.predicates)}",
            )
        elif predicate.domain and not any(d in lineage for d in predicate.domain):
            yield _error(
                "key-outside-domain",
                f"{base}.keys[{i}]",
                f"{identity_key!r} applies to {', '.join(predicate.domain)}, not {key}, "
                "so it is never extracted for this type and the key can never be filled",
            )


def _cycles(ontology: Ontology) -> Iterable[Diagnostic]:
    reported: set[frozenset[str]] = set()
    for key, entity_type in ontology.types.items():
        above = set().union(*(ontology.lineage(p) for p in entity_type.parents))
        if key not in above:
            continue
        members = frozenset(t for t in ontology.lineage(key) if key in ontology.lineage(t))
        if members in reported:
            continue
        reported.add(members)
        shown = (
            f"{key} inherits from itself"
            if len(members) == 1
            else f"{', '.join(sorted(members))} inherit from each other"
        )
        yield _warning(
            "inheritance-cycle",
            f"types.{key}.parents",
            f"{shown}; lineage() tolerates the cycle, but every type in it gets every "
            "other's predicates, which is rarely what was meant",
        )


def _predicate(ontology: Ontology, key: str, predicate: Predicate) -> Iterable[Diagnostic]:
    from odke.ontology import _JSON_TYPES

    base = f"predicates.{key}"
    if predicate.name != key:
        yield _error(
            "name-mismatch",
            f"{base}.name",
            f"is {predicate.name!r} but the entry is keyed {key!r}; facts will carry "
            f"{predicate.name!r} and its identity keys will never be found",
        )
    if predicate.range not in ontology.types and predicate.range not in _JSON_TYPES:
        yield _error(
            "unknown-range",
            f"{base}.range",
            f"{predicate.range!r} is neither an entity type nor a literal type "
            f"({', '.join(sorted(_JSON_TYPES))})"
            f"{_close(predicate.range, [*ontology.types, *_JSON_TYPES])}",
        )
    unknown = [(i, d) for i, d in enumerate(predicate.domain) if d not in ontology.types]
    if predicate.domain and len(unknown) == len(predicate.domain):
        yield _error(
            "unreachable-predicate",
            f"{base}.domain",
            f"names no entity type ({', '.join(predicate.domain)}), so no snippet includes "
            f"{key!r} and it can never be extracted",
        )
        return
    for i, name in unknown:
        yield _warning(
            "unknown-domain",
            f"{base}.domain[{i}]",
            f"{name!r} is not an entity type{_close(name, ontology.types)}; "
            f"{key!r} is still extracted for the rest of its domain",
        )


def _scope(key: str, predicate: Predicate) -> Iterable[Diagnostic]:
    """R4: a declared cardinality scope may only name identity-bearing qualifiers."""
    base = f"predicates.{key}.cardinality_scope"
    for i, name in enumerate(predicate.cardinality_scope):
        qualifier = predicate.qualifiers.get(name)
        if qualifier is None:
            yield _error(
                "scope-undeclared-qualifier",
                f"{base}[{i}]",
                f"{name!r} is not a qualifier of {key!r}{_close(name, predicate.qualifiers)}",
            )
        elif not qualifier.identity:
            yield _error(
                "scope-not-identity",
                f"{base}[{i}]",
                f"{name!r} is reconcilable, not identity-bearing: facts that differ only in "
                f"it are one claim (DECISIONS #11), so it cannot separate one value from "
                "another. Declare it `identity: true` or drop it from the scope",
            )


def _aliases(
    section: str, entries: Iterable[tuple[str, tuple[str, ...]]], kind: str
) -> Iterable[Diagnostic]:
    """A surface form that points at two entries cannot be assigned to either."""
    listed = list(entries)
    owners = {_norm(key): key for key, _ in listed}
    for key, aliases in listed:
        for i, alias in enumerate(aliases):
            owner = owners.setdefault(_norm(alias), key)
            if owner != key:
                yield _error(
                    "duplicate-alias",
                    f"{section}.{key}.aliases[{i}]",
                    f"{alias!r} already refers to the {kind} {owner!r}; a mention of it "
                    f"cannot be assigned to one {kind}",
                )


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("_", " ").split())


def _close(value: str, candidates: Iterable[str]) -> str:
    match = difflib.get_close_matches(value, list(candidates), n=1)
    return f" — did you mean {match[0]!r}?" if match else ""


def _error(code: str, path: str, message: str) -> Diagnostic:
    return Diagnostic(code=code, path=path, message=message, severity="error")


def _warning(code: str, path: str, message: str) -> Diagnostic:
    return Diagnostic(code=code, path=path, message=message, severity="warning")


__all__ = ["Diagnostic", "Severity", "diagnose"]
