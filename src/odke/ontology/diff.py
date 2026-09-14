"""Schema drift, stated as changes and marked by whether they break a live graph.

The moment a graph is live, editing its ontology is an operational event:
someone narrows a range or renames a predicate and existing queries quietly
stop matching. The question a reviewer asks of a schema change is "does this
break anything?", so every change here answers it.

**Breaking** means something valid under the old schema may not be under the
new one: a removed type, predicate or qualifier; a narrowed range or domain; a
changed cardinality; a scope that drops a key; a predicate that becomes
required; changed entity keys; or a qualifier whose identity flips, which
re-partitions `Fact.signature` (DECISIONS #15). Widening a range to an ancestor
type, `integer` to `number`, and every documentation change are listed too, as
compatible.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from odke.ontology import EntityType, Ontology, Predicate

ChangeKind = Literal["added", "removed", "changed"]

# Literal ranges whose old values all fit the new range.
_WIDENINGS = {("integer", "number"), ("integer", "float"), ("float", "number"), ("number", "float")}


class SchemaChange(BaseModel):
    """One difference between two ontologies, addressed like a `Diagnostic`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ChangeKind
    path: str
    breaking: bool
    detail: str = ""

    def __str__(self) -> str:
        flag = "breaking" if self.breaking else "compatible"
        tail = f": {self.detail}" if self.detail else ""
        return f"{flag:<10} {self.kind:<7} {self.path}{tail}"


def diff(old: Ontology, new: Ontology) -> list[SchemaChange]:
    """Every change from `old` to `new`, in declaration order."""
    out: list[SchemaChange] = []
    for field in ("name", "version", "inferred", "frozen_at", "frozen_by"):
        out += _field(field, getattr(old, field), getattr(new, field))
    for key in _union(old.types, new.types):
        path = f"types.{key}"
        if key not in new.types:
            out.append(_removed(path, "entities of this type lose their schema"))
        elif key not in old.types:
            out.append(SchemaChange(kind="added", path=path, breaking=False))
        else:
            out += _type(path, old.types[key], new.types[key])
    for key in _union(old.predicates, new.predicates):
        path = f"predicates.{key}"
        if key not in new.predicates:
            out.append(_removed(path, "no longer extracted; queries on it stop matching new data"))
        elif key not in old.predicates:
            out.append(SchemaChange(kind="added", path=path, breaking=False))
        else:
            out += _predicate(path, old.predicates[key], new.predicates[key], new)
    return out


def _type(path: str, old: EntityType, new: EntityType) -> Iterator[SchemaChange]:
    if old.parents != new.parents:
        lost = [p for p in old.parents if p not in new.parents]
        why = f"no longer inherits from {', '.join(lost)}" if lost else ""
        yield _changed(f"{path}.parents", old.parents, new.parents, breaking=bool(lost), why=why)
    if old.keys != new.keys:
        yield _changed(
            f"{path}.keys",
            old.keys,
            new.keys,
            breaking=True,
            why="identity is decided differently, so existing nodes split or merge",
        )
    for field in ("description", "aliases"):
        yield from _field(f"{path}.{field}", getattr(old, field), getattr(new, field))


def _predicate(
    path: str, old: Predicate, new: Predicate, ontology: Ontology
) -> Iterator[SchemaChange]:
    if old.range != new.range:
        yield _range(f"{path}.range", old.range, new.range, ontology)
    if old.cardinality != new.cardinality:
        why = (
            "subjects already holding several values now conflict"
            if new.cardinality == "single"
            else "readers expecting one value get several"
        )
        yield _changed(
            f"{path}.cardinality", old.cardinality, new.cardinality, breaking=True, why=why
        )
    if old.scope_keys != new.scope_keys:
        # Compared resolved: the store groups by identity keys plus declared ones,
        # so only a key leaving that tuple can turn stored values into conflicts.
        lost = [k for k in old.scope_keys if k not in new.scope_keys]
        why = f"no longer unique per {', '.join(lost)}, so existing values may conflict"
        yield _changed(
            f"{path}.cardinality_scope",
            old.scope_keys,
            new.scope_keys,
            breaking=bool(lost),
            why=why if lost else "",
        )
    else:
        yield from _field(f"{path}.cardinality_scope", old.cardinality_scope, new.cardinality_scope)
    if old.domain != new.domain:
        dropped = [d for d in old.domain if d not in new.domain]
        narrowed = bool(new.domain) and (not old.domain or bool(dropped))
        why = (
            f"was open to every type, now only {', '.join(new.domain)}"
            if not old.domain
            else f"no longer extracted for {', '.join(dropped)}"
        )
        yield _changed(
            f"{path}.domain",
            _Shown(_domain(old.domain)),
            _Shown(_domain(new.domain)),
            breaking=narrowed,
            why=why if narrowed else "",
        )
    if old.required != new.required:
        yield _changed(
            f"{path}.required",
            old.required,
            new.required,
            breaking=new.required,
            why="existing entities without a value no longer conform" if new.required else "",
        )
    yield from _qualifiers(path, old, new)
    for field in ("label", "description", "aliases", "importance", "examples"):
        yield from _field(f"{path}.{field}", getattr(old, field), getattr(new, field))


def _qualifiers(path: str, old: Predicate, new: Predicate) -> Iterator[SchemaChange]:
    for key in _union(old.qualifiers, new.qualifiers):
        qpath = f"{path}.qualifiers.{key}"
        before, after = old.qualifiers.get(key), new.qualifiers.get(key)
        if after is None:
            merged = before is not None and before.identity
            yield _removed(
                qpath,
                "no longer extracted"
                + ("; facts that differed only in it now merge into one claim" if merged else ""),
            )
        elif before is None:
            yield SchemaChange(
                kind="added",
                path=qpath,
                breaking=after.identity,
                detail="identity-bearing, so existing fact signatures split"
                if after.identity
                else "",
            )
        else:
            if before.identity != after.identity:
                yield _changed(
                    f"{qpath}.identity",
                    before.identity,
                    after.identity,
                    breaking=True,
                    why="fact signatures change, so existing claims split or merge",
                )
            yield from _field(f"{qpath}.description", before.description, after.description)


def _range(path: str, old: str, new: str, ontology: Ontology) -> SchemaChange:
    if (old, new) in _WIDENINGS or new in ontology.lineage(old):
        return _changed(path, old, new, breaking=False, why="widened")
    why = (
        "narrowed: values of the wider range may not fit"
        if old in ontology.lineage(new)
        else "existing values no longer fit"
    )
    return _changed(path, old, new, breaking=True, why=why)


class _Shown(str):
    """A value already rendered for display, so `_show` leaves it alone."""


def _domain(domain: tuple[str, ...]) -> str:
    return _show(domain) if domain else "open"


def _field(path: str, old: Any, new: Any) -> list[SchemaChange]:
    return [] if old == new else [_changed(path, old, new, breaking=False)]


def _changed(path: str, old: Any, new: Any, *, breaking: bool, why: str = "") -> SchemaChange:
    detail = f"{_show(old)} → {_show(new)}" + (f" ({why})" if why else "")
    return SchemaChange(kind="changed", path=path, breaking=breaking, detail=detail)


def _removed(path: str, why: str) -> SchemaChange:
    return SchemaChange(kind="removed", path=path, breaking=True, detail=why)


def _show(value: Any) -> str:
    if isinstance(value, _Shown):
        return str(value)
    if isinstance(value, tuple):
        return f"({', '.join(map(str, value))})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _union(old: Iterable[str], new: Iterable[str]) -> list[str]:
    seen = list(old)
    return seen + [k for k in new if k not in seen]


__all__ = ["ChangeKind", "SchemaChange", "diff"]
