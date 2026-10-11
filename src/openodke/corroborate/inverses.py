"""Inverse and symmetric partners: the facts a schema says a stated fact implies.

A passage that says "France contains Brittany" has also said that Brittany is
located in France, and one that names Ada's husband has named his wife. An
extractor states each claim once, in whichever direction the sentence ran, and a
query or a gold set that asks the other way finds nothing. On Re-DocRED that
was 25 gold facts the extractor had in hand (#106). The ontology already knows
the pairs (`Predicate.inverse_of`, `Predicate.symmetric`), so the partner costs
no model call.

`partners(facts, ontology)` returns, for every edge whose predicate has an
inverse or is symmetric, the same claim read the other way:

- subject and object swap, and the predicate becomes its inverse, or stays when
  it is symmetric; the identity keys are the inverse's own;
- evidence, verdict, polarity, confidence, both clocks and the qualifiers are
  the stated fact's. The partner is the same claim told backwards, so it is
  never grounded again, and a gate that refuses the stated fact on its verdict
  refuses the partner too. Its support is counted from the same evidence, so
  the two share it;
- `qualifiers["odke.derived"]` records the rule and the stated fact's
  signature, so a partner can be told from a stated fact, and retired when the
  fact it came from is (#116).

Nothing is derived from a literal property, which has no inverse; from a fact
that was itself derived; or when the batch already states the partner. Then the
stated fact stands on its own evidence, so there is no duplicate whether or not
a corroborator runs. Partners of one claim stated by several sources share a
signature and merge in the corroborator as their sources do (DECISIONS #28).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from uuid import uuid4

from openodke.corroborate.provenance import DERIVED, unstamped
from openodke.ontology import Ontology
from openodke.types import Fact

Signature = tuple[str, str, str, str, str, tuple[tuple[str, str], ...]]


def partners(facts: Iterable[Fact], ontology: Ontology) -> list[Fact]:
    """The partner of every fact the ontology says implies one, unless the batch states it."""
    inverses = ontology.inverses
    if not inverses:
        return []
    batch = list(facts)
    stated = {fact.signature for fact in batch}
    out: list[Fact] = []
    for fact in batch:
        inverse = inverses.get(fact.predicate)
        if inverse is None or fact.object_entity is None or DERIVED in fact.qualifiers:
            continue
        partner = _partner(fact, inverse, ontology)
        if partner.signature not in stated:
            out.append(partner)
    return out


def _partner(fact: Fact, inverse: str, ontology: Ontology) -> Fact:
    # The batch-relative stamps (a contest, a score) belong to the stated fact's
    # own batch; the partner starts where the stated fact started.
    source = unstamped(fact)
    assert source.object_entity is not None
    rule = "symmetric" if inverse == fact.predicate else "inverse"
    return source.model_copy(
        update={
            "id": uuid4().hex,
            "subject": source.object_entity,
            "object_entity": source.subject,
            "predicate": inverse,
            "identity_keys": ontology.identity_keys(inverse),
            "qualifiers": {**source.qualifiers, DERIVED: {"rule": rule, "of": fact.signature}},
        }
    )


def derived_from(fact: Fact) -> Signature | None:
    """The signature of the stated fact this one was derived from, or None if it was stated.

    Read back from `qualifiers["odke.derived"]` as a tuple, whether the fact is
    in memory or was read from JSON, where the tuple became a list.
    """
    stamp = fact.qualifiers.get(DERIVED)
    of = stamp.get("of") if isinstance(stamp, Mapping) else None
    if not isinstance(of, Sequence) or len(of) != 6:
        return None
    subject, kind, predicate, obj, polarity, scoped = of
    pairs: Any = tuple(tuple(pair) for pair in scoped)
    return (str(subject), str(kind), str(predicate), str(obj), str(polarity), pairs)


__all__ = ["Signature", "derived_from", "partners"]
