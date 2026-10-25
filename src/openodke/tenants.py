"""Tenant keys: one store, many tenants, and nothing merged across them (#159).

A store written by more than one tenant must keep their facts apart: two
tenants who state the same claim own two facts, and a store lookup, a store
merge or a retraction for one never reaches the other's. The sink keys each
tenant apart (DECISIONS #44):

- an entity's key in the store is `<tenant>/<key>`;
- a fact's signature is hashed with its tenant;
- each node and relationship carries `tenant`, the tenant's name.

So the uniqueness constraints `bootstrap()` creates, and every MERGE, keep
working on one indexed property, and a key looked up is the tenant's by
construction. In memory nothing changes: one run is one tenant, and its keys
are the keys its extractor gave. A store with no tenant is written as it
always was.

A tenant's name is letters, digits, `_`, `.` and `-`, starting with a letter
or a digit. It never holds `/`, so a scoped key splits at its first `/`.
"""

from __future__ import annotations

import re

# The property naming the tenant of a node or a relationship the sink wrote.
TENANT = "tenant"
SEPARATOR = "/"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def tenant_name(tenant: str | None) -> str | None:
    """`tenant`, checked: None, or a name a scoped key can carry."""
    if tenant is None:
        return None
    if not isinstance(tenant, str) or not _NAME.fullmatch(tenant):
        raise ValueError(
            f"tenant {tenant!r}: letters, digits, '_', '.' and '-', starting with a letter or "
            "a digit"
        )
    return tenant


def scoped(key: str, tenant: str | None) -> str:
    """The key the store holds for `key` in `tenant`: `<tenant>/<key>`, or `key` with none."""
    return key if tenant is None else f"{tenant}{SEPARATOR}{key}"


def unscoped(key: str, tenant: str | None) -> str:
    """The key `scoped` was given, from the one the store holds."""
    prefix = f"{tenant}{SEPARATOR}"
    return key[len(prefix) :] if tenant is not None and key.startswith(prefix) else key


__all__ = ["SEPARATOR", "TENANT", "scoped", "tenant_name", "unscoped"]
