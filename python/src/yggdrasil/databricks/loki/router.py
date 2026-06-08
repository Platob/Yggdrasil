"""Route a natural-language Databricks request to a specialized skill.

In the interactive session, "list the catalogs", "what tables are in
samples.nyctaxi", "describe samples.nyctaxi.trips", "show the warehouses" should
*do the thing* — dispatch the right ``databricks-*`` skill — not just reason
about it. :func:`route` maps the user's line to ``(skill_name, kwargs)`` for the
common, unambiguous read requests, or ``None`` to fall back to reasoning.

Read-only by design: it never routes to a state-changing action (triggering a
job, calling an MCP tool). Those stay explicit (``ygg loki run …``).
"""
from __future__ import annotations

import re
from typing import Optional

__all__ = ["route"]

#: catalog.schema.table and catalog.schema identifiers in a line.
_TRIPLE = re.compile(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b")
_PAIR = re.compile(r"\b([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\b")

_STOP = {"the", "a", "this", "my", "catalog", "schema", "database", "all", "of"}


def _ident_after(text: str, words: tuple[str, ...]) -> Optional[str]:
    """The first identifier token following any of *words* (skips stop-words)."""
    for w in words:
        for m in re.finditer(rf"\b{w}\s+([a-zA-Z0-9_]+)", text, re.I):
            tok = m.group(1)
            if tok.lower() not in _STOP:
                return tok
    return None


def route(text: str) -> "tuple[str, dict] | None":
    """Map a NL Databricks request → ``(skill, kwargs)``, or ``None`` to reason."""
    low = text.lower()

    # A read SQL statement → run it as-is.
    if re.match(r"^\s*(select|with)\b", low) or re.search(r"\bselect\b.+\bfrom\b", low):
        return ("databricks-sql", {"query": text.strip()})

    triple = _TRIPLE.search(text)
    pair = _PAIR.search(text)

    # Describe a specific table (typed columns).
    if triple and any(w in low for w in ("describe", "column", "schema of", "fields", "ddl")):
        c, s, t = triple.groups()
        return ("databricks-tables", {"catalog": c, "schema": s, "table": t})

    if "table" in low:
        if triple:
            c, s, t = triple.groups()
            return ("databricks-tables", {"catalog": c, "schema": s, "table": t})
        if pair:
            c, s = pair.groups()
            return ("databricks-tables", {"catalog": c, "schema": s})
        return ("databricks-tables", {})

    if "schema" in low or "database" in low:
        cat = _ident_after(text, ("in", "of", "for", "from"))
        return ("databricks-schemas", {"catalog": cat} if cat else {})

    if "catalog" in low:
        cat = _ident_after(text, ("in", "of", "for"))
        return ("databricks-catalogs", {"catalog": cat} if cat else {})

    if "warehouse" in low:
        return ("databricks-warehouses", {})
    if "job run" in low or "job-run" in low or re.search(r"\bruns?\b", low):
        return ("databricks-job-runs", {})
    if "job" in low:
        return ("databricks-jobs", {})            # list only — never auto-trigger
    if "cluster" in low:
        return ("databricks-clusters", {})
    if "volume" in low:
        cat = _ident_after(text, ("in", "of", "for"))
        return ("databricks-volumes", {"catalog": cat} if cat else {})
    if "secret" in low or "scope" in low:
        return ("databricks-secrets", {})
    if any(w in low for w in ("who am i", "whoami", "current user", "my identity")):
        return ("databricks-iam", {"what": "me"})
    if "users" in low:
        return ("databricks-iam", {"what": "users"})
    if "groups" in low:
        return ("databricks-iam", {"what": "groups"})
    if "serving" in low or "endpoint" in low:
        return ("databricks-serving", {})
    return None
