"""SQL flavor handling for :mod:`yggdrasil.sql`.

We piggyback on the same :class:`Dialect` enum the expression
backend already uses (:mod:`yggdrasil.data.expr.backends.sql`) so a
predicate parsed via :func:`Expression.from_sql` and a query parsed
via :func:`yggdrasil.sql.sql` agree on identifier quoting, literal
escaping, and ``ILIKE`` availability without a second source of
truth.

Default flavor is **Databricks** — Spark-SQL surface, ``ILIKE``,
backtick identifiers — because that's the closest match to the
shape we already speak elsewhere in the package and to what
analysts in the wild type. Postgres / SQLite / MySQL / ANSI are
pass-throughs to sqlglot.
"""

from __future__ import annotations

from yggdrasil.data.expr.backends.sql import (
    DEFAULT_DIALECT,
    Dialect,
    _resolve_dialect,
)

__all__ = ["Dialect", "DEFAULT_DIALECT", "resolve_dialect"]


def resolve_dialect(spec: "Dialect | str | None") -> Dialect:
    """Coerce *spec* into a :class:`Dialect`, default :data:`DEFAULT_DIALECT`.

    Pass-through for already-:class:`Dialect` values; case-insensitive
    string lookup otherwise. Unknown names raise with the valid
    options enumerated — callers don't have to guess.
    """
    return _resolve_dialect(spec)
