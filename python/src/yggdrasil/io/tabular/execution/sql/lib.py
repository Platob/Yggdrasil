"""Optional-dependency guards for :mod:`yggdrasil.sql`.

The SQL module rests on two optional packages:

- ``sqlglot`` — the parser. We default to the Databricks flavor
  (closest to Spark SQL, very close to Postgres on the SELECT
  surface) and accept any of the dialects sqlglot knows. When
  sqlglot isn't installed we still expose the in-Python
  :class:`yggdrasil.data.expr.Expression` lifters that don't go
  through the parser, but :func:`sql` itself raises a clean
  install hint.
- ``polars`` — the *preferred* execution backend. Most non-trivial
  SQL (joins, aggregations, ``ORDER BY``, CTEs, window functions)
  lands on :class:`polars.SQLContext`; without polars, the executor
  falls back to a small Arrow-only path that handles the
  ``SELECT cols FROM src WHERE pred LIMIT n`` shape.

Each accessor caches its module on first call and surfaces a
helpful "install ``pip install ygg[...]``" error otherwise — same
pattern used by :mod:`yggdrasil.polars.lib` and
:mod:`yggdrasil.postgres.lib`.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "sqlglot_module",
    "sqlglot_expressions",
    "has_sqlglot",
    "polars_module",
    "has_polars",
]


_sqlglot: Any = None
_sqlglot_attempted: bool = False
_sqlglot_expressions: Any = None


def sqlglot_module() -> Any:
    """Return the imported :mod:`sqlglot` module."""
    global _sqlglot, _sqlglot_attempted
    if _sqlglot is not None:
        return _sqlglot
    if _sqlglot_attempted:
        raise ImportError(
            "sqlglot is required for yggdrasil.sql parsing; "
            "install it with `pip install sqlglot` or "
            "`pip install ygg[sql]`."
        )
    _sqlglot_attempted = True
    try:
        import sqlglot as _mod
    except ImportError as exc:
        raise ImportError(
            "sqlglot is required for yggdrasil.sql parsing; "
            "install it with `pip install sqlglot` or "
            "`pip install ygg[sql]`."
        ) from exc
    _sqlglot = _mod
    return _mod


def sqlglot_expressions() -> Any:
    """Return :mod:`sqlglot.expressions` (the AST node module)."""
    global _sqlglot_expressions
    if _sqlglot_expressions is not None:
        return _sqlglot_expressions
    sqlglot_module()
    from sqlglot import expressions as _exprs

    _sqlglot_expressions = _exprs
    return _exprs


def has_sqlglot() -> bool:
    """Whether :mod:`sqlglot` is importable. Probe-only — never raises."""
    global _sqlglot, _sqlglot_attempted
    if _sqlglot is not None:
        return True
    if _sqlglot_attempted:
        return False
    try:
        sqlglot_module()
    except ImportError:
        return False
    return True


def polars_module() -> Any:
    """Return the imported :mod:`polars` module via the canonical guard."""
    from yggdrasil.polars.lib import polars as _pl_proxy

    # The polars proxy is a real module reference once touched; force
    # one attribute lookup so we get the live module back.
    _ = _pl_proxy.__name__  # type: ignore[attr-defined]
    import polars as _pl

    return _pl


def has_polars() -> bool:
    """Whether :mod:`polars` is importable. Probe-only — never raises."""
    try:
        import polars  # noqa: F401
    except Exception:
        return False
    return True
