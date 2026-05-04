"""Optional-dependency guard for the Postgres backend.

Two drivers cooperate:

* :mod:`adbc_driver_postgresql` (preferred) — exposes a DBAPI cursor
  with native Arrow read/write paths (``fetch_arrow_table`` /
  ``adbc_ingest``). Used for everything tabular: query results,
  bulk inserts, schema fetch.
* :mod:`psycopg` (psycopg 3) — used for DDL, lifecycle (``CREATE``
  / ``DROP``), metadata queries against ``information_schema`` /
  ``pg_catalog``, and the fallback path when ADBC isn't installed.

At least one of the two must be importable; ADBC is the canonical
fast path for Arrow movement, but a pure-``psycopg`` install still
works (Arrow tables go through row materialization).

Each accessor caches its module on first call and surfaces a
helpful "install ``pip install ygg[postgres]``" error otherwise —
matching the pattern used by every other engine in the package
(:mod:`yggdrasil.polars.lib`, :mod:`yggdrasil.pandas.lib`).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "psycopg",
    "psycopg_module",
    "adbc_dbapi",
    "adbc_dbapi_module",
    "has_adbc",
    "has_psycopg",
]


_psycopg: Any = None
_psycopg_attempted: bool = False
_adbc: Any = None
_adbc_attempted: bool = False


def psycopg_module() -> Any:
    """Return the imported :mod:`psycopg` (psycopg 3) module."""
    global _psycopg, _psycopg_attempted
    if _psycopg is not None:
        return _psycopg
    if _psycopg_attempted:
        raise ImportError(
            "psycopg (psycopg 3) is required for yggdrasil.postgres; "
            "install it with `pip install ygg[postgres]` or "
            "`pip install psycopg[binary]`."
        )
    _psycopg_attempted = True
    try:
        import psycopg as _mod
    except ImportError as exc:
        raise ImportError(
            "psycopg (psycopg 3) is required for yggdrasil.postgres; "
            "install it with `pip install ygg[postgres]` or "
            "`pip install psycopg[binary]`."
        ) from exc
    _psycopg = _mod
    return _mod


def adbc_dbapi_module() -> Any:
    """Return :mod:`adbc_driver_postgresql.dbapi`.

    ADBC is the fast path for Arrow IO. When it isn't installed,
    callers fall back to the psycopg row-materialization path —
    correct, but materially slower for large result sets.
    """
    global _adbc, _adbc_attempted
    if _adbc is not None:
        return _adbc
    if _adbc_attempted:
        raise ImportError(
            "adbc_driver_postgresql is required for the Arrow-native "
            "Postgres path; install it with `pip install ygg[postgres]` "
            "or `pip install adbc-driver-postgresql`."
        )
    _adbc_attempted = True
    try:
        from adbc_driver_postgresql import dbapi as _dbapi
    except ImportError as exc:
        raise ImportError(
            "adbc_driver_postgresql is required for the Arrow-native "
            "Postgres path; install it with `pip install ygg[postgres]` "
            "or `pip install adbc-driver-postgresql`."
        ) from exc
    _adbc = _dbapi
    return _dbapi


def has_adbc() -> bool:
    """Whether :mod:`adbc_driver_postgresql` is importable.

    Probe-only — does not raise. Used to pick the Arrow-fast path
    over the psycopg fallback at runtime without forcing ADBC as
    a hard dependency.
    """
    global _adbc, _adbc_attempted
    if _adbc is not None:
        return True
    if _adbc_attempted:
        return False
    try:
        adbc_dbapi_module()
    except ImportError:
        return False
    return True


def has_psycopg() -> bool:
    """Whether :mod:`psycopg` (psycopg 3) is importable. Probe-only."""
    global _psycopg, _psycopg_attempted
    if _psycopg is not None:
        return True
    if _psycopg_attempted:
        return False
    try:
        psycopg_module()
    except ImportError:
        return False
    return True


# Convenience singletons matching the rest-of-package pattern
# (``from yggdrasil.polars.lib import polars``). Resolution is
# deferred to first attribute access so importing this module
# alone never fails on a missing optional package.


class _LazyModule:
    __slots__ = ("_loader", "_module")

    def __init__(self, loader):
        self._loader = loader
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            self._module = self._loader()
        return getattr(self._module, name)

    def __bool__(self) -> bool:
        return self._module is not None


psycopg = _LazyModule(psycopg_module)
adbc_dbapi = _LazyModule(adbc_dbapi_module)
