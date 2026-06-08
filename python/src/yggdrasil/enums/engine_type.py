from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional

__all__ = ["EngineType"]

# PARITY: no JS/TS port yet — ``EngineType`` is a backend-only routing concept
# (which compute a Databricks ``Table`` read/write runs on). Add
# ``packages/yggdrasil/enums/engineType.ts`` if it ever needs a client mirror.

# Forgiving aliases → canonical member name. Deliberately NO ``"api"`` alias —
# it's too broad to map unambiguously onto one engine.
_ALIASES: dict[str, str] = {
    "yggdrasil": "YGGDRASIL", "ygg": "YGGDRASIL", "native": "YGGDRASIL",
    "delta": "YGGDRASIL", "deltafolder": "YGGDRASIL", "local": "YGGDRASIL",
    "databricks_sql_warehouse": "DATABRICKS_SQL_WAREHOUSE",
    "sql_warehouse": "DATABRICKS_SQL_WAREHOUSE", "warehouse": "DATABRICKS_SQL_WAREHOUSE",
    "databricks": "DATABRICKS_SQL_WAREHOUSE", "sql": "DATABRICKS_SQL_WAREHOUSE",
    "dbsql": "DATABRICKS_SQL_WAREHOUSE",
    "spark": "SPARK", "pyspark": "SPARK", "databricks_connect": "SPARK", "connect": "SPARK",
}


class EngineType(IntEnum):
    """Compute engine for a Databricks :class:`~yggdrasil.databricks.table.table.Table`
    read / write.

    - :attr:`YGGDRASIL` — yggdrasil's native DeltaFolder (direct ``_delta_log``
      + parquet over UC-vended credentials), no warehouse.
    - :attr:`DATABRICKS_SQL_WAREHOUSE` — the Databricks SQL warehouse.
    - :attr:`SPARK` — a Spark session (Databricks Connect / cluster / notebook).
    """

    YGGDRASIL = 0
    DATABRICKS_SQL_WAREHOUSE = 1
    SPARK = 2

    # ``default=...`` (Ellipsis) is the "raise on failure" sentinel; any other
    # value is returned as the fallback when the input can't be parsed.
    @classmethod
    def from_str(cls, value: Any, *, default: Any = ...) -> "Optional[EngineType]":
        """Coerce an alias string into an :class:`EngineType`.

        Matches a forgiving alias table (``"warehouse"`` / ``"sql"`` →
        ``DATABRICKS_SQL_WAREHOUSE``; ``"ygg"`` / ``"native"`` → ``YGGDRASIL``;
        ``"spark"`` / ``"connect"`` → ``SPARK``). On an unknown string: raise
        :class:`ValueError` when *default* is ``...``, else return *default*.
        """
        key = str(value).strip().lower()
        name = _ALIASES.get(key, key.upper())
        try:
            return cls[name]
        except KeyError:
            if default is ...:
                raise ValueError(f"Unknown {cls.__name__}: {value!r}") from None
            return default

    @classmethod
    def from_numeric(cls, value: Any, *, default: Any = ...) -> "Optional[EngineType]":
        """Coerce an integer code into an :class:`EngineType`.

        On an out-of-range / non-integer code: raise when *default* is ``...``,
        else return *default*. ``bool`` is rejected (it's an ``int`` subclass
        but never a meaningful engine code).
        """
        if isinstance(value, bool):
            if default is ...:
                raise ValueError(f"Cannot coerce bool {value!r} to {cls.__name__}")
            return default
        try:
            return cls(int(value))
        except (ValueError, TypeError):
            if default is ...:
                raise ValueError(f"Unknown {cls.__name__} code: {value!r}") from None
            return default

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "Optional[EngineType]":
        """Coerce an :class:`EngineType` / alias string / int code / ``None``.

        ``None`` (or ``...``) is "unset" and returns ``None``. A string routes
        to :meth:`from_str`, a number to :meth:`from_numeric`; both honour
        *default* (``...`` → raise, else return the fallback).
        """
        if value is None or value is ...:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, float)):  # bool handled inside from_numeric
            return cls.from_numeric(value, default=default)
        return cls.from_str(value, default=default)
