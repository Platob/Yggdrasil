from __future__ import annotations

from enum import IntEnum
from typing import Any, Optional

__all__ = ["EngineName"]

# PARITY: no JS/TS port yet — ``EngineName`` is a backend-only routing concept
# (which compute a Databricks ``Table`` read/write runs on). Add
# ``packages/yggdrasil/enums/engineName.ts`` if it ever needs a client mirror.

# Forgiving aliases → canonical member name.
_ALIASES: dict[str, str] = {
    "yggdrasil": "YGGDRASIL", "ygg": "YGGDRASIL", "native": "YGGDRASIL",
    "delta": "YGGDRASIL", "deltafolder": "YGGDRASIL", "local": "YGGDRASIL",
    "databricks_sql_warehouse": "DATABRICKS_SQL_WAREHOUSE",
    "sql_warehouse": "DATABRICKS_SQL_WAREHOUSE", "warehouse": "DATABRICKS_SQL_WAREHOUSE",
    "databricks": "DATABRICKS_SQL_WAREHOUSE", "sql": "DATABRICKS_SQL_WAREHOUSE",
    "api": "DATABRICKS_SQL_WAREHOUSE", "dbsql": "DATABRICKS_SQL_WAREHOUSE",
    "spark": "SPARK", "pyspark": "SPARK", "databricks_connect": "SPARK", "connect": "SPARK",
}


class EngineName(IntEnum):
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

    @classmethod
    def from_(cls, value: Any, *, default: "Optional[EngineName]" = None) -> "Optional[EngineName]":
        """Coerce an :class:`EngineName` / alias string / int code / ``None``.

        ``None`` (or ``...``) returns *default*. Strings are matched against a
        forgiving alias table (``"warehouse"``, ``"api"`` → ``DATABRICKS_SQL_
        WAREHOUSE``; ``"ygg"``, ``"native"`` → ``YGGDRASIL``; ``"spark"`` →
        ``SPARK``); an unknown value raises :class:`ValueError`.
        """
        if value is None or value is ...:
            return default
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):  # avoid bool-is-int surprises
            raise ValueError(f"Cannot coerce bool {value!r} to EngineName")
        if isinstance(value, int):
            return cls(value)
        key = str(value).strip().lower()
        name = _ALIASES.get(key, key.upper())
        try:
            return cls[name]
        except KeyError:
            raise ValueError(f"Unknown EngineName: {value!r}") from None
