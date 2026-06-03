"""Options for a Databricks :class:`~yggdrasil.databricks.table.table.Table`.

Centralises the table-level read/write knobs in one :class:`CastOptions`
subclass so :meth:`Table.options_class` has a single home to grow.
"""
from __future__ import annotations

import dataclasses

from yggdrasil.data.options import CastOptions
from yggdrasil.enums.engine_type import EngineType

__all__ = ["TableOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class TableOptions(CastOptions):
    """:class:`CastOptions` for a Unity Catalog :class:`Table`.

    Inherits the full cast / projection / predicate / merge-maintenance
    surface (``target``, ``predicate``, ``row_limit``, ``mode``,
    ``match_by``, ``zorder_by``, ``vacuum_hours``, …) and adds the
    table-only routing knob:

    - :attr:`engine` — pick the read/write compute (an :class:`EngineType`):

      * :attr:`~EngineType.YGGDRASIL` — yggdrasil's **native DeltaFolder** (a
        direct ``_delta_log`` + parquet path over UC-vended credentials) when
        the table is Delta-backed. Native writes need an *external* Delta table
        (UC vends read-only credentials for managed tables); a non-Delta or
        managed-Delta write falls back to the warehouse.
      * :attr:`~EngineType.DATABRICKS_SQL_WAREHOUSE` — the SQL warehouse.
      * :attr:`~EngineType.SPARK` — a Spark session.
      * ``None`` (default) — **guess best** per call: an active Spark session →
        ``SPARK``; otherwise a small Delta table (< 128 MiB on disk) →
        ``YGGDRASIL``; a larger one → ``DATABRICKS_SQL_WAREHOUSE``.

    In every case, if the native path can't get UC credentials for the
    table's storage, the read/write transparently falls back to the warehouse.
    """

    #: Read/write compute selector (:class:`EngineType`). ``None`` → guess best
    #: from active Spark + table size.
    engine: "EngineType | None" = None

    def __post_init__(self) -> None:
        # Explicit base call — a ``slots=True`` dataclass replaces the class
        # object, which breaks the zero-arg ``super()`` cell.
        CastOptions.__post_init__(self)
        # Coerce an alias string / int code into a canonical EngineType so
        # ``options.engine == EngineType.SPARK`` comparisons hold downstream.
        if self.engine is not None and not isinstance(self.engine, EngineType):
            object.__setattr__(self, "engine", EngineType.from_(self.engine))
