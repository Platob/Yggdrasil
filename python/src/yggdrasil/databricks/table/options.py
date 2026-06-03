"""Options for a Databricks :class:`~yggdrasil.databricks.table.table.Table`.

Centralises the table-level read/write knobs in one :class:`CastOptions`
subclass so :meth:`Table.options_class` has a single home to grow.
"""
from __future__ import annotations

import dataclasses

from yggdrasil.data.options import CastOptions

__all__ = ["TableOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class TableOptions(CastOptions):
    """:class:`CastOptions` for a Unity Catalog :class:`Table`.

    Inherits the full cast / projection / predicate / merge-maintenance
    surface (``target``, ``predicate``, ``row_limit``, ``mode``,
    ``match_by``, ``zorder_by``, ``vacuum_hours``, …) and adds the
    table-only routing knob:

    - :attr:`use_warehouse` — pick the read/write engine:

      * ``True`` — always go through the SQL warehouse (Databricks).
      * ``False`` — prefer yggdrasil's **native DeltaFolder** (a direct
        ``_delta_log`` + parquet path over UC-vended credentials) whenever
        the table is Delta-backed. Native writes need an *external* Delta
        table (UC vends read-only credentials for managed tables), so a
        managed-Delta write still goes through the warehouse.
      * ``None`` (default) — **guess** per call: an active Spark session →
        Databricks; otherwise a small Delta table (< 128 MiB on disk) →
        native DeltaFolder, a larger one → Databricks. Non-Delta tables
        always use the warehouse.
    """

    #: Read/write engine selector. ``True`` → SQL warehouse, ``False`` →
    #: native :meth:`Table.delta` DeltaFolder (when Delta-backed), ``None``
    #: → guess from active Spark + table size.
    use_warehouse: "bool | None" = None
