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

    - :attr:`prefer_sql` — ``True`` (default) routes reads and writes
      through the SQL warehouse. ``False`` prefers yggdrasil's **native
      DeltaFolder** read/write (a direct ``_delta_log`` + parquet path over
      UC-vended credentials) whenever the table is Delta-backed, bypassing
      the warehouse. Writes fall back to SQL for managed Delta tables
      (UC only vends read-only credentials there), so the native write path
      engages for external Delta tables.
    """

    #: Prefer the SQL warehouse for read/write. When ``False`` and the table
    #: is Delta-backed, prefer the native :meth:`Table.delta` DeltaFolder.
    prefer_sql: bool = True
