"""``TableOptions`` — :class:`CastOptions` for the :class:`Table` TabularIO surface.

Carries everything :class:`Table` needs on its read / write paths that
plain :class:`CastOptions` doesn't already cover.  Inherits the cast,
schema, mode, where, wait, batching, and Spark-session knobs from
:class:`CastOptions`; adds Delta-/UC-flavoured DML knobs that the
``insert_into`` / ``arrow_insert`` / ``spark_insert`` / ``sql_insert``
paths read from.

The fields here exist only because they show up in the public insert
signatures and have no natural home in :class:`CastOptions` (they're
warehouse / Delta semantics, not generic frame-cast policy):

* ``update_cols`` — explicit MATCHED-UPDATE column list for upserts.
  ``None`` means "every non-key column".
* ``prune_by`` / ``prune_values`` — partition-prune key columns and
  pre-collected values.  Cuts the merge ``ON`` predicate to the
  partitions actually touched, which is the difference between
  rewriting one partition and rewriting the table.
* ``zorder_by`` — columns to ``OPTIMIZE … ZORDER BY`` after the write.
* ``optimize_after_merge`` — issue a plain ``OPTIMIZE`` after a merge
  to fold the small files merge produces.
* ``vacuum_hours`` — retention window for the trailing ``VACUUM``;
  ``None`` skips vacuum entirely.
* ``overwrite_schema`` — Delta ``schema.autoMerge.enabled`` for
  Spark-path writes.
* ``spark_options`` — extra Spark session-conf overrides applied for
  the duration of the write.
* ``retry`` — :class:`WaitingConfig` arg installed on each DML
  statement; ignored on the Spark path (driver retries those).

:meth:`TableOptions.check` reuses :meth:`CastOptions.check` and just
threads through the extra fields.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from yggdrasil.data.options import CastOptions
from yggdrasil.dataclasses.waiting import WaitingConfigArg


__all__ = ["TableOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class TableOptions(CastOptions):
    """:class:`CastOptions` extended with :class:`Table`-specific knobs.

    Frozen + slotted (inherited from :class:`CastOptions`) — share one
    instance across threads / tasks; mutate via :meth:`copy` /
    :meth:`replace`.

    All fields default to safe no-ops:

    * ``update_cols = None`` — derive non-key columns from the schema.
    * ``prune_by`` / ``prune_values = None`` — no partition pruning.
    * ``zorder_by`` / ``optimize_after_merge = False`` — no trailing
      OPTIMIZE.
    * ``vacuum_hours = None`` — no trailing VACUUM.
    * ``overwrite_schema = None`` — leave schema-autoMerge at the
      session default.
    * ``spark_options = None`` — no extra Spark-conf overrides.
    * ``retry = None`` — single-shot DML; warehouse default retry off.
    """

    # --- Upsert / merge shape -------------------------------------------
    update_cols: Optional[list[str]] = None

    # --- Partition pruning ----------------------------------------------
    # ``prune_by`` accepts the literal string ``"auto"`` to mean
    # "use the partition columns from the target schema" — Table.*_insert
    # resolves that into a real column list.
    prune_by: "list[str] | str | None" = None
    prune_values: Optional[Mapping[str, tuple[Any, ...]]] = None

    # --- Trailing maintenance -------------------------------------------
    zorder_by: Optional[list[str]] = None
    optimize_after_merge: bool = False
    vacuum_hours: Optional[int] = None

    # --- Engine knobs ----------------------------------------------------
    overwrite_schema: Optional[bool] = None
    spark_options: Optional[dict[str, Any]] = None

    # --- Statement-level retry ------------------------------------------
    # Threaded onto each DML WarehousePreparedStatement on the warehouse
    # path; ignored on Spark (driver-side retry handles it there).
    retry: Optional[WaitingConfigArg] = None
