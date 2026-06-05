"""Operation metadata returned by INSERT / UPDATE / MERGE / DELETE plans.

The :class:`SelectPlan` side hands back a :class:`Tabular` (the read
result). The write-side plans (:class:`InsertPlan`,
:class:`MergePlan`, …) hand back an :class:`OperationResult` — a
small frozen record describing what changed: rows inserted, updated,
deleted, and the post-write target Tabular for downstream chaining.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.io.tabular import Tabular


@dataclasses.dataclass(slots=True, frozen=True)
class OperationResult:
    """Metadata for a write-side plan execution."""

    operation: str  # "INSERT", "UPDATE", "MERGE", "DELETE"
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_deleted: int = 0
    target: "Tabular | None" = None

    @property
    def rows_affected(self) -> int:
        return self.rows_inserted + self.rows_updated + self.rows_deleted

    def to_arrow_tabular(self) -> "Tabular":
        """Materialise this result as a single-row Arrow Tabular.

        Used by :class:`ExecutionPlan`'s :class:`Tabular` contract:
        ``write_plan.read_arrow_table()`` yields the metadata row
        instead of the post-write data.
        """
        from yggdrasil.arrow.tabular import ArrowTabular
        import pyarrow as pa
        return ArrowTabular(pa.table({
            "operation": [self.operation],
            "rows_inserted": [self.rows_inserted],
            "rows_updated": [self.rows_updated],
            "rows_deleted": [self.rows_deleted],
            "rows_affected": [self.rows_affected],
        }))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "rows_inserted": self.rows_inserted,
            "rows_updated": self.rows_updated,
            "rows_deleted": self.rows_deleted,
            "rows_affected": self.rows_affected,
        }
