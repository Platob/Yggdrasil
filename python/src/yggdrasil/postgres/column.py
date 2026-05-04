"""Per-column metadata: a thin dataclass tied to its parent :class:`Table`.

A :class:`Column` is a value object — it doesn't carry a connection
of its own and never queries Postgres directly. The parent
:class:`Table` populates it from ``information_schema.columns`` (or
from a CREATE TABLE round-trip), and DDL helpers (rename, set type,
drop) bounce back through the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pyarrow as pa

from .types import postgres_to_arrow_type

__all__ = ["Column"]


@dataclass(slots=True)
class Column:
    """A single Postgres column.

    Mirrors the rows returned by ``information_schema.columns`` —
    name, declared Postgres type, nullability, default expression,
    and the ordinal position. The :attr:`arrow_type` is derived
    lazily from :attr:`data_type` via :func:`postgres_to_arrow_type`,
    so callers that only need the SQL-level shape don't pay the
    parse cost.
    """

    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    ordinal_position: Optional[int] = None
    comment: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def arrow_type(self) -> pa.DataType:
        """Map :attr:`data_type` to a :class:`pyarrow.DataType`."""
        return postgres_to_arrow_type(self.data_type)

    def to_arrow_field(self) -> pa.Field:
        """Render as a :class:`pyarrow.Field` (preserving nullability)."""
        return pa.field(self.name, self.arrow_type, nullable=self.nullable)

    def __str__(self) -> str:
        nullable = "NULL" if self.nullable else "NOT NULL"
        return f"{self.name} {self.data_type} {nullable}"
