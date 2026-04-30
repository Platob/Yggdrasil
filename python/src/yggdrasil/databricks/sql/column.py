from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping

from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo

from yggdrasil.data import Field
from yggdrasil.databricks.sql.sql_utils import (
    DEFAULT_TAG_COLLATION,
    databricks_tag_literal,
)
from .types import parse_databricks_field

if TYPE_CHECKING:
    from .table import Table

__all__ = ["Column"]


@dataclass
class Column:
    table: "Table"
    name: str
    field: Field = field(repr=False, compare=False, hash=False)

    @classmethod
    def from_api(
        cls,
        table: "Table",
        infos: SQLColumnInfo | CatalogColumnInfo
    ):
        f = parse_databricks_field(infos)
        metadata = {
            b"engine": b"databricks",
            b"catalog_name": table.catalog_name.encode(),
            b"schema_name": table.schema_name.encode(),
            b"table_name": table.table_name.encode(),
        }

        if not f.metadata:
            f.with_metadata(metadata)
        else:
            f.metadata.update(metadata)

        return cls(
            table=table,
            name=f.name,
            field=f,
        )

    @property
    def engine(self):
        return self.table.sql

    @property
    def metadata(self) -> Mapping[bytes, bytes]:
        return self.field.metadata or {}

    def _qcol(self) -> str:
        return f"`{self.name}`"

    def _stable_constraint_name(
        self,
        constraint_name: str | None,
        *parts: object,
    ) -> str:
        """Build a stable constraint name from structured parts."""
        if constraint_name:
            return _safe_constraint_name(constraint_name)
        return _safe_constraint_name(*parts)

    def set_tags_ddl(
        self,
        tags: Mapping[str, str],
        *,
        tag_collation: str | None = None,
    ):
        str_tags = ", ".join(
            f"{databricks_tag_literal(k, collation=tag_collation)} = "
            f"{databricks_tag_literal(v, collation=tag_collation)}"
            for k, v in tags.items() if k and v
        )

        if not str_tags:
            return None

        return (
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"ALTER COLUMN {self._qcol()} SET TAGS ({str_tags})"
        )

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ):
        """Apply column-level tags via the UC ``entity_tag_assignments`` API.

        ``tag_collation`` is accepted for API compatibility and ignored —
        collations only matter for the legacy DDL literal form.
        """
        del tag_collation
        if not tags:
            return self

        from .tags_api import apply_tags

        apply_tags(
            self.table.client,
            entity_type="columns",
            entity_name=f"{self.table.full_name()}.{self.name}",
            tags=tags,
        )
        # Invalidate the table-side column-tag cache so the next read sees
        # the new assignments.
        object.__setattr__(self.table, "_column_tags", None)
        object.__setattr__(self.table, "_column_tags_fetched_at", None)
        return self

    def unset_tags(
        self,
        tag_keys: "Iterable[str]",
        *,
        if_exists: bool = True,
    ):
        """Delete column-level tag assignments by key."""
        from .tags_api import delete_tags

        delete_tags(
            self.table.client,
            entity_type="columns",
            entity_name=f"{self.table.full_name()}.{self.name}",
            tag_keys=tag_keys,
            if_exists=if_exists,
        )
        object.__setattr__(self.table, "_column_tags", None)
        object.__setattr__(self.table, "_column_tags_fetched_at", None)
        return self

    def rename(self, new_name: str) -> "Column":
        """Rename this column in-place (``ALTER TABLE … RENAME COLUMN …``)."""
        new_name = (new_name or "").strip().strip("`")
        if not new_name:
            raise ValueError("Cannot rename column to an empty name")
        if new_name == self.name:
            return self

        self.engine.execute(
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"RENAME COLUMN {self._qcol()} TO `{new_name}`"
        )
        # Frozen dataclass — mutate via object.__setattr__.
        object.__setattr__(self, "name", new_name)
        # Invalidate the parent table's column cache so the next lookup refetches.
        if hasattr(self.table, "_reset_cache"):
            self.table._reset_cache(invalidate_cache=True)
        return self