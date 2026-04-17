from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from databricks.sdk.service.catalog import ColumnInfo as CatalogColumnInfo
from databricks.sdk.service.sql import ColumnInfo as SQLColumnInfo
from yggdrasil.data import Field
from yggdrasil.databricks.sql.sql_utils import (
    DEFAULT_TAG_COLLATION,
    _safe_constraint_name,
    databricks_tag_literal,
    quote_qualified_ident,
)

if TYPE_CHECKING:
    from .columns import Columns
    from .table import Table

__all__ = ["Column"]


@dataclass(frozen=True)
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
        f = Field.from_databricks(infos)
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
        if tags:
            query = self.set_tags_ddl(tags, tag_collation=tag_collation)
            if query:
                self.engine.execute(query)
        return self

    def add_primary_key_ddl(
        self,
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: bool = False,
    ) -> str:
        """
        Build DDL to add a single-column primary key constraint.

        Databricks PK/FK constraints are table constraints, even if this helper is
        exposed from a column object.
        """
        cname = self._stable_constraint_name(
            constraint_name,
            self.table.name,
            self.name,
            "pk",
        )
        rely_clause = " RELY" if rely else ""
        timeseries_clause = " TIMESERIES" if timeseries else ""

        return (
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"ADD CONSTRAINT `{cname}` "
            f"PRIMARY KEY ({self._qcol()}{timeseries_clause})"
            f"{rely_clause}"
        )

    def set_primary_key(
        self,
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: bool = False,
    ):
        query = self.add_primary_key_ddl(
            constraint_name=constraint_name,
            rely=rely,
            timeseries=timeseries,
        )
        self.engine.execute(query)
        return self

    def drop_primary_key_ddl(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""

        return (
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"DROP PRIMARY KEY{if_exists_clause}{cascade_clause}"
        )

    def unset_primary_key(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ):
        query = self.drop_primary_key_ddl(
            if_exists=if_exists,
            cascade=cascade,
        )
        self.engine.execute(query)
        return self

    def add_foreign_key_ddl(
        self,
        *,
        ref_table: "Table",
        ref_column: "str | Column",
        constraint_name: str | None = None,
        rely: bool = False,
        match_full: bool = False,
        on_update_no_action: bool = False,
        on_delete_no_action: bool = False,
    ) -> str:
        """
        Build DDL to add a single-column foreign key constraint.
        """
        ref_column_name = ref_column.name if isinstance(ref_column, Column) else ref_column
        cname = self._stable_constraint_name(
            constraint_name,
            self.table.name,
            self.name,
            ref_table.name,
            ref_column_name,
            "fk",
        )

        options: list[str] = []
        if rely:
            options.append("RELY")
        if match_full:
            options.append("MATCH FULL")
        if on_update_no_action:
            options.append("ON UPDATE NO ACTION")
        if on_delete_no_action:
            options.append("ON DELETE NO ACTION")

        options_sql = f" {' '.join(options)}" if options else ""

        return (
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"ADD CONSTRAINT `{cname}` "
            f"FOREIGN KEY ({self._qcol()}) "
            f"REFERENCES {quote_qualified_ident(ref_table.full_name())} (`{ref_column_name}`)"
            f"{options_sql}"
        )

    def _resolve_ref_args(
        self,
        column: "str | Column | None",
        ref_table: "str | Table | None",
        ref_column: "str | Column | None",
    ) -> tuple["Table", "str | Column"]:
        """Resolve *(ref_table, ref_column)* from the many accepted input forms.

        Resolution order
        ----------------
        1. ``column`` (positional shorthand)

           * ``Column`` object  → ``ref_table = column.table``,
             ``ref_column = column`` (unless already supplied).
           * ``str``            → parsed via
             :meth:`~yggdrasil.databricks.sql.columns.Columns.parse_location`
             using the current column's table as defaults; the table part
             is resolved via :meth:`~yggdrasil.databricks.sql.tables.Tables.find_table`.

        2. ``ref_table`` string → resolved via ``Tables.find_table``.

        3. ``ref_table`` is ``None`` → defaults to ``self.table``
           (self-referencing foreign key).
        """
        from .columns import Columns

        if column is not None:
            if isinstance(column, Column):
                if ref_table is None:
                    ref_table = column.table
                if ref_column is None:
                    ref_column = column

            elif isinstance(column, str):
                svc = Columns(
                    client=self.table.client,
                    catalog_name=self.table.catalog_name,
                    schema_name=self.table.schema_name,
                    table_name=self.table.table_name,
                )
                cat, sch, tbl, col_name = svc.parse_location(column)
                if ref_table is None:
                    ref_table = self.table.client.tables.find_table(
                        catalog_name=cat,
                        schema_name=sch,
                        table_name=tbl,
                    )
                if ref_column is None:
                    ref_column = col_name

        if isinstance(ref_table, str):
            ref_table = self.table.client.tables.find_table(location=ref_table)

        if ref_table is None:
            ref_table = self.table

        return ref_table, ref_column

    def set_foreign_key(
        self,
        column: "str | Column | None" = None,
        *,
        ref_table: "str | Table | None" = None,
        ref_column: "str | Column | None" = None,
        constraint_name: str | None = None,
        rely: bool = False,
        match_full: bool = False,
        on_update_no_action: bool = False,
        on_delete_no_action: bool = False,
    ):
        ref_table, ref_column = self._resolve_ref_args(column, ref_table, ref_column)

        query = self.add_foreign_key_ddl(
            ref_table=ref_table,
            ref_column=ref_column,
            constraint_name=constraint_name,
            rely=rely,
            match_full=match_full,
            on_update_no_action=on_update_no_action,
            on_delete_no_action=on_delete_no_action,
        )
        self.engine.execute(query)
        return self

    def drop_foreign_key_ddl(
        self,
        *,
        if_exists: bool = True,
    ) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""

        return (
            f"ALTER TABLE {self.table.full_name(safe=True)} "
            f"DROP FOREIGN KEY{if_exists_clause} ({self._qcol()})"
        )

    def unset_foreign_key(
        self,
        *,
        if_exists: bool = True,
    ):
        query = self.drop_foreign_key_ddl(if_exists=if_exists)
        self.engine.execute(query)
        return self