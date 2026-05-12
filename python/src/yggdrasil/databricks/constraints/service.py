"""``TableConstraints`` service — sits beside :class:`Tables`.

CRUD over Unity Catalog table constraints.

Read path uses ``INFORMATION_SCHEMA`` via :class:`SQLEngine` (no SDK
endpoint exists for listing). Write path uses the SDK's
``table_constraints`` API.

Field convention
----------------
Constraints are represented as :class:`~yggdrasil.data.data_field.Field`
instances:

- ``field.name`` is the constraint name.
- ``field.children`` are the participating columns
  (single-column constraints unwrap to ``[field]`` itself, per
  :meth:`Field.make_constraint_field`).
- ``field.constraint_key`` is ``True``.
- ``field.primary_key`` is ``True`` for PK Fields,
  ``field.foreign_key`` is ``True`` for FK Fields.
- For foreign keys, the parent reference lives in ``field.metadata``
  under the byte keys ``b"parent_table"`` (full name) and
  ``b"parent_columns"`` (comma-separated column list). Boolean tag
  flags can't carry payload, so this is the only place to put it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.catalog import (
    ForeignKeyConstraint,
    NamedTableConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
)

from yggdrasil.data.data_field import Field
from yggdrasil.databricks.client import DatabricksService

if TYPE_CHECKING:
    from yggdrasil.databricks.sql.table import Table

__all__ = ["TableConstraints"]

logger = logging.getLogger(__name__)

# Metadata keys for the FK parent reference on a constraint Field.
_PARENT_TABLE_KEY: bytes = b"parent_table"
_PARENT_COLUMNS_KEY: bytes = b"parent_columns"

# Single-shot query: every PK/FK column on the target table, plus the
# parent table+column for every FK row. CHECK constraints aren't in
# KEY_COLUMN_USAGE — they need a separate CHECK_CONSTRAINTS query and
# are out of scope for this listing.
#
# Sorting by constraint_name + ordinal_position lets the Python side
# group rows in O(n) and materialize columns in declared order.
_LIST_CONSTRAINTS_SQL = """
SELECT
    tc.constraint_name        AS constraint_name,
    tc.constraint_type        AS constraint_type,
    kcu.column_name           AS child_column,
    kcu.ordinal_position      AS child_ordinal,
    pk_kcu.table_catalog      AS parent_catalog,
    pk_kcu.table_schema       AS parent_schema,
    pk_kcu.table_name         AS parent_table,
    pk_kcu.column_name        AS parent_column,
    pk_kcu.ordinal_position   AS parent_ordinal
FROM {info_schema}.table_constraints AS tc
LEFT JOIN {info_schema}.key_column_usage AS kcu
       ON kcu.constraint_catalog = tc.constraint_catalog
      AND kcu.constraint_schema  = tc.constraint_schema
      AND kcu.constraint_name    = tc.constraint_name
LEFT JOIN {info_schema}.referential_constraints AS rc
       ON rc.constraint_catalog  = tc.constraint_catalog
      AND rc.constraint_schema   = tc.constraint_schema
      AND rc.constraint_name     = tc.constraint_name
LEFT JOIN {info_schema}.key_column_usage AS pk_kcu
       ON pk_kcu.constraint_catalog = rc.unique_constraint_catalog
      AND pk_kcu.constraint_schema  = rc.unique_constraint_schema
      AND pk_kcu.constraint_name    = rc.unique_constraint_name
      AND pk_kcu.ordinal_position   = kcu.position_in_unique_constraint
WHERE tc.table_catalog = '{catalog}'
  AND tc.table_schema  = '{schema}'
  AND tc.table_name    = '{table}'
ORDER BY tc.constraint_name, kcu.ordinal_position
"""


# ============================================================================
# TableConstraints service
# ============================================================================


class TableConstraints(DatabricksService):
    """CRUD over Unity Catalog table constraints, driven by :class:`Field`."""

    # ------------------------------------------------------------------
    # Read API — INFORMATION_SCHEMA via SQL
    # ------------------------------------------------------------------

    def list_constraints(self, resource: "Table | str") -> List[Field]:
        """Return PK/FK constraints registered on *resource* as :class:`Field`.

        Single SQL query against ``<catalog>.information_schema``;
        rows are grouped client-side into one Field per constraint.
        CHECK constraints are out of scope.
        """
        table = self.tables.get(resource)
        rows = self.client.sql.execute(
            _LIST_CONSTRAINTS_SQL.format(
                info_schema=f"`{table.catalog_name}`.information_schema",
                catalog=table.catalog_name,
                schema=table.schema_name,
                table=table.table_name,
            ),
        ).to_pylist()

        return self._rows_to_fields(rows)

    @staticmethod
    def _rows_to_fields(rows: list[dict[str, Any]]) -> List[Field]:
        """Group flat rows into one constraint :class:`Field` per name."""
        # Order is preserved by the SQL ORDER BY; insertion-order dicts
        # do the rest.
        by_name: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            name = row.get("constraint_name")
            if not name:
                continue
            by_name.setdefault(name, []).append(row)

        out: List[Field] = []
        for name, group in by_name.items():
            ctype = (group[0].get("constraint_type") or "").upper()
            child_columns = [
                r["child_column"] for r in group if r.get("child_column")
            ]
            if not child_columns:
                continue  # CHECK or malformed row; skip

            # Build the column Fields using object dtype — type info isn't
            # in the constraints query, and constraint Fields are
            # name-only carriers anyway (the dtype lives on the Table).
            col_fields = [Field(name=c, dtype="object") for c in child_columns]
            constraint = Field.make_constraint_field(col_fields, name=name)

            if ctype == "PRIMARY KEY":
                constraint = constraint.with_primary_key(True, inplace=True)
            elif ctype == "FOREIGN KEY":
                constraint = constraint.with_foreign_key(True, inplace=True)
                # Stamp parent ref into metadata — boolean flags can't
                # carry the parent table + columns, but we have the
                # data here from the rc → pk_kcu join.
                parent_full = _format_parent_table(group[0])
                parent_columns = [
                    r["parent_column"] for r in group if r.get("parent_column")
                ]
                if parent_full or parent_columns:
                    constraint = constraint.with_metadata(
                        metadata={
                            _PARENT_TABLE_KEY: parent_full or "",
                            _PARENT_COLUMNS_KEY: ",".join(parent_columns),
                        },
                        inplace=True,
                    )
            # else: leave as plain constraint_key Field (CHECK fallback)

            out.append(constraint)
        return out

    # ------------------------------------------------------------------
    # Write API — Databricks SDK
    # ------------------------------------------------------------------

    def create_constraint(
        self,
        resource: "Table | str",
        constraint: Field | TableConstraint | PrimaryKeyConstraint
                    | ForeignKeyConstraint | NamedTableConstraint,
    ) -> TableConstraint:
        """Create a constraint on a Unity Catalog table.

        ``constraint`` may be a :class:`Field` (preferred — see module
        docstring), a fully-formed :class:`TableConstraint`, or one of
        the SDK's per-kind constraint objects.
        """
        table = self.tables.get(resource)
        wrapped = self._as_table_constraint(constraint)

        self.client.workspace_client().table_constraints.create(
            full_name_arg=table.full_name(safe=False),
            constraint=wrapped,
        )
        return wrapped

    def delete_constraint(
        self,
        resource: "Table | str",
        constraint: Field | TableConstraint | str,
        *,
        cascade: bool = False,
    ) -> None:
        """Delete a constraint by name. Missing constraints are ignored."""
        table = self.tables.get(resource)
        name = self._constraint_name(constraint)
        if not name:
            raise ValueError(f"Cannot resolve constraint name from {constraint!r}")

        try:
            self.client.workspace_client().table_constraints.delete(
                full_name=table.full_name(safe=False),
                constraint_name=name,
                cascade=cascade,
            )
        except (ResourceDoesNotExist, NotFound):
            logger.debug(
                "Constraint %s on %s already absent",
                name, table.full_name(safe=False),
            )

    # ------------------------------------------------------------------
    # Field → SDK constraint translation
    # ------------------------------------------------------------------

    @staticmethod
    def _as_table_constraint(obj: object) -> TableConstraint:
        """Normalise any accepted input into a :class:`TableConstraint`."""
        if isinstance(obj, TableConstraint):
            return obj
        if isinstance(obj, PrimaryKeyConstraint):
            return TableConstraint(primary_key_constraint=obj)
        if isinstance(obj, ForeignKeyConstraint):
            return TableConstraint(foreign_key_constraint=obj)
        if isinstance(obj, NamedTableConstraint):
            return TableConstraint(named_table_constraint=obj)
        if isinstance(obj, Field):
            return TableConstraints._field_to_table_constraint(obj)
        raise TypeError(
            f"Cannot build TableConstraint from {type(obj).__name__}: {obj!r}"
        )

    @staticmethod
    def _field_to_table_constraint(field: Field) -> TableConstraint:
        """Map a constraint :class:`Field` to its SDK representation.

        Routing:

        - ``field.foreign_key`` → FK; parent ref read from
          ``field.metadata[b"parent_table"]`` /
          ``field.metadata[b"parent_columns"]``.
        - ``field.primary_key`` or just ``field.constraint_key`` → PK.
        - otherwise → :class:`NamedTableConstraint` (CHECK-style).
        """
        child_columns = [c.name for c in field.children] or [field.name]

        if field.foreign_key:
            parent_table, parent_columns = _read_parent_ref(field)
            if not parent_table:
                raise ValueError(
                    f"Foreign key {field.name!r} is missing parent_table metadata; "
                    "stamp it via field.with_metadata({"
                    "b'parent_table': ..., b'parent_columns': ...})"
                )
            return TableConstraint(
                foreign_key_constraint=ForeignKeyConstraint(
                    name=field.name,
                    child_columns=child_columns,
                    parent_table=parent_table,
                    parent_columns=parent_columns or child_columns,
                    rely=True,
                ),
            )

        if field.primary_key or field.constraint_key:
            return TableConstraint(
                primary_key_constraint=PrimaryKeyConstraint(
                    name=field.name,
                    child_columns=child_columns,
                    rely=True,
                ),
            )

        return TableConstraint(
            named_table_constraint=NamedTableConstraint(name=field.name),
        )

    # ------------------------------------------------------------------
    # Name extraction (for delete)
    # ------------------------------------------------------------------

    @staticmethod
    def _constraint_name(obj: object) -> Optional[str]:
        if isinstance(obj, str):
            return obj.strip() or None
        if isinstance(obj, Field):
            return obj.name
        if isinstance(obj, (PrimaryKeyConstraint, ForeignKeyConstraint, NamedTableConstraint)):
            return obj.name
        if isinstance(obj, TableConstraint):
            for sub in (obj.primary_key_constraint,
                        obj.foreign_key_constraint,
                        obj.named_table_constraint):
                if sub is not None and getattr(sub, "name", None):
                    return sub.name
        return None


# ============================================================================
# Module-level helpers
# ============================================================================


def _format_parent_table(row: dict[str, Any]) -> str:
    """Build ``catalog.schema.table`` from a row's parent_* fields."""
    parts = [row.get("parent_catalog"), row.get("parent_schema"), row.get("parent_table")]
    if not all(parts):
        return ""
    return ".".join(parts)


def _read_parent_ref(field: Field) -> tuple[str, list[str]]:
    """Decode the FK parent ref stamped under ``field.metadata``."""
    metadata = field.metadata or {}
    parent_raw = metadata.get(_PARENT_TABLE_KEY)
    cols_raw = metadata.get(_PARENT_COLUMNS_KEY)

    parent = (
        parent_raw.decode("utf-8") if isinstance(parent_raw, bytes)
        else (parent_raw or "")
    )
    cols_str = (
        cols_raw.decode("utf-8") if isinstance(cols_raw, bytes)
        else (cols_raw or "")
    )
    cols = [c.strip() for c in cols_str.split(",") if c.strip()] if cols_str else []
    return parent, cols