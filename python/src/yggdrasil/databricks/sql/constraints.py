"""``TableConstraints`` service — sits beside :class:`Tables` in
``yggdrasil/databricks/sql/tables.py``.

CRUD over Unity Catalog table constraints via the Databricks SDK.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Optional, Sequence

from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.errors.platform import BadRequest, NotFound
from databricks.sdk.service.catalog import (
    ForeignKeyConstraint,
    NamedTableConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
)

import yggdrasil.pickle.json as json_module
from yggdrasil.databricks.client import DatabricksService
from yggdrasil.databricks.sql.sql_utils import quote_ident
from . import Column
from ...data.data_utils import safe_constraint_name

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.schema import Schema
    from .table import Table

__all__ = ["TableConstraints"]

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Metadata probe keys for FK parent lookup
# ============================================================================

_FK_PARENT_KEYS = (b"foreign_key_parent", b"references")
_FK_PARENT_TABLE_KEY = b"parent_table"
_FK_PARENT_COLUMNS_KEY = b"parent_columns"


# ============================================================================
# TableConstraints service
# ============================================================================


@dataclass
class TableConstraints(DatabricksService):
    """CRUD over Unity Catalog table constraints.

    Accepts user-friendly inputs (Column, Field, str, SDK objects) and
    coerces them into the SDK's TableConstraint shape. For FK parent
    info that boolean tag flags don't carry, probes Field.metadata
    first, then falls back to catalog lookup.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_table_constraints(
        self,
        resource: Table | str | None = None,
        *,
        full_name: str | None = None,
        table: Table | None = None,
        constraint: TableConstraint | None = None,
        foreign_key_constraint: Optional[ForeignKeyConstraint | str | Field | Column] = None,
        named_table_constraint: Optional[NamedTableConstraint | str | Field] = None,
        primary_key_constraint: Optional[PrimaryKeyConstraint | str | Field | Column | Sequence[str]] = None,
    ) -> TableConstraint:
        """Create a constraint on a Unity Catalog table.

        Exactly one of ``constraint``, ``foreign_key_constraint``,
        ``named_table_constraint``, or ``primary_key_constraint`` should
        be supplied. If a pre-built ``constraint`` is given it's used
        as-is; otherwise the helper-coerced kind takes precedence.
        """
        table = self.tables.get(table if resource is None else resource)

        if not full_name:
            full_name = table.full_name(safe=False)

        client = self.client.workspace_client().table_constraints

        if constraint is None:
            fk = self._check_foreign_key(foreign_key_constraint, table) if foreign_key_constraint is not None else None
            pk = self._check_primary_key(primary_key_constraint, table) if primary_key_constraint is not None else None
            named = (
                self._check_named_table_constraint(named_table_constraint)
                if named_table_constraint is not None
                else None
            )

            specified = [c for c in (fk, pk, named) if c is not None]
            if len(specified) == 0:
                raise ValueError(
                    "create_table_constraints requires one of: constraint, "
                    "foreign_key_constraint, named_table_constraint, primary_key_constraint"
                )
            if len(specified) > 1:
                raise ValueError(
                    "create_table_constraints accepts only one constraint kind per call; "
                    f"got {len(specified)}"
                )

            constraint = TableConstraint(
                foreign_key_constraint=fk,
                named_table_constraint=named,
                primary_key_constraint=pk,
            )

        client.create(full_name_arg=full_name, constraint=constraint)

        return constraint

    def delete_table_constraint(
        self,
        resource: Table | str | None = None,
        *,
        constraint_name: str | None = None,
        constraint: TableConstraint | str | Field | None = None,
        full_name: str | None = None,
        cascade: bool = False,
    ) -> None:
        """Delete a constraint by name from a Unity Catalog table."""
        table = self.tables.get(table if (table := None) is None and resource is None else resource) \
            if resource is not None else self.tables.get(resource)

        # Simpler: just resolve once
        table = self.tables.get(resource) if resource is not None else None
        if table is None:
            raise ValueError("delete_table_constraint requires `resource` or `table`")

        if not full_name:
            full_name = table.full_name(safe=False)

        if constraint_name is None:
            constraint_name = self._extract_constraint_name(constraint)
        if not constraint_name:
            raise ValueError("delete_table_constraint requires `constraint_name` or a named `constraint`")

        client = self.client.workspace_client().table_constraints
        try:
            client.delete(full_name=full_name, constraint_name=constraint_name, cascade=cascade)
        except (ResourceDoesNotExist, NotFound):
            LOGGER.debug("Constraint %s on %s already absent", constraint_name, full_name)

    def list_table_constraints(
        self,
        resource: Table | str,
    ) -> List[TableConstraint]:
        """Return constraints currently registered on the table.

        Reads them off the cached :class:`TableInfo` since the SDK
        embeds constraints in the table description rather than
        exposing a dedicated list endpoint.
        """
        table = self.tables.get(resource)
        info = table.info() if hasattr(table, "info") else None
        constraints = getattr(info, "table_constraints", None) if info is not None else None
        return list(constraints) if constraints else []

    # ------------------------------------------------------------------
    # Foreign key coercion
    # ------------------------------------------------------------------

    def _check_foreign_key(
        self,
        obj: Any,
        table: Table | str | None = None,
    ) -> ForeignKeyConstraint:
        if isinstance(obj, ForeignKeyConstraint):
            return obj

        if isinstance(obj, Mapping):
            return ForeignKeyConstraint.from_dict(dict(obj))

        # Resolve binding table (the *child* — the table the constraint
        # is being installed on). Used when obj doesn't carry it.
        bound_table = self.tables.get(table, default=None) if table is not None else None

        if isinstance(obj, Column):
            child_table = obj.table or bound_table
            if child_table is None:
                raise ValueError("Column foreign key requires a bound table")
            parent_table, parent_columns = self._resolve_fk_parent(obj, child_table)
            return ForeignKeyConstraint(
                name=safe_constraint_name(obj.name, prefix="fk_"),
                child_columns=[obj.name],
                parent_table=parent_table,
                parent_columns=list(parent_columns),
                rely=True,
            )

        # Field path — child columns come from the field's
        # constraint-key children if it's a struct, else from the field
        # itself. Parent info from metadata or catalog lookup.
        from yggdrasil.data.data_field import Field  # local: avoid circular

        if isinstance(obj, Field):
            if bound_table is None:
                raise ValueError("Field foreign key requires `table` to bind the constraint to")
            child_columns = self._field_child_columns(obj)
            parent_table, parent_columns = self._resolve_fk_parent(obj, bound_table, child_columns=child_columns)
            return ForeignKeyConstraint(
                name=safe_constraint_name(obj.name, prefix="fk_"),
                child_columns=child_columns,
                parent_table=parent_table,
                parent_columns=list(parent_columns),
                rely=True,
            )

        if isinstance(obj, str):
            parts = obj.split(".")
            if len(parts) != 3:
                raise ValueError(
                    f"Foreign key string must be 'catalog.schema.column' (3 parts); got {obj!r}"
                )
            catalog, schema, column = parts
            return ForeignKeyConstraint(
                name=safe_constraint_name(column, prefix="fk_"),
                child_columns=[column],
                parent_table=f"{catalog}.{schema}.{column}",
                parent_columns=[column],
                rely=True,
            )

        raise TypeError(f"Unsupported foreign key spec: {type(obj).__name__}: {obj!r}")

    # ------------------------------------------------------------------
    # Primary key coercion
    # ------------------------------------------------------------------

    def _check_primary_key(
        self,
        obj: Any,
        table: Table | str | None = None,
    ) -> PrimaryKeyConstraint:
        if isinstance(obj, PrimaryKeyConstraint):
            return obj

        if isinstance(obj, Mapping):
            return PrimaryKeyConstraint.from_dict(dict(obj))

        from yggdrasil.data.data_field import Field  # local

        if isinstance(obj, Column):
            return PrimaryKeyConstraint(
                name=safe_constraint_name(obj.name, prefix="pk_"),
                child_columns=[obj.name],
                rely=True,
            )

        if isinstance(obj, Field):
            child_columns = self._field_child_columns(obj)
            return PrimaryKeyConstraint(
                name=safe_constraint_name(obj.name, prefix="pk_"),
                child_columns=child_columns,
                rely=True,
            )

        if isinstance(obj, str):
            return PrimaryKeyConstraint(
                name=safe_constraint_name(obj, prefix="pk_"),
                child_columns=[obj],
                rely=True,
            )

        if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
            cols = [self._column_name(c) for c in obj]
            if not cols:
                raise ValueError("Primary key sequence must be non-empty")
            return PrimaryKeyConstraint(
                name=safe_constraint_name(cols, prefix="pk_"),
                child_columns=cols,
                rely=True,
            )

        raise TypeError(f"Unsupported primary key spec: {type(obj).__name__}: {obj!r}")

    # ------------------------------------------------------------------
    # Named (CHECK-style) constraint coercion
    # ------------------------------------------------------------------

    def _check_named_table_constraint(self, obj: Any) -> NamedTableConstraint:
        if isinstance(obj, NamedTableConstraint):
            return obj

        if isinstance(obj, Mapping):
            return NamedTableConstraint.from_dict(dict(obj))

        if isinstance(obj, str):
            return NamedTableConstraint(name=obj)

        from yggdrasil.data.data_field import Field  # local
        if isinstance(obj, Field):
            return NamedTableConstraint(name=obj.name)

        raise TypeError(f"Unsupported named constraint spec: {type(obj).__name__}: {obj!r}")

    # ------------------------------------------------------------------
    # Field / column helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _column_name(c: Any) -> str:
        if isinstance(c, str):
            return c
        if isinstance(c, Column):
            return c.name
        from yggdrasil.data.data_field import Field
        if isinstance(c, Field):
            return c.name
        raise TypeError(f"Cannot extract column name from {type(c).__name__}: {c!r}")

    @staticmethod
    def _field_child_columns(field: Field) -> List[str]:
        """Extract participating column names from a constraint field.

        Per :meth:`Field.make_constraint_field`, a multi-column
        constraint is wrapped in a struct whose children are the
        participating fields. Single-column constraints are unwrapped.
        """
        from yggdrasil.data.types import StructType  # local

        dtype = getattr(field, "dtype", None) or getattr(field, "data_type", None)

        if isinstance(dtype, StructType):
            children = list(dtype.fields)
            if not children:
                raise ValueError(f"Constraint field {field.name!r} has no children")
            return [c.name for c in children]

        return [field.name]

    # ------------------------------------------------------------------
    # FK parent resolution: metadata → catalog lookup
    # ------------------------------------------------------------------

    def _resolve_fk_parent(
        self,
        source: Field | Column,
        child_table: Table,
        *,
        child_columns: Optional[Sequence[str]] = None,
    ) -> tuple[str, List[str]]:
        """Return ``(parent_table_full_name, parent_columns)``.

        Probes :attr:`Field.metadata` first (priority order documented
        in the module docstring), then falls back to listing existing
        constraints on ``child_table`` and matching by name.
        """
        metadata = getattr(source, "metadata", None) or {}

        # Priority 1 & 2: encoded FK parent under canonical keys
        for key in _FK_PARENT_KEYS:
            raw = metadata.get(key)
            if raw is None:
                continue
            parsed = self._parse_fk_parent_blob(raw)
            if parsed is not None:
                return parsed

        # Priority 3: split keys
        parent_table_raw = metadata.get(_FK_PARENT_TABLE_KEY)
        parent_columns_raw = metadata.get(_FK_PARENT_COLUMNS_KEY)
        if parent_table_raw is not None:
            parent_table = self._decode_str(parent_table_raw)
            parent_columns = self._decode_columns(parent_columns_raw) if parent_columns_raw is not None else None
            if parent_columns:
                return parent_table, list(parent_columns)
            # Fall through to catalog lookup for columns

        # Catalog-lookup fallback: re-installing a known constraint
        catalog_match = self._lookup_existing_fk(source, child_table, child_columns=child_columns)
        if catalog_match is not None:
            return catalog_match

        raise ValueError(
            f"Cannot resolve FK parent for {source.name!r}: no metadata under "
            f"{[k.decode() for k in _FK_PARENT_KEYS]} or {_FK_PARENT_TABLE_KEY.decode()}, "
            "and no existing constraint with that name on the table"
        )

    def _parse_fk_parent_blob(self, raw: Any) -> Optional[tuple[str, List[str]]]:
        """Try to parse a metadata blob holding FK parent info.

        Accepts:
        - a JSON-encoded :class:`Field` (via ``Field.from_json``)
        - a JSON object with ``parent_table`` / ``parent_columns``
        - a plain ``"catalog.schema.table"`` string
        """
        text = self._decode_str(raw)
        if not text:
            return None

        # Plain dotted string
        if "." in text and not text.lstrip().startswith(("{", "[", '"')):
            parts = text.split(".")
            if len(parts) == 3:
                return text, []  # columns deferred to catalog lookup

        # JSON path
        try:
            payload = json_module.loads(text)
        except Exception:
            return None

        if isinstance(payload, Mapping):
            if "parent_table" in payload:
                parent_table = str(payload["parent_table"])
                cols = payload.get("parent_columns") or []
                return parent_table, [str(c) for c in cols]

            # Maybe a serialized Field
            from yggdrasil.data.data_field import Field
            try:
                field = Field.from_json(text)
            except Exception:
                return None
            parent_table = field.metadata.get(_FK_PARENT_TABLE_KEY) if field.metadata else None
            if parent_table is None:
                return None
            return self._decode_str(parent_table), self._field_child_columns(field)

        return None

    def _lookup_existing_fk(
        self,
        source: Field | Column,
        child_table: Table,
        *,
        child_columns: Optional[Sequence[str]] = None,
    ) -> Optional[tuple[str, List[str]]]:
        """Find an existing FK on ``child_table`` whose name matches
        ``source.name``, and return its parent table + columns."""
        existing = self.list_table_constraints(child_table)
        for tc in existing:
            fk = tc.foreign_key_constraint
            if fk is None:
                continue
            if fk.name == source.name:
                return fk.parent_table, list(fk.parent_columns or [])
            # Also match by child-column overlap as a secondary signal
            if child_columns and fk.child_columns and set(fk.child_columns) == set(child_columns):
                return fk.parent_table, list(fk.parent_columns or [])
        return None

    # ------------------------------------------------------------------
    # Metadata decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_str(raw: Any) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)

    def _decode_columns(self, raw: Any) -> List[str]:
        text = self._decode_str(raw)
        try:
            payload = json_module.loads(text)
        except Exception:
            # Comma-separated fallback
            return [c.strip() for c in text.split(",") if c.strip()]
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, Sequence):
            return [str(c) for c in payload]
        raise ValueError(f"Cannot decode parent columns from {text!r}")

    # ------------------------------------------------------------------
    # Constraint name extraction (for delete)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_constraint_name(c: Any) -> Optional[str]:
        if c is None:
            return None
        if isinstance(c, str):
            return c
        NamedTableConstraint
        if isinstance(c, TableConstraint):
            for sub in (c.primary_key_constraint, c.foreign_key_constraint, c.named_table_constraint):
                if sub is not None and getattr(sub, "name", None):
                    return sub.name
            return None
        if isinstance(c, (PrimaryKeyConstraint, ForeignKeyConstraint, NamedTableConstraint)):
            return c.name
        from yggdrasil.data.data_field import Field
        if isinstance(c, Field):
            return c.name
        return None