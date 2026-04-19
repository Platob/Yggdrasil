"""Thin wrapper around the Databricks ``TableConstraintsAPI``.

Documented at
https://docs.databricks.com/api/gcp/workspace/tableconstraints. We use the
SDK's typed helpers so the two endpoints (``create`` / ``delete``) stay
cohesive with the rest of the workspace-client code and are easy to mock
under test.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
)

from .sql_utils import _parse_fk_ref, _safe_constraint_name
from .types import ForeignKeySpec, PrimaryKeySpec

if TYPE_CHECKING:
    from .table import Table

__all__ = [
    "apply_primary_key",
    "apply_foreign_key",
    "delete_constraint",
    "primary_key_constraint_name",
    "foreign_key_constraint_name",
]

logger = logging.getLogger(__name__)


def _table_constraints_api(table: "Table"):
    """Fetch the workspace-client ``table_constraints`` service.

    Kept defensive so older SDKs without the endpoint raise a clear error.
    """
    api = getattr(table.client.workspace_client(), "table_constraints", None)
    if api is None:
        raise RuntimeError(
            "databricks-sdk does not expose 'table_constraints'; upgrade the "
            "SDK to a version that ships the Unity Catalog constraints API."
        )
    return api


def primary_key_constraint_name(
    table: "Table",
    pk_spec: PrimaryKeySpec,
) -> str:
    """Stable default name for a primary-key constraint on *table*."""
    if pk_spec.constraint_name:
        return _safe_constraint_name(pk_spec.constraint_name)
    return _safe_constraint_name(
        f"{table.table_name}_{'_'.join(pk_spec.columns)}_pk"
    )


def foreign_key_constraint_name(
    table: "Table",
    fk_spec: ForeignKeySpec,
) -> str:
    """Stable default name for a foreign-key constraint on *table*."""
    if fk_spec.constraint_name:
        return _safe_constraint_name(fk_spec.constraint_name)

    ref_table, ref_cols = _parse_fk_ref(
        fk_spec.ref,
        default_catalog=table.catalog_name,
        default_schema=table.schema_name,
    )
    return _safe_constraint_name(
        table.table_name,
        fk_spec.column,
        ref_table,
        *ref_cols,
        "fk",
    )


def apply_primary_key(
    table: "Table",
    pk_spec: PrimaryKeySpec,
) -> TableConstraint:
    """Create a ``PRIMARY KEY`` constraint via the SDK."""
    constraint = PrimaryKeyConstraint(
        name=primary_key_constraint_name(table, pk_spec),
        child_columns=list(pk_spec.columns),
        rely=bool(pk_spec.rely) if pk_spec.rely is not None else None,
        timeseries_columns=[pk_spec.timeseries] if pk_spec.timeseries else None,
    )
    return _table_constraints_api(table).create(
        full_name_arg=table.full_name(),
        constraint=TableConstraint(primary_key_constraint=constraint),
    )


def apply_foreign_key(
    table: "Table",
    fk_spec: ForeignKeySpec,
) -> TableConstraint:
    """Create a ``FOREIGN KEY`` constraint via the SDK.

    ``match_full`` / ``on_update_no_action`` / ``on_delete_no_action`` from
    :class:`ForeignKeySpec` have no mapping in the REST API (Databricks only
    supports RELY on FKs), so they are dropped with a debug log. The spec
    is kept to preserve user-facing DDL ergonomics.
    """
    ref_table, ref_cols = _parse_fk_ref(
        fk_spec.ref,
        default_catalog=table.catalog_name,
        default_schema=table.schema_name,
    )
    if any(
        getattr(fk_spec, attr, False)
        for attr in ("match_full", "on_update_no_action", "on_delete_no_action")
    ):
        logger.debug(
            "ForeignKeySpec MATCH FULL / ON UPDATE / ON DELETE flags are not "
            "supported by the table_constraints API; ignored for %s.%s",
            table.full_name(), fk_spec.column,
        )
    constraint = ForeignKeyConstraint(
        name=foreign_key_constraint_name(table, fk_spec),
        child_columns=[fk_spec.column],
        parent_table=ref_table,
        parent_columns=list(ref_cols),
        rely=bool(fk_spec.rely) if fk_spec.rely is not None else None,
    )
    return _table_constraints_api(table).create(
        full_name_arg=table.full_name(),
        constraint=TableConstraint(foreign_key_constraint=constraint),
    )


def delete_constraint(
    table: "Table",
    constraint_name: str,
    *,
    cascade: bool = False,
    if_exists: bool = True,
) -> None:
    """Delete a constraint by name; swallow ``NotFound`` under ``if_exists``."""
    try:
        _table_constraints_api(table).delete(
            full_name=table.full_name(),
            constraint_name=constraint_name,
            cascade=bool(cascade),
        )
    except NotFound:
        if not if_exists:
            raise
