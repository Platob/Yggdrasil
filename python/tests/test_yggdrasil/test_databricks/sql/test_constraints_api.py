"""Unit tests for the Unity Catalog ``table_constraints`` API wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
)

from yggdrasil.databricks.sql.constraints_api import (
    apply_foreign_key,
    apply_primary_key,
    delete_constraint,
    foreign_key_constraint_name,
    primary_key_constraint_name,
)
from yggdrasil.databricks.sql.types import ForeignKeySpec, PrimaryKeySpec


class _TableConstraintsAPIStub:
    def __init__(self, *, delete_raises: Exception | None = None) -> None:
        self.created: list[tuple[str, TableConstraint]] = []
        self.deleted: list[tuple[str, str, bool]] = []
        self._delete_raises = delete_raises

    def create(self, *, full_name_arg, constraint):
        self.created.append((full_name_arg, constraint))
        return constraint

    def delete(self, *, full_name, constraint_name, cascade):
        self.deleted.append((full_name, constraint_name, cascade))
        if self._delete_raises is not None:
            raise self._delete_raises


def _make_table(api: _TableConstraintsAPIStub):
    workspace = SimpleNamespace(table_constraints=api)
    return SimpleNamespace(
        catalog_name="main",
        schema_name="analytics",
        table_name="trades",
        client=SimpleNamespace(workspace_client=lambda: workspace),
        full_name=lambda safe=False: "main.analytics.trades",
    )


def test_apply_primary_key_calls_create_with_mapped_fields():
    api = _TableConstraintsAPIStub()
    table = _make_table(api)

    spec = PrimaryKeySpec(
        columns=["trade_id", "trade_date"],
        rely=True,
        timeseries="trade_date",
    )

    result = apply_primary_key(table, spec)

    assert len(api.created) == 1
    full_name, constraint = api.created[0]
    assert full_name == "main.analytics.trades"
    pk = constraint.primary_key_constraint
    assert pk is not None
    assert pk.child_columns == ["trade_id", "trade_date"]
    assert pk.rely is True
    assert pk.timeseries_columns == ["trade_date"]
    assert pk.name == primary_key_constraint_name(table, spec)
    assert result.primary_key_constraint.name == pk.name


def test_apply_foreign_key_calls_create_with_mapped_fields():
    api = _TableConstraintsAPIStub()
    table = _make_table(api)

    spec = ForeignKeySpec(
        column="book_id",
        ref="main.refined.books.id",
        rely=True,
    )

    apply_foreign_key(table, spec)

    assert len(api.created) == 1
    full_name, constraint = api.created[0]
    assert full_name == "main.analytics.trades"
    fk = constraint.foreign_key_constraint
    assert fk is not None
    assert fk.child_columns == ["book_id"]
    assert fk.parent_table == "main.refined.books"
    assert fk.parent_columns == ["id"]
    assert fk.rely is True
    assert fk.name == foreign_key_constraint_name(table, spec)


def test_apply_foreign_key_resolves_partial_refs_against_table_defaults():
    api = _TableConstraintsAPIStub()
    table = _make_table(api)

    apply_foreign_key(
        table,
        ForeignKeySpec(column="book_id", ref="books.id"),
    )

    _, constraint = api.created[0]
    fk = constraint.foreign_key_constraint
    assert fk.parent_table == "main.analytics.books"
    assert fk.parent_columns == ["id"]


def test_delete_constraint_swallows_not_found_when_if_exists():
    api = _TableConstraintsAPIStub(delete_raises=NotFound("missing"))
    table = _make_table(api)

    delete_constraint(table, "missing_fk", cascade=False, if_exists=True)

    assert api.deleted == [("main.analytics.trades", "missing_fk", False)]


def test_delete_constraint_propagates_not_found_when_required():
    api = _TableConstraintsAPIStub(delete_raises=NotFound("missing"))
    table = _make_table(api)

    with pytest.raises(NotFound):
        delete_constraint(table, "missing_fk", if_exists=False)


def test_missing_table_constraints_api_raises_clearly():
    table = SimpleNamespace(
        catalog_name="main",
        schema_name="analytics",
        table_name="trades",
        client=SimpleNamespace(
            workspace_client=lambda: SimpleNamespace(),
        ),
        full_name=lambda safe=False: "main.analytics.trades",
    )

    with pytest.raises(RuntimeError, match="table_constraints"):
        apply_primary_key(table, PrimaryKeySpec(columns=["id"]))
