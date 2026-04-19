"""
Unit tests for :attr:`Table.tags` and :attr:`Table.column_tags` — the
lazy, TTL-cached entity-tag assignments that replaced the old
``TableAllInfos`` snapshot.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import (
    ColumnInfo,
    EntityTagAssignment,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
    TableConstraint,
    TableInfo,
    TableRowFilter,
)

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Table, Tables


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_client():
    client = MagicMock(spec=DatabricksClient)
    client.base_url.to_string.return_value = "https://adb-123.azuredatabricks.net"
    client.base_url.with_path.side_effect = lambda p: MagicMock(
        to_string=lambda: f"https://adb-123.azuredatabricks.net{p}"
    )
    return client


@pytest.fixture()
def mock_ws(mock_client):
    ws = MagicMock()
    mock_client.workspace_client.return_value = ws
    return ws


@pytest.fixture()
def table_infos():
    return TableInfo(
        catalog_name="main",
        schema_name="sales",
        name="orders",
        full_name="main.sales.orders",
        table_id="tbl-123",
        owner="alice@example.com",
        comment="Orders fact table",
        properties={"delta.enableChangeDataFeed": "true"},
        columns=[
            ColumnInfo(name="id", position=0),
            ColumnInfo(name="amount", position=1),
        ],
        row_filter=TableRowFilter(
            function_name="main.sec.row_filter",
            input_column_names=["id"],
        ),
    )


@pytest.fixture()
def tables(mock_client):
    return Tables(client=mock_client, catalog_name="main", schema_name="sales")


@pytest.fixture()
def primed_table(mock_client, tables, table_infos):
    """Table with a fresh basic ``_infos`` cache — no network access required."""
    tb = Table(
        service=tables,
        catalog_name="main",
        schema_name="sales",
        table_name="orders",
    )
    tb._store_infos(table_infos)
    mock_client.sql.return_value.tables = tables
    mock_client.tables = tables
    return tb


def _tag(entity_type: str, entity_name: str, key: str, value: str) -> EntityTagAssignment:
    return EntityTagAssignment(
        entity_type=entity_type,
        entity_name=entity_name,
        tag_key=key,
        tag_value=value,
    )


# ── Table.tags — lazy table-level entity tag assignments ──────────────────────


class TestTableTags:
    def test_tags_fetches_once_and_caches(self, primed_table, mock_ws):
        mock_ws.entity_tag_assignments.list.return_value = iter(
            [_tag("tables", "main.sales.orders", "pii", "true")]
        )

        first = primed_table.tags
        second = primed_table.tags

        assert first is second
        assert len(first) == 1
        assert first[0].tag_key == "pii"
        mock_ws.entity_tag_assignments.list.assert_called_once_with(
            entity_type="tables", entity_name="main.sales.orders",
        )

    def test_tags_swallow_api_errors_and_return_empty(self, primed_table, mock_ws):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("tags API off")

        assert primed_table.tags == ()

    def test_missing_tag_api_returns_empty(self, primed_table, mock_ws):
        del mock_ws.entity_tag_assignments

        assert primed_table.tags == ()

    def test_tags_refetch_after_expiry(self, primed_table, mock_ws):
        mock_ws.entity_tag_assignments.list.side_effect = [
            iter([_tag("tables", "main.sales.orders", "pii", "true")]),
            iter([_tag("tables", "main.sales.orders", "pii", "false")]),
        ]

        first = primed_table.tags
        assert first[0].tag_value == "true"

        # Age the tags cache past the TTL horizon.
        object.__setattr__(primed_table, "_tags_fetched_at", time.time() - 10_000)

        second = primed_table.tags
        assert second[0].tag_value == "false"
        assert mock_ws.entity_tag_assignments.list.call_count == 2


# ── Table.column_tags — lazy per-column entity tag assignments ────────────────


class TestTableColumnTags:
    def test_column_tags_collects_only_tagged_columns(self, primed_table, mock_ws):
        def fake_list(*, entity_type, entity_name, **_):
            if entity_type == "columns" and entity_name.endswith(".id"):
                return iter([_tag("columns", entity_name, "primary_key", "true")])
            return iter([])

        mock_ws.entity_tag_assignments.list.side_effect = fake_list

        result = primed_table.column_tags

        assert set(result) == {"id"}
        assert result["id"][0].tag_key == "primary_key"

    def test_column_tags_caches(self, primed_table, mock_ws):
        mock_ws.entity_tag_assignments.list.return_value = iter([])

        first = primed_table.column_tags
        second = primed_table.column_tags

        assert first is second
        # One call per column on the first access, none on the second.
        assert mock_ws.entity_tag_assignments.list.call_count == 2

    def test_column_tags_swallow_errors(self, primed_table, mock_ws):
        mock_ws.entity_tag_assignments.list.side_effect = RuntimeError("boom")

        assert primed_table.column_tags == {}


# ── Table.infos — basic TableInfo TTL cache ───────────────────────────────────


class TestTableInfosCache:
    def test_infos_expired_cache_triggers_refetch(self, primed_table, mock_ws):
        # Age the basic-infos cache past the TTL horizon.
        object.__setattr__(primed_table, "_infos_fetched_at", time.time() - 10_000)

        refreshed = TableInfo(
            catalog_name="main",
            schema_name="sales",
            name="orders",
            table_id="tbl-123",
            owner="carol@example.com",
            columns=[ColumnInfo(name="id", position=0)],
        )
        mock_ws.tables.get.return_value = refreshed

        result = primed_table.infos
        assert result is refreshed

    def test_reset_cache_clears_all_lazy_fields(
        self, primed_table, table_infos, mock_ws
    ):
        mock_ws.entity_tag_assignments.list.return_value = iter([])

        _ = primed_table.tags
        _ = primed_table.column_tags
        assert primed_table._tags is not None
        assert primed_table._column_tags is not None

        primed_table._reset_cache()

        assert primed_table._infos is None
        assert primed_table._columns is None
        assert primed_table._tags is None
        assert primed_table._tags_fetched_at is None
        assert primed_table._column_tags is None
        assert primed_table._column_tags_fetched_at is None


# ── Table.data_schema — enrichment with tags + constraints ────────────────────


class TestDataSchemaEnrichment:
    def test_data_schema_stamps_primary_key_and_foreign_key_and_tags(
        self, primed_table, mock_ws
    ):
        # Primary key on `id`, foreign key from `amount` to `finance.payments.id`.
        primed_table._infos.table_constraints = [
            TableConstraint(
                primary_key_constraint=PrimaryKeyConstraint(
                    name="orders_id_pk",
                    child_columns=["id"],
                ),
            ),
            TableConstraint(
                foreign_key_constraint=ForeignKeyConstraint(
                    name="orders_amount_fk",
                    child_columns=["amount"],
                    parent_table="main.finance.payments",
                    parent_columns=["id"],
                ),
            ),
        ]

        # Table + column tags returned by the workspace client.
        def _fake_list(*, entity_type, entity_name, **_):
            if entity_type == "tables":
                return iter([
                    _tag("tables", entity_name, "domain", "power"),
                ])
            if entity_type == "columns" and entity_name.endswith(".amount"):
                return iter([
                    _tag("columns", entity_name, "owner", "nika"),
                ])
            return iter([])

        mock_ws.entity_tag_assignments.list.side_effect = _fake_list

        schema = primed_table.data_schema

        id_field = schema["id"]
        amount_field = schema["amount"]

        id_tags = id_field.tags or {}
        amount_tags = amount_field.tags or {}

        assert id_tags.get(b"primary_key") == b"true"
        assert b"foreign_key" not in id_tags

        assert amount_tags.get(b"foreign_key") == b"main.finance.payments.id"
        assert amount_tags.get(b"owner") == b"nika"

        # Schema-level metadata picks up the table tag and constraint summary.
        assert schema.metadata is not None
        assert schema.metadata[b"primary_key"] == b"id"
        assert schema.metadata[b"foreign_key"] == b"amount=main.finance.payments.id"
        assert schema.metadata[b"tag:domain"] == b"power"
