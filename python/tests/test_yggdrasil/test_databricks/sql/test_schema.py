"""
Tests for :class:`~yggdrasil.databricks.sql.schema.Schema`.

Structure
---------
Unit tests (no live workspace)
    ``mock_ws`` fixture stubs ``workspace_client()`` so lifecycle methods
    are exercised without any network call.

Integration tests (``requires_databricks``)
    :class:`TestSchemaIntegration` inherits :class:`DatabricksCase` and
    is skipped automatically when ``DATABRICKS_HOST`` is not set.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound, DatabricksError
from databricks.sdk.service.catalog import SchemaInfo, TableInfo

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Catalog, Catalogs, Schema, Table, Tables

from ..conftest import DatabricksCase, requires_databricks

# ── helpers ───────────────────────────────────────────────────────────────────


def _sch_info(catalog: str = "main", name: str = "sales", **kw) -> SchemaInfo:
    return SchemaInfo(
        catalog_name=catalog, name=name,
        comment=kw.get("comment"), owner=kw.get("owner"),
    )


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
    """Stubbed WorkspaceClient returned by ``mock_client.workspace_client()``."""
    ws = MagicMock()
    mock_client.workspace_client.return_value = ws
    return ws


@pytest.fixture()
def cats(mock_client):
    return Catalogs(client=mock_client)


@pytest.fixture()
def sch(cats):
    return cats.schema("main.sales")


@pytest.fixture()
def mock_tables(mock_client):
    svc = MagicMock(spec=Tables)
    svc.table.return_value = MagicMock(spec=Table)
    svc.list_tables.return_value = iter([])
    mock_client.tables = svc
    return svc


# ===========================================================================
# Unit — Schema identity & navigation
# ===========================================================================


class TestSchemaIdentity:
    def test_full_name_unquoted(self, sch):
        assert sch.full_name() == "main.sales"

    def test_full_name_safe_backtick(self, sch):
        assert sch.full_name(safe=True) == "`main`.`sales`"

    def test_full_name_safe_custom_quote(self, sch):
        assert sch.full_name(safe='"') == '"main"."sales"'

    def test_str(self, sch):
        assert str(sch) == "main.sales"

    def test_repr(self, sch):
        assert "Schema<" in repr(sch)

    def test_equality_same_schema(self, cats):
        # Same service instance → dataclass __eq__ compares service+catalog+schema
        a = cats.schema("main.sales")
        b = cats.schema("main.sales")
        assert a == b

    def test_inequality_different_schema(self, cats):
        a = cats.schema("main.sales")
        c = cats.schema("main.analytics")
        assert a != c

    def test_inequality_different_catalog(self, cats):
        a = cats.schema("main.sales")
        b = cats.schema("staging.sales")
        assert a != b

    def test_unhashable(self, sch):
        """Schema is mutable — it must not be hashable."""
        with pytest.raises(TypeError, match="unhashable"):
            hash(sch)


class TestSchemaNavigation:
    def test_getitem_returns_table(self, sch, mock_tables):
        tbl = sch["orders"]
        mock_tables.table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders"
        )
        assert tbl is mock_tables.table.return_value

    def test_table_method_returns_table(self, sch, mock_tables):
        tbl = sch.table("orders")
        mock_tables.table.assert_called_once_with(
            catalog_name="main", schema_name="sales", table_name="orders"
        )

    def test_catalog_property_navigates_up(self, sch):
        parent = sch.catalog
        assert isinstance(parent, Catalog)
        assert parent.catalog_name == "main"

    def test_tables_delegates_to_client_tables(self, sch, mock_tables):
        list(sch.tables())
        mock_tables.list_tables.assert_called_once_with(
            catalog_name="main", schema_name="sales"
        )

    def test_subscript_chains_to_table(self, cats, mock_tables):
        """cats["main.sales"]["orders"] → Table."""
        sch = cats["main.sales"]
        tbl = sch["orders"]
        assert tbl is mock_tables.table.return_value

    def test_url_contains_catalog_and_schema(self, sch):
        url = sch.url
        url_str = url.to_string() if hasattr(url, "to_string") else str(url)
        assert "main" in url_str and "sales" in url_str


# ===========================================================================
# Unit — Schema infos & cache
# ===========================================================================


class TestSchemaInfos:
    def test_fetches_from_api_on_first_access(self, sch, mock_ws):
        info = _sch_info(comment="test schema")
        mock_ws.schemas.get.return_value = info
        result = sch.infos
        assert result is info
        mock_ws.schemas.get.assert_called_once_with(full_name="main.sales")

    def test_second_access_uses_cache(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        _ = sch.infos
        _ = sch.infos
        assert mock_ws.schemas.get.call_count == 1

    def test_cache_expires_after_ttl(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        _ = sch.infos
        object.__setattr__(sch, "_infos_fetched_at", time.time() - sch._infos_ttl - 1)
        _ = sch.infos
        assert mock_ws.schemas.get.call_count == 2

    def test_clear_resets_cache(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        _ = sch.infos
        sch.clear()
        assert sch._infos is None and sch._infos_fetched_at is None

    def test_comment_property(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info(comment="important")
        assert sch.comment == "important"

    def test_owner_property(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info(owner="alice")
        assert sch.owner == "alice"


class TestSchemaExists:
    def test_exists_true(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        assert sch.exists is True

    def test_exists_false_on_not_found(self, sch, mock_ws):
        mock_ws.schemas.get.side_effect = NotFound("not found")
        assert sch.exists is False


# ===========================================================================
# Unit — Schema lifecycle
# ===========================================================================


class TestSchemaCreate:
    def test_create_calls_api(self, sch, mock_ws):
        mock_ws.schemas.create.return_value = _sch_info(comment="c")
        sch.create(comment="c")
        mock_ws.schemas.create.assert_called_once_with(
            catalog_name="main", name="sales",
            comment="c", properties=None, storage_root=None,
        )

    def test_create_caches_returned_info(self, sch, mock_ws):
        info = _sch_info()
        mock_ws.schemas.create.return_value = info
        sch.create()
        assert sch._infos is info

    def test_create_if_not_exists_silences_already_exists(self, sch, mock_ws):
        mock_ws.schemas.create.side_effect = DatabricksError("already exists")
        sch.create(if_not_exists=True)   # must not raise

    def test_create_re_raises_other_errors(self, sch, mock_ws):
        mock_ws.schemas.create.side_effect = DatabricksError("permission denied")
        with pytest.raises(DatabricksError):
            sch.create(if_not_exists=True)

    def test_ensure_created_calls_create_when_missing(self, sch, mock_ws):
        mock_ws.schemas.get.side_effect = NotFound("not found")
        mock_ws.schemas.create.return_value = _sch_info()
        sch.ensure_created()
        mock_ws.schemas.create.assert_called_once()

    def test_ensure_created_skips_when_exists(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        sch.ensure_created()
        mock_ws.schemas.create.assert_not_called()


class TestSchemaDelete:
    def test_delete_calls_api(self, sch, mock_ws):
        sch.delete()
        mock_ws.schemas.delete.assert_called_once_with(
            full_name="main.sales", force=False
        )

    def test_delete_force(self, sch, mock_ws):
        sch.delete(force=True)
        mock_ws.schemas.delete.assert_called_once_with(
            full_name="main.sales", force=True
        )

    def test_delete_resets_cache(self, sch, mock_ws):
        mock_ws.schemas.get.return_value = _sch_info()
        _ = sch.infos
        sch.delete()
        assert sch._infos is None

    def test_delete_raise_error_false_suppresses(self, sch, mock_ws):
        mock_ws.schemas.delete.side_effect = DatabricksError("gone")
        sch.delete(raise_error=False)   # must not raise

    def test_delete_raise_error_true_propagates(self, sch, mock_ws):
        mock_ws.schemas.delete.side_effect = DatabricksError("gone")
        with pytest.raises(DatabricksError):
            sch.delete(raise_error=True)


class TestSchemaUpdate:
    def test_update_sends_kwargs(self, sch, mock_ws):
        mock_ws.schemas.update.return_value = _sch_info(comment="new", owner="bob")
        sch.update(comment="new", owner="bob")
        mock_ws.schemas.update.assert_called_once_with(
            full_name="main.sales", comment="new", owner="bob"
        )

    def test_update_refreshes_cache(self, sch, mock_ws):
        info = _sch_info(comment="updated")
        mock_ws.schemas.update.return_value = info
        sch.update(comment="updated")
        assert sch._infos is info

    def test_update_omits_none_kwargs(self, sch, mock_ws):
        mock_ws.schemas.update.return_value = _sch_info()
        sch.update(comment="only-comment")
        _, kwargs = mock_ws.schemas.update.call_args
        assert "owner" not in kwargs
        assert "properties" not in kwargs


class TestSchemaTags:
    def test_set_tags_ddl_produces_alter_statement(self, sch):
        ddl = sch.set_tags_ddl({"env": "prod"})
        assert "ALTER SCHEMA" in ddl
        assert "`main`.`sales`" in ddl
        assert "'env' = 'prod'" in ddl

    def test_set_tags_ddl_empty_raises(self, sch):
        with pytest.raises(ValueError):
            sch.set_tags_ddl({})

    def test_set_tags_executes_sql(self, sch, mock_client):
        mock_sql = MagicMock()
        mock_client.sql = mock_sql
        sch.set_tags({"env": "prod"})
        mock_sql.execute.assert_called_once()
        ddl_arg = mock_sql.execute.call_args[0][0]
        assert "ALTER SCHEMA" in ddl_arg

    def test_set_tags_none_is_noop(self, sch, mock_client):
        mock_sql = MagicMock()
        mock_client.sql = mock_sql
        sch.set_tags(None)
        mock_sql.execute.assert_not_called()


# ===========================================================================
# Unit — Tables / Table shortcuts to Schema
# ===========================================================================


class TestTablesSchemaHelper:
    def test_returns_schema(self, mock_client):
        svc = Tables(client=mock_client, catalog_name="main", schema_name="sales")
        sch = svc.schema("main.sales")
        assert isinstance(sch, Schema)
        assert sch.catalog_name == "main" and sch.schema_name == "sales"

    def test_uses_service_defaults_when_no_arg(self, mock_client):
        svc = Tables(client=mock_client, catalog_name="main", schema_name="sales")
        sch = svc.schema()
        assert sch.catalog_name == "main" and sch.schema_name == "sales"

    def test_keyword_overrides(self, mock_client):
        svc = Tables(client=mock_client, catalog_name="main", schema_name="sales")
        sch = svc.schema(catalog_name="staging", schema_name="raw")
        assert sch.catalog_name == "staging" and sch.schema_name == "raw"

    def test_service_tables_shorthand(self, mock_client):
        mock_tbl = MagicMock(spec=Tables)
        mock_client.tables = mock_tbl
        svc = Tables(client=mock_client)
        assert svc.tables is mock_tbl


class TestTableSchemaProperty:
    def test_navigates_up_to_schema(self, mock_client):
        svc = MagicMock(spec=Tables)
        svc.client = mock_client
        tbl = Table(service=svc, catalog_name="main", schema_name="sales", table_name="orders")
        ts = tbl.schema
        assert isinstance(ts, Schema)
        assert ts.catalog_name == "main" and ts.schema_name == "sales"

    def test_table_catalog_consistent_with_schema(self, mock_client):
        svc = MagicMock(spec=Tables)
        svc.client = mock_client
        tbl = Table(service=svc, catalog_name="main", schema_name="sales", table_name="orders")
        assert tbl.catalog.catalog_name == tbl.schema.catalog_name


# ===========================================================================
# Integration — requires live Databricks workspace
# ===========================================================================

pytestmark_integration = [requires_databricks, pytest.mark.integration]


class TestSchemaIntegration(DatabricksCase):
    """Integration tests that hit a real Unity Catalog workspace."""

    pytestmark = pytestmark_integration

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cats = cls.workspace.catalogs
        # Pick the first available catalog that has at least one schema
        cls.first_cat = None
        cls.first_sch = None
        for cat in cls.cats.list():
            schemas = list(cat.schemas())
            if schemas:
                cls.first_cat = cat
                cls.first_sch = schemas[0]
                break

    # ── read-only probes ───────────────────────────────────────────────────

    def test_schema_exists_is_true(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        assert self.first_sch.exists is True

    def test_schema_infos_has_name_and_catalog(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        info = self.first_sch.infos
        assert info.name == self.first_sch.schema_name
        assert info.catalog_name == self.first_sch.catalog_name

    def test_schema_full_name(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        expected = f"{self.first_sch.catalog_name}.{self.first_sch.schema_name}"
        assert self.first_sch.full_name() == expected

    def test_schema_catalog_navigates_up(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        parent = self.first_sch.catalog
        assert isinstance(parent, Catalog)
        assert parent.catalog_name == self.first_sch.catalog_name

    def test_schema_tables_returns_table_objects(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        tables = list(self.first_sch.tables())
        assert all(isinstance(t, Table) for t in tables)

    def test_schema_subscript_returns_table(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        tables = list(self.first_sch.tables())
        if not tables:
            self.skipTest("No tables in the selected schema")
        tbl = self.first_sch[tables[0].table_name]
        assert isinstance(tbl, Table)
        assert tbl.table_name == tables[0].table_name

    def test_catalogs_2part_subscript_reaches_schema(self):
        if self.first_sch is None:
            self.skipTest("No schemas available in any catalog")
        key = f"{self.first_sch.catalog_name}.{self.first_sch.schema_name}"
        result = self.cats[key]
        assert isinstance(result, Schema)
        assert result.full_name() == self.first_sch.full_name()

