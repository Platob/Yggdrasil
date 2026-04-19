"""
Tests for :class:`~yggdrasil.databricks.sql.catalog.Catalog` and
:class:`~yggdrasil.databricks.sql.catalogs.Catalogs`.

Structure
---------
Unit tests (no live workspace)
    ``mock_ws`` fixture stubs ``workspace_client()`` so lifecycle methods
    are exercised without any network call.

Integration tests (``requires_databricks``)
    :class:`TestCatalogIntegration` inherits :class:`DatabricksCase` and
    is skipped automatically when ``DATABRICKS_HOST`` is not set.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound, DatabricksError
from databricks.sdk.service.catalog import CatalogInfo, SchemaInfo

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Catalog, Catalogs, Schema, Table, Tables
from ..conftest import DatabricksCase, requires_databricks


# ── helpers ───────────────────────────────────────────────────────────────────


def _cat_info(name: str = "main", **kw) -> CatalogInfo:
    return CatalogInfo(name=name, comment=kw.get("comment"), owner=kw.get("owner"))


def _sch_info(catalog: str = "main", name: str = "sales", **kw) -> SchemaInfo:
    return SchemaInfo(catalog_name=catalog, name=name, comment=kw.get("comment"))


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
def cat(cats):
    return cats.catalog("main")


@pytest.fixture()
def mock_tables(mock_client):
    svc = MagicMock(spec=Tables)
    svc.table.return_value = MagicMock(spec=Table)
    mock_client.tables = svc
    return svc


# ===========================================================================
# Unit — Catalogs collection
# ===========================================================================


class TestCatalogsGetitem:
    def test_1part_returns_catalog(self, cats):
        result = cats["main"]
        assert isinstance(result, Catalog)
        assert result.catalog_name == "main"

    def test_2part_returns_schema(self, cats):
        result = cats["main.sales"]
        assert isinstance(result, Schema)
        assert result.catalog_name == "main"
        assert result.schema_name == "sales"

    def test_3part_returns_table(self, cats, mock_tables):
        result = cats["main.sales.orders"]
        mock_tables.table.assert_called_once_with(location="main.sales.orders")
        assert result is mock_tables.table.return_value

    def test_4part_returns_column(self, cats, mock_client):
        mock_columns = MagicMock()
        mock_client.columns = mock_columns
        result = cats["main.sales.orders.price"]
        mock_columns.column.assert_called_once_with("main.sales.orders.price")
        assert result is mock_columns.column.return_value

    def test_backtick_stripped(self, cats):
        result = cats["`main`.`sales`"]
        assert isinstance(result, Schema)
        assert result.catalog_name == "main"
        assert result.schema_name == "sales"

    def test_too_many_parts_raises(self, cats):
        with pytest.raises(KeyError, match="1- to 4-part"):
            _ = cats["a.b.c.d.e"]


class TestCatalogsIter:
    def test_iter_delegates_to_list(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [_cat_info("main"), _cat_info("hive")]
        result = list(iter(cats))
        assert [c.catalog_name for c in result] == ["main", "hive"]


class TestCatalogsSetitem:
    def test_setitem_renames_catalog(self, cats, mock_ws):
        mock_ws.catalogs.update.return_value = _cat_info("renamed")
        cats["main"] = "renamed"
        mock_ws.catalogs.update.assert_called_once_with(
            name="main", new_name="renamed",
        )

    def test_setitem_renames_schema_via_2part_key(self, cats, mock_ws):
        mock_ws.schemas.update.return_value = None
        cats["main.sales"] = "sales_v2"
        mock_ws.schemas.update.assert_called_once_with(
            full_name="main.sales", new_name="sales_v2",
        )


class TestCatalogsFactories:
    def test_catalog_factory(self, cats):
        cat = cats.catalog("staging")
        assert isinstance(cat, Catalog)
        assert cat.catalog_name == "staging"

    def test_schema_factory(self, cats):
        sch = cats.schema("main.analytics")
        assert isinstance(sch, Schema)
        assert sch.catalog_name == "main" and sch.schema_name == "analytics"

    def test_schema_factory_bad_name_raises(self, cats):
        with pytest.raises(ValueError, match="two-part"):
            cats.schema("onlyonepart")

    def test_table_factory_delegates(self, cats, mock_tables):
        cats.table("main.sales.orders")
        mock_tables.table.assert_called_once_with(location="main.sales.orders")


class TestCatalogsParseLocation:
    @pytest.mark.parametrize("location,expected", [
        ("main.sales.orders", ("main", "sales", "orders")),
        ("main.sales",        ("main", "sales", None)),
        ("main",              ("main", None,    None)),
    ])
    def test_from_string(self, cats, location, expected):
        assert cats.parse_location(location) == expected

    def test_kwargs_override_location(self, cats):
        c, s, t = cats.parse_location("main.sales.orders", catalog_name="override")
        assert c == "override" and s == "sales" and t == "orders"

    def test_pure_kwargs(self, cats):
        assert cats.parse_location(catalog_name="a", schema_name="b", table_name="c") == ("a", "b", "c")


class TestCatalogsList:
    def test_list_yields_catalog_objects(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [_cat_info("main"), _cat_info("hive")]
        result = list(cats.list_catalogs())
        assert len(result) == 2
        assert all(isinstance(c, Catalog) for c in result)
        assert [c.catalog_name for c in result] == ["main", "hive"]

    def test_list_sets_infos_on_yielded_objects(self, cats, mock_ws):
        info = _cat_info("main", comment="prod")
        mock_ws.catalogs.list.return_value = [info]
        (cat,) = cats.list_catalogs()
        assert cat._infos is info

    def test_list_populates_module_cache(self, cats, mock_ws):
        from yggdrasil.databricks.sql.catalogs import _CATALOG_INFO_CACHE
        mock_ws.catalogs.list.return_value = [_cat_info("main")]
        list(cats.list_catalogs(use_cache=True))
        host = mock_client_url = "https://adb-123.azuredatabricks.net"
        assert any("main" in k for k in _CATALOG_INFO_CACHE._store)  # noqa: SLF001

    def test_list_name_exact_filter(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [
            _cat_info("main"),
            _cat_info("staging"),
        ]
        result = list(cats.list_catalogs(name="staging"))
        assert [c.catalog_name for c in result] == ["staging"]

    def test_list_name_glob_is_case_insensitive(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [
            _cat_info("Prod_Main"),
            _cat_info("prod_staging"),
            _cat_info("dev_main"),
        ]
        result = list(cats.list_catalogs(name="prod_*"))
        assert [c.catalog_name for c in result] == ["Prod_Main", "prod_staging"]

    def test_list_name_star_matches_all(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [
            _cat_info("main"),
            _cat_info("hive"),
        ]
        result = list(cats.list_catalogs(name="*"))
        assert [c.catalog_name for c in result] == ["main", "hive"]

    def test_list_name_middle_wildcard(self, cats, mock_ws):
        mock_ws.catalogs.list.return_value = [
            _cat_info("prefix_a_cat"),
            _cat_info("prefix_b_cat"),
            _cat_info("other"),
        ]
        result = list(cats.list_catalogs(name="prefix_*_cat"))
        assert [c.catalog_name for c in result] == ["prefix_a_cat", "prefix_b_cat"]


# ===========================================================================
# Unit — Catalog resource
# ===========================================================================


class TestCatalogNavigation:
    def test_getitem_returns_schema(self, cat):
        sch = cat["sales"]
        assert isinstance(sch, Schema)
        assert sch.catalog_name == "main" and sch.schema_name == "sales"

    def test_schema_method(self, cat):
        sch = cat.schema("analytics")
        assert isinstance(sch, Schema) and sch.schema_name == "analytics"

    def test_table_delegates(self, cat, mock_tables):
        cat.table("sales.orders")
        mock_tables.table.assert_called_once_with(
            location="sales.orders", catalog_name="main",
            schema_name=None, table_name=None,
        )

    def test_schemas_yields_schema_objects(self, cat, mock_ws):
        mock_ws.schemas.list.return_value = [
            _sch_info("main", "sales"),
            _sch_info("main", "analytics"),
        ]
        result = list(cat.schemas())
        assert len(result) == 2
        assert all(isinstance(s, Schema) for s in result)
        assert [s.schema_name for s in result] == ["sales", "analytics"]

    def test_schemas_sets_infos_on_yielded(self, cat, mock_ws):
        info = _sch_info("main", "sales", comment="test")
        mock_ws.schemas.list.return_value = [info]
        (sch,) = cat.schemas()
        assert sch._infos is info

    def test_str_returns_catalog_name(self, cat):
        assert str(cat) == "main"

    def test_full_name(self, cat):
        assert cat.full_name() == "main"

    def test_iter_delegates_to_schemas(self, cat, mock_ws):
        mock_ws.schemas.list.return_value = [
            _sch_info("main", "sales"),
            _sch_info("main", "analytics"),
        ]
        result = list(iter(cat))
        assert [s.schema_name for s in result] == ["sales", "analytics"]

    def test_setitem_renames_child_schema(self, cat, mock_ws):
        cat["sales"] = "sales_v2"
        mock_ws.schemas.update.assert_called_once_with(
            full_name="main.sales", new_name="sales_v2",
        )


class TestCatalogRename:
    def test_rename_calls_update_with_new_name(self, cat, mock_ws):
        mock_ws.catalogs.update.return_value = _cat_info("new_main")
        cat.rename("new_main")
        mock_ws.catalogs.update.assert_called_once_with(
            name="main", new_name="new_main",
        )
        assert cat.catalog_name == "new_main"

    def test_rename_noop_on_same_name(self, cat, mock_ws):
        cat.rename("main")
        mock_ws.catalogs.update.assert_not_called()

    def test_rename_empty_raises(self, cat):
        with pytest.raises(ValueError):
            cat.rename("")

    def test_rename_strips_backticks(self, cat, mock_ws):
        mock_ws.catalogs.update.return_value = _cat_info("x")
        cat.rename("`x`")
        mock_ws.catalogs.update.assert_called_once_with(
            name="main", new_name="x",
        )


class TestCatalogInfos:
    def test_fetches_from_api_on_first_access(self, cat, mock_ws):
        info = _cat_info("main", comment="prod")
        mock_ws.catalogs.get.return_value = info
        assert cat.infos is info
        mock_ws.catalogs.get.assert_called_once_with("main")

    def test_second_access_uses_cache(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        _ = cat.infos
        _ = cat.infos
        assert mock_ws.catalogs.get.call_count == 1

    def test_cache_expires_after_ttl(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        _ = cat.infos
        # Force expiry by backdating the timestamp
        object.__setattr__(cat, "_infos_fetched_at", time.time() - cat._infos_ttl - 1)
        _ = cat.infos
        assert mock_ws.catalogs.get.call_count == 2

    def test_clear_resets_cache(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        _ = cat.infos
        cat.clear()
        assert cat._infos is None
        assert cat._infos_fetched_at is None

    def test_comment_property(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main", comment="hello")
        assert cat.comment == "hello"

    def test_owner_property(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main", owner="alice")
        assert cat.owner == "alice"


class TestCatalogExists:
    def test_exists_true_when_api_returns_info(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        assert cat.exists is True

    def test_exists_false_on_not_found(self, cat, mock_ws):
        mock_ws.catalogs.get.side_effect = NotFound("not found")
        assert cat.exists is False


class TestCatalogCreate:
    def test_create_calls_api(self, cat, mock_ws):
        mock_ws.catalogs.create.return_value = _cat_info("main", comment="c")
        cat.create(comment="c")
        mock_ws.catalogs.create.assert_called_once_with(
            name="main", comment="c", properties=None, storage_root=None,
        )

    def test_create_caches_returned_info(self, cat, mock_ws):
        info = _cat_info("main")
        mock_ws.catalogs.create.return_value = info
        cat.create()
        assert cat._infos is info

    def test_create_if_not_exists_silences_already_exists(self, cat, mock_ws):
        mock_ws.catalogs.create.side_effect = DatabricksError("already exists")
        cat.create(if_not_exists=True)   # must not raise

    def test_create_re_raises_other_errors(self, cat, mock_ws):
        mock_ws.catalogs.create.side_effect = DatabricksError("permission denied")
        with pytest.raises(DatabricksError):
            cat.create(if_not_exists=True)

    def test_ensure_created_calls_create_when_missing(self, cat, mock_ws):
        mock_ws.catalogs.get.side_effect = NotFound("not found")
        mock_ws.catalogs.create.return_value = _cat_info("main")
        cat.ensure_created()
        mock_ws.catalogs.create.assert_called_once()

    def test_ensure_created_skips_when_exists(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        cat.ensure_created()
        mock_ws.catalogs.create.assert_not_called()


class TestCatalogDelete:
    def test_delete_calls_api(self, cat, mock_ws):
        cat.delete()
        mock_ws.catalogs.delete.assert_called_once_with(name="main", force=False)

    def test_delete_force(self, cat, mock_ws):
        cat.delete(force=True)
        mock_ws.catalogs.delete.assert_called_once_with(name="main", force=True)

    def test_delete_resets_cache(self, cat, mock_ws):
        mock_ws.catalogs.get.return_value = _cat_info("main")
        _ = cat.infos                     # populate cache
        cat.delete()
        assert cat._infos is None

    def test_delete_raise_error_false_suppresses(self, cat, mock_ws):
        mock_ws.catalogs.delete.side_effect = DatabricksError("gone")
        cat.delete(raise_error=False)     # must not raise

    def test_delete_raise_error_true_propagates(self, cat, mock_ws):
        mock_ws.catalogs.delete.side_effect = DatabricksError("gone")
        with pytest.raises(DatabricksError):
            cat.delete(raise_error=True)


class TestCatalogUpdate:
    def test_update_sends_kwargs(self, cat, mock_ws):
        mock_ws.catalogs.update.return_value = _cat_info("main", comment="new", owner="bob")
        cat.update(comment="new", owner="bob")
        mock_ws.catalogs.update.assert_called_once_with(
            name="main", comment="new", owner="bob"
        )

    def test_update_refreshes_cache(self, cat, mock_ws):
        info = _cat_info("main", comment="updated")
        mock_ws.catalogs.update.return_value = info
        cat.update(comment="updated")
        assert cat._infos is info

    def test_update_omits_none_kwargs(self, cat, mock_ws):
        mock_ws.catalogs.update.return_value = _cat_info("main")
        cat.update(comment="only-comment")
        _, kwargs = mock_ws.catalogs.update.call_args
        assert "owner" not in kwargs
        assert "properties" not in kwargs


class TestCatalogTags:
    def test_set_tags_ddl_produces_alter_statement(self, cat):
        ddl = cat.set_tags_ddl({"env": "prod", "team": "data"})
        assert ddl.startswith("ALTER CATALOG `main` SET TAGS")
        assert (
            "'env' = 'prod'" in ddl
            or "'team' = 'data'" in ddl
        )

    def test_set_tags_ddl_empty_raises(self, cat):
        with pytest.raises(ValueError):
            cat.set_tags_ddl({})

    def test_set_tags_executes_sql(self, cat, mock_client):
        mock_sql = MagicMock()
        mock_client.sql = mock_sql
        cat.set_tags({"env": "prod"})
        mock_sql.execute.assert_called_once()
        ddl_arg = mock_sql.execute.call_args[0][0]
        assert "ALTER CATALOG" in ddl_arg

    def test_set_tags_none_is_noop(self, cat, mock_client):
        mock_sql = MagicMock()
        mock_client.sql = mock_sql
        cat.set_tags(None)
        mock_sql.execute.assert_not_called()


# ===========================================================================
# Unit — Tables / Table shortcuts to Catalog
# ===========================================================================


class TestTablesCatalogHelper:
    def test_returns_catalog(self, mock_client):
        svc = Tables(client=mock_client, catalog_name="main")
        cat = svc.catalog("main")
        assert isinstance(cat, Catalog) and cat.catalog_name == "main"

    def test_uses_service_default_when_no_arg(self, mock_client):
        svc = Tables(client=mock_client, catalog_name="main")
        cat = svc.catalog()
        assert cat.catalog_name == "main"


class TestTableCatalogProperty:
    def test_navigates_up(self, mock_client):
        svc = MagicMock(spec=Tables)
        svc.client = mock_client
        tbl = Table(service=svc, catalog_name="main", schema_name="sales", table_name="orders")
        tc = tbl.catalog
        assert isinstance(tc, Catalog) and tc.catalog_name == "main"

    def test_service_shorthand(self, mock_client):
        mock_cats = MagicMock(spec=Catalogs)
        mock_client.catalogs = mock_cats
        svc = Tables(client=mock_client)
        assert svc.catalogs is mock_cats


# ===========================================================================
# Integration — requires live Databricks workspace
# ===========================================================================

pytestmark_integration = [requires_databricks, pytest.mark.integration]


class TestCatalogIntegration(DatabricksCase):
    """Integration tests that hit a real Unity Catalog workspace."""

    pytestmark = pytestmark_integration

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.cats = cls.workspace.catalogs

    # ── read-only probes (safe to run on any workspace) ────────────────────

    def test_list_yields_at_least_one_catalog(self):
        result = list(self.cats.list_catalogs())
        assert len(result) >= 1
        assert all(isinstance(c, Catalog) for c in result)

    def test_catalog_subscript_returns_catalog(self):
        (first, *_) = self.cats.list_catalogs()
        cat = self.cats[first.catalog_name]
        assert isinstance(cat, Catalog)
        assert cat.catalog_name == first.catalog_name

    def test_catalog_infos_has_name(self):
        (first, *_) = self.cats.list_catalogs()
        assert first.infos.name == first.catalog_name

    def test_catalog_exists_is_true(self):
        (first, *_) = self.cats.list_catalogs()
        assert first.exists is True

    def test_catalog_schemas_returns_schemas(self):
        (first, *_) = self.cats.list_catalogs()
        schemas = list(first.schemas())
        assert all(isinstance(s, Schema) for s in schemas)

    def test_catalog_schema_subscript_chains_to_schema(self):
        (first, *_) = self.cats.list_catalogs()
        schemas = list(first.schemas())
        if not schemas:
            self.skipTest("No schemas in the first catalog")
        sch = first[schemas[0].schema_name]
        assert isinstance(sch, Schema)
        assert sch.catalog_name == first.catalog_name

    def test_three_level_subscript_chain(self):
        """client.catalogs["cat"]["schema"]["table"] resolves correctly."""
        (first, *_) = self.cats.list_catalogs()
        schemas = list(first.schemas())
        if not schemas:
            self.skipTest("No schemas in the first catalog")
        tables = list(schemas[0].tables())
        if not tables:
            self.skipTest("No tables in the first schema")
        tbl = self.cats[first.catalog_name][schemas[0].schema_name][tables[0].table_name]
        from yggdrasil.databricks.sql import Table
        assert isinstance(tbl, Table)

    def test_four_level_subscript_chain(self):
        """client.catalogs["cat"]["schema"]["table"]["column"] resolves to a Column."""
        (first, *_) = self.cats.list_catalogs()
        schemas = list(first.schemas())
        if not schemas:
            self.skipTest("No schemas in the first catalog")
        tables = list(schemas[0].tables())
        if not tables:
            self.skipTest("No tables in the first schema")
        cols = tables[0].columns
        if not cols:
            self.skipTest("No columns in the first table")
        col = (
            self.cats[first.catalog_name]
            [schemas[0].schema_name]
            [tables[0].table_name]
            [cols[0].name]
        )
        from yggdrasil.databricks.sql import Column
        assert isinstance(col, Column)

