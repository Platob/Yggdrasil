"""Tests for :class:`Catalogs` and :class:`Schemas` module-level cache
injection — verifying that warm-cache paths avoid remote round trips and
that TTL timestamps are correctly stamped on cache hits.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.catalog.catalogs import Catalogs, _CATALOG_INFO_CACHE
from yggdrasil.databricks.schema.schemas import Schemas, _SCHEMA_INFO_CACHE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches():
    _CATALOG_INFO_CACHE.clear()
    _SCHEMA_INFO_CACHE.clear()
    yield
    _CATALOG_INFO_CACHE.clear()
    _SCHEMA_INFO_CACHE.clear()


@pytest.fixture(autouse=True)
def _clear_catalog_singletons():
    from yggdrasil.databricks.catalog.catalog import Catalog
    Catalog._INSTANCES.clear()
    yield
    Catalog._INSTANCES.clear()


@pytest.fixture(autouse=True)
def _clear_schema_singletons():
    from yggdrasil.databricks.schema.schema import Schema
    Schema._INSTANCES.clear()
    yield
    Schema._INSTANCES.clear()


def _mock_client(host: str = "adb-test.azuredatabricks.net") -> MagicMock:
    client = MagicMock()
    client.host = host
    client.base_url = MagicMock()
    client.base_url.host = host
    return client


def _catalog_info(name: str = "main") -> SimpleNamespace:
    return SimpleNamespace(name=name, catalog_type="MANAGED_CATALOG")


def _schema_info(catalog: str = "main", schema: str = "sales") -> SimpleNamespace:
    return SimpleNamespace(
        name=schema, catalog_name=catalog, full_name=f"{catalog}.{schema}",
    )


# ---------------------------------------------------------------------------
# Catalogs.catalog — module-level cache injection
# ---------------------------------------------------------------------------


class TestCatalogsCacheInjection:

    def test_cache_hit_skips_remote_call(self) -> None:
        client = _mock_client()
        svc = Catalogs(client=client)
        info = _catalog_info("main")
        # Pre-populate the module-level cache.
        _CATALOG_INFO_CACHE[svc._cache_key("main")] = info

        cat = svc.catalog("main")
        # CatalogInfo was injected from the cache — no SDK call needed.
        assert cat._infos is info
        client.workspace_client.return_value.catalogs.get.assert_not_called()

    def test_cache_hit_stamps_fetch_timestamp(self) -> None:
        # When info is injected from the module cache, ``_infos_fetched_at``
        # must be set so the Catalog.infos TTL guard doesn't immediately
        # expire the entry and fire another remote fetch.
        client = _mock_client()
        svc = Catalogs(client=client)
        info = _catalog_info("main")
        _CATALOG_INFO_CACHE[svc._cache_key("main")] = info

        before = time.time()
        cat = svc.catalog("main")
        after = time.time()

        assert cat._infos_fetched_at is not None
        assert before <= cat._infos_fetched_at <= after

    def test_cache_miss_returns_lazy_handle(self) -> None:
        # Nothing in the cache yet — catalog handle is returned without
        # ``_infos`` pre-loaded (lazy fetch on first ``.infos`` access).
        client = _mock_client()
        svc = Catalogs(client=client)
        cat = svc.catalog("missing")
        assert cat._infos is None
        assert cat._infos_fetched_at is None

    def test_list_catalogs_populates_cache(self) -> None:
        client = _mock_client()
        info = _catalog_info("prod")
        client.workspace_client.return_value.catalogs.list.return_value = [info]
        svc = Catalogs(client=client)

        cats = list(svc.list_catalogs(use_cache=True))
        assert len(cats) == 1
        # Subsequent catalog() lookup must find the entry in the cache.
        key = svc._cache_key("prod")
        assert _CATALOG_INFO_CACHE.get(key) is info

    def test_list_catalogs_then_catalog_avoids_remote(self) -> None:
        client = _mock_client()
        info = _catalog_info("prod")
        client.workspace_client.return_value.catalogs.list.return_value = [info]
        svc = Catalogs(client=client)

        list(svc.list_catalogs(use_cache=True))
        # The catalog() call should see the warm cache entry and avoid
        # calling ``catalogs.get``.
        cat = svc.catalog("prod")
        assert cat._infos is info
        client.workspace_client.return_value.catalogs.get.assert_not_called()


# ---------------------------------------------------------------------------
# Schemas.schema — module-level cache injection
# ---------------------------------------------------------------------------


class TestSchemasCacheInjection:

    def test_cache_hit_skips_remote_call(self) -> None:
        client = _mock_client()
        svc = Schemas(client=client)
        info = _schema_info("main", "sales")
        _SCHEMA_INFO_CACHE[svc._cache_key("main", "sales")] = info

        sch = svc.schema("main.sales")
        assert sch._infos is info
        client.workspace_client.return_value.schemas.get.assert_not_called()

    def test_cache_hit_stamps_fetch_timestamp(self) -> None:
        client = _mock_client()
        svc = Schemas(client=client)
        info = _schema_info("main", "sales")
        _SCHEMA_INFO_CACHE[svc._cache_key("main", "sales")] = info

        before = time.time()
        sch = svc.schema("main.sales")
        after = time.time()

        assert sch._infos_fetched_at is not None
        assert before <= sch._infos_fetched_at <= after

    def test_list_schemas_populates_cache(self) -> None:
        client = _mock_client()
        info = _schema_info("main", "analytics")
        info.catalog_name = "main"
        client.workspace_client.return_value.schemas.list.return_value = [info]
        svc = Schemas(client=client)

        list(svc.list(catalog_name="main", use_cache=True))
        key = svc._cache_key("main", "analytics")
        assert _SCHEMA_INFO_CACHE.get(key) is info

    def test_find_cache_hit_avoids_remote(self) -> None:
        client = _mock_client()
        svc = Schemas(client=client)
        info = _schema_info("main", "finance")
        _SCHEMA_INFO_CACHE[svc._cache_key("main", "finance")] = info

        sch = svc.find("main.finance", cache_ttl=300.0)
        assert sch is not None
        assert sch._infos is info
        client.workspace_client.return_value.schemas.get.assert_not_called()

    def test_find_cache_bypass_hits_remote(self) -> None:
        client = _mock_client()
        info = _schema_info("main", "finance")
        info.catalog_name = "main"
        info.name = "finance"
        client.workspace_client.return_value.schemas.get.return_value = info
        svc = Schemas(client=client)

        # Pre-populate the cache, then bypass it.
        _SCHEMA_INFO_CACHE[svc._cache_key("main", "finance")] = info
        sch = svc.find("main.finance", cache_ttl=None)
        assert sch is not None
        client.workspace_client.return_value.schemas.get.assert_called_once()


# ---------------------------------------------------------------------------
# Cache key consistency
# ---------------------------------------------------------------------------


class TestCacheKeyConsistency:
    """Catalog and Schema cache keys use ``client.host`` (pre-normalized),
    not ``base_url.to_string()`` (builds a URL object on every call)."""

    def test_catalog_cache_key_uses_host(self) -> None:
        client = _mock_client("myworkspace.azuredatabricks.net")
        svc = Catalogs(client=client)
        key = svc._cache_key("main")
        assert key == "myworkspace.azuredatabricks.net|main"
        # ``base_url.to_string()`` must NOT have been called.
        client.base_url.to_string.assert_not_called()

    def test_schema_cache_key_uses_host(self) -> None:
        client = _mock_client("myworkspace.azuredatabricks.net")
        svc = Schemas(client=client)
        key = svc._cache_key("main", "sales")
        assert key == "myworkspace.azuredatabricks.net|main.sales"
        client.base_url.to_string.assert_not_called()
