"""External-table DDL + default-location resolution.

``Table.create`` / ``Table.sql_create`` emit ``CREATE EXTERNAL TABLE …
USING <fmt> LOCATION '…'`` for external tables, and default the LOCATION to
the catalog's governed UC ``storage_root`` when the caller omits one. These
are pure-DDL / pure-resolution checks — no workspace round-trips (the SQL
surface and the catalog lookup are mocked).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import DataSourceFormat

from yggdrasil.databricks.client import _CATALOG_STORAGE_ROOTS
from yggdrasil.databricks.table.table import Table
from yggdrasil.data.schema import Schema, field


def _schema() -> Schema:
    return Schema([
        field("id", "string", nullable=False),
        field("name", "string"),
        field("value", "float64"),
        field("updated_at", "timestamp[us]"),
    ])


def _mock_table(name: str = "trading_tgp_dev.ygg_integration.ext_tb") -> tuple[Table, list[str]]:
    cat, sch, tbl = name.split(".")
    t = Table(service=MagicMock(name="Tables"), catalog_name=cat, schema_name=sch, table_name=tbl)
    captured: list[str] = []
    t.sql.execute = lambda stmt, **kw: captured.append(stmt)
    t._apply_post_create_constraints = lambda *a, **k: None  # type: ignore[method-assign]
    return t, captured


class TestExternalDDL:
    def test_external_delta_ddl_matches_reference_shape(self) -> None:
        t, captured = _mock_table()
        t.sql_create(
            _schema(),
            storage_location="s3://bucket/loc/ext_tb",
            data_source_format=DataSourceFormat.DELTA,
            record_ygg_properties=False,
        )
        ddl = captured[0]
        # No EXTERNAL keyword — LOCATION makes it external (Databricks' own form).
        assert ddl.startswith(
            "CREATE TABLE IF NOT EXISTS "
            "`trading_tgp_dev`.`ygg_integration`.`ext_tb` ("
        )
        assert "EXTERNAL" not in ddl
        assert "`id` STRING NOT NULL" in ddl
        assert "`updated_at` TIMESTAMP" in ddl
        assert "USING DELTA" in ddl
        assert "LOCATION 's3://bucket/loc/ext_tb'" in ddl
        # AUTO is managed-only, so an external table specifies explicit
        # liquid-clustering columns — defaulting to the first column.
        assert "CLUSTER BY (`id`)" in ddl
        assert "CLUSTER BY AUTO" not in ddl

    def test_external_clusters_by_primary_key_when_present(self) -> None:
        t, captured = _mock_table()
        sch = Schema([
            field("id", "string", nullable=False, tags={"primary_key": "true"}),
            field("name", "string"),
        ])
        t.sql_create(
            sch, storage_location="s3://bucket/loc/ext_tb",
            data_source_format=DataSourceFormat.DELTA, record_ygg_properties=False,
        )
        assert "CLUSTER BY (`id`)" in captured[0]

    def test_external_explicit_cluster_tag_wins(self) -> None:
        t, captured = _mock_table()
        sch = Schema([
            field("id", "string", nullable=False),
            field("name", "string", tags={"cluster_by": "true"}),
        ])
        t.sql_create(
            sch, storage_location="s3://bucket/loc/ext_tb",
            data_source_format=DataSourceFormat.DELTA, record_ygg_properties=False,
        )
        assert "CLUSTER BY (`name`)" in captured[0]

    def test_managed_ddl_unchanged(self) -> None:
        t, captured = _mock_table()
        t.sql_create(_schema(), data_source_format=DataSourceFormat.DELTA, record_ygg_properties=False)
        ddl = captured[0]
        assert ddl.startswith("CREATE TABLE IF NOT EXISTS")
        assert "EXTERNAL" not in ddl
        assert "LOCATION" not in ddl
        assert "CLUSTER BY AUTO" in ddl

    def test_or_replace_external_drops_keyword_keeps_location(self) -> None:
        # ``CREATE OR REPLACE EXTERNAL TABLE`` isn't valid grammar; the
        # LOCATION clause still makes it external.
        t, captured = _mock_table()
        t.sql_create(
            _schema(),
            storage_location="s3://bucket/loc/ext_tb",
            or_replace=True,
            record_ygg_properties=False,
        )
        ddl = captured[0]
        assert ddl.startswith("CREATE OR REPLACE TABLE")
        assert "LOCATION 's3://bucket/loc/ext_tb'" in ddl


class TestDefaultStorageLocation:
    def _client(self, catalog: str, storage_root: str | None):
        from yggdrasil.databricks.client import DatabricksClient

        c = DatabricksClient(catalog_name=catalog)
        ws = MagicMock()
        ws.catalogs.get.return_value = MagicMock(storage_root=storage_root)
        c.workspace_client = lambda: ws  # type: ignore[method-assign]
        _CATALOG_STORAGE_ROOTS.pop((c.host, catalog), None)
        return c

    def test_resolves_catalog_storage_root(self) -> None:
        c = self._client("cat_a", "s3://root/uc/cat_a")
        assert c.default_storage_location() == "s3://root/uc/cat_a/"
        assert c.default_storage_location("s/t") == "s3://root/uc/cat_a/s/t"

    def test_suffix_leading_slash_normalized(self) -> None:
        c = self._client("cat_b", "s3://root/uc/cat_b/")
        assert c.default_storage_location("/s/t") == "s3://root/uc/cat_b/s/t"

    def test_explicit_catalog_overrides(self) -> None:
        c = self._client("cat_c", "s3://root/uc/cat_c")
        # Override resolves a different catalog (re-mock the lookup).
        c.workspace_client().catalogs.get.return_value = MagicMock(storage_root="s3://root/uc/other")
        _CATALOG_STORAGE_ROOTS.pop((c.host, "other"), None)
        assert c.default_storage_location("x", catalog_name="other") == "s3://root/uc/other/x"

    def test_raises_without_catalog(self) -> None:
        from yggdrasil.databricks.client import DatabricksClient

        c = DatabricksClient(catalog_name=None)
        with pytest.raises(ValueError, match="needs a catalog"):
            c.default_storage_location()

    def test_raises_when_catalog_has_no_storage_root(self) -> None:
        c = self._client("cat_d", None)
        with pytest.raises(ValueError, match="no storage_root"):
            c.default_storage_location()


class TestDefaultExternalLocation:
    def test_falls_back_to_catalog_root_when_schema_has_no_storage(self) -> None:
        t = Table(
            service=MagicMock(name="Tables"),
            catalog_name="cat_e", schema_name="sch", table_name="tbl",
        )
        # Schema advertises no storage location → NotImplementedError → fall
        # back to the client's catalog-root default.
        t.schema_storage_location = MagicMock(side_effect=NotImplementedError)  # type: ignore[method-assign]
        t.client.default_storage_location = MagicMock(return_value="s3://root/uc/cat_e/sch/tbl")  # type: ignore[method-assign]
        assert t._default_external_location() == "s3://root/uc/cat_e/sch/tbl"
        t.client.default_storage_location.assert_called_once_with(
            suffix="sch/tbl", catalog_name="cat_e",
        )
