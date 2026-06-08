"""External-table DDL + default-location resolution.

``Table.create`` / ``Table.sql_create`` emit ``CREATE EXTERNAL TABLE …
USING <fmt> LOCATION '…'`` for external tables, and default the LOCATION to
the schema's governed UC storage root when the caller omits one. These are
pure-DDL / pure-resolution checks — no workspace round-trips (the SQL surface
and the schema lookup are mocked).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import DataSourceFormat

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


class TestDefaultExternalLocation:
    def test_uses_schema_storage_root(self) -> None:
        t = Table(
            service=MagicMock(name="Tables"),
            catalog_name="cat_e", schema_name="sch", table_name="tbl",
        )
        t.schema_storage_location = MagicMock(  # type: ignore[method-assign]
            return_value="s3://root/uc/cat_e/sch",
        )
        assert t._default_external_location() == "s3://root/uc/cat_e/sch/tables/tbl"

    def test_propagates_when_schema_has_no_storage(self) -> None:
        t = Table(
            service=MagicMock(name="Tables"),
            catalog_name="cat_e", schema_name="sch", table_name="tbl",
        )
        # Schema advertises no storage location → NotImplementedError is
        # surfaced (no catalog-root fabrication); the caller must pin a location.
        t.schema_storage_location = MagicMock(  # type: ignore[method-assign]
            side_effect=NotImplementedError,
        )
        with pytest.raises(NotImplementedError):
            t._default_external_location()
