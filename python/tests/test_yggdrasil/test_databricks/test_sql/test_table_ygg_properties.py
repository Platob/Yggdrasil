"""Tests for the ``ygg.*`` TBLPROPERTIES emitted by table creation.

The same helper feeds both create paths
(:meth:`Table.sql_create` and :meth:`Table.api_create`) so the two
surfaces stay symmetric. These tests pin the property surface — name
set, per-field JSON dump shape, fingerprint stability — without
touching a real Databricks workspace.
"""
from __future__ import annotations

import json
import unittest

from databricks.sdk.service.catalog import DataSourceFormat, TableType

from yggdrasil.data.enums import MimeTypes
from yggdrasil.data.schema import Schema, field
from yggdrasil.databricks.sql.table import (
    YGG_PROPERTY_PREFIX,
    YGG_SCHEMA_FIELD_PREFIX,
    YGG_SCHEMA_FIELD_SUFFIX,
    _build_ygg_properties,
    _resolve_format_mime,
    _ygg_schema_key,
)
from yggdrasil.version import __version__ as ygg_version


def _schema() -> Schema:
    return Schema([
        field("id", "int64", tags={"primary_key": "true"}),
        field("region", "string", tags={"partition_by": "true"}),
        field("bucket", "string", tags={"cluster_by": "true"}),
        field("amount", "decimal(18, 2)"),
    ])


class TestYggProperties(unittest.TestCase):

    def test_emits_required_keys(self) -> None:
        props = _build_ygg_properties(
            _schema(),
            engine="sql",
            data_source_format=DataSourceFormat.DELTA,
            table_type=TableType.MANAGED,
        )
        for key in (
            "ygg.version",
            "ygg.created_at",
            "ygg.engine",
            "ygg.mime_type",
            "ygg.table_type",
            "ygg.field_count",
            "ygg.schema_fingerprint",
            "ygg.partition_columns",
            "ygg.cluster_columns",
            "ygg.primary_keys",
        ):
            self.assertIn(key, props)

        self.assertEqual(props["ygg.engine"], "sql")
        self.assertEqual(props["ygg.mime_type"], MimeTypes.DELTA.value)
        self.assertEqual(props["ygg.table_type"], "MANAGED")
        self.assertEqual(props["ygg.field_count"], "4")
        self.assertEqual(props["ygg.partition_columns"], "region")
        self.assertEqual(props["ygg.cluster_columns"], "bucket")
        self.assertEqual(props["ygg.primary_keys"], "id")
        self.assertEqual(props["ygg.version"], str(ygg_version))
        # raw enum string is replaced by the ygg MimeType categorization.
        self.assertNotIn("ygg.format", props)

    def test_no_giant_schema_json_blob(self) -> None:
        """Per-field properties replace the legacy ``ygg.schema_json`` dump."""
        props = _build_ygg_properties(
            _schema(),
            engine="sql",
            data_source_format=DataSourceFormat.DELTA,
        )
        self.assertNotIn("ygg.schema_json", props)
        self.assertNotIn("ygg.schema", props)

    def test_per_field_schema_keys(self) -> None:
        props = _build_ygg_properties(
            _schema(),
            engine="api",
            data_source_format=DataSourceFormat.DELTA,
            table_type=TableType.EXTERNAL,
            storage_location="s3://bucket/path",
        )
        schema_keys = sorted(k for k in props if k.startswith(YGG_SCHEMA_FIELD_PREFIX))
        self.assertEqual(
            schema_keys,
            sorted([
                "ygg.schema[id]",
                "ygg.schema[region]",
                "ygg.schema[bucket]",
                "ygg.schema[amount]",
            ]),
        )
        # Each per-field value is a self-contained JSON document; its
        # ``name`` matches the bracketed field name in the key.
        for key in schema_keys:
            payload = json.loads(props[key])
            unwrapped = key.removeprefix(YGG_SCHEMA_FIELD_PREFIX).removesuffix(
                YGG_SCHEMA_FIELD_SUFFIX
            )
            self.assertEqual(payload["name"], unwrapped)
            self.assertIn("dtype", payload)
            self.assertIn("nullable", payload)

        self.assertEqual(props["ygg.engine"], "api")
        self.assertEqual(props["ygg.table_type"], "EXTERNAL")
        self.assertEqual(props["ygg.storage_location"], "s3://bucket/path")

    def test_schema_field_prefix_is_namespaced(self) -> None:
        """Per-field keys live under the documented bracketed shape."""
        self.assertTrue(YGG_SCHEMA_FIELD_PREFIX.startswith(YGG_PROPERTY_PREFIX))
        self.assertEqual(YGG_SCHEMA_FIELD_PREFIX, "ygg.schema[")
        self.assertEqual(YGG_SCHEMA_FIELD_SUFFIX, "]")
        self.assertEqual(_ygg_schema_key("amount"), "ygg.schema[amount]")
        # Bracket wrap survives field names containing the property
        # separator without colliding with adjacent keys.
        self.assertEqual(
            _ygg_schema_key("user.first_name"),
            "ygg.schema[user.first_name]",
        )

    def test_fingerprint_stable_across_engines(self) -> None:
        """SQL and API paths agree on the schema fingerprint for the same shape."""
        s = _schema()
        sql_props = _build_ygg_properties(
            s, engine="sql", data_source_format=DataSourceFormat.DELTA,
        )
        api_props = _build_ygg_properties(
            s, engine="api", data_source_format=DataSourceFormat.DELTA,
        )
        self.assertEqual(
            sql_props["ygg.schema_fingerprint"],
            api_props["ygg.schema_fingerprint"],
        )
        # Per-field payloads must also match — only the engine tag and the
        # creation timestamp differ between calls.
        sql_field_keys = {k: sql_props[k] for k in sql_props if k.startswith(YGG_SCHEMA_FIELD_PREFIX)}
        api_field_keys = {k: api_props[k] for k in api_props if k.startswith(YGG_SCHEMA_FIELD_PREFIX)}
        self.assertEqual(sql_field_keys, api_field_keys)

    def test_fingerprint_changes_with_schema(self) -> None:
        s1 = Schema([field("id", "int64")])
        s2 = Schema([field("id", "int64"), field("v", "string")])
        p1 = _build_ygg_properties(s1, engine="sql")
        p2 = _build_ygg_properties(s2, engine="sql")
        self.assertNotEqual(p1["ygg.schema_fingerprint"], p2["ygg.schema_fingerprint"])

    def test_constraint_fields_excluded_from_schema_dump(self) -> None:
        """FK/CHECK constraint rows are not columns and don't get a per-field key."""
        s = Schema([
            field("id", "int64"),
            field(
                "fk_orders_customer",
                "string",
                tags={"constraint_key": "true", "foreign_key": "customers.id"},
            ),
        ])
        props = _build_ygg_properties(s, engine="sql")
        self.assertIn("ygg.schema[id]", props)
        self.assertNotIn("ygg.schema[fk_orders_customer]", props)
        self.assertEqual(props["ygg.field_count"], "1")

    def test_optional_keys_only_when_relevant(self) -> None:
        """Empty partitioning/cluster/PK collections drop their props."""
        s = Schema([field("id", "int64"), field("v", "string")])
        props = _build_ygg_properties(s, engine="sql")
        self.assertNotIn("ygg.partition_columns", props)
        self.assertNotIn("ygg.cluster_columns", props)
        self.assertNotIn("ygg.primary_keys", props)
        self.assertNotIn("ygg.storage_location", props)

    def test_format_resolves_to_ygg_mimetype(self) -> None:
        """Common storage flavors map to a registered ygg MimeType."""
        cases = [
            (DataSourceFormat.DELTA, MimeTypes.DELTA),
            (DataSourceFormat.PARQUET, MimeTypes.PARQUET),
            (DataSourceFormat.AVRO, MimeTypes.AVRO),
            (DataSourceFormat.ORC, MimeTypes.ORC),
            (DataSourceFormat.CSV, MimeTypes.CSV),
            (DataSourceFormat.JSON, MimeTypes.JSON),
            (DataSourceFormat.ICEBERG, MimeTypes.ICEBERG),
        ]
        for fmt, expected in cases:
            with self.subTest(format=fmt.value):
                self.assertIs(_resolve_format_mime(fmt), expected)

    def test_databricks_connector_formats_resolve(self) -> None:
        """Databricks-specific connector formats now have their own MimeType."""
        cases = [
            (DataSourceFormat.HIVE, MimeTypes.HIVE),
            (DataSourceFormat.DELTASHARING, MimeTypes.DELTASHARING),
            (DataSourceFormat.DELTA_UNIFORM_HUDI, MimeTypes.DELTA_UNIFORM_HUDI),
            (DataSourceFormat.DELTA_UNIFORM_ICEBERG, MimeTypes.DELTA_UNIFORM_ICEBERG),
            (DataSourceFormat.UNITY_CATALOG, MimeTypes.UNITY_CATALOG),
            (DataSourceFormat.DATABRICKS_FORMAT, MimeTypes.DATABRICKS_FORMAT),
            (DataSourceFormat.MYSQL_FORMAT, MimeTypes.MYSQL_FORMAT),
            (DataSourceFormat.POSTGRESQL_FORMAT, MimeTypes.POSTGRESQL_FORMAT),
            (DataSourceFormat.SNOWFLAKE_FORMAT, MimeTypes.SNOWFLAKE_FORMAT),
            (DataSourceFormat.BIGQUERY_FORMAT, MimeTypes.BIGQUERY_FORMAT),
            (DataSourceFormat.REDSHIFT_FORMAT, MimeTypes.REDSHIFT_FORMAT),
            (DataSourceFormat.VECTOR_INDEX_FORMAT, MimeTypes.VECTOR_INDEX_FORMAT),
        ]
        for fmt, expected in cases:
            with self.subTest(format=fmt.value):
                self.assertIs(_resolve_format_mime(fmt), expected)

    def test_every_data_source_format_resolves(self) -> None:
        """Coverage guard: no ``DataSourceFormat`` value should silently fall back."""
        uc_table = MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE
        unresolved: list[str] = []
        for fmt in DataSourceFormat:
            mime = _resolve_format_mime(fmt)
            # UNITY_CATALOG is a real category (foreign UC view), not the
            # default fallback, so the only legitimate hit on the UC-table
            # mime would mean a new enum value yggdrasil hasn't catalogued.
            if mime is uc_table and fmt is not DataSourceFormat.UNITY_CATALOG:
                unresolved.append(fmt.value)
        self.assertEqual(unresolved, [])

    def test_format_none_falls_back_to_uc_table_mime(self) -> None:
        self.assertIs(
            _resolve_format_mime(None),
            MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE,
        )

    def test_all_values_are_strings(self) -> None:
        """TBLPROPERTIES values must be strings on both create paths."""
        props = _build_ygg_properties(
            _schema(),
            engine="sql",
            data_source_format=DataSourceFormat.DELTA,
            table_type=TableType.MANAGED,
            storage_location="s3://b/p",
        )
        for k, v in props.items():
            self.assertIsInstance(k, str, f"non-string key: {k!r}")
            self.assertIsInstance(v, str, f"non-string value at {k!r}: {v!r}")
