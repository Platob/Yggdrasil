"""Tests for the ``ygg.schema[<field>]`` TBLPROPERTIES emitted on table creation.

The same helper feeds both create paths
(:meth:`Table.sql_create` and :meth:`Table.api_create`) so the two
surfaces stay symmetric. Everything else a reader might want
(``table_type``, ``storage_location``, partition / cluster / primary
columns, ``data_source_format``) is already first-class on UC's
``TableInfo`` — the only field UC can't reconstruct is yggdrasil's
own per-field schema dump, so that's all this helper emits now.
"""
from __future__ import annotations

import json
import unittest

from yggdrasil.data.schema import Schema, field
from yggdrasil.databricks.table.table import (
    YGG_SCHEMA_FIELD_PREFIX,
    YGG_SCHEMA_FIELD_SUFFIX,
    _build_ygg_properties,
    _ygg_schema_key,
)


def _schema() -> Schema:
    return Schema([
        field("id", "int64", tags={"primary_key": "true"}),
        field("region", "string", tags={"partition_by": "true"}),
        field("bucket", "string", tags={"cluster_by": "true"}),
        field("amount", "decimal(18, 2)"),
    ])


class TestYggProperties(unittest.TestCase):

    def test_only_per_field_schema_keys_are_emitted(self) -> None:
        props = _build_ygg_properties(_schema())
        self.assertEqual(
            sorted(props),
            sorted([
                "ygg.schema[id]",
                "ygg.schema[region]",
                "ygg.schema[bucket]",
                "ygg.schema[amount]",
            ]),
        )

    def test_per_field_payload_shape(self) -> None:
        props = _build_ygg_properties(_schema())
        for key, value in props.items():
            payload = json.loads(value)
            unwrapped = key.removeprefix(YGG_SCHEMA_FIELD_PREFIX).removesuffix(
                YGG_SCHEMA_FIELD_SUFFIX
            )
            self.assertEqual(payload["name"], unwrapped)
            self.assertIn("dtype", payload)
            self.assertIn("nullable", payload)

    def test_schema_field_prefix_brackets_field_name(self) -> None:
        self.assertEqual(YGG_SCHEMA_FIELD_PREFIX, "ygg.schema[")
        self.assertEqual(YGG_SCHEMA_FIELD_SUFFIX, "]")
        self.assertEqual(_ygg_schema_key("amount"), "ygg.schema[amount]")
        # Bracket wrap survives field names containing the property
        # separator without colliding with adjacent keys.
        self.assertEqual(
            _ygg_schema_key("user.first_name"),
            "ygg.schema[user.first_name]",
        )

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
        props = _build_ygg_properties(s)
        self.assertEqual(list(props), ["ygg.schema[id]"])

    def test_all_values_are_strings(self) -> None:
        """TBLPROPERTIES values must be strings on both create paths."""
        props = _build_ygg_properties(_schema())
        for k, v in props.items():
            self.assertIsInstance(k, str, f"non-string key: {k!r}")
            self.assertIsInstance(v, str, f"non-string value at {k!r}: {v!r}")


if __name__ == "__main__":
    unittest.main()
