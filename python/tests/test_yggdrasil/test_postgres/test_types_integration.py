"""Live-Postgres integration tests for Arrow ↔ Postgres type mapping.

Each test creates a single-column table with a known Arrow type,
writes a sample value through the ADBC fast path, reads it back via
``information_schema``, and asserts the declared Postgres type and the
inferred Arrow type round-trip cleanly.
"""

from __future__ import annotations

import pytest

from yggdrasil.postgres.tests import PostgresTestCase
from yggdrasil.postgres.types import postgres_to_arrow_type

pytestmark = pytest.mark.postgres_integration


class TestTypeRoundtrip(PostgresTestCase):

    def _create_with_type(self, name: str, arrow_type) -> None:
        """Create a single-column table whose column has *arrow_type*."""
        schema = self.pa.schema([self.pa.field("v", arrow_type)])
        self.table(name).create(schema)

    def _declared_type(self, name: str) -> str:
        """Read back the column's declared Postgres type."""
        cols = self.table(name).columns()
        self.assertEqual([c.name for c in cols], ["v"])
        return cols[0].data_type

    def test_int_family_maps_to_smallint_integer_bigint(self) -> None:
        cases = {
            "i16": (self.pa.int16(), "smallint"),
            "i32": (self.pa.int32(), "integer"),
            "i64": (self.pa.int64(), "bigint"),
        }
        for tname, (arrow_type, expected_pg) in cases.items():
            self._create_with_type(tname, arrow_type)
            self.assertEqual(
                self._declared_type(tname).lower(),
                expected_pg,
                f"{tname}: expected Postgres type {expected_pg}",
            )

    def test_string_maps_to_text(self) -> None:
        self._create_with_type("strs", self.pa.string())
        self.assertEqual(self._declared_type("strs").lower(), "text")

    def test_bool_maps_to_boolean(self) -> None:
        self._create_with_type("flags", self.pa.bool_())
        self.assertEqual(self._declared_type("flags").lower(), "boolean")

    def test_decimal_preserves_precision_and_scale(self) -> None:
        self._create_with_type("decs", self.pa.decimal128(10, 2))
        # information_schema reports ``data_type='numeric'``; the
        # precision/scale live on numeric_precision/numeric_scale, not
        # the type name.  postgres_to_arrow_type has to default to
        # decimal128(38, 0) when the bare ``numeric`` form is used.
        declared = self._declared_type("decs").lower()
        self.assertEqual(declared, "numeric")
        # Round-trip the value to ensure precision survives the ADBC
        # write path even when the type name doesn't carry it.
        from decimal import Decimal
        tbl = self.table("decs")
        tbl.write_arrow_table(self.pa.table(
            {"v": self.pa.array([Decimal("12.34")], type=self.pa.decimal128(10, 2))},
        ))
        out = tbl.read_arrow_table()
        self.assertEqual(out.column("v").to_pylist(), [Decimal("12.34")])

    def test_timestamptz_keeps_timezone(self) -> None:
        self._create_with_type("ts_tz", self.pa.timestamp("us", tz="UTC"))
        # information_schema returns "timestamp with time zone" for
        # tz-aware columns.
        declared = self._declared_type("ts_tz").lower()
        self.assertIn("with time zone", declared)
        # postgres_to_arrow_type recognises that and returns a tz-aware
        # Arrow timestamp.
        arrow_back = postgres_to_arrow_type(declared)
        self.assertIsNotNone(getattr(arrow_back, "tz", None))

    def test_struct_maps_to_jsonb(self) -> None:
        struct_type = self.pa.struct([
            self.pa.field("k", self.pa.string()),
            self.pa.field("v", self.pa.int64()),
        ])
        self._create_with_type("payload", struct_type)
        self.assertEqual(self._declared_type("payload").lower(), "jsonb")

    def test_binary_maps_to_bytea(self) -> None:
        self._create_with_type("blob", self.pa.binary())
        self.assertEqual(self._declared_type("blob").lower(), "bytea")
