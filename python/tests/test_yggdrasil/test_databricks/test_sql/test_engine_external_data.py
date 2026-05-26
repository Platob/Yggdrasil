"""Unit tests for ``_coerce_external_data_for_spark``.

The helper normalizes ``external_data`` into per-alias
:class:`ExternalStatementData` entries that the Spark execution path
can register / substitute.  It mirrors the warehouse path's permissive
input contract so a single ``external_data=`` argument works in
either mode without coercion at the call site.

These tests do *not* require Databricks credentials — they exercise
pure Python coercion logic only.
"""
from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.enums import Scheme
from yggdrasil.data.statement import ExternalStatementData
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.sql.engine import _coerce_external_data_for_spark
from yggdrasil.io.tabular import ArrowTabular
from yggdrasil.url import URL


class TestCoerceExternalDataForSpark(unittest.TestCase):

    def _vp(self, path: str = "/cat/sch/vol/foo.parquet") -> VolumePath:
        return VolumePath(url=URL(scheme=Scheme.DATABRICKS_VOLUME, path=path))

    # ------------------------------------------------------------------
    # Empty / falsy inputs short-circuit to None
    # ------------------------------------------------------------------

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(_coerce_external_data_for_spark(None))

    def test_empty_mapping_returns_none(self) -> None:
        self.assertIsNone(_coerce_external_data_for_spark({}))

    # ------------------------------------------------------------------
    # VolumePath → text-substitute as parquet.`<full>`
    # ------------------------------------------------------------------

    def test_volume_path_substitutes_as_parquet_path(self) -> None:
        vp = self._vp("/cat/sch/vol/data.parquet")
        out = _coerce_external_data_for_spark({"alias": vp})
        self.assertIsNotNone(out)
        entry = out["alias"]
        self.assertEqual(entry.text_key, "alias")
        self.assertIsNone(entry.tabular)
        self.assertEqual(
            entry.text_value, "parquet.`/Volumes/cat/sch/vol/data.parquet`",
        )

    # ------------------------------------------------------------------
    # ExternalStatementData → pass-through (key normalization on mismatch)
    # ------------------------------------------------------------------

    def test_external_statement_data_passes_through(self) -> None:
        entry = ExternalStatementData(
            "alias", text_value="(VALUES (1)) AS t(id)",
        )
        out = _coerce_external_data_for_spark({"alias": entry})
        self.assertIs(out["alias"], entry)

    def test_external_statement_data_rebinds_when_key_differs(self) -> None:
        """If a caller stuffs an ``ExternalStatementData`` whose
        ``text_key`` doesn't match the dict key, the dict key wins —
        substitution is driven by the key everywhere downstream."""
        entry = ExternalStatementData("orig", text_value="t1")
        out = _coerce_external_data_for_spark({"alias": entry})
        self.assertIsNot(out["alias"], entry)
        self.assertEqual(out["alias"].text_key, "alias")
        self.assertEqual(out["alias"].text_value, "t1")

    # ------------------------------------------------------------------
    # Tabular → bound for temp-view registration
    # ------------------------------------------------------------------

    def test_tabular_is_bound_directly(self) -> None:
        data = pa.table({"id": [1, 2, 3]})
        tabular = ArrowTabular(data)
        out = _coerce_external_data_for_spark({"alias": tabular})
        self.assertIs(out["alias"].tabular, tabular)
        self.assertIsNone(out["alias"].text_value)

    # ------------------------------------------------------------------
    # str → text_value only (caller already staged it)
    # ------------------------------------------------------------------

    def test_string_is_text_value(self) -> None:
        out = _coerce_external_data_for_spark(
            {"alias": "(VALUES (1), (2)) AS t(id)"},
        )
        self.assertIsNone(out["alias"].tabular)
        self.assertEqual(
            out["alias"].text_value, "(VALUES (1), (2)) AS t(id)",
        )

    # ------------------------------------------------------------------
    # (tabular, text_value) tuple → both fields set
    # ------------------------------------------------------------------

    def test_tuple_pair_sets_both_fields(self) -> None:
        data = pa.table({"id": [1]})
        tabular = ArrowTabular(data)
        out = _coerce_external_data_for_spark(
            {"alias": (tabular, "view_x")},
        )
        self.assertIs(out["alias"].tabular, tabular)
        self.assertEqual(out["alias"].text_value, "view_x")

    # ------------------------------------------------------------------
    # Raw frames are wrapped in ArrowTabular so the Spark side has a
    # real Tabular to register.
    # ------------------------------------------------------------------

    def test_raw_arrow_table_is_wrapped_in_arrow_tabular(self) -> None:
        data = pa.table({"id": [10, 20, 30]})
        out = _coerce_external_data_for_spark({"alias": data})
        self.assertIsInstance(out["alias"].tabular, ArrowTabular)
        self.assertIsNone(out["alias"].text_value)

    def test_raw_record_batch_is_wrapped_in_arrow_tabular(self) -> None:
        batch = pa.record_batch([pa.array([1, 2])], names=["id"])
        out = _coerce_external_data_for_spark({"alias": batch})
        self.assertIsInstance(out["alias"].tabular, ArrowTabular)

    def test_unsupported_value_raises_typeerror_with_context(self) -> None:
        class WeirdNotAFrame:
            pass

        with self.assertRaises(TypeError) as ctx:
            _coerce_external_data_for_spark({"alias": WeirdNotAFrame()})
        msg = str(ctx.exception)
        self.assertIn("alias", msg)
        self.assertIn("WeirdNotAFrame", msg)

    # ------------------------------------------------------------------
    # Multi-alias dict: keys preserved, each value coerced independently
    # ------------------------------------------------------------------

    def test_multi_alias_mapping_coerces_each_value(self) -> None:
        vp = self._vp("/c/s/v/x.parquet")
        out = _coerce_external_data_for_spark(
            {
                "vol": vp,
                "raw": pa.table({"a": [1]}),
                "txt": "alias_text",
            }
        )
        self.assertEqual(set(out), {"vol", "raw", "txt"})
        self.assertTrue(out["vol"].text_value.startswith("parquet.`"))
        self.assertIsInstance(out["raw"].tabular, ArrowTabular)
        self.assertEqual(out["txt"].text_value, "alias_text")
