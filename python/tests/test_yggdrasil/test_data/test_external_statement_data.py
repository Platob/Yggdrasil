"""Unit tests for :class:`ExternalStatementData` and the
:meth:`PreparedStatement.apply_external_substitution` chokepoint.

These tests do not require a live Databricks workspace or a running
SparkSession; they exercise the dataclass shape, the coercion helper
on :class:`PreparedStatement`, and the substitution semantics that
every engine relies on.
"""

from __future__ import annotations

import unittest

import pyarrow as pa

from yggdrasil.data.statement import (
    ExternalStatementData,
    PreparedStatement,
)
from yggdrasil.io.tabular import ArrowTabular


def _arrow_tabular() -> ArrowTabular:
    table = pa.table({"id": [1, 2], "name": ["a", "b"]})
    return ArrowTabular(table)


class TestExternalStatementDataConstruction(unittest.TestCase):

    def test_text_value_only_is_allowed(self) -> None:
        entry = ExternalStatementData("foo", text_value="my_table")
        self.assertEqual(entry.text_key, "foo")
        self.assertEqual(entry.text_value, "my_table")
        self.assertIsNone(entry.tabular)

    def test_tabular_only_defers_text_value(self) -> None:
        tab = _arrow_tabular()
        entry = ExternalStatementData("foo", tabular=tab)
        self.assertEqual(entry.text_key, "foo")
        self.assertIsNone(entry.text_value)
        self.assertIs(entry.tabular, tab)

    def test_neither_tabular_nor_text_value_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one of"):
            ExternalStatementData("foo")

    def test_blank_text_key_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            ExternalStatementData("", text_value="t")

    def test_invalid_identifier_text_key_raises(self) -> None:
        # Bracket / hyphen / dot would all break ``{name}`` substitution.
        for bad in ("a-b", "a b", "1abc", "$x"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "valid identifier"):
                    ExternalStatementData(bad, text_value="t")

    def test_from_passes_through_existing_instance(self) -> None:
        entry = ExternalStatementData("foo", text_value="t")
        self.assertIs(ExternalStatementData.from_(entry), entry)

    def test_from_wraps_tabular_under_text_key(self) -> None:
        tab = _arrow_tabular()
        entry = ExternalStatementData.from_(tab, text_key="x")
        self.assertEqual(entry.text_key, "x")
        self.assertIs(entry.tabular, tab)
        self.assertIsNone(entry.text_value)

    def test_from_wraps_string_as_text_value(self) -> None:
        entry = ExternalStatementData.from_("VALUES (1, 2)", text_key="x")
        self.assertEqual(entry.text_value, "VALUES (1, 2)")
        self.assertIsNone(entry.tabular)

    def test_from_requires_text_key_for_non_instance(self) -> None:
        with self.assertRaisesRegex(ValueError, "text_key is required"):
            ExternalStatementData.from_("VALUES (1)")


class TestPreparedStatementExternalData(unittest.TestCase):
    """``PreparedStatement.__init__`` accepts a flexible ``external_data``
    mapping and normalizes every value into an
    :class:`ExternalStatementData`."""

    def test_dict_of_text_values_is_accepted(self) -> None:
        stmt = PreparedStatement(
            "SELECT * FROM {a}",
            external_data={"a": "VALUES (1)"},
        )
        self.assertIn("a", stmt.external_data)
        self.assertEqual(stmt.external_data["a"].text_value, "VALUES (1)")

    def test_dict_of_tabulars_is_accepted(self) -> None:
        tab = _arrow_tabular()
        stmt = PreparedStatement(
            "SELECT * FROM {a}",
            external_data={"a": tab},
        )
        self.assertIs(stmt.external_data["a"].tabular, tab)
        self.assertIsNone(stmt.external_data["a"].text_value)

    def test_explicit_external_statement_data_passes_through(self) -> None:
        entry = ExternalStatementData("a", text_value="VALUES (2)")
        stmt = PreparedStatement(
            "SELECT * FROM {a}",
            external_data={"a": entry},
        )
        self.assertIs(stmt.external_data["a"], entry)

    def test_dict_key_overrides_entry_key_on_mismatch(self) -> None:
        # If the caller built an ExternalStatementData under one name but
        # placed it under a different mapping key, the mapping key wins:
        # substitution is driven by the dict key everywhere downstream.
        entry = ExternalStatementData("typo", text_value="VALUES (1)")
        stmt = PreparedStatement(
            "SELECT * FROM {a}",
            external_data={"a": entry},
        )
        self.assertEqual(stmt.external_data["a"].text_key, "a")
        self.assertEqual(stmt.external_data["a"].text_value, "VALUES (1)")

    def test_empty_external_data_yields_none(self) -> None:
        stmt = PreparedStatement("SELECT 1", external_data={})
        self.assertIsNone(stmt.external_data)


class TestApplyExternalSubstitution(unittest.TestCase):

    def test_no_external_data_returns_text_unchanged(self) -> None:
        out = PreparedStatement.apply_external_substitution(
            "SELECT * FROM t", None,
        )
        self.assertEqual(out, "SELECT * FROM t")

    def test_substitutes_every_alias(self) -> None:
        external = {
            "a": ExternalStatementData("a", text_value="left_view"),
            "b": ExternalStatementData("b", text_value="right_view"),
        }
        out = PreparedStatement.apply_external_substitution(
            "SELECT * FROM {a} JOIN {b} USING (id)",
            external,
        )
        self.assertEqual(
            out, "SELECT * FROM left_view JOIN right_view USING (id)",
        )

    def test_unset_text_value_raises(self) -> None:
        # ``text_value`` is None until the engine materializes the binding
        # — substitution must not silently leave a placeholder behind.
        external = {"a": ExternalStatementData("a", tabular=_arrow_tabular())}
        with self.assertRaisesRegex(ValueError, "text_value is unset"):
            PreparedStatement.apply_external_substitution(
                "SELECT * FROM {a}", external,
            )

    def test_substitution_does_not_mutate_text(self) -> None:
        # Sanity: helper returns a new string; doesn't touch any input.
        text = "SELECT * FROM {a}"
        external = {"a": ExternalStatementData("a", text_value="my_view")}
        out = PreparedStatement.apply_external_substitution(text, external)
        self.assertEqual(text, "SELECT * FROM {a}")
        self.assertEqual(out, "SELECT * FROM my_view")

    def test_substitution_replaces_every_occurrence(self) -> None:
        external = {"a": ExternalStatementData("a", text_value="t")}
        out = PreparedStatement.apply_external_substitution(
            "SELECT * FROM {a} UNION ALL SELECT * FROM {a}", external,
        )
        self.assertEqual(out, "SELECT * FROM t UNION ALL SELECT * FROM t")


class TestPreparedStatementCrossClassCoercion(unittest.TestCase):
    """``PreparedStatement.from_`` carries ``external_data`` across when
    coercing one subclass into another."""

    class _OtherPreparedStatement(PreparedStatement):
        pass

    def test_external_data_preserved_on_cross_class_coerce(self) -> None:
        src = PreparedStatement(
            "SELECT * FROM {a}",
            external_data={"a": ExternalStatementData("a", text_value="t")},
        )
        coerced = self._OtherPreparedStatement.from_(src)
        self.assertIsInstance(coerced, self._OtherPreparedStatement)
        self.assertEqual(coerced.text, src.text)
        self.assertEqual(set(coerced.external_data), {"a"})
        self.assertEqual(coerced.external_data["a"].text_value, "t")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
