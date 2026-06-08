"""Behaviors for :meth:`Table.safe_name` — the centralized
"raw string → Unity-Catalog-safe identifier" builder.

Every caller that needs to turn arbitrary text (URL paths, free-text
keys, composed names from upstream metadata) into a UC table name
routes through this single classmethod. The tests pin:

* the sanitization pipeline (lowercase, collapse non-alphanumeric runs,
  strip surrounding ``_``, fallback ``"root"`` for empty input);
* the 255-char Unity Catalog ceiling (via :func:`safe_table_name`);
* the ``logger.warning`` emitted when the input is rewritten — silent
  when the identifier is already safe, loud once when sanitization or
  truncation kicks in.
"""

from __future__ import annotations

import logging
import unittest

from yggdrasil.databricks.sql.sql_utils import MAX_TABLE_NAME_LEN
from yggdrasil.databricks.table.table import Table


class TestTableSafeName(unittest.TestCase):
    """Pure-python checks — no workspace, no skip."""

    # ── sanitization pipeline ────────────────────────────────────────────

    def test_already_safe_name_round_trips_unchanged(self) -> None:
        self.assertEqual(Table.safe_name("orders"), "orders")
        self.assertEqual(Table.safe_name("raw_orders_v2"), "raw_orders_v2")

    def test_empty_falls_back_to_root(self) -> None:
        self.assertEqual(Table.safe_name(""), "root")
        self.assertEqual(Table.safe_name("/"), "root")
        self.assertEqual(Table.safe_name(None), "root")

    def test_collapses_non_alphanumeric_runs(self) -> None:
        self.assertEqual(
            Table.safe_name("/api/v1/users.json?id=42"),
            "api_v1_users_json_id_42",
        )

    def test_lowercases_input(self) -> None:
        self.assertEqual(Table.safe_name("/Path/MixedCase"), "path_mixedcase")

    def test_strips_surrounding_underscores(self) -> None:
        # Pure separators collapse to ``_``; the strip catches the
        # leading/trailing artefacts so the result starts/ends on an
        # identifier character.
        self.assertEqual(Table.safe_name("/.._foo_../"), "foo")

    # ── length cap ───────────────────────────────────────────────────────

    def test_long_input_stays_within_uc_limit(self) -> None:
        long_path = "/" + "/".join("seg" + str(i) for i in range(200))
        out = Table.safe_name(long_path)
        self.assertLessEqual(len(out), MAX_TABLE_NAME_LEN)

    def test_distinct_overflows_get_distinct_digests(self) -> None:
        long_path = "/" + "/".join("seg" + str(i) for i in range(200))
        self.assertNotEqual(
            Table.safe_name(long_path),
            Table.safe_name(long_path + "X"),
        )

    # ── warning channel ──────────────────────────────────────────────────

    def test_safe_input_does_not_warn(self) -> None:
        """Steady-state path: a name that already conforms must not
        leak a warning on every call — that would drown real signal
        in any pipeline with a tight Table-by-name loop."""
        with self.assertLogs("yggdrasil.databricks.table.table", level="WARNING") as cap:
            Table.safe_name("already_safe")
            # ``assertLogs`` requires at least one record; emit a
            # bookkeeping marker so the context-manager body is valid,
            # then assert nothing else fired.
            logging.getLogger("yggdrasil.databricks.table.table").warning("marker")
        self.assertEqual(len(cap.records), 1)
        self.assertEqual(cap.records[0].getMessage(), "marker")

    def test_sanitization_warns_with_reason(self) -> None:
        with self.assertLogs("yggdrasil.databricks.table.table", level="WARNING") as cap:
            out = Table.safe_name("/api/v1/users")
        self.assertEqual(out, "api_v1_users")
        # Exactly one warning, naming both the input and the result so
        # an operator can trace where the rewrite came from.
        self.assertEqual(len(cap.records), 1)
        msg = cap.records[0].getMessage()
        self.assertIn("/api/v1/users", msg)
        self.assertIn("api_v1_users", msg)
        self.assertIn("non-identifier-chars", msg)

    def test_truncation_warns_with_reason(self) -> None:
        # Build a name that's already identifier-safe but blows past
        # the 255-char ceiling so the warning attributes the rewrite
        # to truncation, not sanitization.
        long_name = "a" * (MAX_TABLE_NAME_LEN + 50)
        with self.assertLogs("yggdrasil.databricks.table.table", level="WARNING") as cap:
            out = Table.safe_name(long_name)
        self.assertLessEqual(len(out), MAX_TABLE_NAME_LEN)
        self.assertEqual(len(cap.records), 1)
        self.assertIn("truncated", cap.records[0].getMessage())
