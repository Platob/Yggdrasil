"""MERGE / upsert strategies (match-by keys, partition pruning, latest-wins).

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.databricks.table.table import Table
from yggdrasil.enums import Mode

from ._sql_integration import SQLIntegrationCase


class TestSQLMergeStrategy(SQLIntegrationCase):
    """Each save-mode + ``match_by`` combination through ``Table.insert``.

    Reuses the catalog/schema fixture and per-test cleanup from
    :class:`TestSQLEngineIntegration` and adds a small helper to
    seed a fresh table with a known initial row set. The expected
    DML each branch generates is documented inline in
    ``yggdrasil.databricks.table.insert._build_dml_statements``; this
    suite verifies the *observable* outcome (row counts + values).
    """

    @staticmethod
    def _initial_data() -> pa.Table:
        """Three rows with keys 1/2/3 — the seed every test starts from."""
        return pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "label": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
                "amount": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
            }
        )

    def _seed(self, prefix: str) -> Table:
        """Build a unique table, create it, and seed with ``_initial_data``."""
        data = self._initial_data()
        table = self._unique_table(prefix)
        self._ensure_table(table, data.schema)
        # OVERWRITE so the seed is deterministic regardless of any
        # auto-create state the helper may have left behind.
        table.insert(data, mode=Mode.OVERWRITE)
        return table

    def _read_rows(self, table: Table) -> list[dict]:
        return self.engine.execute(
            f"SELECT id, label, amount FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

    # ------------------------------------------------------------------
    # MERGE — full UPSERT (update matching, insert new)
    # ------------------------------------------------------------------

    def test_merge_upsert_updates_matching_and_inserts_new(self) -> None:
        """``Mode.UPSERT`` + ``match_by=['id']`` → MERGE INTO with both
        WHEN MATCHED UPDATE and WHEN NOT MATCHED INSERT branches. Rows
        with overlapping keys take the source values; new keys land."""
        table = self._seed("merge_upsert")

        update = pa.table(
            {
                # id=2 overlaps (must be updated); id=4 is new (must be inserted).
                "id": pa.array([2, 4], type=pa.int64()),
                "label": pa.array(["beta-2", "delta"], type=pa.string()),
                "amount": pa.array([200.0, 40.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.UPSERT, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha", "amount": 10.0},
                {"id": 2, "label": "beta-2", "amount": 200.0},
                {"id": 3, "label": "gamma", "amount": 30.0},
                {"id": 4, "label": "delta", "amount": 40.0},
            ],
        )

    def test_merge_alias_matches_upsert(self) -> None:
        """``Mode.MERGE`` is treated as a synonym for ``UPSERT`` —
        same MERGE-with-update branch."""
        table = self._seed("merge_alias")

        update = pa.table(
            {
                "id": pa.array([1], type=pa.int64()),
                "label": pa.array(["alpha-merged"], type=pa.string()),
                "amount": pa.array([100.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.MERGE, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(rows[0], {"id": 1, "label": "alpha-merged", "amount": 100.0})
        self.assertEqual(len(rows), 3)

    # ------------------------------------------------------------------
    # MERGE — INSERT-ONLY branches
    # ------------------------------------------------------------------

    def test_merge_append_with_match_by_skips_existing_keys(self) -> None:
        """``Mode.APPEND`` + ``match_by`` → MERGE WHEN NOT MATCHED INSERT
        only; rows whose key already exists are dropped silently."""
        table = self._seed("merge_append_keyed")

        update = pa.table(
            {
                # id=2 already exists → skipped; id=5 is new → inserted.
                "id": pa.array([2, 5], type=pa.int64()),
                "label": pa.array(["beta-DROPPED", "epsilon"], type=pa.string()),
                "amount": pa.array([999.0, 50.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.APPEND, match_by=["id"])

        rows = self._read_rows(table)
        # Original beta still wins because APPEND+match_by never updates.
        self.assertEqual(rows[1], {"id": 2, "label": "beta", "amount": 20.0})
        self.assertEqual(
            rows[-1], {"id": 5, "label": "epsilon", "amount": 50.0}
        )
        self.assertEqual(len(rows), 4)

    def test_merge_auto_with_match_by_skips_existing_keys(self) -> None:
        """``Mode.AUTO`` + ``match_by`` behaves the same as APPEND —
        insert-only MERGE."""
        table = self._seed("merge_auto_keyed")

        update = pa.table(
            {
                "id": pa.array([3, 6], type=pa.int64()),
                "label": pa.array(["gamma-DROPPED", "zeta"], type=pa.string()),
                "amount": pa.array([999.0, 60.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.AUTO, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(rows[2], {"id": 3, "label": "gamma", "amount": 30.0})
        self.assertEqual(rows[-1], {"id": 6, "label": "zeta", "amount": 60.0})
        self.assertEqual(len(rows), 4)

    # ------------------------------------------------------------------
    # TRUNCATE + match_by → keyed DELETE then INSERT
    # ------------------------------------------------------------------

    def test_truncate_with_match_by_deletes_matching_then_inserts(self) -> None:
        table = self._seed("merge_truncate_keyed")

        update = pa.table(
            {
                # Wipe id=1 and id=3 (matched by key) and re-insert with new values.
                # id=2 is untouched because the source doesn't carry that key.
                "id": pa.array([1, 3], type=pa.int64()),
                "label": pa.array(["alpha-v2", "gamma-v2"], type=pa.string()),
                "amount": pa.array([11.0, 33.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.TRUNCATE, match_by=["id"])

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha-v2", "amount": 11.0},
                {"id": 2, "label": "beta", "amount": 20.0},
                {"id": 3, "label": "gamma-v2", "amount": 33.0},
            ],
        )

    def test_truncate_without_match_by_wipes_table(self) -> None:
        """Plain ``Mode.TRUNCATE`` without keys: full TRUNCATE + INSERT."""
        table = self._seed("merge_truncate_full")

        replacement = pa.table(
            {
                "id": pa.array([99], type=pa.int64()),
                "label": pa.array(["only"], type=pa.string()),
                "amount": pa.array([0.0], type=pa.float64()),
            }
        )


        table.insert(replacement, mode=Mode.TRUNCATE)

        rows = self._read_rows(table)
        self.assertEqual(rows, [{"id": 99, "label": "only", "amount": 0.0}])

    # ------------------------------------------------------------------
    # safe_merge=True — sidestep MERGE entirely (DELETE+INSERT semantics)
    # ------------------------------------------------------------------

    def test_safe_merge_upsert_round_trip(self) -> None:
        """``safe_merge=True`` + UPSERT runs keyed DELETE + INSERT
        instead of MERGE; outcome is identical for non-overlapping
        writers."""
        table = self._seed("safe_merge_upsert")

        update = pa.table(
            {
                "id": pa.array([2, 4], type=pa.int64()),
                "label": pa.array(["beta-safe", "delta-safe"], type=pa.string()),
                "amount": pa.array([222.0, 44.0], type=pa.float64()),
            }
        )
        table.insert(
            update, mode=Mode.UPSERT, match_by=["id"], safe_merge=True,
        )

        rows = self._read_rows(table)
        self.assertEqual(
            rows,
            [
                {"id": 1, "label": "alpha", "amount": 10.0},
                {"id": 2, "label": "beta-safe", "amount": 222.0},
                {"id": 3, "label": "gamma", "amount": 30.0},
                {"id": 4, "label": "delta-safe", "amount": 44.0},
            ],
        )

    def test_safe_merge_append_uses_anti_join_insert(self) -> None:
        """``safe_merge=True`` + APPEND runs ``INSERT ... WHERE NOT
        EXISTS`` against the target — same observable result as the
        native MERGE insert-only branch."""
        table = self._seed("safe_merge_append")

        update = pa.table(
            {
                "id": pa.array([1, 7], type=pa.int64()),
                "label": pa.array(["alpha-DROPPED", "eta"], type=pa.string()),
                "amount": pa.array([999.0, 70.0], type=pa.float64()),
            }
        )
        table.insert(
            update, mode=Mode.APPEND, match_by=["id"], safe_merge=True,
        )

        rows = self._read_rows(table)
        # id=1 untouched; id=7 added.
        self.assertEqual(rows[0], {"id": 1, "label": "alpha", "amount": 10.0})
        self.assertEqual(rows[-1], {"id": 7, "label": "eta", "amount": 70.0})
        self.assertEqual(len(rows), 4)

    # ------------------------------------------------------------------
    # Multi-key match_by
    # ------------------------------------------------------------------

    def test_merge_upsert_composite_key(self) -> None:
        """``match_by`` accepts multiple columns — MERGE ON joins on the
        full key tuple."""
        table = self._unique_table("merge_composite")
        seed = pa.table(
            {
                "tenant": pa.array(["a", "a", "b"], type=pa.string()),
                "id": pa.array([1, 2, 1], type=pa.int64()),
                "value": pa.array([10.0, 20.0, 100.0], type=pa.float64()),
            }
        )
        self._ensure_table(table, seed.schema)
        table.insert(seed, mode=Mode.OVERWRITE)

        update = pa.table(
            {
                # (a, 2) overlaps → updated; (b, 2) is new → inserted.
                "tenant": pa.array(["a", "b"], type=pa.string()),
                "id": pa.array([2, 2], type=pa.int64()),
                "value": pa.array([222.0, 200.0], type=pa.float64()),
            }
        )
        table.insert(update, mode=Mode.UPSERT, match_by=["tenant", "id"])

        rows = self.engine.execute(
            f"SELECT tenant, id, value FROM {table.full_name(safe=True)} "
            "ORDER BY tenant, id"
        ).to_arrow_table().to_pylist()
        self.assertEqual(
            rows,
            [
                {"tenant": "a", "id": 1, "value": 10.0},
                {"tenant": "a", "id": 2, "value": 222.0},
                {"tenant": "b", "id": 1, "value": 100.0},
                {"tenant": "b", "id": 2, "value": 200.0},
            ],
        )


# =====================================================================
# Schema fill — source columns are a subset of the target schema
# =====================================================================
