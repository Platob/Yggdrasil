"""Insert column-alignment: missing columns filled, extra columns rejected / projected.

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.databricks.table.table import Table
from yggdrasil.enums import Mode

from ._sql_integration import LOGGER, SQLIntegrationCase


class TestSQLInsertFillMissingColumns(SQLIntegrationCase):
    """Insert source rows whose schema is a strict subset of the target.

    The Arrow staging cast (``cast_arrow_tabular`` in
    ``yggdrasil.data.types.nested.struct_arrow``) materialises a default
    array for every target column the source doesn't carry, so the
    staged Parquet — and therefore the ``INSERT INTO target (cols)
    SELECT ...`` projection that ``arrow_insert`` builds against the
    target schema — always has a value for every target column. These
    tests pin that contract by inserting subset payloads through the
    public ``Table.insert`` API and reading the rows back: missing
    columns must land as ``NULL`` for inserted rows, and the
    ``update_column_names`` knob must protect target-only columns on
    MERGE-matched rows.
    """

    @staticmethod
    def _wide_schema() -> pa.Schema:
        """Target schema with one key + three nullable payload columns.

        The key (``id``) is non-nullable so the tests can fail loudly
        if the missing-column fill ever leaks onto the key.
        """
        return pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("label", pa.string()),
                pa.field("amount", pa.float64()),
                pa.field("active", pa.bool_()),
            ]
        )

    def _read_rows(self, table: Table) -> list[dict]:
        return self.engine.execute(
            f"SELECT id, label, amount, active "
            f"FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_arrow_table().to_pylist()

    # ------------------------------------------------------------------
    # APPEND — fill one missing column with NULL
    # ------------------------------------------------------------------

    def test_append_fills_single_missing_column_with_null(self) -> None:
        """Source carries id+label+active; target also has ``amount``.
        Inserted rows must land with ``amount = NULL``."""
        table = self._unique_table("fill_one_append")
        table.ensure_created(self._wide_schema())

        partial = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["a", "b"], type=pa.string()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(partial, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "a", "amount": None, "active": True},
                {"id": 2, "label": "b", "amount": None, "active": False},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND — fill multiple missing columns with NULL
    # ------------------------------------------------------------------

    def test_append_fills_multiple_missing_columns_with_null(self) -> None:
        """Source carries the key column only — every payload column
        on the target must land as ``NULL``."""
        table = self._unique_table("fill_many_append")
        table.ensure_created(self._wide_schema())

        partial = pa.table(
            {"id": pa.array([10, 20], type=pa.int64())}
        )
        table.insert(partial, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 10, "label": None, "amount": None, "active": None},
                {"id": 20, "label": None, "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND — source columns in a different order than target
    # ------------------------------------------------------------------

    def test_append_reorders_source_columns_to_target_schema(self) -> None:
        """Cast is by name, not by position — a reordered + truncated
        source still lines up with the target columns."""
        table = self._unique_table("fill_reorder_append")
        table.ensure_created(self._wide_schema())

        reordered = pa.table(
            {
                # Intentionally label, id (reversed) and no amount/active.
                "label": pa.array(["x", "y"], type=pa.string()),
                "id": pa.array([100, 200], type=pa.int64()),
            }
        )
        table.insert(reordered, mode=Mode.APPEND)

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 100, "label": "x", "amount": None, "active": None},
                {"id": 200, "label": "y", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # OVERWRITE — fill missing columns on a replace
    # ------------------------------------------------------------------

    def test_overwrite_fills_missing_columns_with_null(self) -> None:
        """``Mode.OVERWRITE`` drops the target before insert; the
        subset replacement must keep the wider target schema and fill
        the absent columns with NULL."""
        table = self._unique_table("fill_overwrite")
        wide = self._wide_schema()
        LOGGER.debug(
            "Creating wide target for overwrite test %r (schema_columns=%d, schema_names=%r)",
            table, len(wide.names), wide.names,
        )
        table.ensure_created(wide)
        LOGGER.debug(
            "Created wide target for overwrite test %r (remote_columns=%r)",
            table, table.collect_schema().names,
        )

        # Seed so the OVERWRITE actually replaces something.
        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["seed-a", "seed-b"], type=pa.string()),
                "amount": pa.array([1.5, 2.5], type=pa.float64()),
                "active": pa.array([True, True], type=pa.bool_()),
            }
        )
        LOGGER.debug(
            "Seeding overwrite test %r (mode=OVERWRITE, rows=%d, source_names=%r)",
            table, seed.num_rows, seed.schema.names,
        )
        table.insert(seed, mode=Mode.OVERWRITE)
        LOGGER.debug(
            "Seeded overwrite test %r (remote_columns=%r, rows=%r)",
            table, table.collect_schema().names, self._read_rows(table),
        )

        partial = pa.table(
            {
                "id": pa.array([5, 6], type=pa.int64()),
                "label": pa.array(["fresh-a", "fresh-b"], type=pa.string()),
            }
        )
        LOGGER.debug(
            "Overwriting with partial source %r (mode=OVERWRITE, rows=%d, source_names=%r, missing=%r)",
            table, partial.num_rows, partial.schema.names,
            [n for n in wide.names if n not in partial.schema.names],
        )
        table.insert(partial, mode=Mode.OVERWRITE)

        result = self.engine.execute(
            f"SELECT id, label FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_pylist()

        self.assertEqual(
            result,
            [
                {"id": 5, "label": "fresh-a"},
                {"id": 6, "label": "fresh-b"},
            ],
        )

    # ------------------------------------------------------------------
    # UPSERT — match by key, missing source columns become NULL on the
    # matched rows (incoming wins, full row replaced)
    # ------------------------------------------------------------------

    def test_upsert_missing_columns_become_null_on_match(self) -> None:
        """MERGE ... WHEN MATCHED THEN UPDATE SET T.col = S.col touches
        every non-key column. A source missing ``amount`` / ``active``
        therefore overwrites the matched row's payload columns with
        the NULL the staging cast filled in — the documented MERGE
        semantics for callers that didn't scope updates with
        ``update_column_names``."""
        table = self._unique_table("fill_upsert")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        # id=2 overlaps → updated; id=3 is new → inserted. Both should
        # land with ``amount = NULL`` and ``active = NULL`` because the
        # source doesn't carry those columns.
        partial = pa.table(
            {
                "id": pa.array([2, 3], type=pa.int64()),
                "label": pa.array(["beta-2", "gamma"], type=pa.string()),
            }
        )
        table.insert(partial, mode=Mode.UPSERT, match_by=["id"])

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                {"id": 2, "label": "beta-2", "amount": None, "active": None},
                {"id": 3, "label": "gamma", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # UPSERT + update_column_names — protect target-only payload columns
    # ------------------------------------------------------------------

    def test_upsert_update_column_names_protects_target_only_columns(self) -> None:
        """``update_column_names=['label']`` restricts the MERGE UPDATE
        SET clause to just ``label``. Matched rows keep their existing
        ``amount`` / ``active`` even though the source is missing them;
        unmatched rows still land with the staged NULL fills."""
        table = self._unique_table("fill_upsert_scoped")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, False], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        partial = pa.table(
            {
                "id": pa.array([2, 3], type=pa.int64()),
                "label": pa.array(["beta-2", "gamma"], type=pa.string()),
            }
        )
        table.insert(
            partial,
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=["label"],
        )

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                # id=2: label updated; amount/active preserved by scope.
                {"id": 2, "label": "beta-2", "amount": 20.0, "active": False},
                # id=3: new row → NULL fill stands.
                {"id": 3, "label": "gamma", "amount": None, "active": None},
            ],
        )

    # ------------------------------------------------------------------
    # APPEND + match_by — insert-only MERGE drops keyed dupes; survivors
    # carry the NULL fill for missing columns
    # ------------------------------------------------------------------

    def test_keyed_append_fills_missing_columns_for_new_rows(self) -> None:
        """Insert-only MERGE: existing keys are skipped, new keys land
        with the staged NULL fills for columns the source omitted."""
        table = self._unique_table("fill_append_keyed")
        table.ensure_created(self._wide_schema())

        seed = pa.table(
            {
                "id": pa.array([1, 2], type=pa.int64()),
                "label": pa.array(["alpha", "beta"], type=pa.string()),
                "amount": pa.array([10.0, 20.0], type=pa.float64()),
                "active": pa.array([True, True], type=pa.bool_()),
            }
        )
        table.insert(seed, mode=Mode.OVERWRITE)

        partial = pa.table(
            {
                # id=2 already exists → skipped (existing row kept).
                # id=5 is new → inserted with NULL amount / active.
                "id": pa.array([2, 5], type=pa.int64()),
                "label": pa.array(["beta-DROPPED", "epsilon"], type=pa.string()),
            }
        )
        table.insert(partial, mode=Mode.APPEND, match_by=["id"])

        self.assertEqual(
            self._read_rows(table),
            [
                {"id": 1, "label": "alpha", "amount": 10.0, "active": True},
                # Original row wins on key conflict.
                {"id": 2, "label": "beta", "amount": 20.0, "active": True},
                # New row carries the staged NULL fill.
                {"id": 5, "label": "epsilon", "amount": None, "active": None},
            ],
        )


# =====================================================================
# Concurrent writes
# =====================================================================
