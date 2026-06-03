"""Live integration for :meth:`WarehouseStatementResult._scan_polars_frame`.

The mocked suite (``test_warehouse_polars_lazy.py``) stubs the Statement
Execution API and the external-link HTTP session.  This suite drives the
*real* warehouse path end to end: a SQL statement runs on the configured
SQL warehouse, results come back over ``EXTERNAL_LINKS`` as Arrow IPC
chunks, and the autonomous :class:`~polars.LazyFrame` re-streams those
chunks on each collect with projection / predicate / row-count pushdown.

``engine="api"`` is forced on every execute so the result is always a
:class:`WarehouseStatementResult` (the lazy override lives there), never
the Spark path.

Skipped unless ``DATABRICKS_HOST`` (and credentials) are set — see
:class:`SQLIntegrationCase`.
"""
from __future__ import annotations

import polars as pl

from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult

from ._sql_integration import SQLIntegrationCase


# Four-row fixture mirroring the mocked suite: a bigint ``id`` and a
# string ``name``, materialized inline so the test owns no table.
_FOUR_ROWS = (
    "SELECT * FROM VALUES "
    "(CAST(1 AS BIGINT), 'a'), "
    "(CAST(2 AS BIGINT), 'b'), "
    "(CAST(3 AS BIGINT), 'c'), "
    "(CAST(4 AS BIGINT), 'd') AS t(id, name)"
)


class TestWarehousePolarsLazyIntegration(SQLIntegrationCase):
    """Autonomous polars LazyFrame over a live warehouse result."""

    def _result(self, text: str) -> WarehouseStatementResult:
        """Run *text* on the warehouse (API engine) and return the result.

        Asserts the concrete type so a Spark-path regression (which has
        no lazy override) fails loudly rather than silently exercising
        the base ``scan_pyarrow_dataset``.
        """
        result = self.engine.execute(text, engine="api")
        self.assertIsInstance(result, WarehouseStatementResult)
        return result

    # ------------------------------------------------------------------
    # Scan construction is lazy
    # ------------------------------------------------------------------

    def test_scan_returns_lazyframe_with_schema(self) -> None:
        lf = self._result(_FOUR_ROWS).scan_polars_frame()

        self.assertIsInstance(lf, pl.LazyFrame)
        self.assertEqual(lf.collect_schema().names(), ["id", "name"])

    # ------------------------------------------------------------------
    # Collect materializes the streamed rows
    # ------------------------------------------------------------------

    def test_collect_materializes_rows(self) -> None:
        df = self._result(_FOUR_ROWS).scan_polars_frame().collect()

        self.assertEqual(df.shape, (4, 2))
        self.assertEqual(df["id"].to_list(), [1, 2, 3, 4])
        self.assertEqual(df["name"].to_list(), ["a", "b", "c", "d"])

    def test_lazyframe_is_re_collectable(self) -> None:
        """The base ``scan_pyarrow_dataset`` path drains on first collect;
        the autonomous frame re-streams the external-link chunks on every
        collect, so two collects of the same frame agree."""
        lf = self._result(_FOUR_ROWS).scan_polars_frame()

        first = lf.collect()
        second = lf.collect()

        self.assertTrue(first.equals(second))
        self.assertEqual(first["id"].to_list(), [1, 2, 3, 4])

    # ------------------------------------------------------------------
    # Pushdown — projection / predicate / row-count
    # ------------------------------------------------------------------

    def test_projection_pushdown(self) -> None:
        df = self._result(_FOUR_ROWS).scan_polars_frame().select("id").collect()

        self.assertEqual(df.columns, ["id"])
        self.assertEqual(df["id"].to_list(), [1, 2, 3, 4])

    def test_predicate_pushdown(self) -> None:
        df = (
            self._result(_FOUR_ROWS)
            .scan_polars_frame()
            .filter(pl.col("id") >= 3)
            .collect()
        )

        self.assertEqual(df["id"].to_list(), [3, 4])

    def test_n_rows_pushdown(self) -> None:
        df = self._result(_FOUR_ROWS).scan_polars_frame().head(2).collect()

        self.assertEqual(df["id"].to_list(), [1, 2])

    def test_composes_in_a_larger_plan(self) -> None:
        df = (
            self._result(_FOUR_ROWS)
            .scan_polars_frame()
            .filter(pl.col("id") % 2 == 0)
            .select("name")
            .collect()
        )

        self.assertEqual(df["name"].to_list(), ["b", "d"])

    # ------------------------------------------------------------------
    # Empty result still carries the manifest schema
    # ------------------------------------------------------------------

    def test_empty_result_yields_schema_bearing_empty_frame(self) -> None:
        df = self._result(_FOUR_ROWS + " WHERE 1 = 0").scan_polars_frame().collect()

        self.assertEqual(df.height, 0)
        self.assertEqual(df.columns, ["id", "name"])

    # ------------------------------------------------------------------
    # Multi-chunk: a result large enough to span several external links
    # still streams in full, and pushdown short-circuits the fetch.
    # ------------------------------------------------------------------

    def test_large_result_streams_all_rows(self) -> None:
        n = 250_000
        text = f"SELECT id, CAST(id AS STRING) AS name FROM range(0, {n})"

        lf = self._result(text).scan_polars_frame()

        self.assertEqual(lf.collect().height, n)

        # ``head`` stops streaming once the row budget is met — the frame
        # must still be re-collectable after an early-terminated collect.
        head = lf.head(5).collect()
        self.assertEqual(head["id"].to_list(), [0, 1, 2, 3, 4])

    def test_head_does_not_drain_the_whole_stream(self) -> None:
        """A tiny ``head`` over a large result must *fetch* only a little.

        Correctness alone (``head(5) == [0..4]``) doesn't prove the frame
        streamed lazily — the base ``scan_pyarrow_dataset`` reader would
        also return the right 5 rows after eagerly pulling every chunk.
        This asserts the *fetch* short-circuits: we count how many Arrow
        batches ``_read_arrow_batches`` actually pulls from the warehouse
        and require ``head(5)`` to pull far fewer than a full collect.

        ``byte_size`` is forced small so the 250k-row result flushes as
        many batches (the default 32 MiB cap would buffer the whole ~4 MiB
        result into a single batch, hiding the early-termination signal).
        """
        n = 250_000
        text = f"SELECT id, CAST(id AS STRING) AS name FROM range(0, {n})"
        result = self._result(text)

        # Count the batches the lazy generator pulls. The closure in
        # ``_scan_polars_frame`` reads ``self._read_arrow_batches`` at
        # collect time, so shadowing the bound method on the instance with
        # a counting wrapper instruments every pull — and chaining through
        # the original generator keeps the executor's cancel-on-close fetch
        # behaviour intact.
        original = result._read_arrow_batches
        pulled = 0

        def counting(options):
            nonlocal pulled
            for batch in original(options):
                pulled += 1
                yield batch

        result._read_arrow_batches = counting

        # Small flush size → the full result spans many batches.
        flush_bytes = 256 * 1024
        head = result.scan_polars_frame(byte_size=flush_bytes).head(5).collect()
        self.assertEqual(head["id"].to_list(), [0, 1, 2, 3, 4])
        pulled_for_head = pulled

        pulled = 0
        full = result.scan_polars_frame(byte_size=flush_bytes).collect()
        self.assertEqual(full.height, n)
        pulled_for_full = pulled

        # The full collect spans many batches; the head pulls only the
        # first one (or two, on a flush boundary) before the row budget is
        # met and the stream is cancelled.
        self.assertGreaterEqual(pulled_for_full, 5)
        self.assertLessEqual(pulled_for_head, 2)
        self.assertLess(pulled_for_head, pulled_for_full)

    def test_eager_read_polars_frame_still_works(self) -> None:
        """The eager ``read_polars_frame`` surface is unaffected by the
        lazy override."""
        df = self._result(_FOUR_ROWS).read_polars_frame()

        self.assertEqual(df.shape, (4, 2))
        self.assertEqual(df["id"].to_list(), [1, 2, 3, 4])
