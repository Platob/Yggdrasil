"""Pandas read/write round-trip tests against a managed remote path.

Targets :class:`yggdrasil.databricks.fs.volume_path.VolumePath` — the
``/Volumes/<cat>/<sch>/<vol>/...`` UC-managed namespace that
``DatabricksClient.dbfs_path()`` returns for managed paths.

Why we don't use ``DatabricksCase`` here
----------------------------------------
The integration suite in :mod:`tests.test_yggdrasil.test_databricks.fs.test_integration`
gates on ``DATABRICKS_HOST`` so it only runs against a live workspace.
That's appropriate for end-to-end coverage, but the *path code itself*
(``open`` → BytesIO transaction buffer → format dispatch by
extension → flush via ``write_stream``) has no Databricks-specific
behaviour beyond the SDK transport hooks. We exercise it offline by
plugging an in-memory store into the SDK seam, so this file runs in
the standard unit-test pass and CI can catch regressions without a
workspace.

The accompanying :class:`BenchmarkPandasRemotePath` times the same
round-trip across a few payload sizes; it's gated by
``YGG_BENCH_REMOTE_PATH=1`` so the regular suite stays fast.
"""

from __future__ import annotations

import os
import time
import unittest
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from yggdrasil.pandas.tests import PandasTestCase

from ._inmemory_volume import InMemoryVolumePath as _InMemoryVolumePath


# ---------------------------------------------------------------------------
# Round-trip — write_pandas_frame / read_pandas_frame
# ---------------------------------------------------------------------------


class TestPandasRemotePath(PandasTestCase):
    """Round-trip ``write_pandas_frame`` / ``read_pandas_frame`` on a
    managed remote path.

    These pin the contract that ``DatabricksClient.dbfs_path(...)``
    consumers rely on: a path returned from the workspace can be
    written to with ``write_pandas_frame`` and round-tripped with
    ``read_pandas_frame`` (the same shape the deleted ``DatabricksIO``
    classes used to expose). Regression here would surface to every
    caller using the path's ``TabularIO`` surface for tabular IO.
    """

    def setUp(self) -> None:
        super().setUp()
        _InMemoryVolumePath.reset()

    def _path(self, name: str) -> _InMemoryVolumePath:
        return _InMemoryVolumePath(f"/Volumes/cat/sch/vol/{name}")

    def _seed_parquet(
        self, p: _InMemoryVolumePath, df,
    ) -> None:
        """Pre-populate the in-memory store with parquet bytes.

        Used by the read-only tests so they don't depend on the write
        path also working — the two are exercised separately.
        """
        import io as _io
        sink = _io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), sink)
        _InMemoryVolumePath._STORE[p.full_path()] = sink.getvalue()

    # ---- read_pandas_frame --------------------------------------------

    def test_read_pandas_frame_parquet(self) -> None:
        """Reading parquet from the volume returns the original frame."""
        p = self._path("read_basic.parquet")
        df = self.df({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self._seed_parquet(p, df)

        got = p.read_pandas_frame()

        self.assertFrameEqual(got, df)
        self.assertEqual(_InMemoryVolumePath.download_count, 1)

    def test_read_pandas_frame_passes_kwargs(self) -> None:
        """``**kwargs`` flow through to the read path without raising."""
        p = self._path("read_kwargs.parquet")
        df = self.df({"a": [10, 20, 30]})
        self._seed_parquet(p, df)

        got = p.read_pandas_frame(columns=["a"])
        self.assertFrameEqual(got, df)

    def test_read_pandas_frame_missing_path(self) -> None:
        """Reading a path that was never written produces an empty
        DataFrame (the framework's standard "no batches → empty table"
        contract from :meth:`TabularIO._read_arrow_table`)."""
        p = self._path("never_written.parquet")
        got = p.read_pandas_frame()
        self.assertEqual(len(got), 0)

    # ---- write_pandas_frame --------------------------------------------
    #
    # The Path/Holder refactor wired ``Path.pwrite`` so writes against a
    # non-local backing land in the transaction :class:`BytesIO`,
    # whose ``dirty`` bit drives ``close`` → :meth:`Path._pwrite`
    # commit. End-to-end round trips therefore work through the SDK
    # seam without any per-format leaf glue.

    def test_write_pandas_frame_uploads(self) -> None:
        """``write_pandas_frame(df)`` must actually push bytes to the
        remote store. Regression test for a flush path that left the
        transaction buffer's dirty bit unset, so close became a no-op
        and the upload silently dropped."""
        p = self._path("write_uploads.parquet")
        df = self.df({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        p.write_pandas_frame(df)

        self.assertIn(
            p.full_path(),
            _InMemoryVolumePath._STORE,
            "write_pandas_frame did not upload — the remote store is "
            "empty after the call. Most likely cause: the BytesIO "
            "transaction buffer was not marked dirty by the format "
            "leaf's _write_arrow_batches, so close() → commit() bailed "
            "out before reaching write_stream.",
        )
        payload = _InMemoryVolumePath._STORE[p.full_path()]
        self.assertGreater(len(payload), 0)
        self.assertEqual(payload[:4], b"PAR1", "expected parquet magic")
        self.assertGreaterEqual(_InMemoryVolumePath.upload_count, 1)

    def test_write_then_read_round_trip(self) -> None:
        """End-to-end: write_pandas_frame → read_pandas_frame returns
        the same frame, going through the live SDK seam each way."""
        p = self._path("round_trip.parquet")
        df = self.df({
            "i": [1, 2, 3, 4],
            "s": ["a", "b", "c", "d"],
            "f": [1.5, 2.5, 3.5, 4.5],
        })

        p.write_pandas_frame(df)
        got = p.read_pandas_frame()

        self.assertFrameEqual(got, df)

    def test_write_overwrites_existing(self) -> None:
        """A second ``write_pandas_frame`` replaces the prior payload —
        the contract is OVERWRITE, not APPEND."""
        p = self._path("overwrite.parquet")
        first = self.df({"a": [1, 2]})
        second = self.df({"a": [10, 20, 30]})

        p.write_pandas_frame(first)
        p.write_pandas_frame(second)

        got = p.read_pandas_frame()
        self.assertFrameEqual(got, second)

    def test_write_empty_frame(self) -> None:
        """Writing an empty DataFrame is allowed and round-trips with
        zero rows. Schemas are preserved so a downstream reader sees the
        column shape, not an empty dict."""
        p = self._path("empty.parquet")
        df = self.df({"a": pa.array([], type=pa.int64()).to_pandas(),
                      "b": pa.array([], type=pa.string()).to_pandas()})

        p.write_pandas_frame(df)

        # Either zero uploads (write was elided) or a single upload of
        # a header-only parquet — both are reasonable. What's NOT OK
        # is writing some other shape than the input.
        if p.full_path() in _InMemoryVolumePath._STORE:
            got = p.read_pandas_frame()
            self.assertEqual(len(got), 0)
            self.assertEqual(list(got.columns), list(df.columns))


# ---------------------------------------------------------------------------
# Benchmark — opt-in via YGG_BENCH_REMOTE_PATH=1
# ---------------------------------------------------------------------------


def _bench_enabled() -> bool:
    return os.environ.get("YGG_BENCH_REMOTE_PATH", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


@unittest.skipUnless(
    _bench_enabled(),
    "Set YGG_BENCH_REMOTE_PATH=1 to run the managed-remote-path "
    "pandas-IO benchmark.",
)
class BenchmarkPandasRemotePath(PandasTestCase):
    """Time ``write_pandas_frame`` / ``read_pandas_frame`` round-trips
    on a managed remote path across several payload sizes.

    Uses the in-memory ``_InMemoryVolumePath`` so the timing reflects
    framework overhead (Arrow batch construction, parquet encode,
    transaction-buffer flush, write_stream marshal) rather than network
    latency — that's what we actually want to measure for regressions
    in the path-IO machinery. Live-network timing belongs in a separate
    integration benchmark.

    Output is printed via ``self._print``; pytest captures it by default,
    pass ``-s`` to see it.
    """

    # Row counts to sweep. Kept modest so the benchmark stays under a
    # few seconds even under CI jitter.
    SIZES: ClassVar[tuple] = (1_000, 10_000, 100_000)

    def setUp(self) -> None:
        super().setUp()
        _InMemoryVolumePath.reset()

    def _print(self, line: str) -> None:
        # unittest test runners send stdout through capture; use
        # write+flush so ``-s`` shows it live.
        import sys
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _make_frame(self, n: int):
        pd = self.pd
        return pd.DataFrame({
            "id": range(n),
            "value": [float(i) * 0.5 for i in range(n)],
            "label": [f"row-{i}" for i in range(n)],
        })

    def _seed_parquet(self, p: _InMemoryVolumePath, df) -> int:
        """Direct in-memory seeding — bypasses ``write_pandas_frame`` so
        the read benchmark stays useful even when the write path is
        broken (see ``TestPandasRemotePath.test_write_pandas_frame_uploads``).
        """
        import io as _io
        sink = _io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), sink)
        payload = sink.getvalue()
        _InMemoryVolumePath._STORE[p.full_path()] = payload
        return len(payload)

    def test_round_trip_sweep(self) -> None:
        self._print(
            f"{'rows':>10} | {'write (ms)':>11} | {'read (ms)':>11} | "
            f"{'bytes':>10} | {'write MB/s':>11} | {'read MB/s':>10}"
        )
        self._print("-" * 78)

        for n in self.SIZES:
            df = self._make_frame(n)
            p = _InMemoryVolumePath(f"/Volumes/cat/sch/vol/bench_{n}.parquet")

            t0 = time.perf_counter()
            p.write_pandas_frame(df)
            t_write = time.perf_counter() - t0

            payload = _InMemoryVolumePath._STORE.get(p.full_path(), b"")
            if not payload:
                # Write path silently dropped (see comment on
                # TestPandasRemotePath); seed directly so the read
                # benchmark still measures something useful.
                size = self._seed_parquet(p, df)
            else:
                size = len(payload)

            t0 = time.perf_counter()
            got = p.read_pandas_frame()
            t_read = time.perf_counter() - t0

            self.assertEqual(len(got), n)

            mb = size / (1024 * 1024)
            w_mbps = mb / t_write if t_write > 0 else float("inf")
            r_mbps = mb / t_read if t_read > 0 else float("inf")
            self._print(
                f"{n:>10} | {t_write * 1000:>11.2f} | {t_read * 1000:>11.2f} | "
                f"{size:>10} | {w_mbps:>11.2f} | {r_mbps:>10.2f}"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
