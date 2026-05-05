"""TabularIO integration tests against a local path and a mocked remote.

Pins the same surface — :class:`yggdrasil.io.buffer.base.TabularIO`
operations (``read_pandas_frame`` / ``write_arrow_table`` /
``read_polars_frame`` / ``write_pylist`` / …) — across two backends:

- :class:`yggdrasil.io.fs.local_path.LocalPath` — real files in a
  per-test tmpdir. Exercises the local-FS fast path where
  ``BytesIO._commit`` early-returns (writes hit the kernel via
  ``os.pwrite``) so there's no transaction-buffer flush at all.

- :class:`._inmemory_volume.InMemoryVolumePath` — in-memory backing
  swapped in at the SDK seam, so the path machinery uses the
  non-local code path (transaction buffer + commit-on-close) without
  needing a Databricks workspace. Lets the same assertions exercise
  the remote-write contract from a unit test.

Test methods live on :class:`_TabularIOPathMixin` so both backends
run them. The remote class wraps every ``test_write_*`` /
``test_*_round_trip`` with :func:`unittest.expectedFailure` because
the format-leaf write currently bypasses ``BytesIO.write`` and leaves
the transaction buffer's dirty bit clear; close → commit becomes a
no-op and nothing lands on the remote. See
:mod:`test_pandas_remote_path` for the full diagnosis. Removing the
``expectedFailure`` markers on the remote class once the flush bug
is fixed will surface ``unexpectedSuccess`` and force a cleanup pass.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

# Import the in-memory volume helper first — it pulls in
# ``yggdrasil.databricks.fs`` → ``yggdrasil.io.buffer`` → ``yggdrasil.io.fs``
# in the correct order. ``from yggdrasil.io.fs import LocalPath`` as the
# very first import of the test module deadlocks the buffer ↔ fs cycle.
from ._inmemory_volume import InMemoryVolumePath  # noqa: I001

from yggdrasil.io.fs import LocalPath
from yggdrasil.pandas.tests import PandasTestCase
from yggdrasil.polars.lib import polars as pl

# ---------------------------------------------------------------------------
# Mixin — backend-agnostic TabularIO assertions
# ---------------------------------------------------------------------------

class _TabularIOPathMixin:
    """Shared TabularIO round-trip suite.

    Subclasses provide :meth:`_make_path` (and may override
    :meth:`_seed_parquet` if direct seeding is needed for read tests
    on a backend whose write path is broken).

    The mixin is deliberately not a :class:`unittest.TestCase` —
    concrete subclasses combine it with ``PandasTestCase`` (which
    pulls in pandas+pyarrow auto-import and the ``self.df`` /
    ``assertFrameEqual`` helpers).
    """

    def _make_path(self, name: str):  # pragma: no cover - abstract
        raise NotImplementedError

    def _seed_parquet(self, p, table: pa.Table) -> int:
        """Default: write via the path itself.

        The remote subclass overrides this to bypass the (currently
        broken) write path and seed the in-memory store directly, so
        the read-side tests stay meaningful regardless of the
        write-side regression.
        """
        p.write_arrow_table(table)
        return p.size

    # ------------------------------------------------------------------
    # Arrow
    # ------------------------------------------------------------------

    def test_arrow_table_round_trip(self) -> None:
        """``write_arrow_table`` / ``read_arrow_table`` round-trip."""
        p = self._make_path("arrow_table.parquet")
        table = pa.table({
            "i": pa.array([1, 2, 3], type=pa.int64()),
            "s": pa.array(["a", "b", "c"], type=pa.string()),
        })

        p.write_arrow_table(table)
        got = p.read_arrow_table()

        self.assertEqual(got.to_pydict(), table.to_pydict())

    def test_arrow_batches_streaming(self) -> None:
        """Streaming write/read via ``write_arrow_batches`` /
        ``read_arrow_batches`` — same shape as the Arrow IPC and Spark
        loaders use under the hood."""
        p = self._make_path("arrow_batches.parquet")
        table = pa.table({"i": list(range(10))})
        # Force two batches to exercise the streaming path.
        batches = table.to_batches(max_chunksize=4)
        self.assertGreater(len(batches), 1)

        p.write_arrow_batches(iter(batches))
        got_batches = list(p.read_arrow_batches())

        self.assertGreaterEqual(len(got_batches), 1)
        roundtripped = pa.Table.from_batches(got_batches)
        self.assertEqual(roundtripped.to_pydict(), table.to_pydict())

    # ------------------------------------------------------------------
    # Pandas
    # ------------------------------------------------------------------

    def test_pandas_frame_round_trip(self) -> None:
        p = self._make_path("pandas.parquet")
        df = self.df({
            "i": [1, 2, 3, 4],
            "s": ["a", "b", "c", "d"],
            "f": [1.5, 2.5, 3.5, 4.5],
        })

        p.write_pandas_frame(df)
        got = p.read_pandas_frame()

        self.assertFrameEqual(got, df)

    def test_pandas_to_pandas_alias(self) -> None:
        """``to_pandas`` is the documented alias for
        ``read_pandas_frame`` — pin the parity so refactors don't
        silently drop one side."""
        p = self._make_path("to_pandas.parquet")
        df = self.df({"a": [10, 20, 30]})
        # Seed via the helper so this test doesn't depend on the
        # write path also working.
        self._seed_parquet(p, pa.Table.from_pandas(df, preserve_index=False))

        got = p.to_pandas()
        self.assertFrameEqual(got, df)

    # ------------------------------------------------------------------
    # Polars
    # ------------------------------------------------------------------

    def test_polars_frame_round_trip(self) -> None:
        
        p = self._make_path("polars.parquet")
        df = pl.DataFrame({
            "i": [1, 2, 3],
            "s": ["x", "y", "z"],
        })

        p.write_polars_frame(df)
        got = pl.from_arrow(p.read_arrow_table())

        # Compare via Arrow to sidestep polars-version dtype quirks.
        self.assertEqual(got.to_dict(as_series=False), df.to_dict(as_series=False))

    def test_polars_lazy_scan(self) -> None:
        """``scan_polars_frame`` returns a :class:`pl.LazyFrame` over
        the path; collecting it yields the original rows."""
        
        p = self._make_path("polars_lazy.parquet")
        df = pl.DataFrame({"i": [1, 2, 3]})
        # Seed via Arrow so this test isn't blocked by the polars
        # write path on remote.
        self._seed_parquet(p, df.to_arrow())

        scanned = p.scan_polars_frame()
        self.assertIsInstance(scanned, pl.LazyFrame)

        materialized = scanned.collect()
        self.assertEqual(
            materialized.to_dict(as_series=False),
            df.to_dict(as_series=False),
        )

    # ------------------------------------------------------------------
    # pylist / pydict
    # ------------------------------------------------------------------

    def test_pylist_round_trip(self) -> None:
        p = self._make_path("pylist.parquet")
        rows = [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]

        p.write_pylist(rows)
        got = p.read_pylist()

        self.assertEqual(got, rows)

    def test_pydict_round_trip(self) -> None:
        p = self._make_path("pydict.parquet")
        data = {"id": [1, 2, 3], "name": ["a", "b", "c"]}

        p.write_pydict(data)
        got = p.read_pydict()

        self.assertEqual(got, data)

    # ------------------------------------------------------------------
    # Cross-format reads (write parquet, read pandas / polars / pylist)
    # ------------------------------------------------------------------

    def test_read_after_seed_pandas(self) -> None:
        """Path seeded with parquet bytes is readable as pandas."""
        p = self._make_path("seed_then_pandas.parquet")
        df = self.df({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        self._seed_parquet(p, pa.Table.from_pandas(df, preserve_index=False))

        got = p.read_pandas_frame()
        self.assertFrameEqual(got, df)

    def test_read_after_seed_polars(self) -> None:
        
        p = self._make_path("seed_then_polars.parquet")
        df = pl.DataFrame({"a": [1, 2, 3]})
        self._seed_parquet(p, df.to_arrow())

        got = pl.from_arrow(p.read_arrow_table())
        self.assertEqual(got.to_dict(as_series=False), df.to_dict(as_series=False))

    def test_read_after_seed_pylist(self) -> None:
        p = self._make_path("seed_then_pylist.parquet")
        rows = [{"a": 1}, {"a": 2}]
        self._seed_parquet(p, pa.Table.from_pylist(rows))

        got = p.read_pylist()
        self.assertEqual(got, rows)

# ---------------------------------------------------------------------------
# LocalPath — real files in a tmpdir
# ---------------------------------------------------------------------------

class TestTabularIOLocalPath(_TabularIOPathMixin, PandasTestCase):
    """Run the TabularIO suite against a real local-FS path.

    Local paths are the "happy path" for the buffer machinery: writes
    go straight to the file via ``os.pwrite`` and ``BytesIO._commit``
    early-returns on ``is_local``, so no transaction-buffer flush is
    needed. Anything that breaks here breaks everywhere.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tabularto_tmp = tempfile.mkdtemp(prefix="ygg-tabulario-local-")

    def tearDown(self) -> None:
        shutil.rmtree(self._tabularto_tmp, ignore_errors=True)
        super().tearDown()

    def _make_path(self, name: str) -> LocalPath:
        return LocalPath(os.path.join(self._tabularto_tmp, name))

# ---------------------------------------------------------------------------
# Mocked remote VolumePath — non-local code path, in-memory backing
# ---------------------------------------------------------------------------

def _xfail_inherited(*method_names: str):
    """Class decorator: replace each named inherited test method on
    the subclass with an :func:`unittest.expectedFailure`-wrapped
    version, so writing the override boilerplate doesn't dwarf the
    test bodies on the mixin.
    """
    def deco(cls):
        for name in method_names:
            orig = getattr(cls, name, None)
            if orig is None:
                raise AttributeError(
                    f"{cls.__name__} has no inherited method {name!r} "
                    "to mark as expectedFailure"
                )

            def make_wrapper(method):
                @unittest.expectedFailure
                def _wrapped(self, *args, **kwargs):
                    return method(self, *args, **kwargs)
                _wrapped.__name__ = method.__name__
                _wrapped.__qualname__ = f"{cls.__name__}.{method.__name__}"
                _wrapped.__doc__ = method.__doc__
                return _wrapped

            setattr(cls, name, make_wrapper(orig))
        return cls
    return deco

@_xfail_inherited(
    "test_arrow_table_round_trip",
    "test_arrow_batches_streaming",
    "test_pandas_frame_round_trip",
    "test_polars_frame_round_trip",
    "test_pylist_round_trip",
    "test_pydict_round_trip",
)
class TestTabularIORemoteMock(_TabularIOPathMixin, PandasTestCase):
    """Run the TabularIO suite against a mocked managed remote path.

    Same assertions as the local class, but routed through the
    non-local path machinery (``open_io`` → BytesIO transaction buffer
    → ``write_stream`` on flush). The write tests are currently
    expected to fail; the read tests pass because they seed the store
    directly via :meth:`_seed_parquet` instead of going through the
    broken write path.
    """

    def setUp(self) -> None:
        super().setUp()
        InMemoryVolumePath.reset()

    def _make_path(self, name: str) -> InMemoryVolumePath:
        return InMemoryVolumePath(f"/Volumes/cat/sch/vol/{name}")

    def _seed_parquet(self, p, table: pa.Table) -> int:
        """Direct seed — bypasses the broken write path so the
        read-side tests still have meaningful coverage."""
        import io as _io
        sink = _io.BytesIO()
        pq.write_table(table, sink)
        payload = sink.getvalue()
        InMemoryVolumePath._STORE[p.full_path()] = payload
        return len(payload)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
