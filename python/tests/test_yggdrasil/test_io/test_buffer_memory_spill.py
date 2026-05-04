"""Auto-spill tests for :class:`MemoryArrowIO`.

The holder consolidates batches to a local Arrow IPC file once the
in-memory footprint crosses ``spill_bytes`` and re-attaches via
:func:`pyarrow.memory_map` for zero-copy reads. Tests pin both the
state-machine transitions (in-memory → spilled → spilled-with-tail
→ re-spilled) and the cleanup contract (owned spill files unlinked
on close; caller-supplied paths preserved).
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.io.buffer.memory import MemoryArrowIO
from yggdrasil.io.enums import Mode


def _wide_table(rows: int, *, value: int = 1) -> pa.Table:
    """Build a table large enough to cross small spill thresholds.

    Three int64 columns × ``rows`` rows ≈ 24 ``rows`` bytes — easy
    knob for tests that want to control whether the holder spills.
    """
    return pa.table({
        "a": pa.array([value] * rows, type=pa.int64()),
        "b": pa.array([value + 1] * rows, type=pa.int64()),
        "c": pa.array([value + 2] * rows, type=pa.int64()),
    })


class TestSpillBasics(unittest.TestCase):
    def test_small_payload_stays_in_memory(self) -> None:
        io = MemoryArrowIO(_wide_table(8))
        try:
            self.assertFalse(io.spilled)
            self.assertEqual(io.num_rows, 8)
        finally:
            io.close()

    def test_threshold_triggers_spill(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            self.assertTrue(io.spilled)
            # Spill file lives on disk and is one of ours.
            self.assertIsNotNone(io._spill_path)
            path = pathlib.Path(str(io._spill_path))
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".arrow")
            # In-memory tail is empty post-consolidation.
            self.assertEqual(len(io._batches), 0)
            # Reads still serve every row.
            out = io.read_arrow_table()
            self.assertEqual(out.num_rows, 10_000)
        finally:
            io.close()

    def test_disable_spill_with_zero_threshold(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=0)
        try:
            self.assertFalse(io.spilled)
            self.assertEqual(len(io._batches), 1)
        finally:
            io.close()


class TestAppendAfterSpill(unittest.TestCase):
    def test_small_append_keeps_in_memory_tail(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            self.assertTrue(io.spilled)
            io.write_arrow_table(
                _wide_table(5),
                options=CastOptions(mode=Mode.APPEND),
            )
            # Tail too small to re-spill — sits in memory next to the
            # spilled mmap-backed table.
            self.assertTrue(io.spilled)
            self.assertEqual(len(io._batches), 1)
            self.assertEqual(io.num_rows, 10_005)
            self.assertEqual(io.read_arrow_table().num_rows, 10_005)
        finally:
            io.close()

    def test_large_append_re_spills(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            first_path = str(io._spill_path)
            io.write_arrow_table(
                _wide_table(20_000, value=42),
                options=CastOptions(mode=Mode.APPEND),
            )
            self.assertTrue(io.spilled)
            self.assertEqual(len(io._batches), 0)
            self.assertEqual(io.num_rows, 30_000)
            # New consolidated file on a different path; old one
            # unlinked.
            self.assertNotEqual(str(io._spill_path), first_path)
            self.assertFalse(pathlib.Path(first_path).exists())
        finally:
            io.close()

    def test_overwrite_after_spill_clears_state(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            old_path = str(io._spill_path)
            io.write_arrow_table(
                _wide_table(3),
                options=CastOptions(mode=Mode.OVERWRITE),
            )
            self.assertFalse(io.spilled)
            self.assertEqual(io.num_rows, 3)
            self.assertFalse(pathlib.Path(old_path).exists())
        finally:
            io.close()


class TestCleanupContract(unittest.TestCase):
    def test_owned_spill_unlinks_on_close(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        path = pathlib.Path(str(io._spill_path))
        self.assertTrue(path.exists())
        io.close()
        self.assertFalse(path.exists())

    def test_caller_supplied_path_preserved_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "stash.arrow")
            io = MemoryArrowIO(
                _wide_table(10_000),
                spill_bytes=1024,
                spill_path=target,
            )
            try:
                self.assertTrue(io.spilled)
                self.assertEqual(str(io._spill_path), target)
                self.assertFalse(io._owns_spill_path)
                self.assertTrue(os.path.exists(target))
            finally:
                io.close()
            # Caller's file survives close — they own it.
            self.assertTrue(os.path.exists(target))

    def test_caller_path_rewritten_on_re_spill(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = os.path.join(td, "stash.arrow")
            io = MemoryArrowIO(
                _wide_table(10_000),
                spill_bytes=1024,
                spill_path=target,
            )
            try:
                first_size = os.path.getsize(target)
                io.write_arrow_table(
                    _wide_table(20_000, value=7),
                    options=CastOptions(mode=Mode.APPEND),
                )
                # Same path, larger payload.
                self.assertEqual(str(io._spill_path), target)
                self.assertGreater(os.path.getsize(target), first_size)
                self.assertEqual(io.num_rows, 30_000)
            finally:
                io.close()


class TestReadEngineSurfaces(unittest.TestCase):
    """Spilled holders still serve every TabularIO read path."""

    def test_read_polars_frame_after_spill(self) -> None:
        try:
            import polars  # noqa: F401
        except ImportError:
            self.skipTest("polars not installed")
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            self.assertTrue(io.spilled)
            df = io.read_polars_frame()
            self.assertEqual(df.height, 10_000)
        finally:
            io.close()

    def test_read_arrow_batches_yields_spilled_then_tail(self) -> None:
        io = MemoryArrowIO(_wide_table(10_000), spill_bytes=1024)
        try:
            io.write_arrow_table(
                _wide_table(5, value=99),
                options=CastOptions(mode=Mode.APPEND),
            )
            batches = list(io.read_arrow_batches())
            total = sum(b.num_rows for b in batches)
            self.assertEqual(total, 10_005)
        finally:
            io.close()


if __name__ == "__main__":
    unittest.main()
