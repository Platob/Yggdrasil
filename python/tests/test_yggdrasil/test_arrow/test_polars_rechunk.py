"""Tests for :func:`yggdrasil.polars.cast.rechunk_polars_frames`.

Mirrors the byte/row-size cases exercised against
:func:`yggdrasil.arrow.cast.rechunk_arrow_batches` in
``test_arrow/test_cast.py``, plus polars-specific concerns
(lazy input/output, frame-type preservation, multi-frame coalesce).
"""
from __future__ import annotations

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.polars.cast import rechunk_polars_frame, rechunk_polars_frames
from yggdrasil.polars.tests import PolarsTestCase


class TestRechunkPolarsFrames(PolarsTestCase, ArrowTestCase):
    def _df(self, n: int):
        return self.df({"a": list(range(n))})

    def test_passthrough_when_no_knobs(self) -> None:
        df = self._df(4)
        out = list(rechunk_polars_frames([df]))
        self.assertEqual([f.height for f in out], [4])
        self.assertIsInstance(out[0], self.pl.DataFrame)

    def test_row_size_only_slices(self) -> None:
        out = list(rechunk_polars_frames([self._df(7)], row_size=3))
        self.assertEqual([f.height for f in out], [3, 3, 1])

    def test_row_size_drops_empty_frames(self) -> None:
        empty = self.df({"a": []}, schema={"a": self.pl.Int64})
        out = list(rechunk_polars_frames([empty, self._df(3)], row_size=2))
        self.assertEqual([f.height for f in out], [2, 1])

    def test_byte_size_emits_chunks(self) -> None:
        out = list(rechunk_polars_frames([self._df(100)], byte_size=64))
        self.assertEqual(sum(f.height for f in out), 100)
        self.assertGreater(len(out), 1)

    def test_byte_and_row_caps_pick_minimum(self) -> None:
        out = list(rechunk_polars_frames([self._df(10)], byte_size=10_000, row_size=4))
        self.assertEqual([f.height for f in out], [4, 4, 2])

    def test_concat_buffers_under_target(self) -> None:
        small = [self._df(1) for _ in range(5)]
        out = list(rechunk_polars_frames(small, byte_size=10_000))
        self.assertEqual(sum(f.height for f in out), 5)
        self.assertEqual(len(out), 1)

    def test_lazy_input_is_collected(self) -> None:
        lf = self.lazy({"a": list(range(7))})
        out = list(rechunk_polars_frames([lf], row_size=2))
        self.assertEqual([f.height for f in out], [2, 2, 2, 1])
        for f in out:
            self.assertIsInstance(f, self.pl.DataFrame)

    def test_lazy_output(self) -> None:
        out = list(rechunk_polars_frames([self._df(7)], row_size=3, lazy=True))
        self.assertEqual([type(f).__name__ for f in out], ["LazyFrame"] * 3)
        self.assertEqual([f.collect().height for f in out], [3, 3, 1])

    def test_single_frame_helper(self) -> None:
        out = list(rechunk_polars_frame(self._df(11), row_size=3))
        self.assertEqual([f.height for f in out], [3, 3, 3, 2])

    def test_round_trip_preserves_values(self) -> None:
        df = self.df({"a": list(range(10)), "b": [f"v{i}" for i in range(10)]})
        out = list(rechunk_polars_frames([df], row_size=4))
        reassembled = self.pl.concat(out)
        self.assertFrameEqual(reassembled, df)
