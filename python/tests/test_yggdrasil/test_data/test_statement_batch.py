"""Unit tests for ``StatementBatch`` pooling and lifecycle methods."""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass, field

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.statement import Statement as BaseStatement, StatementBatch


@dataclass
class _StubStatement(BaseStatement):
    """Minimal ``Statement`` used to exercise batch orchestration."""

    name: str = ""
    fail: bool = False
    sleep_on_start: float = 0.0
    _started: bool = field(default=False, init=False)
    _cancelled: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)

    @property
    def done(self) -> bool:
        return self._finished or self._cancelled

    @property
    def failed(self) -> bool:
        return self.fail

    def raise_for_status(self) -> None:
        if self.fail:
            raise RuntimeError(f"{self.name} failed")

    def refresh_status(self) -> None:
        return None

    def start(self, *, wait=True, raise_error=True, **_kwargs) -> "_StubStatement":
        self._started = True
        if self.sleep_on_start:
            time.sleep(self.sleep_on_start)
        if self.fail:
            self._finished = True
            if raise_error:
                raise RuntimeError(f"{self.name} failed")
            return self
        self._finished = True
        return self

    def cancel(self) -> "_StubStatement":
        self._cancelled = True
        return self

    def collect_schema(self, full: bool = False) -> Schema:
        return Schema.from_any_fields([], metadata={})

    def to_arrow_reader(self, **_: object) -> pa.RecordBatchReader:  # pragma: no cover
        raise NotImplementedError


class TestResolveParallel(unittest.TestCase):
    def test_false_maps_to_one(self):
        self.assertEqual(StatementBatch._resolve_parallel(False), 1)

    def test_zero_maps_to_one(self):
        self.assertEqual(StatementBatch._resolve_parallel(0), 1)

    def test_true_maps_to_four(self):
        self.assertEqual(StatementBatch._resolve_parallel(True), 4)

    def test_int_is_preserved(self):
        self.assertEqual(StatementBatch._resolve_parallel(3), 3)

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            StatementBatch._resolve_parallel("2")


class TestStatementBatchStart(unittest.TestCase):
    def _build(self, n: int = 3, **kwargs):
        stmts = [_StubStatement(name=f"s{i}", **kwargs) for i in range(n)]
        return StatementBatch.from_results(stmts), stmts

    def test_sequential_start_calls_each(self):
        batch, stmts = self._build(3)
        batch.start(parallel=False)
        for s in stmts:
            self.assertTrue(s._started)
            self.assertTrue(s._finished)

    def test_pooled_start_completes_every_statement(self):
        batch, stmts = self._build(6, sleep_on_start=0.01)
        batch.start(parallel=2)
        batch.wait()
        for s in stmts:
            self.assertTrue(s._finished)

    def test_pool_caps_concurrency(self):
        running = 0
        peak = 0
        lock = threading.Lock()

        @dataclass
        class _TrackingStatement(_StubStatement):
            def start(self, *, wait=True, raise_error=True, **_kwargs):
                nonlocal running, peak
                with lock:
                    running += 1
                    peak = max(peak, running)
                try:
                    time.sleep(0.02)
                finally:
                    with lock:
                        running -= 1
                self._started = True
                self._finished = True
                return self

        stmts = [_TrackingStatement(name=f"s{i}") for i in range(8)]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=3)
        batch.wait()

        self.assertLessEqual(peak, 3)
        self.assertGreaterEqual(peak, 2)

    def test_start_twice_raises(self):
        batch, _ = self._build(2, sleep_on_start=0.01)
        batch.start(parallel=2)
        with self.assertRaises(RuntimeError):
            batch.start(parallel=2)
        batch.wait()


class TestStatementBatchErrorPropagation(unittest.TestCase):
    def test_pool_error_cancels_remaining(self):
        stmts = [
            _StubStatement(name="ok0", sleep_on_start=0.02),
            _StubStatement(name="boom", fail=True),
            _StubStatement(name="slow", sleep_on_start=1.0),
            _StubStatement(name="slower", sleep_on_start=1.0),
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2)

        with self.assertRaises(RuntimeError):
            batch.wait()

        self.assertTrue(stmts[2]._cancelled or stmts[3]._cancelled)

    def test_sequential_error_cancels_later_statements(self):
        stmts = [
            _StubStatement(name="ok"),
            _StubStatement(name="boom", fail=True),
            _StubStatement(name="after1"),
            _StubStatement(name="after2"),
        ]
        batch = StatementBatch.from_results(stmts)

        with self.assertRaises(RuntimeError):
            batch.start(parallel=False)

        self.assertTrue(stmts[0]._finished)
        self.assertTrue(stmts[2]._cancelled)
        self.assertTrue(stmts[3]._cancelled)

    def test_raise_error_false_swallows_pool_errors(self):
        stmts = [
            _StubStatement(name="ok"),
            _StubStatement(name="boom", fail=True),
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2, raise_error=False)
        # Returning cleanly is the assertion here.
        batch.wait(raise_error=False)


class TestStatementBatchCancel(unittest.TestCase):
    def test_cancel_calls_each_statement(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(3)]
        batch = StatementBatch.from_results(stmts)
        batch.cancel()
        for s in stmts:
            self.assertTrue(s._cancelled)

    def test_cancel_tears_down_pool(self):
        stmts = [
            _StubStatement(name=f"s{i}", sleep_on_start=0.5) for i in range(4)
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2)
        batch.cancel()
        self.assertIsNone(batch._executor)
        self.assertEqual(len(batch._futures), 0)


class TestStatementBatchWaitNoPool(unittest.TestCase):
    def test_wait_without_start_iterates_sequentially(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(3)]
        for s in stmts:
            s._finished = True
        batch = StatementBatch.from_results(stmts)
        # No pool active — wait should return cleanly.
        batch.wait(wait=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
