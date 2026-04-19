"""Unit tests for ``StatementBatch`` pooling and lifecycle methods."""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass, field

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult as BaseStatementResult,
)


@dataclass
class _StubStatement(BaseStatementResult):
    """Deterministic ``StatementResult`` that models submit + poll semantics.

    ``start`` records submission and a completion deadline; ``refresh_status``
    flips ``done`` once the deadline passes, so the batch's polling loop can
    drive the statement to a terminal state without any real backend.
    """

    name: str = ""
    fail: bool = False
    duration: float = 0.0
    _submitted_at: float = field(default=0.0, init=False)
    _started: bool = field(default=False, init=False)
    _cancelled: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def done(self) -> bool:
        return self._finished or self._cancelled

    @property
    def failed(self) -> bool:
        return self._finished and self.fail

    def raise_for_status(self) -> None:
        if self._finished and self.fail:
            raise RuntimeError(f"{self.name} failed")

    def refresh_status(self) -> None:
        if self._cancelled or self._finished:
            return
        if self._started and (time.time() - self._submitted_at) >= self.duration:
            self._finished = True

    def start(self, *, wait=False, raise_error=True, **_kwargs) -> "_StubStatement":
        if self._started:
            return self
        self._started = True
        self._submitted_at = time.time()
        if self.duration <= 0:
            self._finished = True
        if wait:
            self.wait(wait=wait, raise_error=raise_error)
        return self

    def cancel(self) -> "_StubStatement":
        if not self._finished:
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

    def test_sequential_start_runs_to_completion(self):
        batch, stmts = self._build(3)
        batch.start(parallel=False)
        for s in stmts:
            self.assertTrue(s._started)
            self.assertTrue(s._finished)

    def test_sequential_waits_all_but_last_when_wait_false(self):
        """``wait=False`` only affects the final statement; earlier ones wait."""
        # Non-zero duration + wait=False means the statement does NOT reach a
        # terminal state from a single ``start`` call — ``wait`` must be True
        # for that to happen.  We use this to probe which statements actually
        # got a ``wait(wait=True)``.
        stmts = [_StubStatement(name=f"s{i}", duration=0.01) for i in range(3)]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=False, wait=False)

        # All submitted.
        for s in stmts:
            self.assertTrue(s._started)

        # Earlier two MUST be fully done (batch forced wait=True).
        self.assertTrue(stmts[0]._finished)
        self.assertTrue(stmts[1]._finished)

        # The last one respected wait=False — still in-flight on the "backend".
        self.assertFalse(stmts[2]._finished)

    def test_pooled_start_fills_window_only(self):
        batch, stmts = self._build(6, duration=0.05)
        batch.start(parallel=2)
        # Only the first two are submitted; the rest wait in the queue.
        self.assertTrue(stmts[0]._started)
        self.assertTrue(stmts[1]._started)
        self.assertFalse(stmts[2]._started)
        self.assertFalse(stmts[3]._started)
        self.assertEqual(len(batch._in_flight), 2)
        self.assertEqual(len(batch._pending_queue), 4)

        batch.wait(wait={"interval": 0.005, "max_interval": 0.02, "timeout": 10})
        for s in stmts:
            self.assertTrue(s._started)
            self.assertTrue(s._finished)
        self.assertEqual(len(batch._in_flight), 0)
        self.assertEqual(len(batch._pending_queue), 0)

    def test_pool_caps_in_flight(self):
        stmts = [
            _StubStatement(name=f"s{i}", duration=0.03) for i in range(8)
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=3)

        self.assertLessEqual(len(batch._in_flight), 3)
        started_count = sum(1 for s in stmts if s._started)
        self.assertEqual(started_count, 3)

        batch.wait(wait={"interval": 0.005, "max_interval": 0.02, "timeout": 10})
        for s in stmts:
            self.assertTrue(s._finished)

    def test_start_with_wait_true_drains_inline(self):
        batch, stmts = self._build(4, duration=0.02)
        batch.start(parallel=2, wait={"interval": 0.005, "timeout": 10})
        for s in stmts:
            self.assertTrue(s._finished)
        self.assertEqual(len(batch._in_flight), 0)
        self.assertEqual(len(batch._pending_queue), 0)

    def test_start_twice_raises(self):
        batch, _ = self._build(3, duration=0.1)
        batch.start(parallel=2)
        try:
            with self.assertRaises(RuntimeError):
                batch.start(parallel=2)
        finally:
            batch.cancel()


class TestStatementBatchErrorPropagation(unittest.TestCase):
    def test_pool_error_cancels_remaining(self):
        stmts = [
            _StubStatement(name="ok0", duration=0.03),
            _StubStatement(name="boom", fail=True, duration=0.0),
            _StubStatement(name="queued1", duration=1.0),
            _StubStatement(name="queued2", duration=1.0),
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2)

        with self.assertRaises(RuntimeError):
            batch.wait(wait={"interval": 0.005, "timeout": 10})

        # Queued entries should never have been submitted.
        self.assertFalse(stmts[2]._started)
        self.assertFalse(stmts[3]._started)
        # Their cancel() should still have been called as cleanup.
        self.assertTrue(stmts[2]._cancelled)
        self.assertTrue(stmts[3]._cancelled)

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

    def test_raise_error_false_swallows_failures(self):
        stmts = [
            _StubStatement(name="ok", duration=0.01),
            _StubStatement(name="boom", fail=True, duration=0.0),
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2, raise_error=False)
        batch.wait(
            wait={"interval": 0.005, "timeout": 10},
            raise_error=False,
        )


class TestStatementBatchCancel(unittest.TestCase):
    def test_cancel_calls_each_statement(self):
        stmts = [_StubStatement(name=f"s{i}", duration=5.0) for i in range(3)]
        batch = StatementBatch.from_results(stmts)
        batch.cancel()
        for s in stmts:
            self.assertTrue(s._cancelled)

    def test_cancel_tears_down_pool(self):
        stmts = [
            _StubStatement(name=f"s{i}", duration=5.0) for i in range(4)
        ]
        batch = StatementBatch.from_results(stmts)
        batch.start(parallel=2)
        batch.cancel()
        self.assertEqual(len(batch._in_flight), 0)
        self.assertEqual(len(batch._pending_queue), 0)
        self.assertIsNone(batch._pool_runner)


class TestStatementBatchWaitNoPool(unittest.TestCase):
    def test_wait_without_start_iterates_sequentially(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(3)]
        for s in stmts:
            s._started = True
            s._finished = True
        batch = StatementBatch.from_results(stmts)
        batch.wait(wait=False)


class TestStatementBatchStartCancels(unittest.TestCase):
    """``start`` cancels every remaining statement on any inner error."""

    def test_sequential_runner_error_cancels_all(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(4)]

        def bad_runner(result):
            if result.name == "s1":
                raise RuntimeError("submit failed")
            result._started = True
            result._finished = True

        batch = StatementBatch.from_results(stmts)
        with self.assertRaises(RuntimeError):
            batch.start(parallel=False, runner=bad_runner)

        # s0 already done, s1 was the failing one; s2/s3 must have been
        # cancelled when the batch unwound.
        self.assertTrue(stmts[2]._cancelled)
        self.assertTrue(stmts[3]._cancelled)

    def test_pool_runner_error_cancels_everything(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(5)]

        def bad_runner(result):
            raise RuntimeError("submit failed")

        batch = StatementBatch.from_results(stmts)
        with self.assertRaises(RuntimeError):
            batch.start(parallel=3, runner=bad_runner)

        # After the teardown, every result in the batch has been cancelled.
        for s in stmts:
            self.assertTrue(s._cancelled)
        # Pool bookkeeping is wiped.
        self.assertFalse(batch._in_flight)
        self.assertFalse(batch._pending_queue)


class TestStatementBatchEngineDefaultRunner(unittest.TestCase):
    """When ``engine`` is bound to the batch, ``start`` uses ``engine.execute``."""

    def test_engine_is_used_as_default_runner(self):
        stmts = [_StubStatement(name=f"s{i}") for i in range(3)]

        class _FakeEngine:
            def __init__(self):
                self.calls = []

            def execute(self, result, *, wait, raise_error, **kw):
                self.calls.append((result, wait, raise_error, kw))
                result._started = True
                result._finished = True
                return result

        engine = _FakeEngine()
        batch = StatementBatch.from_results(stmts, engine=engine)
        batch.start(parallel=False)

        self.assertEqual(len(engine.calls), 3)
        for result, wait, raise_error, kw in engine.calls:
            self.assertFalse(wait)
            self.assertTrue(raise_error)
            self.assertEqual(kw, {})

    def test_explicit_runner_overrides_engine_default(self):
        stmts = [_StubStatement(name="only")]

        class _FakeEngine:
            def execute(self, *a, **kw):  # pragma: no cover - should not run
                raise AssertionError("engine.execute should be bypassed")

        seen = []

        def custom(result):
            seen.append(result)
            result._started = True
            result._finished = True

        batch = StatementBatch.from_results(stmts, engine=_FakeEngine())
        batch.start(parallel=False, runner=custom)
        self.assertEqual(seen, stmts)


class _StubBatch(StatementBatch):
    """Batch subclass that builds :class:`_StubStatement` results."""

    def factory(self, statement):
        return _StubStatement(statement=statement)


class TestFromStatements(unittest.TestCase):
    def test_from_statements_uses_subclass_factory(self):
        configs = [
            PreparedStatement(text="SELECT 1"),
            PreparedStatement(text="SELECT 2", parameters={"x": 1}),
        ]
        batch = _StubBatch.from_statements(configs)
        assert list(batch.results.keys()) == ["0", "1"]
        assert batch["0"].statement.text == "SELECT 1"
        assert batch["1"].statement.parameters == {"x": 1}
        assert batch.statements["0"] is configs[0]

    def test_from_statements_accepts_mapping(self):
        configs = {
            "a": "SELECT 1",
            "b": PreparedStatement(text="SELECT 2"),
        }
        batch = _StubBatch.from_statements(configs)
        assert list(batch.results.keys()) == ["a", "b"]
        assert batch["a"].text == "SELECT 1"
        assert batch["b"].text == "SELECT 2"

    def test_from_statements_without_engine_or_override_raises(self):
        with self.assertRaises(NotImplementedError):
            StatementBatch.from_statements([PreparedStatement(text="SELECT 1")])

    def test_from_statements_delegates_to_engine_when_bound(self):
        built = []

        class _FakeEngine:
            def statement_result(self, cfg):
                result = _StubStatement(statement=cfg, name=cfg.text)
                built.append(result)
                return result

        batch = StatementBatch.from_statements(
            ["SELECT 1", "SELECT 2"],
            engine=_FakeEngine(),
        )
        assert len(built) == 2
        assert list(batch.results.values()) == built


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
