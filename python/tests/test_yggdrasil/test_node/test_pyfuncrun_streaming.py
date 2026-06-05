"""Tests for the streaming PyFuncRun service.

Validate the non-blocking POST + live stdout/stderr + real cancel/timeout
chain. No pytest-asyncio dependency — each test runs the coroutine with
``asyncio.run``.
"""
from __future__ import annotations

import asyncio
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

from yggdrasil.node.api.schemas.pyfunc import PyFuncCreate
from yggdrasil.node.api.schemas.pyfuncrun import PyFuncRunCreate
from yggdrasil.node.api.services.pyenv import PyEnvService
from yggdrasil.node.api.services.pyfunc import PyFuncService
from yggdrasil.node.api.services.pyfuncrun import PyFuncRunService
from yggdrasil.node.config import Settings


def _make_settings(tmp_home: Path, **overrides) -> Settings:
    base = dict(
        node_id="test-node",
        node_home=tmp_home,
        front_home=tmp_home,
        max_log_lines_per_stream=1000,
        run_heartbeat_interval=0.2,
        run_cancel_grace_seconds=0.5,
        max_python_timeout=600.0,
    )
    base.update(overrides)
    return Settings(**base)


def _build_services(settings: Settings) -> tuple[PyEnvService, PyFuncService, PyFuncRunService]:
    pyenv = PyEnvService(settings)
    pyfunc = PyFuncService(settings)
    pyfuncrun = PyFuncRunService(settings, pyenv, pyfunc)
    return pyenv, pyfunc, pyfuncrun


async def _register_func(pyfunc: PyFuncService, name: str, code: str) -> int:
    resp = await pyfunc.create(PyFuncCreate(name=name, code=code))
    return resp.func.id


class StreamingPyFuncRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_home = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- 1. Fast success ---------------------------------------------------

    def test_fast_success(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            fid = await _register_func(pyfunc, "hello", "print('hi')\n")
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            final = await pyfuncrun.wait(resp.run.id, timeout=10)
            self.assertEqual(final.status, "completed")
            self.assertEqual(final.returncode, 0)
            self.assertIn("hi", final.stdout or "")

        asyncio.run(run())

    # --- 2. POST returns before completion --------------------------------

    def test_create_returns_immediately(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            fid = await _register_func(
                pyfunc, "sleeper",
                "import time\ntime.sleep(2)\nprint('done')\n",
            )
            t0 = time.monotonic()
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 1.0, f"create() took {elapsed:.2f}s — should be near-instant")
            self.assertIn(resp.run.status, ("pending", "running"))
            final = await pyfuncrun.wait(resp.run.id, timeout=10)
            self.assertEqual(final.status, "completed")

        asyncio.run(run())

    # --- 3. Stdout streams BEFORE completion -------------------------------

    def test_stdout_streams_live(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            code = (
                "import sys, time\n"
                "print('first', flush=True)\n"
                "time.sleep(1.0)\n"
                "print('second', flush=True)\n"
            )
            fid = await _register_func(pyfunc, "tick", code)
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))

            saw_stdout_before_complete = False
            saw_complete = False
            t_first_stdout: float | None = None
            t_start = time.monotonic()
            async for event in pyfuncrun.stream_logs(resp.run.id):
                if event["type"] == "stdout":
                    if t_first_stdout is None:
                        t_first_stdout = time.monotonic() - t_start
                    if not saw_complete:
                        saw_stdout_before_complete = True
                elif event["type"] == "complete":
                    saw_complete = True
                    break
            self.assertTrue(saw_stdout_before_complete, "no stdout event arrived before complete")
            self.assertIsNotNone(t_first_stdout)
            self.assertLess(t_first_stdout, 0.9, f"first stdout event arrived at {t_first_stdout:.2f}s — should be near-instant")

        asyncio.run(run())

    # --- 4. Cancel kills the process --------------------------------------

    def test_cancel_kills_process(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            code = "while True:\n    pass\n"
            fid = await _register_func(pyfunc, "spin", code)
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            # Wait for process to actually spawn
            for _ in range(50):
                entry = await pyfuncrun.get(resp.run.id)
                if entry.pid:
                    break
                await asyncio.sleep(0.05)
            self.assertIsNotNone(entry.pid)
            pid = entry.pid

            final = await pyfuncrun.cancel(resp.run.id)
            self.assertEqual(final.status, "cancelled")
            self.assertTrue(final.cancellation_requested)

            # Verify the OS process is actually gone.
            try:
                os.kill(pid, 0)
                # If we got here the process still exists — give it a beat to reap.
                await asyncio.sleep(0.3)
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            except ProcessLookupError:
                pass  # already dead, ideal

        asyncio.run(run())

    # --- 5. Timeout kills the process -------------------------------------

    def test_timeout_kills_process(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            code = "import time\ntime.sleep(60)\n"
            fid = await _register_func(pyfunc, "slow", code)
            t0 = time.monotonic()
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid, timeout=0.5))
            final = await pyfuncrun.wait(resp.run.id, timeout=5)
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 3.0, f"timeout was supposed to fire in 0.5s, run took {elapsed:.2f}s total")
            self.assertEqual(final.status, "failed")
            self.assertIn("timed out", final.error or "")
            # Make sure fractional-second timeouts aren't truncated to "0s".
            self.assertIn("0.5", final.error or "")

        asyncio.run(run())

    # --- 6. Failing run ---------------------------------------------------

    def test_failing_run(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            code = "raise SystemExit(2)\n"
            fid = await _register_func(pyfunc, "boom", code)
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            final = await pyfuncrun.wait(resp.run.id, timeout=10)
            self.assertEqual(final.status, "failed")
            self.assertEqual(final.returncode, 2)

        asyncio.run(run())

    # --- 7. Bounded logs --------------------------------------------------

    def test_bounded_logs(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home, max_log_lines_per_stream=10)
            _, pyfunc, pyfuncrun = _build_services(settings)
            code = (
                "for i in range(50):\n"
                "    print(f'line-{i}', flush=True)\n"
            )
            fid = await _register_func(pyfunc, "many", code)
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            final = await pyfuncrun.wait(resp.run.id, timeout=15)
            self.assertEqual(final.status, "completed")
            self.assertTrue(final.stdout_truncated)
            lines = (final.stdout or "").splitlines()
            self.assertEqual(len(lines), 10)
            # The last lines retained should be the latest ones.
            self.assertEqual(lines[-1], "line-49")
            self.assertEqual(lines[0], "line-40")

        asyncio.run(run())

    # --- 8. Runtime cleanup after completion -----------------------------

    def test_runtime_cleanup_after_completion(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            fid = await _register_func(pyfunc, "quick_clean", "print('done')\n")
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid))
            await pyfuncrun.wait(resp.run.id, timeout=10)
            # The supervisor pops _runtimes before signaling completion, so by
            # the time wait() returns the runtime must already be gone — no
            # zombie deques / replay buffers accumulate across thousands of runs.
            self.assertNotIn(resp.run.id, pyfuncrun._runtimes)
            # The entry stays in history so /get and terminal replay still work.
            entry = await pyfuncrun.get(resp.run.id)
            self.assertEqual(entry.status, "completed")

        asyncio.run(run())

    # --- 9. Error event precedes complete --------------------------------

    def test_error_event_precedes_complete(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            fid = await _register_func(pyfunc, "slow_err", "import time\ntime.sleep(60)\n")
            resp = await pyfuncrun.create(PyFuncRunCreate(func_id=fid, timeout=0.3))
            event_types: list[str] = []
            async for event in pyfuncrun.stream_logs(resp.run.id):
                event_types.append(event["type"])
                if event["type"] == "complete":
                    break
            self.assertIn("error", event_types)
            self.assertIn("complete", event_types)
            self.assertLess(
                event_types.index("error"),
                event_types.index("complete"),
                f"error event must precede complete; got {event_types}",
            )

        asyncio.run(run())

    # --- 10. Concurrent short runs ----------------------------------------

    def test_concurrent_short_runs(self) -> None:
        async def run() -> None:
            settings = _make_settings(self.tmp_home)
            _, pyfunc, pyfuncrun = _build_services(settings)
            fid = await _register_func(pyfunc, "quick", "print('ok')\n")
            responses = await asyncio.gather(*[
                pyfuncrun.create(PyFuncRunCreate(func_id=fid))
                for _ in range(5)
            ])
            ids = {r.run.id for r in responses}
            self.assertEqual(len(ids), 5)
            finals = await asyncio.gather(*[
                pyfuncrun.wait(rid, timeout=15) for rid in ids
            ])
            for f in finals:
                self.assertEqual(f.status, "completed")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
