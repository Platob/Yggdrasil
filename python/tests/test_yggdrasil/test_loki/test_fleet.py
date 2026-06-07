"""Tests for the Loki fleet — delegating tasks to process agents + monitoring."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki.fleet import Fleet


def _ok_cmd(answer: str = "did it", files=("a.py",), tokens=1000, cost=0.01) -> list[str]:
    """A stand-in agent that prints a ``do --json`` transcript and exits 0."""
    import json
    payload = json.dumps({"completed": True, "answer": answer,
                          "files_changed": list(files), "steps": [1, 2],
                          "usage": {"total_tokens": tokens, "cost_usd": cost}})
    return [sys.executable, "-c", f"print({payload!r})"]


def _fail_cmd() -> list[str]:
    return [sys.executable, "-c", "import sys; sys.stderr.write('boom: nope\\n'); sys.exit(1)"]


def _slow_cmd() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(30)"]


class TestFleet(unittest.TestCase):
    def test_do_command_builds_ygg_loki_do(self):
        cmd = Fleet().do_command("fix the bug", root="/tmp/x", engine="claude",
                                 max_steps=5, allow_web=True, read_only=True)
        self.assertIn("yggdrasil.loki.cli", cmd)
        self.assertIn("do", cmd)
        self.assertIn("fix the bug", cmd)
        self.assertIn("--json", cmd)
        self.assertEqual(cmd[cmd.index("--root") + 1], "/tmp/x")
        self.assertEqual(cmd[cmd.index("--engine") + 1], "claude")
        self.assertIn("--read-only", cmd)
        self.assertIn("--allow-web", cmd)

    def test_spawn_monitor_parses_success_and_failure(self):
        fleet = Fleet()
        fleet.spawn("one", cmd=_ok_cmd(answer="alpha"))
        fleet.spawn("two", cmd=_fail_cmd())
        ticks = []
        fleet.monitor(lambda agents: ticks.append(len(agents)), interval=0.02)
        self.assertTrue(fleet.all_done())
        self.assertGreater(len(ticks), 0)
        a, b = fleet.agents
        self.assertEqual(a.status, "done")
        self.assertEqual(a.answer, "alpha")
        self.assertEqual(a.files_changed, ["a.py"])
        self.assertEqual(a.steps, 2)
        self.assertEqual(b.status, "failed")
        self.assertIn("boom", b.stderr_tail)

    def test_summary_rows(self):
        fleet = Fleet()
        fleet.spawn("one", cmd=_ok_cmd())
        fleet.monitor(interval=0.02)
        row = fleet.summary()[0]
        self.assertEqual(row["status"], "done")
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["files_changed"], ["a.py"])
        self.assertIn("elapsed", row)

    def test_timeout_cancels_survivors(self):
        fleet = Fleet()
        fleet.spawn("slow", cmd=_slow_cmd())
        fleet.monitor(interval=0.05, timeout=0.3)
        self.assertEqual(fleet.agents[0].status, "timeout")
        self.assertFalse(fleet.agents[0].running)

    def test_kpis_aggregate_steps_tokens_cost(self):
        fleet = Fleet()
        fleet.spawn("one", cmd=_ok_cmd(tokens=1000, cost=0.01))
        fleet.spawn("two", cmd=_ok_cmd(tokens=500, cost=0.02))
        fleet.monitor(interval=0.02)
        k = fleet.kpis()
        self.assertEqual(k["done"], 2)
        self.assertEqual(k["steps"], 4)              # 2 each
        self.assertEqual(k["tokens"], 1500)
        self.assertAlmostEqual(k["cost"], 0.03)

    def test_max_parallel_queues_extra_tasks(self):
        fleet = Fleet(max_parallel=2)
        # spawn_all builds real ygg commands; inject fast stand-ins via _pending
        # so the cap behaviour is what's exercised, not a live agent.
        for t in ("a", "b", "c", "d"):
            fleet._pending.append((t, {"cmd": _ok_cmd()}))
        fleet._launch_pending()
        self.assertLessEqual(len(fleet.running()), 2)   # only the cap runs at once
        self.assertEqual(fleet.queued(), 2)             # the rest wait
        fleet.monitor(interval=0.02)
        self.assertEqual(len(fleet.agents), 4)          # all eventually launched
        self.assertTrue(all(a.ok for a in fleet.agents))
        self.assertEqual(fleet.queued(), 0)

    def test_non_json_output_still_completes(self):
        fleet = Fleet()
        fleet.spawn("noisy", cmd=[sys.executable, "-c", "print('hello, not json')"])
        fleet.monitor(interval=0.02)
        # rc 0, no parseable transcript → treated as done with no result.
        self.assertEqual(fleet.agents[0].status, "done")
        self.assertIsNone(fleet.agents[0].result)


class TestLiveDisplay(unittest.TestCase):
    def test_noop_off_tty(self):
        import io
        from contextlib import redirect_stdout

        from yggdrasil.cli import style
        buf = io.StringIO()
        with patch.object(style, "_IS_TTY", False), redirect_stdout(buf):
            live = style.LiveDisplay()
            live.update(["a", "b"])
        self.assertEqual(buf.getvalue(), "")

    def test_redraws_in_place_on_tty(self):
        import io
        from contextlib import redirect_stdout

        from yggdrasil.cli import style
        buf = io.StringIO()
        with patch.object(style, "_IS_TTY", True), redirect_stdout(buf):
            live = style.LiveDisplay()
            live.update(["x", "y"])          # first: no cursor move
            live.update(["x", "z"])          # second: move up 2 lines (CSI 2F)
        out = buf.getvalue()
        self.assertIn("\033[2F", out)        # cursor-up to top of the 2-line block
        self.assertIn("\033[2K", out)        # each line cleared


class TestDecomposeAndDelegate(unittest.TestCase):
    def test_decompose_parses_json_array(self):
        from yggdrasil.loki import Loki

        loki = Loki()
        with patch.object(loki, "reason", return_value='Sure:\n["task a", "task b", ""]\n'):
            tasks = loki.decompose("build X")
        self.assertEqual(tasks, ["task a", "task b"])

    def test_decompose_falls_back_to_goal(self):
        from yggdrasil.loki import Loki

        loki = Loki()
        with patch.object(loki, "reason", return_value="no json here"):
            self.assertEqual(loki.decompose("just do it"), ["just do it"])

    def test_delegate_skill_goal_path_decomposes_then_delegates(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.skills import DelegateSkill

        loki = Loki()
        summaries = [{"status": "done"}, {"status": "failed"}]
        with patch.object(loki, "decompose", return_value=["a", "b"]) as dec, \
                patch.object(loki, "delegate", return_value=summaries) as deleg:
            res = DelegateSkill().run(loki, goal="big goal")
        dec.assert_called_once()
        deleg.assert_called_once()
        self.assertEqual(res["completed"], 1)
        self.assertEqual(res["failed"], 1)

    def test_delegate_skill_requires_tasks_or_goal(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.skills import DelegateSkill

        with self.assertRaises(ValueError):
            DelegateSkill().run(Loki())


if __name__ == "__main__":
    unittest.main()
