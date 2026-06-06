"""Tests for local agent replication (child processes)."""
from __future__ import annotations

import sys
import unittest

from yggdrasil.loki import Loki
from yggdrasil.loki.behavior import REGISTRY, LokiBehavior, register


@register
class _SquareBehavior(LokiBehavior):
    name = "square-test"
    description = "square a number in a child process"

    def run(self, agent, *, n, **_):
        return {"n": n, "square": n * n, "pid": __import__("os").getpid()}


@unittest.skipUnless(sys.platform.startswith("linux"), "fork-based replication")
class TestReplication(unittest.TestCase):
    def test_spawn_runs_in_a_child_process(self):
        import os

        loki = Loki()
        rep = loki.spawn("square-test", n=6)
        self.assertIn(rep, loki.replicas)
        response = rep.result(timeout=30)
        self.assertEqual(response.data["square"], 36)
        # It really ran in a different process.
        self.assertNotEqual(response.data["pid"], os.getpid())
        self.assertEqual(rep.status, "done")

    def test_map_fans_out_in_parallel(self):
        loki = Loki()
        reps = loki.map("square-test", [2, 3, 4], arg="n")
        results = sorted(r.result(timeout=30).data["square"] for r in reps)
        self.assertEqual(results, [4, 9, 16])

    def test_failure_is_reported(self):
        loki = Loki()
        rep = loki.spawn("does-not-exist-behavior")
        with self.assertRaises(RuntimeError):
            rep.result(timeout=30)
        self.assertEqual(rep.status, "failed")


if __name__ == "__main__":
    unittest.main()
