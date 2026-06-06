"""Tests for Loki session workspaces (+ purge) and self-compressing memory."""
from __future__ import annotations

import shutil
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from yggdrasil.loki.memory import LokiMemory
from yggdrasil.loki.session import LokiSession


class TestLokiSession(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("/tmp") / f"ygg-sess-{time.time_ns()}"
        self.base = self.tmp / "session"
        self._patch = patch.multiple(
            "yggdrasil.loki.session", BASE=self.base)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_creates_isolated_tree(self):
        s = LokiSession.start()
        self.assertTrue(s.workspace.is_dir())
        self.assertTrue(s.memory_dir.is_dir())
        self.assertTrue(s.cache_dir.is_dir())
        self.assertTrue(str(s.dir).startswith(str(self.base)))

    def test_purge_keeps_recent_and_drops_excess(self):
        made = []
        for _ in range(5):
            made.append(LokiSession.start(purge=False))
            time.sleep(0.01)
        current = made[-1]
        removed = LokiSession.purge(keep=2, max_age_days=999, exclude=current.dir)
        self.assertGreaterEqual(len(removed), 1)
        self.assertTrue(current.dir.is_dir())            # current never purged
        self.assertLessEqual(len(LokiSession.list()), 3)

    def test_purge_drops_old_by_age(self):
        old = LokiSession.start(purge=False)
        past = time.time() - 30 * 86400
        import os
        os.utime(old.dir, (past, past))
        LokiSession.purge(keep=99, max_age_days=14)
        self.assertFalse(old.dir.is_dir())


class TestLokiMemory(unittest.TestCase):
    def setUp(self):
        self.dir = Path("/tmp") / f"ygg-mem-{time.time_ns()}"
        self.dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_context_and_persist(self):
        m = LokiMemory(self.dir / "m.json")
        m.add("user", "implement X")
        m.add("assistant", "wrote x.py")
        ctx = m.system_context()
        self.assertIn("implement X", ctx)
        self.assertIn("wrote x.py", ctx)
        # reload from disk
        self.assertEqual(len(LokiMemory(self.dir / "m.json").turns), 2)

    def test_empty_context_is_none(self):
        self.assertIsNone(LokiMemory().system_context())

    def test_compress_folds_old_turns_into_synthesis(self):
        m = LokiMemory(keep_recent=2, compress_chars=50)
        for i in range(6):
            m.add("user", f"turn {i} " + "x" * 20)

        class _Agent:
            def reason(self, prompt, *, engine=None, tier=None, system=None):
                return "SYNTH: earlier turns summarized"

        self.assertTrue(m.maybe_compress(_Agent()))
        self.assertEqual(m.synthesis, "SYNTH: earlier turns summarized")
        self.assertEqual(len(m.turns), 2)                # only recent kept
        self.assertIn("SYNTH", m.system_context())

    def test_compress_skips_without_engine(self):
        m = LokiMemory(keep_recent=2, compress_chars=10)
        for i in range(6):
            m.add("user", "x" * 30)

        class _NoEngine:
            def reason(self, *a, **k):
                raise RuntimeError("no engine")

        self.assertFalse(m.maybe_compress(_NoEngine()))
        self.assertEqual(len(m.turns), 6)                # untouched


if __name__ == "__main__":
    unittest.main()
