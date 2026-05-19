"""Live-integration tests for ``ygg-genie`` (the :class:`GenieCLI`).

Skipped unless ``DATABRICKS_HOST`` (plus matching credentials) and
``DATABRICKS_GENIE_SPACE_ID`` are exported. Exercises the same agent
flows that :mod:`test_databricks.test_genie.test_integration` covers,
but routed through the CLI's parse → defaults-merge → REPL plumbing
so the wiring between argparse, the agent, and on-disk artifacts
is verified end-to-end against a real workspace.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional

from yggdrasil.cli.databricks.genie import GenieCLI, main

from ...test_databricks import DatabricksIntegrationCase


_DEFAULT_TEXT_QUESTION = "Hello, what data do you have access to?"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


class GenieCLIIntegrationCase(DatabricksIntegrationCase):
    """Common setup: tmp output dir, configured space id, no-color CLI."""

    def setUp(self) -> None:
        super().setUp()
        space_id = _env("DATABRICKS_GENIE_SPACE_ID")
        if not space_id:
            raise unittest.SkipTest(
                "DATABRICKS_GENIE_SPACE_ID not set — skipping ygg-genie "
                "integration tests."
            )
        self.space_id = space_id
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ygg-genie-cli-int-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        # Reset the cached agent so each test starts with a clean history.
        self.client.genie._agent = None

        self.captured: list[str] = []
        self.cli = GenieCLI(
            client=self.client,
            color=False,
            input_fn=self._fail_on_input,
            output_fn=self.captured.append,
        )

    @staticmethod
    def _fail_on_input(_prompt: str) -> str:
        raise EOFError  # any test that hits the REPL prompt is a bug

    def _out(self) -> str:
        return "\n".join(self.captured)


class TestGenieCLIOneShot(GenieCLIIntegrationCase):
    """``-q``: ask, render answer, exit. No REPL."""

    def test_ask_once_prints_answer(self):
        question = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        # We bypass argparse for setUp simplicity — call ask_once directly
        # against the already-wired CLI instance. The argparse → defaults
        # wiring is covered by a separate end-to-end test below.
        from dataclasses import replace

        self.client.genie.defaults = replace(
            self.client.genie.defaults,
            space_id=self.space_id,
            agent_output_dir=str(self.tmpdir),
        )
        rc = self.cli.ask_once(question)
        self.assertEqual(rc, 0)
        out = self._out()
        # Either text or SQL must show up in the rendered answer.
        self.assertTrue(
            "status:" in out and ("msg:" in out or "SQL:" in out),
            msg=f"unexpected CLI output:\n{out}",
        )


class TestGenieCLIMain(GenieCLIIntegrationCase):
    """End-to-end through ``main(argv)``: argparse + defaults merge + run."""

    def test_main_one_shot_writes_json(self):
        import io
        from unittest.mock import patch

        question = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            rc = main([
                "--space-id", self.space_id,
                "--output-dir", str(self.tmpdir),
                "--auto-save",
                "--auto-save-format", "json",
                "--no-color",
                "-q", question,
            ])
        self.assertEqual(rc, 0)
        # The CLI should have rendered an answer header to stdout.
        self.assertIn("status:", buf.getvalue())
        # And auto-save fired exactly once, regardless of whether Genie
        # produced SQL (json format is envelope-only).
        written = list(self.tmpdir.rglob("*.json"))
        self.assertEqual(len(written), 1, msg=f"unexpected files: {written!r}")
