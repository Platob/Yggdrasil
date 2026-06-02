"""Smoke tests for the Genie CLIs.

Covers ``ygg databricks genie …`` (the sub-command dispatcher) and the
standalone ``ygg-genie`` agent console script. Help paths assert clean
``--help`` exits; behaviour paths patch ``DatabricksClient`` so the
dispatch + handler wiring is exercised without a live workspace.
"""
from __future__ import annotations

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main as dbks_main
from yggdrasil.cli.databricks.genie import main as genie_main

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _strip(text: str) -> str:
    """Drop ANSI SGR escapes so assertions match on plain content."""
    return _ANSI.sub("", text)


# ---------------------------------------------------------------------------
# Help paths
# ---------------------------------------------------------------------------


class TestGenieCliHelp(unittest.TestCase):
    def test_genie_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            dbks_main(["genie", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_genie_ask_help(self):
        with self.assertRaises(SystemExit) as ctx:
            dbks_main(["genie", "ask", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_genie_agent_help(self):
        with self.assertRaises(SystemExit) as ctx:
            dbks_main(["genie", "agent", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_genie_no_action_prints_help(self):
        # ``genie`` with no sub-action returns 1 (prints help).
        self.assertEqual(dbks_main(["genie"]), 1)

    def test_ygg_genie_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            genie_main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


# ---------------------------------------------------------------------------
# Behaviour paths (patched client)
# ---------------------------------------------------------------------------


def _answer(*, text=None, sql=None, has_query=False, failed=False):
    ans = MagicMock()
    ans.text = text
    ans.sql = sql
    ans.has_query = has_query
    ans.failed = failed
    ans.error = None
    ans.questions = ()
    return ans


class TestGenieCliBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        patcher = patch(
            "yggdrasil.databricks.client.DatabricksClient", return_value=self.client,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_spaces_lists(self):
        s1 = MagicMock(space_id="a", title="Sales")
        s2 = MagicMock(space_id="b", title="Ops")
        self.client.genie.list_spaces.return_value = [s1, s2]
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = dbks_main(["genie", "spaces"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("a\tSales", out)
        self.assertIn("b\tOps", out)

    def test_ask_prints_text_and_sql(self):
        self.client.genie.ask.return_value = _answer(
            text="Revenue is up.", sql="SELECT 1", has_query=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = dbks_main(["genie", "ask", "how much revenue?", "--space", "sp-1"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Revenue is up.", out)
        self.assertIn("SELECT 1", out)
        self.client.genie.ask.assert_called_once_with(
            "how much revenue?", space_id="sp-1",
        )

    def test_agent_runs_and_prints_summary(self):
        run = MagicMock()
        run.summary.return_value = "Goal: x\n[1] (you) x"
        run.data_answer = None
        agent = MagicMock()
        agent.run.return_value = run
        self.client.genie.agent.return_value = agent

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = dbks_main(["genie", "agent", "why the dip?", "--space", "sp-1", "--max-turns", "3"])
        self.assertEqual(rc, 0)
        self.client.genie.agent.assert_called_once_with(space_id="sp-1", max_turns=3)
        agent.run.assert_called_once_with("why the dip?")
        self.assertIn("Goal: x", buf.getvalue())

    def test_ygg_genie_agent_default_mode(self):
        run = MagicMock()
        goal_turn = MagicMock(autonomous=False, question="why the dip?")
        goal_turn.answer = _answer(text="Revenue fell on lower volume.")
        run.turns = [goal_turn]
        run.data_answer = None
        agent = MagicMock()
        agent.run.return_value = run
        self.client.genie.agent.return_value = agent

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = genie_main(["--space", "sp-1", "why the dip?"])
        self.assertEqual(rc, 0)
        # planner defaults to None (heuristic) when --planner is omitted.
        self.client.genie.agent.assert_called_once_with(
            space_id="sp-1", planner=None, max_turns=4,
        )
        agent.run.assert_called_once_with("why the dip?")
        out = _strip(buf.getvalue())
        self.assertIn("why the dip?", out)
        self.assertIn("Revenue fell on lower volume.", out)

    def test_ygg_genie_planner_flag_threads_through(self):
        run = MagicMock(turns=[], data_answer=None)
        agent = MagicMock()
        agent.run.return_value = run
        self.client.genie.agent.return_value = agent

        with redirect_stdout(io.StringIO()):
            rc = genie_main([
                "--space", "sp-1", "--planner", "databricks-claude-sonnet-4", "explain churn",
            ])
        self.assertEqual(rc, 0)
        self.client.genie.agent.assert_called_once_with(
            space_id="sp-1", planner="databricks-claude-sonnet-4", max_turns=4,
        )

    def test_ygg_genie_one_shot_ask(self):
        self.client.genie.ask.return_value = _answer(text="42", failed=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = genie_main(["--space", "sp-1", "--ask", "the answer?"])
        self.assertEqual(rc, 0)
        self.client.genie.ask.assert_called_once_with("the answer?", space_id="sp-1")
        self.assertIn("42", _strip(buf.getvalue()))

    def test_ygg_genie_requires_space(self):
        # No --space and no env → exit code 2.
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("YGG_GENIE_SPACE", None)
            rc = genie_main(["why the dip?"])
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# Interactive console (GenieConsole) — driven with scripted input
# ---------------------------------------------------------------------------


class TestGenieConsole(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.genie.space.return_value.title = "Sales"

    def _run(self, lines: list[str]) -> str:
        from yggdrasil.cli.databricks.genie import GenieConsole

        console = GenieConsole(self.client, "sp-1")
        buf = io.StringIO()
        with redirect_stdout(buf), patch("builtins.input", side_effect=lines):
            console.run()
        return _strip(buf.getvalue())

    def test_help_then_quit(self):
        out = self._run(["/help", "/quit"])
        self.assertIn("/agent", out)
        self.assertIn("/sql", out)

    def test_sql_command_runs_sql(self):
        import pyarrow as pa

        self.client.sql.execute.return_value.to_arrow_table.return_value = pa.table(
            {"n": [1, 2]}
        )
        out = self._run(["/sql SELECT 1", "/quit"])
        self.client.sql.execute.assert_called_once_with("SELECT 1")
        self.assertIn("n", out)

    def test_plain_text_asks_genie(self):
        ans = _answer(text="Revenue is up.")
        self.client.genie.space.return_value.start_conversation.return_value = (
            MagicMock(), ans,
        )
        out = self._run(["how is revenue?", "/quit"])
        self.assertIn("Revenue is up.", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
