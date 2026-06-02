"""Smoke tests for the ``ygg databricks genie`` console.

Help paths assert clean ``--help`` exits; behaviour paths patch
``DatabricksClient`` so the dispatch + handler wiring (and the rich
rendering) is exercised without a live workspace.
"""
from __future__ import annotations

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main as dbks_main

_ANSI = re.compile(r"\033\[[0-9;]*m")


def _strip(text: str) -> str:
    """Drop ANSI SGR escapes so assertions match on plain content."""
    return _ANSI.sub("", text)


def _answer(*, text=None, sql=None, has_query=False, failed=False):
    ans = MagicMock()
    ans.text = text
    ans.sql = sql
    ans.has_query = has_query
    ans.failed = failed
    ans.error = None
    ans.questions = ()
    return ans


def _turn(question, *, autonomous, answer):
    t = MagicMock(autonomous=autonomous, question=question)
    t.answer = answer
    return t


# ---------------------------------------------------------------------------
# Help paths
# ---------------------------------------------------------------------------


class TestGenieCliHelp(unittest.TestCase):
    def _help_exits_zero(self, argv):
        with self.assertRaises(SystemExit) as ctx:
            dbks_main(argv)
        self.assertEqual(ctx.exception.code, 0)

    def test_genie_help(self):
        self._help_exits_zero(["genie", "--help"])

    def test_genie_ask_help(self):
        self._help_exits_zero(["genie", "ask", "--help"])

    def test_genie_agent_help(self):
        self._help_exits_zero(["genie", "agent", "--help"])

    def test_genie_console_help(self):
        self._help_exits_zero(["genie", "console", "--help"])

    def test_genie_no_action_prints_help(self):
        # ``genie`` with no sub-action returns 1 (prints help).
        self.assertEqual(dbks_main(["genie"]), 1)


# ---------------------------------------------------------------------------
# Behaviour paths (patched client)
# ---------------------------------------------------------------------------


class TestGenieCliBehaviour(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        patcher = patch(
            "yggdrasil.databricks.client.DatabricksClient", return_value=self.client,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, argv) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = dbks_main(argv)
        return rc, _strip(buf.getvalue())

    def test_spaces_lists(self):
        self.client.genie.list_spaces.return_value = [
            MagicMock(space_id="a", title="Sales"),
            MagicMock(space_id="b", title="Ops"),
        ]
        rc, out = self._run(["genie", "spaces"])
        self.assertEqual(rc, 0)
        self.assertIn("Sales", out)
        self.assertIn("Ops", out)

    def test_ask_prints_text_and_sql(self):
        self.client.genie.ask.return_value = _answer(text="Revenue is up.", sql="SELECT 1")
        rc, out = self._run(["genie", "ask", "how much revenue?", "--space", "sp-1"])
        self.assertEqual(rc, 0)
        self.assertIn("Revenue is up.", out)
        self.assertIn("SELECT 1", out)
        self.client.genie.ask.assert_called_once_with("how much revenue?", space_id="sp-1")

    def test_agent_runs_and_renders_transcript(self):
        run = MagicMock(data_answer=None)
        run.turns = [_turn("why the dip?", autonomous=False,
                           answer=_answer(text="Lower volume."))]
        agent = MagicMock()
        agent.run.return_value = run
        self.client.genie.agent.return_value = agent

        rc, out = self._run(["genie", "agent", "why the dip?", "--space", "sp-1", "--max-turns", "3"])
        self.assertEqual(rc, 0)
        # planner defaults to None (heuristic) when --planner is omitted.
        self.client.genie.agent.assert_called_once_with(
            space_id="sp-1", planner=None, max_turns=3,
        )
        agent.run.assert_called_once_with("why the dip?")
        self.assertIn("why the dip?", out)
        self.assertIn("Lower volume.", out)

    def test_agent_planner_flag_threads_through(self):
        run = MagicMock(turns=[], data_answer=None)
        agent = MagicMock()
        agent.run.return_value = run
        self.client.genie.agent.return_value = agent

        rc, _out = self._run([
            "genie", "agent", "explain churn", "--space", "sp-1",
            "--planner", "databricks-claude-sonnet-4",
        ])
        self.assertEqual(rc, 0)
        self.client.genie.agent.assert_called_once_with(
            space_id="sp-1", planner="databricks-claude-sonnet-4", max_turns=4,
        )

    def test_console_no_space_creates_default(self):
        # No --space → a default space is ensured and the console opens.
        self.client.genie.ensure_default_space.return_value = MagicMock(
            space_id="auto-1", title="Yggdrasil Genie",
        )
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("YGG_GENIE_SPACE", None)
            with patch("builtins.input", side_effect=EOFError):
                rc, _out = self._run(["genie", "console"])
        self.assertEqual(rc, 0)
        self.client.genie.ensure_default_space.assert_called_once_with()

    def test_console_no_space_no_default_fails(self):
        # No space and no default creatable (no catalog/schema) → exit 2.
        self.client.genie.ensure_default_space.side_effect = ValueError("no catalog/schema")
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("YGG_GENIE_SPACE", None)
            rc, _out = self._run(["genie", "console"])
        self.assertEqual(rc, 2)

    def test_ask_auto_creates_default_space(self):
        self.client.genie.ensure_default_space.return_value = MagicMock(
            space_id="auto-1", title="Yggdrasil Genie",
        )
        self.client.genie.ask.return_value = _answer(text="42")
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("YGG_GENIE_SPACE", None)
            rc, out = self._run(["genie", "ask", "the answer?"])
        self.assertEqual(rc, 0)
        self.client.genie.ask.assert_called_once_with("the answer?", space_id="auto-1")
        self.assertIn("42", out)

    def test_create_builds_space_from_discovered_tables(self):
        self.client.genie.discover_tables.return_value = ["c.s.t1", "c.s.t2"]
        self.client.genie.create_space.return_value = MagicMock(
            space_id="new-1", title="Yggdrasil Genie",
        )
        rc, out = self._run(["genie", "create", "--catalog", "c", "--schema", "s"])
        self.assertEqual(rc, 0)
        self.client.genie.discover_tables.assert_called_once_with(catalog="c", schema="s")
        self.client.genie.create_space.assert_called_once_with(
            tables=["c.s.t1", "c.s.t2"], title=None, warehouse_id=None,
        )
        self.assertIn("new-1", out)

    def test_create_explicit_tables(self):
        self.client.genie.create_space.return_value = MagicMock(
            space_id="new-2", title="My Space",
        )
        rc, _out = self._run([
            "genie", "create", "--tables", "a.b.c, a.b.d", "--title", "My Space",
        ])
        self.assertEqual(rc, 0)
        self.client.genie.discover_tables.assert_not_called()
        self.client.genie.create_space.assert_called_once_with(
            tables=["a.b.c", "a.b.d"], title="My Space", warehouse_id=None,
        )


# ---------------------------------------------------------------------------
# Interactive console (GenieConsole) — driven with scripted input
# ---------------------------------------------------------------------------


class TestGenieConsole(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.genie.space.return_value.title = "Sales"

    def _run(self, lines: list[str]) -> str:
        from yggdrasil.databricks.cli.services.genie import GenieConsole

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

        self.client.sql.execute.return_value.to_arrow_table.return_value = pa.table({"n": [1, 2]})
        out = self._run(["/sql SELECT 1", "/quit"])
        self.client.sql.execute.assert_called_once_with("SELECT 1")
        self.assertIn("n", out)

    def test_plain_text_asks_genie(self):
        ans = _answer(text="Revenue is up.")
        self.client.genie.space.return_value.start_conversation.return_value = (MagicMock(), ans)
        out = self._run(["how is revenue?", "/quit"])
        self.assertIn("Revenue is up.", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
