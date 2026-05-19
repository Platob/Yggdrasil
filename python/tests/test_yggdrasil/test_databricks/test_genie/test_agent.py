"""Tests for :class:`yggdrasil.databricks.genie.GenieAgent`.

Exercises the local-orchestration surface on top of the Genie service —
output-directory resolution, save dispatch (parquet / csv / arrow / json /
text), the ``run`` / ``chat`` flows, history bookkeeping, and the safe-by-
default tool registry.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pyarrow as pa
from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieMessage,
    GenieQueryAttachment,
    MessageStatus,
    TextAttachment,
)

from yggdrasil.databricks.genie import AGENT_SAVE_FORMATS, GenieAgent
from yggdrasil.databricks.tests import DatabricksTestCase


def _build_completed_message(
    *,
    space_id: str = "space-1",
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    text: str | None = "answer",
    query: str | None = None,
    attachment_id: str = "att-1",
    statement_id: str = "stmt-1",
) -> GenieMessage:
    attachments = [
        GenieAttachment(
            attachment_id=attachment_id,
            text=TextAttachment(id="tid", content=text) if text else None,
            query=(
                GenieQueryAttachment(
                    id="qid",
                    query=query,
                    statement_id=statement_id,
                )
                if query
                else None
            ),
        )
    ]
    return GenieMessage(
        space_id=space_id,
        conversation_id=conversation_id,
        content="q",
        message_id=message_id,
        id=message_id,
        status=MessageStatus.COMPLETED,
        attachments=attachments,
    )


class GenieAgentTestCase(DatabricksTestCase):
    """Base — wires the Genie API mock and isolates the output directory."""

    def setUp(self):
        super().setUp()
        self.genie_api = self.workspace_client.genie
        self.genie = self.client.genie
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

        # Direct the agent's output dir at a per-test tmp folder so
        # write tests never spam the user's real ~/.cache.
        self.tmp_dir = Path(self._make_tmp())
        self.genie.defaults = replace(
            self.genie.defaults,
            agent_output_dir=str(self.tmp_dir),
        )

    def _make_tmp(self) -> str:
        import tempfile

        d = tempfile.mkdtemp(prefix="ygg-genie-agent-")
        self.addCleanup(self._rm_tree, d)
        return d

    @staticmethod
    def _rm_tree(d: str) -> None:
        import shutil

        shutil.rmtree(d, ignore_errors=True)

    @property
    def agent(self) -> GenieAgent:
        return self.genie.agent

    def _start_returns(self, message: GenieMessage) -> None:
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            space_id=message.space_id,
        )


class TestAgentSingleton(GenieAgentTestCase):
    def test_agent_property_is_cached(self):
        first = self.genie.agent
        second = self.genie.agent
        self.assertIs(first, second)

    def test_agent_history_starts_empty(self):
        self.assertEqual(self.agent.history, [])
        self.assertIsNone(self.agent.last())

    def test_repr_describes_state(self):
        rep = repr(self.agent)
        self.assertIn("GenieAgent", rep)
        self.assertIn("history=0", rep)


class TestAgentOutputDir(GenieAgentTestCase):
    def test_output_dir_uses_configured_value(self):
        self.assertEqual(self.agent.output_dir, self.tmp_dir)

    def test_output_dir_falls_back_to_xdg(self):
        # Drop the explicit override.
        self.genie.defaults = replace(self.genie.defaults, agent_output_dir=None)
        # Inject an XDG_CACHE_HOME so we don't depend on the real env.
        xdg = self._make_tmp()
        import os

        prior = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = xdg
        try:
            resolved = self.agent.output_dir
        finally:
            if prior is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prior
        self.assertEqual(resolved, Path(xdg) / "yggdrasil" / "genie")

    def test_resolve_path_sanitises_components(self):
        message = _build_completed_message(
            space_id="space/with slash",
            conversation_id="conv:weird?",
            message_id="msg id",
        )
        self._start_returns(message)
        answer = self.genie.ask("q")
        resolved = self.agent.resolve_path(answer, format="parquet")
        # Slashes / colons / spaces collapse to '_' so the directory tree
        # stays flat under output_dir.
        self.assertEqual(resolved.parent.parent.parent, self.tmp_dir)
        self.assertNotIn("/", resolved.parent.name)
        self.assertNotIn(":", resolved.parent.name)
        self.assertTrue(resolved.name.endswith(".parquet"))


class TestAgentRun(GenieAgentTestCase):
    def test_run_appends_to_history(self):
        self._start_returns(_build_completed_message(text="hello"))
        answer = self.agent.run("hi")
        self.assertIs(self.agent.last(), answer)
        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(answer.text, "hello")

    def test_run_does_not_save_without_query_attachment(self):
        self._start_returns(_build_completed_message(text="just words", query=None))
        self.agent.run("hi", save=True)
        # No files written for a text-only answer.
        self.assertEqual(list(self.tmp_dir.rglob("*")), [])

    def test_run_saves_when_auto_save_default_set(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            agent_auto_save=True,
            agent_auto_save_format="json",
        )
        self._start_returns(_build_completed_message(text="hi"))
        self.agent.run("question")
        written = sorted(self.tmp_dir.rglob("*.json"))
        self.assertEqual(len(written), 1)


class TestAgentChat(GenieAgentTestCase):
    def test_chat_reuses_conversation_id_across_steps(self):
        first = _build_completed_message(
            message_id="m1", conversation_id="conv-A", text="first",
        )
        second = _build_completed_message(
            message_id="m2", conversation_id="conv-A", text="second",
        )
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: first,
            response=MagicMock(),
            conversation_id="conv-A",
            message_id="m1",
            space_id="space-1",
        )
        self.genie_api.create_message.return_value = Wait(
            waiter=lambda **kwargs: second,
            response=MagicMock(),
            conversation_id="conv-A",
            message_id="m2",
            space_id="space-1",
        )

        answers = self.agent.chat("a", "b")
        self.assertEqual([a.message_id for a in answers], ["m1", "m2"])
        self.genie_api.create_message.assert_called_once_with(
            space_id="space-1",
            conversation_id="conv-A",
            content="b",
        )

    def test_chat_respects_max_steps(self):
        self._start_returns(_build_completed_message(text="hi"))
        self.genie_api.create_message.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(message_id="m2"),
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="m2",
            space_id="space-1",
        )
        answers = self.agent.chat("a", "b", "c", max_steps=1)
        self.assertEqual(len(answers), 1)

    def test_chat_max_steps_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.agent.chat("a", max_steps=0)

    def test_chat_no_questions_returns_empty(self):
        self.assertEqual(self.agent.chat(), [])


class TestAgentSaveFormats(GenieAgentTestCase):
    """Each save format produces a readable file in the right place."""

    def _ask_with_table(self, table: pa.Table) -> "object":
        # Genie's start_conversation returns a message that *references* a
        # query attachment; the per-answer fetch_query_result method goes
        # through `service.api.get_message_attachment_query_result`.
        # We pre-build the answer and then stub that fetch by hand.
        message = _build_completed_message(
            text="hi", query="SELECT 1", attachment_id="att-x", statement_id="stmt-x",
        )
        self._start_returns(message)
        answer = self.genie.ask("q")
        # Bypass the API + warehouse round trip — install a fake
        # WarehouseStatementResult that just returns the in-memory table.
        fake_result = MagicMock()
        fake_result.read_arrow_table.return_value = table
        answer._statement_result_cache = fake_result
        return answer

    def test_save_parquet(self):
        table = pa.table({"x": [1, 2, 3]})
        answer = self._ask_with_table(table)
        path = self.agent.save(answer, format="parquet")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        import pyarrow.parquet as pq
        round_trip = pq.read_table(str(path))
        self.assertEqual(round_trip.column("x").to_pylist(), [1, 2, 3])

    def test_save_csv(self):
        table = pa.table({"x": [1, 2]})
        answer = self._ask_with_table(table)
        path = self.agent.save(answer, format="csv")
        self.assertTrue(path.exists())
        self.assertIn("x", path.read_text())

    def test_save_arrow(self):
        table = pa.table({"x": [1.0, 2.0]})
        answer = self._ask_with_table(table)
        path = self.agent.save(answer, format="arrow")
        self.assertTrue(path.exists())
        from pyarrow import feather
        self.assertEqual(feather.read_table(str(path)).column("x").to_pylist(), [1.0, 2.0])

    def test_save_json_does_not_require_query(self):
        # Text-only answer — json save is metadata-only.
        self._start_returns(_build_completed_message(text="hello world", query=None))
        answer = self.genie.ask("q")
        path = self.agent.save(answer, format="json")
        self.assertTrue(path.exists())

        from yggdrasil.pickle import json as ygg_json
        payload = ygg_json.loads(path.read_bytes())
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(payload["status"], "COMPLETED")

    def test_save_text_only(self):
        self._start_returns(_build_completed_message(text="hi there"))
        answer = self.genie.ask("q")
        path = self.agent.save(answer, format="text")
        self.assertEqual(path.read_text(), "hi there")

    def test_save_unknown_format_raises(self):
        self._start_returns(_build_completed_message(text="hi"))
        answer = self.genie.ask("q")
        with self.assertRaises(ValueError) as ctx:
            self.agent.save(answer, format="xlsx")
        self.assertIn("xlsx", str(ctx.exception))

    def test_save_returns_none_when_no_table(self):
        self._start_returns(_build_completed_message(text="hi", query=None))
        answer = self.genie.ask("q")
        # Parquet save with no query attachment — returns None, no file.
        self.assertIsNone(self.agent.save(answer, format="parquet"))

    def test_save_to_explicit_path(self):
        table = pa.table({"x": [1]})
        answer = self._ask_with_table(table)
        target = self.tmp_dir / "nested" / "deep" / "out.parquet"
        result = self.agent.save(answer, format="parquet", path=target)
        self.assertEqual(result, target)
        self.assertTrue(target.exists())

    def test_all_documented_formats_listed(self):
        self.assertIn("parquet", AGENT_SAVE_FORMATS)
        self.assertIn("csv", AGENT_SAVE_FORMATS)
        self.assertIn("arrow", AGENT_SAVE_FORMATS)
        self.assertIn("json", AGENT_SAVE_FORMATS)
        self.assertIn("text", AGENT_SAVE_FORMATS)


class TestAgentTools(GenieAgentTestCase):
    def test_default_tools_registered(self):
        for name in (
            "arrow_table", "polars", "pandas",
            "save", "save_parquet", "save_csv", "save_arrow",
            "save_json", "save_text",
            "ask", "chat", "inspect", "url", "refresh", "execute_query",
            "feedback",
            "output_dir", "history", "last", "reset", "defaults",
        ):
            self.assertIn(name, self.agent.tools, msg=name)

    def test_destructive_tools_not_registered_by_default(self):
        # The agent must NOT auto-expose space / conversation / message
        # deletion — those go through the service explicitly.
        for name in ("trash_space", "delete_space", "delete_conversation", "delete_message"):
            self.assertNotIn(name, self.agent.tools, msg=name)

    def test_register_custom_tool(self):
        self.agent.register_tool("shout", lambda s: s.upper())
        self.assertEqual(self.agent.run_tool("shout", "ok"), "OK")

    def test_register_non_callable_rejected(self):
        with self.assertRaises(TypeError):
            self.agent.register_tool("bad", 42)

    def test_unknown_tool_raises_keyerror_with_options(self):
        with self.assertRaises(KeyError) as ctx:
            self.agent.run_tool("nope")
        # Hint to the caller — which tools ARE registered.
        self.assertIn("nope", str(ctx.exception))

    def test_unregister_tool(self):
        self.agent.unregister_tool("save_text")
        self.assertNotIn("save_text", self.agent.tools)

    def test_inspect_returns_summary_dict(self):
        self._start_returns(_build_completed_message(
            text="hello", query="SELECT 1", statement_id="stmt-9",
        ))
        answer = self.genie.ask("q")
        info = self.agent.inspect(answer)
        self.assertEqual(info["text"], "hello")
        self.assertEqual(info["query"], "SELECT 1")
        self.assertEqual(info["statement_id"], "stmt-9")
        self.assertEqual(info["status"], "COMPLETED")

    def test_reset_clears_history(self):
        self._start_returns(_build_completed_message(text="hi"))
        self.agent.run("q")
        self.assertEqual(len(self.agent.history), 1)
        self.agent.reset()
        self.assertEqual(self.agent.history, [])
