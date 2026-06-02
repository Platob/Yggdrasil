"""Unit tests for the Databricks Genie service, resources, and agent.

Exercises space / conversation resolution, the GenieAnswer wrapper
(text / sql / suggested-questions / status / result fetch), the
autonomous agent's follow-up loop, and the defaults dataclass — all on
mocked SDK calls.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from databricks.sdk.errors import NotFound
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieConversationSummary,
    GenieMessage,
    GenieQueryAttachment,
    GenieSpace as SdkGenieSpace,
    GenieSuggestedQuestionsAttachment,
    TextAttachment,
)

from yggdrasil.databricks.genie import (
    DEFAULT_GENIE_WAIT,
    AgentRun,
    Genie,
    GenieAgent,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)
from yggdrasil.databricks.tests import DatabricksTestCase

#: Module path for patching the default-space cache helpers.
_SVC = "yggdrasil.databricks.genie.service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_attachment(content: str) -> GenieAttachment:
    return GenieAttachment(attachment_id="t1", text=TextAttachment(content=content))


def _query_attachment(sql: str, *, description: str = "the query") -> GenieAttachment:
    return GenieAttachment(
        attachment_id="q1",
        query=GenieQueryAttachment(query=sql, description=description),
    )


def _suggestions(*questions: str) -> GenieAttachment:
    return GenieAttachment(
        attachment_id="s1",
        suggested_questions=GenieSuggestedQuestionsAttachment(questions=list(questions)),
    )


def _message(
    *,
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    content: str = "the question",
    attachments=None,
    status: str = "COMPLETED",
) -> GenieMessage:
    from databricks.sdk.service.dashboards import MessageStatus

    return GenieMessage(
        space_id="space-1",
        conversation_id=conversation_id,
        message_id=message_id,
        content=content,
        attachments=attachments or [],
        status=MessageStatus(status),
    )


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class GenieTestCase(DatabricksTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie_api = self.workspace_client.genie

    @property
    def genie(self) -> Genie:
        return self.client.genie


# ---------------------------------------------------------------------------
# Wiring + defaults
# ---------------------------------------------------------------------------


class TestWiring(GenieTestCase):
    def test_client_genie_cached(self):
        g = self.client.genie
        self.assertIsInstance(g, Genie)
        self.assertIs(self.client.genie, g)

    def test_service_shortcut_via_inherited_property(self):
        self.assertIs(self.client.sql.genie, self.client.genie)

    def test_defaults(self):
        d = self.genie.defaults
        self.assertIsInstance(d, GenieDefaults)
        self.assertIs(d.wait, DEFAULT_GENIE_WAIT)
        self.assertIsNone(d.space_id)

    def test_defaults_replace(self):
        self.genie.defaults = replace(self.genie.defaults, space_id="space-9")
        self.assertEqual(self.genie.defaults.space_id, "space-9")

    def test_space_requires_id(self):
        with self.assertRaises(ValueError):
            self.genie.space()

    def test_space_uses_default(self):
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")
        sp = self.genie.space()
        self.assertIsInstance(sp, GenieSpace)
        self.assertEqual(sp.space_id, "space-1")

    def test_space_explore_url(self):
        self.assertEqual(
            self.genie.space("space-1").explore_url.to_string(),
            "https://test.databricks.net/genie/rooms/space-1",
        )


# ---------------------------------------------------------------------------
# Spaces listing
# ---------------------------------------------------------------------------


class TestSpaces(GenieTestCase):
    def test_list_spaces_paginates(self):
        page1 = MagicMock()
        page1.spaces = [SdkGenieSpace(space_id="a", title="Sales")]
        page1.next_page_token = "tok"
        page2 = MagicMock()
        page2.spaces = [SdkGenieSpace(space_id="b", title="Ops")]
        page2.next_page_token = None
        self.genie_api.list_spaces.side_effect = [page1, page2]

        spaces = list(self.genie.list_spaces())
        self.assertEqual([s.space_id for s in spaces], ["a", "b"])
        self.assertEqual(spaces[0].title, "Sales")
        self.assertEqual(self.genie_api.list_spaces.call_count, 2)

    def test_find_space_by_title(self):
        page = MagicMock()
        page.spaces = [
            SdkGenieSpace(space_id="a", title="Sales"),
            SdkGenieSpace(space_id="b", title="Ops"),
        ]
        page.next_page_token = None
        self.genie_api.list_spaces.return_value = page
        match = self.genie.find_space(title="ops")
        self.assertIsNotNone(match)
        self.assertEqual(match.space_id, "b")

    def test_space_infos_caches(self):
        self.genie_api.get_space.return_value = SdkGenieSpace(
            space_id="space-1", title="Sales", warehouse_id="wh-1",
        )
        sp = self.genie.space("space-1")
        self.assertEqual(sp.title, "Sales")
        self.assertEqual(sp.warehouse_id, "wh-1")
        _ = sp.infos
        self.genie_api.get_space.assert_called_once_with(space_id="space-1")

    def test_space_exists_false_on_not_found(self):
        self.genie_api.get_space.side_effect = NotFound("missing")
        self.assertFalse(self.genie.space("space-x").exists())


# ---------------------------------------------------------------------------
# Space creation
# ---------------------------------------------------------------------------


class TestCreateSpace(GenieTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie_api.create_space.return_value = SdkGenieSpace(
            space_id="new-1", title="Yggdrasil Genie", warehouse_id="wh-1",
        )

    def test_create_space_builds_serialized_from_tables(self):
        from yggdrasil.pickle import json as ygg_json

        with patch.object(
            self.client.warehouses, "find_default",
            return_value=MagicMock(warehouse_id="wh-default"),
        ):
            space = self.genie.create_space(tables=["c.s.a", "c.s.b"], title="My Space")

        self.assertEqual(space.space_id, "new-1")
        call = self.genie_api.create_space.call_args
        self.assertEqual(call.kwargs["warehouse_id"], "wh-default")
        self.assertEqual(call.kwargs["title"], "My Space")
        parsed = ygg_json.loads(call.kwargs["serialized_space"])
        self.assertEqual(parsed, {
            "version": 2,
            "data_sources": {"tables": [{"identifier": "c.s.a"}, {"identifier": "c.s.b"}]},
        })

    def test_create_space_uses_default_warehouse_from_defaults(self):
        self.genie.defaults = replace(self.genie.defaults, warehouse_id="wh-cfg")
        self.genie.create_space(tables=["c.s.a"])
        self.assertEqual(
            self.genie_api.create_space.call_args.kwargs["warehouse_id"], "wh-cfg",
        )

    def test_create_space_default_title(self):
        self.genie.defaults = replace(self.genie.defaults, warehouse_id="wh")
        self.genie.create_space(tables=["c.s.a"])
        self.assertEqual(
            self.genie_api.create_space.call_args.kwargs["title"], "Yggdrasil Genie",
        )

    def test_create_space_requires_tables_or_serialized(self):
        self.genie.defaults = replace(self.genie.defaults, warehouse_id="wh")
        with self.assertRaises(ValueError):
            self.genie.create_space()

    def test_create_space_requires_a_warehouse(self):
        with patch.object(self.client.warehouses, "find_default", return_value=None):
            with self.assertRaises(ValueError):
                self.genie.create_space(tables=["c.s.a"])

    def test_discover_tables_via_show_tables(self):
        import pyarrow as pa

        result = MagicMock()
        result.to_arrow_table.return_value = pa.table({
            "database": ["s", "s", "s"],
            "tableName": ["a", "b", "tmp"],
            "isTemporary": [False, False, True],
        })
        with patch.object(self.client.sql, "execute", return_value=result) as ex:
            tables = self.genie.discover_tables(catalog="c", schema="s")
        ex.assert_called_once_with("SHOW TABLES IN `c`.`s`")
        # Temp tables are skipped; names are three-part.
        self.assertEqual(tables, ["c.s.a", "c.s.b"])

    def test_discover_tables_requires_catalog_schema(self):
        with self.assertRaises(ValueError):
            self.genie.discover_tables()

    def test_ensure_default_space_cache_hit(self):
        # A cached id that still exists is reused without listing or creating.
        self.genie_api.get_space.return_value = SdkGenieSpace(
            space_id="cached", title="Yggdrasil Genie",
        )
        with patch(f"{_SVC}._read_default_space", return_value="cached"):
            space = self.genie.ensure_default_space()
        self.assertEqual(space.space_id, "cached")
        self.genie_api.create_space.assert_not_called()
        self.genie_api.list_spaces.assert_not_called()

    def test_ensure_default_space_reuses_existing_by_title(self):
        page = MagicMock()
        page.spaces = [SdkGenieSpace(space_id="exists", title="Yggdrasil Genie")]
        page.next_page_token = None
        self.genie_api.list_spaces.return_value = page

        with patch(f"{_SVC}._read_default_space", return_value=None), \
                patch(f"{_SVC}._write_default_space") as wr:
            space = self.genie.ensure_default_space()
        self.assertEqual(space.space_id, "exists")
        self.genie_api.create_space.assert_not_called()
        wr.assert_called_once()  # cache the found id

    def test_ensure_default_space_creates_when_missing(self):
        page = MagicMock(spaces=[], next_page_token=None)
        self.genie_api.list_spaces.return_value = page
        with patch(f"{_SVC}._read_default_space", return_value=None), \
                patch(f"{_SVC}._write_default_space") as wr, \
                patch.object(self.genie, "discover_tables", return_value=["c.s.a"]), \
                patch.object(self.client.warehouses, "find_default",
                             return_value=MagicMock(warehouse_id="wh")):
            space = self.genie.ensure_default_space()
        self.assertEqual(space.space_id, "new-1")
        self.genie_api.create_space.assert_called_once()
        wr.assert_called_once_with(  # cache the created id
            wr.call_args.args[0], "new-1",
        )

    def test_space_trash(self):
        self.genie.space("space-1").trash()
        self.genie_api.trash_space.assert_called_once_with(space_id="space-1")

    def test_space_trash_missing_ok(self):
        self.genie_api.trash_space.side_effect = NotFound("gone")
        self.genie.space("space-1").trash(missing_ok=True)


# ---------------------------------------------------------------------------
# Ask / conversation
# ---------------------------------------------------------------------------


class TestAsk(GenieTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

    def test_ask_starts_conversation_and_waits(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("Revenue was $1.2M.")],
        )
        answer = self.genie.ask("how much revenue?")
        self.assertIsInstance(answer, GenieAnswer)
        call = self.genie_api.start_conversation_and_wait.call_args
        self.assertEqual(call.kwargs["space_id"], "space-1")
        self.assertEqual(call.kwargs["content"], "how much revenue?")
        self.assertEqual(answer.text, "Revenue was $1.2M.")
        self.assertTrue(answer.is_complete)
        self.assertFalse(answer.has_query)

    def test_ask_with_query_attachment(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[
                _text_attachment("Here are the top customers."),
                _query_attachment("SELECT name FROM customers LIMIT 5", description="top 5"),
            ],
        )
        answer = self.genie.ask("top customers")
        self.assertTrue(answer.has_query)
        self.assertEqual(answer.sql, "SELECT name FROM customers LIMIT 5")
        self.assertEqual(answer.description, "top 5")
        self.assertEqual(answer.attachment_id, "q1")

    def test_ask_suggested_questions(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            content="vague", attachments=[_suggestions("by region?", "by month?")],
        )
        answer = self.genie.ask("show me sales")
        self.assertEqual(answer.questions, ("by region?", "by month?"))
        self.assertIsNone(answer.text)

    def test_start_conversation_returns_live_handle(self):
        self.genie_api.start_conversation_and_wait.return_value = _message()
        conv, answer = self.genie.space().start_conversation("hi")
        self.assertIsInstance(conv, GenieConversation)
        self.assertEqual(conv.conversation_id, "conv-1")

    def test_conversation_follow_up(self):
        self.genie_api.start_conversation_and_wait.return_value = _message()
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2", attachments=[_text_attachment("EMEA only.")],
        )
        conv, _ = self.genie.space().start_conversation("revenue")
        follow = conv.ask("just EMEA")
        call = self.genie_api.create_message_and_wait.call_args
        self.assertEqual(call.kwargs["conversation_id"], "conv-1")
        self.assertEqual(call.kwargs["content"], "just EMEA")
        self.assertEqual(follow.text, "EMEA only.")

    def test_conversations_listing(self):
        resp = MagicMock()
        resp.conversations = [
            GenieConversationSummary(conversation_id="c1", title="t1"),
            GenieConversationSummary(conversation_id="c2", title="t2"),
        ]
        self.genie_api.list_conversations.return_value = resp
        convs = list(self.genie.space().conversations())
        self.assertEqual([c.conversation_id for c in convs], ["c1", "c2"])

    def test_failed_answer(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(status="FAILED")
        answer = self.genie.ask("boom")
        self.assertTrue(answer.failed)
        self.assertFalse(answer.is_complete)


# ---------------------------------------------------------------------------
# Result materialisation
# ---------------------------------------------------------------------------


class TestResult(GenieTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

    def _query_result_response(self):
        """Build a Genie query-result response with an inline statement body."""
        from databricks.sdk.service.sql import (
            ColumnInfo,
            ColumnInfoTypeName,
            ResultData,
            ResultManifest,
            ResultSchema,
            StatementResponse,
        )

        statement = StatementResponse(
            statement_id="stmt-9",
            manifest=ResultManifest(
                schema=ResultSchema(columns=[
                    ColumnInfo(name="table_name", type_name=ColumnInfoTypeName.STRING),
                    ColumnInfo(name="record_count", type_name=ColumnInfoTypeName.LONG),
                ]),
                total_row_count=2,
            ),
            result=ResultData(data_array=[["a", "10"], ["b", "20"]], row_count=2),
        )
        resp = MagicMock()
        resp.statement_response = statement
        return resp

    def test_result_none_for_text_only(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("just text")],
        )
        answer = self.genie.ask("q")
        self.assertIsNone(answer.statement_response)
        self.assertIsNone(answer.to_arrow())
        self.assertEqual(answer.rows(), [])

    def test_result_fetches_and_converts_inline(self):
        import pyarrow as pa

        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_query_attachment("SELECT 1")],
        )
        self.genie_api.get_message_attachment_query_result.return_value = (
            self._query_result_response()
        )

        answer = self.genie.ask("top customers")
        table = answer.to_arrow()

        self.genie_api.get_message_attachment_query_result.assert_called_once_with(
            space_id="space-1", conversation_id="conv-1",
            message_id="msg-1", attachment_id="q1",
        )
        self.assertEqual(table.column_names, ["table_name", "record_count"])
        # LONG column is cast to int64; string stays string.
        self.assertEqual(table.schema.field("record_count").type, pa.int64())
        self.assertEqual(table.schema.field("table_name").type, pa.string())
        self.assertEqual(answer.rows(), [
            {"table_name": "a", "record_count": 10},
            {"table_name": "b", "record_count": 20},
        ])
        self.assertEqual(answer.row_count, 2)

    def test_result_table_cached(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_query_attachment("SELECT 1")],
        )
        self.genie_api.get_message_attachment_query_result.return_value = (
            self._query_result_response()
        )
        answer = self.genie.ask("q")
        _ = answer.to_arrow()
        _ = answer.to_arrow()
        # The expensive fetch happens once; the table is cached.
        self.genie_api.get_message_attachment_query_result.assert_called_once()


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class TestFeedback(GenieTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

    def test_thumbs_up(self):
        from databricks.sdk.service.dashboards import GenieFeedbackRating

        self.genie_api.start_conversation_and_wait.return_value = _message()
        answer = self.genie.ask("q")
        answer.thumbs_up(comment="nice")
        call = self.genie_api.send_message_feedback.call_args
        self.assertEqual(call.kwargs["rating"], GenieFeedbackRating.POSITIVE)
        self.assertEqual(call.kwargs["comment"], "nice")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TestAgent(GenieTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

    def test_agent_handle(self):
        agent = self.genie.agent()
        self.assertIsInstance(agent, GenieAgent)
        self.assertEqual(agent.space.space_id, "space-1")

    def test_agent_stops_on_query_answer(self):
        # Opening question immediately yields a data answer → one turn.
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[
                _text_attachment("Top customers:"),
                _query_attachment("SELECT name FROM c"),
            ],
        )
        run = self.genie.agent().run("top customers")
        self.assertIsInstance(run, AgentRun)
        self.assertEqual(len(run.turns), 1)
        self.assertFalse(run.turns[0].autonomous)
        self.assertEqual(run.sql, "SELECT name FROM c")
        self.genie_api.create_message_and_wait.assert_not_called()

    def test_agent_follows_suggestions_until_data(self):
        # Turn 1: only suggestions → agent picks the first.
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_suggestions("by region?", "by month?")],
        )
        # Turn 2: the follow-up yields a query answer → stop.
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2",
            attachments=[_query_attachment("SELECT region, sales FROM t")],
        )
        run = self.genie.agent(max_turns=4).run("show sales")
        self.assertEqual(len(run.turns), 2)
        self.assertFalse(run.turns[0].autonomous)
        self.assertTrue(run.turns[1].autonomous)
        self.assertEqual(run.turns[1].question, "by region?")
        self.assertEqual(run.sql, "SELECT region, sales FROM t")
        # The agent asked exactly one follow-up.
        self.genie_api.create_message_and_wait.assert_called_once()

    def test_agent_respects_max_turns(self):
        # Always suggestions, never a data answer → bounded by max_turns.
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_suggestions("more?")],
        )
        self.genie_api.create_message_and_wait.return_value = _message(
            attachments=[_suggestions("even more?")],
        )
        run = self.genie.agent(max_turns=3).run("vague goal")
        self.assertEqual(len(run.turns), 3)
        # Opening turn + 2 autonomous follow-ups == 3 turns.
        self.assertEqual(self.genie_api.create_message_and_wait.call_count, 2)

    def test_agent_stops_when_no_suggestions(self):
        # Plain text answer, no query, no suggestions → nothing to follow.
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("I don't know.")],
        )
        run = self.genie.agent(max_turns=4).run("???")
        self.assertEqual(len(run.turns), 1)
        self.genie_api.create_message_and_wait.assert_not_called()

    def test_agent_summary_renders_transcript(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_suggestions("by region?")],
        )
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2",
            attachments=[
                _text_attachment("By region:"),
                _query_attachment("SELECT region FROM t"),
            ],
        )
        run = self.genie.agent().run("sales")
        summary = run.summary()
        self.assertIn("Goal: sales", summary)
        self.assertIn("(agent) by region?", summary)
        self.assertIn("SELECT region FROM t", summary)

    def test_agent_no_follow_when_disabled(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_suggestions("by region?")],
        )
        run = self.genie.agent(follow_suggestions=False, max_turns=4).run("sales")
        self.assertEqual(len(run.turns), 1)
        self.genie_api.create_message_and_wait.assert_not_called()

    def test_agent_callable_planner_drives_until_done(self):
        # A callable planner returns the next question, then None (= done).
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("opening")],
        )
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2", attachments=[_query_attachment("SELECT 1")],
        )
        seen = []

        def planner(run):
            seen.append(len(run.turns))
            return "drill deeper" if len(run.turns) < 2 else None

        run = self.genie.agent(planner=planner, max_turns=5).run("goal")
        self.assertEqual(len(run.turns), 2)
        self.assertTrue(run.turns[1].autonomous)
        self.assertEqual(run.turns[1].question, "drill deeper")
        self.assertEqual(seen, [1, 2])


class TestAgentLLMPlanner(GenieTestCase):
    """The fully-autonomous planner brain: a Model Serving LLM picks each step."""

    def setUp(self) -> None:
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")
        self.serving_api = self.workspace_client.serving_endpoints

    def _planner_reply(self, text: str):
        from databricks.sdk.service.serving import (
            ChatMessage,
            ChatMessageRole,
            QueryEndpointResponse,
            V1ResponseChoiceElement,
        )

        return QueryEndpointResponse(
            choices=[V1ResponseChoiceElement(
                index=0,
                message=ChatMessage(role=ChatMessageRole.ASSISTANT, content=text),
            )],
        )

    def test_llm_planner_asks_then_stops_on_done(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("opening answer")],
        )
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2", attachments=[_query_attachment("SELECT 1")],
        )
        # Planner: first decide a follow-up, then say DONE.
        self.serving_api.query.side_effect = [
            self._planner_reply("break it down by region"),
            self._planner_reply("DONE"),
        ]

        run = self.genie.agent(
            planner="databricks-claude-sonnet-4", max_turns=5,
        ).run("explain sales")

        self.assertEqual(len(run.turns), 2)
        self.assertEqual(run.turns[1].question, "break it down by region")
        self.assertTrue(run.turns[1].autonomous)
        # The planner LLM was queried against the serving endpoint.
        first_call = self.serving_api.query.call_args_list[0]
        self.assertEqual(first_call.kwargs["name"], "databricks-claude-sonnet-4")

    def test_agent_true_planner_uses_default_endpoint(self):
        from yggdrasil.databricks.genie.agent import DEFAULT_PLANNER_ENDPOINT

        agent = self.genie.agent(planner=True)
        self.assertEqual(agent.planner, DEFAULT_PLANNER_ENDPOINT)

    def test_llm_planner_strips_bullet_prefix(self):
        self.genie_api.start_conversation_and_wait.return_value = _message(
            attachments=[_text_attachment("opening")],
        )
        self.genie_api.create_message_and_wait.return_value = _message(
            message_id="msg-2", attachments=[_query_attachment("SELECT 1")],
        )
        self.serving_api.query.side_effect = [
            self._planner_reply("- what about EMEA?\nsome trailing chatter"),
            self._planner_reply("DONE"),
        ]
        run = self.genie.agent(planner="ep", max_turns=4).run("goal")
        self.assertEqual(run.turns[1].question, "what about EMEA?")
