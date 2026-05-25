"""Unit tests for the Databricks Genie service and resources.

Exercises the simplest ``client.genie.ask("…")`` path plus the conversation /
space / feedback flows on top of mocked SDK calls.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace
from unittest.mock import MagicMock

from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieMessage,
    GenieQueryAttachment,
    GenieStartConversationResponse,
    MessageStatus,
    TextAttachment,
)

from yggdrasil.databricks.genie import (
    DEFAULT_MANAGED_SPACE_TITLE,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_WAIT,
    Genie,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
    build_serialized_space,
)
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.dataclasses import WaitingConfig


def _build_completed_message(
    *,
    space_id: str = "space-1",
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    content: str = "question",
    text: str | None = "Genie's natural-language reply",
    query: str | None = None,
    attachment_id: str = "att-1",
    statement_id: str = "stmt-1",
) -> GenieMessage:
    attachments: list[GenieAttachment] = [
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
        content=content,
        message_id=message_id,
        id=message_id,
        status=MessageStatus.COMPLETED,
        attachments=attachments,
    )


class GenieTestCase(DatabricksTestCase):
    """Helper base that exposes a Genie-API mock."""

    def setUp(self):
        super().setUp()
        # workspace_client.genie returns the GenieAPI mock
        self.genie_api = self.workspace_client.genie

    @property
    def genie(self) -> Genie:
        return self.client.genie


class TestGenieDefaults(GenieTestCase):
    """Service-level default configuration."""

    def test_defaults_attached_to_service(self):
        self.assertIsInstance(self.genie.defaults, GenieDefaults)
        self.assertIs(self.genie.defaults.wait, DEFAULT_WAIT)
        self.assertEqual(self.genie.defaults.wait.timeout, DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(self.genie.defaults.auto_pick_space)

    def test_defaults_override_in_place(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id="space-x",
            wait=WaitingConfig.from_(60),
        )
        self.assertEqual(self.genie.defaults.space_id, "space-x")
        self.assertEqual(self.genie.defaults.timeout, dt.timedelta(seconds=60))


class TestGenieAsk(GenieTestCase):
    """The simplest ``Genie.ask`` flow."""

    def setUp(self):
        super().setUp()
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")

        # Build a fake Wait[GenieMessage] returned by start_conversation
        self.completed = _build_completed_message(
            message_id="msg-1",
            text="There were 1,234 orders last month.",
        )
        self.start_response = GenieStartConversationResponse(
            message_id="msg-1",
            conversation_id="conv-1",
            conversation=MagicMock(id="conv-1"),
            message=MagicMock(id="msg-1"),
        )
        self.start_waiter = Wait(
            waiter=lambda **kwargs: self.completed,
            response=self.start_response,
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )
        self.genie_api.start_conversation.return_value = self.start_waiter

    def test_ask_starts_conversation_and_returns_answer(self):
        answer = self.genie.ask("How many orders last month?")

        self.assertIsInstance(answer, GenieAnswer)
        self.assertEqual(answer.space_id, "space-1")
        self.assertEqual(answer.conversation_id, "conv-1")
        self.assertEqual(answer.message_id, "msg-1")
        self.assertEqual(answer.text, "There were 1,234 orders last month.")
        self.assertTrue(answer.is_completed)
        self.genie_api.start_conversation.assert_called_once_with(
            space_id="space-1",
            content="How many orders last month?",
        )

    def test_ask_uses_default_timeout(self):
        original_result = self.start_waiter.result
        captured_timeout = []

        def _record(timeout=None, callback=None):
            captured_timeout.append(timeout)
            return self.completed

        self.start_waiter.result = _record  # type: ignore[method-assign]
        try:
            self.genie.ask("Question")
        finally:
            self.start_waiter.result = original_result  # type: ignore[method-assign]

        self.assertEqual(
            captured_timeout[0],
            dt.timedelta(seconds=DEFAULT_TIMEOUT_SECONDS),
        )

    def test_ask_with_explicit_wait_seconds(self):
        captured = []

        def _record(timeout=None, callback=None):
            captured.append(timeout)
            return self.completed

        self.start_waiter.result = _record  # type: ignore[method-assign]
        self.genie.ask("Q", wait=30)
        self.assertEqual(captured[0], dt.timedelta(seconds=30))

    def test_ask_with_explicit_wait_config(self):
        captured = []

        def _record(timeout=None, callback=None):
            captured.append(timeout)
            return self.completed

        self.start_waiter.result = _record  # type: ignore[method-assign]
        self.genie.ask("Q", wait=WaitingConfig(timeout=45.0))
        self.assertEqual(captured[0], dt.timedelta(seconds=45))

    def test_ask_continues_conversation_when_id_given(self):
        create_waiter = Wait(
            waiter=lambda **kwargs: self.completed,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-2",
            space_id="space-1",
        )
        self.genie_api.create_message.return_value = create_waiter

        answer = self.genie.ask("Follow-up", conversation_id="conv-1")

        self.assertEqual(answer.conversation_id, "conv-1")
        self.genie_api.create_message.assert_called_once_with(
            space_id="space-1",
            conversation_id="conv-1",
            content="Follow-up",
        )
        self.genie_api.start_conversation.assert_not_called()


class TestGenieSpaceResolution(GenieTestCase):
    """Default space id resolution rules."""

    def test_explicit_space_id_wins(self):
        self.genie.defaults = replace(self.genie.defaults, space_id="default-space")
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(),
            response=MagicMock(),
            conversation_id="c",
            message_id="m",
            space_id="explicit",
        )
        self.genie.ask("q", space_id="explicit")
        self.genie_api.start_conversation.assert_called_once_with(
            space_id="explicit",
            content="q",
        )

    def test_auto_pick_disabled_requires_space(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id=None,
            auto_pick_space=False,
        )
        with self.assertRaises(ValueError):
            self.genie.ask("q")

    def test_auto_pick_uses_first_listed_space(self):
        self.genie.defaults = replace(self.genie.defaults, space_id=None)

        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="space-A", title="Alpha"),
        ]
        self.genie_api.list_spaces.return_value = spaces_page
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(),
            response=MagicMock(),
            conversation_id="c",
            message_id="m",
            space_id="space-A",
        )

        self.genie.ask("q")

        self.genie_api.start_conversation.assert_called_once_with(
            space_id="space-A",
            content="q",
        )

    def test_auto_pick_filters_by_space_name(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id=None,
            space_name="Beta",
        )

        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="space-A", title="Alpha"),
            MagicMock(space_id="space-B", title="Beta"),
        ]
        self.genie_api.list_spaces.return_value = spaces_page
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(),
            response=MagicMock(),
            conversation_id="c",
            message_id="m",
            space_id="space-B",
        )

        self.genie.ask("q")

        self.genie_api.start_conversation.assert_called_once_with(
            space_id="space-B",
            content="q",
        )


class TestGenieSpaceResource(GenieTestCase):
    """``GenieSpace`` helpers."""

    def test_space_ask_routes_to_service(self):
        message = _build_completed_message(
            space_id="space-9", conversation_id="conv-9", message_id="msg-9", text="hi",
        )
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id="conv-9",
            message_id="msg-9",
            space_id="space-9",
        )
        space = self.genie.space("space-9")
        answer = space.ask("hello")
        self.assertEqual(answer.text, "hi")
        self.assertEqual(answer.space_id, "space-9")

    def test_start_conversation_returns_conversation_and_answer(self):
        message = _build_completed_message(
            space_id="space-5", conversation_id="conv-5", message_id="msg-5",
        )
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id="conv-5",
            message_id="msg-5",
            space_id="space-5",
        )
        space = self.genie.space("space-5")
        conv, answer = space.start_conversation("hi")

        self.assertIsInstance(conv, GenieConversation)
        self.assertEqual(conv.conversation_id, "conv-5")
        self.assertEqual(answer.conversation_id, "conv-5")


class TestGenieAnswerFollowUp(GenieTestCase):
    """``GenieAnswer.ask`` continues the same conversation."""

    def test_follow_up_uses_create_message(self):
        # First ask → start_conversation
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(message_id="m1"),
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="m1",
            space_id="space-1",
        )
        first = self.genie.ask("first")

        # Second ask via the answer → create_message in same conversation
        self.genie_api.create_message.return_value = Wait(
            waiter=lambda **kwargs: _build_completed_message(message_id="m2"),
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="m2",
            space_id="space-1",
        )
        second = first.ask("second")

        self.assertEqual(second.message_id, "m2")
        self.assertEqual(second.conversation_id, "conv-1")
        self.genie_api.create_message.assert_called_once_with(
            space_id="space-1",
            conversation_id="conv-1",
            content="second",
        )


class TestGenieAnswerAttachments(GenieTestCase):
    """Query attachment unwrapping."""

    def test_query_attachment_exposes_sql(self):
        message = _build_completed_message(
            text=None,
            query="SELECT count(*) FROM orders",
            attachment_id="att-9",
            statement_id="stmt-99",
        )
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id="c",
            message_id="m",
            space_id="s",
        )

        self.genie.defaults = replace(self.genie.defaults, space_id="s")
        answer = self.genie.ask("show orders")

        self.assertEqual(answer.query, "SELECT count(*) FROM orders")
        self.assertEqual(answer.statement_id, "stmt-99")
        self.assertEqual(answer.attachment_id, "att-9")
        # No text on this one
        self.assertIsNone(answer.text)


class TestGenieFeedback(GenieTestCase):
    """Feedback path accepts strings or enum values."""

    def test_feedback_string_converted_to_enum(self):
        from databricks.sdk.service.dashboards import GenieFeedbackRating

        message = _build_completed_message(
            space_id="s", conversation_id="c", message_id="m",
        )
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id="c",
            message_id="m",
            space_id="s",
        )
        self.genie.defaults = replace(self.genie.defaults, space_id="s")
        answer = self.genie.ask("q")

        answer.feedback("positive", comment="nice")

        self.genie_api.send_message_feedback.assert_called_once_with(
            space_id="s",
            conversation_id="c",
            message_id="m",
            rating=GenieFeedbackRating.POSITIVE,
            comment="nice",
        )


class TestGenieManagedDefaults(GenieTestCase):
    """New ``GenieDefaults`` fields covering auto-create + cleanup."""

    def test_managed_defaults_have_sensible_off_state(self):
        d = self.genie.defaults
        self.assertFalse(d.auto_create_space)
        self.assertFalse(d.cleanup_dead_spaces)
        self.assertEqual(d.managed_space_title, DEFAULT_MANAGED_SPACE_TITLE)
        self.assertEqual(d.managed_space_tables, ())
        self.assertIsNone(d.managed_space_description)
        self.assertIsNone(d.managed_space_parent_path)

    def test_build_serialized_space_minimal(self):
        from yggdrasil.pickle import json as ygg_json

        payload = build_serialized_space(tables=("main.sales.orders",))
        body = ygg_json.loads(payload)
        self.assertEqual(body["version"], 1)
        self.assertEqual(
            body["data_sources"]["tables"],
            [{"identifier": "main.sales.orders"}],
        )
        self.assertNotIn("instructions", body)

    def test_build_serialized_space_with_instructions(self):
        from yggdrasil.pickle import json as ygg_json

        payload = build_serialized_space(
            tables=("main.sales.orders",),
            text_instructions=("Be brief.", "Quote dollar amounts."),
        )
        body = ygg_json.loads(payload)
        texts = body["instructions"]["text_instructions"]
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0]["content"], ["Be brief."])
        # Each instruction gets a fresh 32-char hex id (uuid4 hex).
        self.assertEqual(len(texts[0]["id"]), 32)
        self.assertNotEqual(texts[0]["id"], texts[1]["id"])


class TestGenieEnsureSpace(GenieTestCase):
    """``Genie.ensure_space`` resolve / auto-create flow."""

    def test_returns_existing_when_space_id_set(self):
        self.genie.defaults = replace(self.genie.defaults, space_id="space-X")
        space = self.genie.ensure_space()
        self.assertEqual(space.space_id, "space-X")
        # Did not list or create.
        self.genie_api.list_spaces.assert_not_called()
        self.genie_api.create_space.assert_not_called()

    def test_picks_matching_managed_title_over_first(self):
        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="other-id", title="Other"),
            MagicMock(space_id="managed-id", title=DEFAULT_MANAGED_SPACE_TITLE),
        ]
        self.genie_api.list_spaces.return_value = spaces_page

        space = self.genie.ensure_space()
        self.assertEqual(space.space_id, "managed-id")
        # Caches resolved id back on defaults so the next ask() is cheap.
        self.assertEqual(self.genie.defaults.space_id, "managed-id")
        self.genie_api.create_space.assert_not_called()

    def test_auto_create_requires_tables(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            auto_create_space=True,
            warehouse_id="wh-1",
        )
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[])
        with self.assertRaises(ValueError) as ctx:
            self.genie.ensure_space()
        self.assertIn("managed_space_tables", str(ctx.exception))

    def test_auto_create_calls_create_space(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            auto_create_space=True,
            warehouse_id="wh-1",
            managed_space_tables=("main.sales.orders",),
            managed_space_description="auto",
        )
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[])
        created = MagicMock(space_id="new-id", title=DEFAULT_MANAGED_SPACE_TITLE)
        self.genie_api.create_space.return_value = created

        space = self.genie.ensure_space()
        self.assertEqual(space.space_id, "new-id")
        self.assertEqual(self.genie.defaults.space_id, "new-id")

        call = self.genie_api.create_space.call_args
        self.assertEqual(call.kwargs["warehouse_id"], "wh-1")
        self.assertEqual(call.kwargs["title"], DEFAULT_MANAGED_SPACE_TITLE)
        self.assertEqual(call.kwargs["description"], "auto")
        # serialized_space carries our minimal payload.
        self.assertIn("main.sales.orders", call.kwargs["serialized_space"])

    def test_missing_space_without_auto_create_raises(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id=None,
            auto_create_space=False,
        )
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[])
        with self.assertRaises(ValueError) as ctx:
            self.genie.ensure_space()
        self.assertIn("auto_create_space", str(ctx.exception))


class TestResolveTableIdentifiers(GenieTestCase):
    """``Genie.resolve_table_identifiers`` resolves short names."""

    def _make_table_mock(self, catalog, schema, table):
        m = MagicMock()
        m.catalog_name = catalog
        m.schema_name = schema
        m.table_name = table
        return m

    def test_fully_qualified_passes_through(self):
        result = self.genie.resolve_table_identifiers(
            ("main.sales.orders",),
        )
        self.assertEqual(result, ("main.sales.orders",))

    def test_single_part_resolves_unique_match(self):
        self.workspace_client.tables.list.return_value = [
            MagicMock(
                name="trading_tgp_prd",
                catalog_name="trading",
                schema_name="tgp",
                table_id="t1",
            ),
        ]
        tbl = self._make_table_mock("trading", "tgp", "trading_tgp_prd")
        self.client.tables.list_tables = MagicMock(return_value=iter([tbl]))

        result = self.genie.resolve_table_identifiers(
            ("trading_tgp_prd",),
        )
        self.assertEqual(result, ("trading.tgp.trading_tgp_prd",))

    def test_two_part_resolves_unique_match(self):
        tbl = self._make_table_mock("main", "sales", "orders")
        self.client.tables.list_tables = MagicMock(return_value=iter([tbl]))

        result = self.genie.resolve_table_identifiers(
            ("sales.orders",),
        )
        self.assertEqual(result, ("main.sales.orders",))

    def test_single_part_no_match_raises(self):
        self.client.tables.list_tables = MagicMock(return_value=iter([]))

        with self.assertRaises(ValueError) as ctx:
            self.genie.resolve_table_identifiers(("nonexistent",))
        self.assertIn("not found", str(ctx.exception))

    def test_single_part_ambiguous_raises(self):
        tbl1 = self._make_table_mock("cat_a", "s1", "orders")
        tbl2 = self._make_table_mock("cat_b", "s2", "orders")
        self.client.tables.list_tables = MagicMock(
            return_value=iter([tbl1, tbl2]),
        )

        with self.assertRaises(ValueError) as ctx:
            self.genie.resolve_table_identifiers(("orders",))
        self.assertIn("ambiguous", str(ctx.exception).lower())
        self.assertIn("cat_a.s1.orders", str(ctx.exception))
        self.assertIn("cat_b.s2.orders", str(ctx.exception))

    def test_mixed_short_and_qualified(self):
        tbl = self._make_table_mock("prd", "data", "events")
        self.client.tables.list_tables = MagicMock(return_value=iter([tbl]))

        result = self.genie.resolve_table_identifiers(
            ("main.sales.orders", "events"),
        )
        self.assertEqual(
            result,
            ("main.sales.orders", "prd.data.events"),
        )

    def test_auto_create_resolves_short_names(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            auto_create_space=True,
            warehouse_id="wh-1",
            managed_space_tables=("trading_tgp_prd",),
        )
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[])

        tbl = self._make_table_mock("trading", "tgp", "trading_tgp_prd")
        self.client.tables.list_tables = MagicMock(return_value=iter([tbl]))

        created = MagicMock(
            space_id="new-id",
            title=DEFAULT_MANAGED_SPACE_TITLE,
        )
        self.genie_api.create_space.return_value = created

        space = self.genie.ensure_space()
        self.assertEqual(space.space_id, "new-id")
        call = self.genie_api.create_space.call_args
        self.assertIn("trading.tgp.trading_tgp_prd", call.kwargs["serialized_space"])

    def test_create_space_with_tables_resolves_and_builds(self):
        """``genie.create_space(tables=["trading_tgp_prd"])`` works end-to-end."""
        self.genie.defaults = replace(
            self.genie.defaults, warehouse_id="wh-1",
        )
        tbl = self._make_table_mock("trading", "tgp", "trading_tgp_prd")
        self.client.tables.list_tables = MagicMock(return_value=iter([tbl]))

        created = MagicMock(space_id="direct-id")
        self.genie_api.create_space.return_value = created

        space = self.genie.create_space(
            tables=["trading_tgp_prd"],
            title="My Space",
        )
        self.assertEqual(space.space_id, "direct-id")
        call = self.genie_api.create_space.call_args
        self.assertIn("trading.tgp.trading_tgp_prd", call.kwargs["serialized_space"])
        self.assertEqual(call.kwargs["title"], "My Space")

    def test_create_space_with_tables_and_instructions(self):
        self.genie.defaults = replace(
            self.genie.defaults, warehouse_id="wh-1",
        )
        created = MagicMock(space_id="s1")
        self.genie_api.create_space.return_value = created

        space = self.genie.create_space(
            tables=["main.sales.orders"],
            instructions=["Be brief.", "Use USD."],
        )
        self.assertEqual(space.space_id, "s1")
        payload = self.genie_api.create_space.call_args.kwargs["serialized_space"]
        self.assertIn("main.sales.orders", payload)
        self.assertIn("Be brief.", payload)
        self.assertIn("Use USD.", payload)

    def test_create_space_rejects_tables_and_serialized_space(self):
        with self.assertRaises(ValueError) as ctx:
            self.genie.create_space(
                tables=["main.sales.orders"],
                serialized_space='{"version": 1}',
                warehouse_id="wh-1",
            )
        self.assertIn("not both", str(ctx.exception))

    def test_create_space_requires_tables_or_serialized_space(self):
        with self.assertRaises(ValueError) as ctx:
            self.genie.create_space(warehouse_id="wh-1")
        self.assertIn("required", str(ctx.exception))


class TestGenieCleanupDeadSpaces(GenieTestCase):
    """Duplicate managed-title spaces get trashed."""

    def test_no_op_when_no_duplicates(self):
        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="only", title=DEFAULT_MANAGED_SPACE_TITLE),
        ]
        self.genie_api.list_spaces.return_value = spaces_page
        self.assertEqual(self.genie.cleanup_dead_spaces(), [])
        self.genie_api.trash_space.assert_not_called()

    def test_trashes_duplicates_keeping_default(self):
        self.genie.defaults = replace(self.genie.defaults, space_id="keeper")
        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="dup-A", title=DEFAULT_MANAGED_SPACE_TITLE),
            MagicMock(space_id="keeper", title=DEFAULT_MANAGED_SPACE_TITLE),
            MagicMock(space_id="dup-B", title=DEFAULT_MANAGED_SPACE_TITLE),
            MagicMock(space_id="other", title="Not Managed"),
        ]
        self.genie_api.list_spaces.return_value = spaces_page

        trashed = self.genie.cleanup_dead_spaces()
        self.assertEqual(sorted(trashed), ["dup-A", "dup-B"])
        trashed_calls = {
            call.kwargs["space_id"]
            for call in self.genie_api.trash_space.call_args_list
        }
        self.assertEqual(trashed_calls, {"dup-A", "dup-B"})

    def test_trashes_duplicates_keeping_first_when_no_default(self):
        # No defaults.space_id — survivor is the first listed entry.
        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="first", title=DEFAULT_MANAGED_SPACE_TITLE),
            MagicMock(space_id="second", title=DEFAULT_MANAGED_SPACE_TITLE),
        ]
        self.genie_api.list_spaces.return_value = spaces_page
        trashed = self.genie.cleanup_dead_spaces()
        self.assertEqual(trashed, ["second"])

    def test_cleanup_runs_from_ensure_space_when_flag_on(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id="keeper",
            cleanup_dead_spaces=True,
        )
        spaces_page = MagicMock()
        spaces_page.spaces = [
            MagicMock(space_id="keeper", title=DEFAULT_MANAGED_SPACE_TITLE),
            MagicMock(space_id="dup", title=DEFAULT_MANAGED_SPACE_TITLE),
        ]
        self.genie_api.list_spaces.return_value = spaces_page

        self.genie.ensure_space()
        self.genie_api.trash_space.assert_called_once_with(space_id="dup")
