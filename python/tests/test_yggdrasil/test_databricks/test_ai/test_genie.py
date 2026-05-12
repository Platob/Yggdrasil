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

from yggdrasil.databricks.ai.genie import (
    DEFAULT_TIMEOUT_SECONDS,
    Genie,
    GenieAnswer,
    GenieConversation,
    GenieDefaults,
    GenieSpace,
)
from yggdrasil.databricks.tests import DatabricksTestCase


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
        self.assertEqual(self.genie.defaults.timeout_seconds, DEFAULT_TIMEOUT_SECONDS)
        self.assertTrue(self.genie.defaults.auto_pick_space)

    def test_defaults_override_in_place(self):
        self.genie.defaults = replace(
            self.genie.defaults,
            space_id="space-x",
            timeout_seconds=60.0,
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

    def test_ask_with_explicit_timeout(self):
        captured = []

        def _record(timeout=None, callback=None):
            captured.append(timeout)
            return self.completed

        self.start_waiter.result = _record  # type: ignore[method-assign]
        self.genie.ask("Q", timeout_seconds=30)
        self.assertEqual(captured[0], dt.timedelta(seconds=30))

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
