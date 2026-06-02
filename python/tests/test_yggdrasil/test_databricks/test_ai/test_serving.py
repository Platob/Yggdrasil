"""Unit tests for the Databricks model-serving service and resources.

Exercises endpoint resolution, the chat/complete query path, message
normalization (mappings, pairs, system prepend), defaults injection, and
:class:`ChatResult` parsing — all on top of a mocked
``workspace_client.serving_endpoints`` SDK boundary.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from yggdrasil.databricks.ai import (
    DEFAULT_SERVING_ENDPOINT,
    ChatResult,
    DatabricksAI,
    ModelServing,
    ServingDefaults,
    ServingEndpoint,
)
from yggdrasil.databricks.tests import DatabricksTestCase


def _chat_response(content: str, *, model: str = "m", finish: str = "stop") -> SimpleNamespace:
    """A minimal stand-in for the SDK ``QueryEndpointResponse`` chat shape."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish,
        text=None,
    )
    return SimpleNamespace(
        choices=[choice],
        model=model,
        usage=SimpleNamespace(as_dict=lambda: {"total_tokens": 7}),
        predictions=None,
    )


class TestModelServing(DatabricksTestCase):

    @property
    def serving_api(self):
        return self.workspace_client.serving_endpoints

    def test_reachable_via_client_ai(self):
        self.assertIsInstance(self.client.ai, DatabricksAI)
        self.assertIsInstance(self.client.ai.serving, ModelServing)
        # cached on the umbrella
        self.assertIs(self.client.ai.serving, self.client.ai.serving)

    def test_endpoint_defaults_to_foundation_model(self):
        ep = self.client.ai.serving.endpoint()
        self.assertIsInstance(ep, ServingEndpoint)
        self.assertEqual(ep.endpoint_name, DEFAULT_SERVING_ENDPOINT)

    def test_complete_calls_query_and_parses(self):
        self.serving_api.query.return_value = _chat_response("optimized!")
        result = self.client.ai.serving.complete("hi", system="be terse", max_tokens=64)

        self.assertIsInstance(result, ChatResult)
        self.assertEqual(result.content, "optimized!")
        self.assertEqual(result.model, "m")
        self.assertEqual(result.usage, {"total_tokens": 7})

        self.serving_api.query.assert_called_once()
        kwargs = self.serving_api.query.call_args.kwargs
        self.assertEqual(kwargs["name"], DEFAULT_SERVING_ENDPOINT)
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["temperature"], 0.0)  # from defaults
        # system message is prepended, then the user turn
        messages = kwargs["messages"]
        self.assertEqual([m.role for m in messages], [ChatMessageRole.SYSTEM, ChatMessageRole.USER])
        self.assertEqual(messages[0].content, "be terse")
        self.assertEqual(messages[1].content, "hi")

    def test_chat_normalizes_message_shapes(self):
        self.serving_api.query.return_value = _chat_response("ok")
        self.client.ai.serving.chat(
            [
                {"role": "user", "content": "a"},
                ("assistant", "b"),
                ChatMessage(role=ChatMessageRole.USER, content="c"),
            ]
        )
        messages = self.serving_api.query.call_args.kwargs["messages"]
        self.assertEqual([(m.role, m.content) for m in messages], [
            (ChatMessageRole.USER, "a"),
            (ChatMessageRole.ASSISTANT, "b"),
            (ChatMessageRole.USER, "c"),
        ])

    def test_named_endpoint_overrides_default(self):
        self.serving_api.query.return_value = _chat_response("ok")
        self.client.ai.serving.complete("x", endpoint_name="my-llm")
        self.assertEqual(self.serving_api.query.call_args.kwargs["name"], "my-llm")

    def test_defaults_are_replaceable(self):
        serving = self.client.ai.serving
        serving.defaults = replace(serving.defaults, endpoint_name="custom", temperature=0.5)
        self.serving_api.query.return_value = _chat_response("ok")
        serving.complete("x")
        kwargs = self.serving_api.query.call_args.kwargs
        self.assertEqual(kwargs["name"], "custom")
        self.assertEqual(kwargs["temperature"], 0.5)

    def test_chatresult_falls_back_to_predictions(self):
        resp = SimpleNamespace(choices=[], model="m", usage=None, predictions=["pred-text"])
        self.assertEqual(ChatResult.from_response(resp).content, "pred-text")

    def test_endpoint_missing_when_not_found(self):
        from databricks.sdk.errors import NotFound

        self.serving_api.get.side_effect = NotFound("nope")
        self.assertIsNone(self.client.ai.serving.find_endpoint(name="ghost"))
