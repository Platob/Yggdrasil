"""Unit tests for the Databricks Model Serving service and resources.

Exercises endpoint lifecycle, the served-entity builders (UC models,
agents, external LLMs), the "max-config" defaults (AI Gateway usage
tracking + inference tables + scale-to-zero), the query data-plane
(chat / complete / embed), and the ``QueryEndpointResponse`` wrapper —
all on top of mocked SDK calls.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

from databricks.sdk.errors import AlreadyExists, NotFound
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
    EndpointCoreConfigInput,
    EndpointCoreConfigOutput,
    EndpointState,
    EndpointStateConfigUpdate,
    EndpointStateReady,
    ExternalModelProvider,
    QueryEndpointResponse,
    ServedEntityInput,
    ServedEntityOutput,
    ServedModelInputWorkloadType,
    ServingEndpoint as SdkServingEndpoint,
    ServingEndpointDetailed,
    V1ResponseChoiceElement,
)

from yggdrasil.databricks.ai import (
    DEFAULT_SERVING_WAIT,
    DatabricksAI,
    ModelServing,
    Served,
    ServingDefaults,
    ServingEndpoint,
    ServingQueryResult,
)
from yggdrasil.databricks.tests import DatabricksTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detailed(
    *,
    name: str = "my-llm",
    ready: EndpointStateReady = EndpointStateReady.READY,
    config_update: EndpointStateConfigUpdate = EndpointStateConfigUpdate.NOT_UPDATING,
    entity_names: tuple[str, ...] = ("gpt-4o",),
    endpoint_url: str = "https://test.databricks.net/serving-endpoints/my-llm/invocations",
) -> ServingEndpointDetailed:
    return ServingEndpointDetailed(
        name=name,
        endpoint_url=endpoint_url,
        state=EndpointState(ready=ready, config_update=config_update),
        config=EndpointCoreConfigOutput(
            served_entities=[ServedEntityOutput(name=n) for n in entity_names],
        ),
    )


def _chat_response(text: str = "hello back", *, n: int = 1) -> QueryEndpointResponse:
    return QueryEndpointResponse(
        id="q-1",
        model="my-llm",
        served_model_name="gpt-4o",
        choices=[
            V1ResponseChoiceElement(
                index=i,
                message=ChatMessage(role=ChatMessageRole.ASSISTANT, content=f"{text}-{i}"),
                finish_reason="stop",
            )
            for i in range(n)
        ],
    )


# ---------------------------------------------------------------------------
# Test base
# ---------------------------------------------------------------------------


class ServingTestCase(DatabricksTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.serving_api = self.workspace_client.serving_endpoints

    @property
    def serving(self) -> ModelServing:
        return self.client.ai.serving


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring(ServingTestCase):
    def test_client_ai_serving_cached(self):
        s = self.client.ai.serving
        self.assertIsInstance(s, ModelServing)
        self.assertIs(self.client.ai.serving, s)

    def test_ai_umbrella_is_singleton(self):
        self.assertIsInstance(self.client.ai, DatabricksAI)
        self.assertIs(self.client.sql.ai, self.client.ai)

    def test_endpoint_handle(self):
        ep = self.serving.endpoint("my-llm")
        self.assertIsInstance(ep, ServingEndpoint)
        self.assertEqual(ep.name, "my-llm")
        self.assertIn("my-llm", repr(ep))

    def test_explore_url(self):
        ep = self.serving.endpoint("my-llm")
        self.assertEqual(
            ep.explore_url.to_string(),
            "https://test.databricks.net/ml/endpoints/my-llm",
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults(ServingTestCase):
    def test_defaults_are_maximal(self):
        d = self.serving.defaults
        self.assertIsInstance(d, ServingDefaults)
        self.assertIs(d.wait, DEFAULT_SERVING_WAIT)
        self.assertTrue(d.scale_to_zero_enabled)
        self.assertTrue(d.enable_usage_tracking)
        self.assertTrue(d.enable_inference_table)
        self.assertEqual(d.workload_size, "Small")
        self.assertEqual(d.workload_type, "CPU")

    def test_defaults_replace_in_place(self):
        self.serving.defaults = replace(
            self.serving.defaults, workload_size="Medium", enable_inference_table=False,
        )
        self.assertEqual(self.serving.defaults.workload_size, "Medium")
        self.assertFalse(self.serving.defaults.enable_inference_table)


# ---------------------------------------------------------------------------
# Served-entity builders
# ---------------------------------------------------------------------------


class TestServedBuilders(ServingTestCase):
    def test_openai_wraps_external_model_with_secret_ref(self):
        e = Served.openai("gpt-4o", api_key_secret="llm/openai_key")
        self.assertIsInstance(e, ServedEntityInput)
        self.assertEqual(e.name, "gpt-4o")
        self.assertEqual(e.external_model.provider, ExternalModelProvider.OPENAI)
        self.assertEqual(e.external_model.task, "llm/v1/chat")
        self.assertEqual(
            e.external_model.openai_config.openai_api_key,
            "{{secrets/llm/openai_key}}",
        )

    def test_openai_keeps_full_template_secret(self):
        e = Served.openai("gpt-4o", api_key_secret="{{secrets/scope/key}}")
        self.assertEqual(
            e.external_model.openai_config.openai_api_key, "{{secrets/scope/key}}",
        )

    def test_anthropic_builder(self):
        e = Served.anthropic("claude-3-5-sonnet", api_key_secret="llm/anthropic")
        self.assertEqual(e.external_model.provider, ExternalModelProvider.ANTHROPIC)
        self.assertEqual(
            e.external_model.anthropic_config.anthropic_api_key,
            "{{secrets/llm/anthropic}}",
        )

    def test_uc_model_builder(self):
        e = Served.uc_model("main.agents.rag", 3, workload_size="Large")
        self.assertEqual(e.entity_name, "main.agents.rag")
        self.assertEqual(e.entity_version, "3")
        self.assertEqual(e.workload_size, "Large")

    def test_amazon_bedrock_builder(self):
        e = Served.amazon_bedrock(
            "claude", region="us-east-1", bedrock_provider="anthropic",
            access_key_id_secret="aws/key", secret_access_key_secret="aws/secret",
        )
        cfg = e.external_model.amazon_bedrock_config
        self.assertEqual(cfg.aws_region, "us-east-1")
        self.assertEqual(cfg.aws_access_key_id, "{{secrets/aws/key}}")

    def test_service_exposes_served_namespace(self):
        self.assertIs(self.serving.served, Served)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate(ServingTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.serving.defaults = replace(
            self.serving.defaults,
            inference_table_catalog="main",
            inference_table_schema="serving_logs",
        )
        wait_handle = MagicMock()
        wait_handle.response = _detailed()
        self.serving_api.create.return_value = wait_handle

    def test_create_external_model_assembles_gateway_and_traffic(self):
        ep = self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/openai_key")],
        )
        self.assertIs(ep.infos, self.serving_api.create.return_value.response)
        call = self.serving_api.create.call_args
        self.assertEqual(call.kwargs["name"], "my-llm")

        config: EndpointCoreConfigInput = call.kwargs["config"]
        self.assertEqual(len(config.served_entities), 1)
        # Single entity → 100% traffic route built automatically.
        routes = config.traffic_config.routes
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].served_model_name, "gpt-4o")
        self.assertEqual(routes[0].traffic_percentage, 100)

        # Max-config gateway: usage tracking on + inference table on.
        gw = call.kwargs["ai_gateway"]
        self.assertTrue(gw.usage_tracking_config.enabled)
        self.assertTrue(gw.inference_table_config.enabled)
        self.assertEqual(gw.inference_table_config.catalog_name, "main")
        self.assertEqual(gw.inference_table_config.schema_name, "serving_logs")

    def test_create_fills_workload_defaults_on_custom_model(self):
        self.serving.endpoint("rag").create(
            served_entities=[Served.uc_model("main.agents.rag", 2)],
        )
        config: EndpointCoreConfigInput = self.serving_api.create.call_args.kwargs["config"]
        entity = config.served_entities[0]
        self.assertEqual(entity.workload_size, "Small")
        self.assertEqual(entity.workload_type, ServedModelInputWorkloadType.CPU)
        self.assertTrue(entity.scale_to_zero_enabled)

    def test_create_does_not_set_workload_on_external_entity(self):
        self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
        )
        config: EndpointCoreConfigInput = self.serving_api.create.call_args.kwargs["config"]
        entity = config.served_entities[0]
        self.assertIsNone(entity.workload_size)
        self.assertIsNone(entity.scale_to_zero_enabled)

    def test_create_requires_entities(self):
        with self.assertRaises(ValueError):
            self.serving.endpoint("my-llm").create()

    def test_create_inference_table_skipped_without_catalog(self):
        self.serving.defaults = replace(
            self.serving.defaults,
            inference_table_catalog=None, inference_table_schema=None,
        )
        # client has no bound catalog/schema in this test → capture skipped.
        self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
        )
        gw = self.serving_api.create.call_args.kwargs["ai_gateway"]
        self.assertIsNone(gw.inference_table_config)
        self.assertTrue(gw.usage_tracking_config.enabled)

    def test_create_merges_tags(self):
        self.serving.defaults = replace(self.serving.defaults, tags={"team": "ml"})
        self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
            tags={"env": "prod"},
        )
        tags = self.serving_api.create.call_args.kwargs["tags"]
        as_dict = {t.key: t.value for t in tags}
        # Service default tags (ServiceName) are merged in too; the
        # service-level + per-call tags layer on top.
        self.assertEqual(as_dict.get("team"), "ml")
        self.assertEqual(as_dict.get("env"), "prod")
        self.assertEqual(as_dict.get("ServiceName"), "modelserving")

    def test_create_missing_ok_swallows_already_exists(self):
        self.serving_api.create.side_effect = AlreadyExists("dup")
        self.serving_api.get.return_value = _detailed()
        ep = self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
            missing_ok=True,
        )
        self.assertEqual(ep.infos.name, "my-llm")

    def test_create_missing_ok_false_propagates(self):
        self.serving_api.create.side_effect = AlreadyExists("dup")
        with self.assertRaises(AlreadyExists):
            self.serving.endpoint("my-llm").create(
                served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
                missing_ok=False,
            )

    def test_create_waits_when_requested(self):
        self.serving_api.wait_get_serving_endpoint_not_updating.return_value = _detailed()
        self.serving.endpoint("my-llm").create(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
            wait=True,
        )
        self.serving_api.wait_get_serving_endpoint_not_updating.assert_called_once()

    def test_serve_openai_convenience(self):
        ep = self.serving.endpoint("my-llm").serve_openai("gpt-4o", api_key_secret="llm/k")
        self.assertIsInstance(ep, ServingEndpoint)
        self.serving_api.create.assert_called_once()

    def test_serve_uc_model_convenience(self):
        self.serving.endpoint("rag").serve_uc_model("main.agents.rag", 1)
        config = self.serving_api.create.call_args.kwargs["config"]
        self.assertEqual(config.served_entities[0].entity_name, "main.agents.rag")


# ---------------------------------------------------------------------------
# ensure_created / update / delete / wait
# ---------------------------------------------------------------------------


class TestLifecycle(ServingTestCase):
    def test_ensure_created_skips_when_exists(self):
        self.serving_api.get.return_value = _detailed()
        self.serving.endpoint("my-llm").ensure_created(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
        )
        self.serving_api.create.assert_not_called()

    def test_ensure_created_creates_when_missing(self):
        self.serving_api.get.side_effect = NotFound("missing")
        wait_handle = MagicMock()
        wait_handle.response = _detailed()
        self.serving_api.create.return_value = wait_handle
        self.serving.endpoint("my-llm").ensure_created(
            served_entities=[Served.openai("gpt-4o", api_key_secret="llm/k")],
        )
        self.serving_api.create.assert_called_once()

    def test_update_config(self):
        wait_handle = MagicMock()
        wait_handle.response = _detailed(entity_names=("gpt-4o-mini",))
        self.serving_api.update_config.return_value = wait_handle
        self.serving.endpoint("my-llm").update_config(
            served_entities=[Served.openai("gpt-4o-mini", api_key_secret="llm/k")],
        )
        call = self.serving_api.update_config.call_args
        self.assertEqual(call.kwargs["name"], "my-llm")
        self.assertEqual(len(call.kwargs["served_entities"]), 1)
        self.assertEqual(call.kwargs["traffic_config"].routes[0].served_model_name, "gpt-4o-mini")

    def test_delete(self):
        ep = self.serving.endpoint("my-llm")
        ep._details = _detailed()
        ep.delete()
        self.serving_api.delete.assert_called_once_with(name="my-llm")
        self.assertIsNone(ep._details)

    def test_delete_missing_ok(self):
        self.serving_api.delete.side_effect = NotFound("gone")
        self.serving.endpoint("my-llm").delete(missing_ok=True)

    def test_delete_missing_not_ok_raises(self):
        self.serving_api.delete.side_effect = NotFound("gone")
        with self.assertRaises(NotFound):
            self.serving.endpoint("my-llm").delete()

    def test_wait_ready_routes_through_sdk_helper(self):
        import datetime as dt

        self.serving_api.wait_get_serving_endpoint_not_updating.return_value = _detailed()
        ep = self.serving.endpoint("my-llm").wait_ready(wait=30)
        call = self.serving_api.wait_get_serving_endpoint_not_updating.call_args
        self.assertEqual(call.kwargs["name"], "my-llm")
        self.assertEqual(call.kwargs["timeout"], dt.timedelta(seconds=30))
        self.assertEqual(ep.infos.name, "my-llm")

    def test_state_and_readiness(self):
        self.serving_api.get.return_value = _detailed(
            ready=EndpointStateReady.NOT_READY,
            config_update=EndpointStateConfigUpdate.IN_PROGRESS,
        )
        ep = self.serving.endpoint("my-llm")
        self.assertEqual(ep.state, "IN_PROGRESS")
        self.assertEqual(ep.ready, "NOT_READY")
        self.assertFalse(ep.is_ready)

    def test_exists_false_on_not_found(self):
        self.serving_api.get.side_effect = NotFound("missing")
        self.assertFalse(self.serving.endpoint("my-llm").exists())

    def test_infos_caches(self):
        self.serving_api.get.return_value = _detailed()
        ep = self.serving.endpoint("my-llm")
        self.assertIs(ep.infos, ep.infos)
        self.serving_api.get.assert_called_once_with(name="my-llm")

    def test_served_entity_names(self):
        self.serving_api.get.return_value = _detailed(entity_names=("a", "b"))
        self.assertEqual(self.serving.endpoint("my-llm").served_entity_names, ("a", "b"))


# ---------------------------------------------------------------------------
# Tags / gateway / ops
# ---------------------------------------------------------------------------


class TestManagement(ServingTestCase):
    def test_add_tags(self):
        self.serving.endpoint("my-llm").add_tags({"env": "prod"})
        call = self.serving_api.patch.call_args
        self.assertEqual(call.kwargs["name"], "my-llm")
        self.assertEqual({t.key: t.value for t in call.kwargs["add_tags"]}, {"env": "prod"})

    def test_delete_tags(self):
        self.serving.endpoint("my-llm").delete_tags(["env", "team"])
        call = self.serving_api.patch.call_args
        self.assertEqual(call.kwargs["delete_tags"], ["env", "team"])

    def test_logs_uses_first_entity(self):
        self.serving_api.get.return_value = _detailed(entity_names=("gpt-4o",))
        self.serving.endpoint("my-llm").logs()
        self.serving_api.logs.assert_called_once_with(
            name="my-llm", served_model_name="gpt-4o",
        )

    def test_logs_explicit_entity(self):
        self.serving.endpoint("my-llm").logs(served_model_name="explicit")
        self.serving_api.logs.assert_called_once_with(
            name="my-llm", served_model_name="explicit",
        )

    def test_metrics_and_openapi(self):
        self.serving.endpoint("my-llm").metrics()
        self.serving.endpoint("my-llm").openapi()
        self.serving_api.export_metrics.assert_called_once_with(name="my-llm")
        self.serving_api.get_open_api.assert_called_once_with(name="my-llm")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListing(ServingTestCase):
    def test_list_endpoints(self):
        self.serving_api.list.return_value = iter([
            SdkServingEndpoint(name="a"),
            SdkServingEndpoint(name="b"),
        ])
        eps = list(self.serving.list_endpoints())
        self.assertEqual([e.name for e in eps], ["a", "b"])

    def test_find_endpoint_match(self):
        self.serving_api.get.return_value = _detailed(name="my-llm")
        ep = self.serving.find_endpoint(name="my-llm")
        self.assertIsNotNone(ep)
        self.assertEqual(ep.name, "my-llm")

    def test_find_endpoint_missing(self):
        self.serving_api.get.side_effect = NotFound("missing")
        self.assertIsNone(self.serving.find_endpoint(name="nope"))


# ---------------------------------------------------------------------------
# Query data-plane
# ---------------------------------------------------------------------------


class TestQuery(ServingTestCase):
    def test_chat_string_shortcut(self):
        self.serving_api.query.return_value = _chat_response("hi there")
        result = self.serving.endpoint("my-llm").chat("Hello!")
        self.assertIsInstance(result, ServingQueryResult)
        call = self.serving_api.query.call_args
        self.assertEqual(call.kwargs["name"], "my-llm")
        msgs = call.kwargs["messages"]
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].role, ChatMessageRole.USER)
        self.assertEqual(msgs[0].content, "Hello!")
        self.assertEqual(result.text, "hi there-0")

    def test_chat_message_list_and_params(self):
        self.serving_api.query.return_value = _chat_response()
        self.serving.endpoint("my-llm").chat(
            [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
            max_tokens=64, temperature=0.2, stop=["\n"],
        )
        call = self.serving_api.query.call_args
        self.assertEqual([m.role for m in call.kwargs["messages"]],
                         [ChatMessageRole.SYSTEM, ChatMessageRole.USER])
        self.assertEqual(call.kwargs["max_tokens"], 64)
        self.assertEqual(call.kwargs["temperature"], 0.2)
        self.assertEqual(call.kwargs["stop"], ["\n"])

    def test_chat_accepts_chatmessage_objects(self):
        self.serving_api.query.return_value = _chat_response()
        self.serving.endpoint("my-llm").chat(
            [ChatMessage(role=ChatMessageRole.USER, content="hi")],
        )
        msgs = self.serving_api.query.call_args.kwargs["messages"]
        self.assertEqual(msgs[0].content, "hi")

    def test_complete(self):
        self.serving_api.query.return_value = QueryEndpointResponse(
            choices=[V1ResponseChoiceElement(index=0, text="completed")],
        )
        result = self.serving.endpoint("my-llm").complete("Once upon", max_tokens=10)
        self.assertEqual(self.serving_api.query.call_args.kwargs["prompt"], "Once upon")
        self.assertEqual(result.text, "completed")

    def test_embed(self):
        from databricks.sdk.service.serving import (
            EmbeddingsV1ResponseEmbeddingElement,
        )

        self.serving_api.query.return_value = QueryEndpointResponse(
            data=[
                EmbeddingsV1ResponseEmbeddingElement(embedding=[0.1, 0.2, 0.3], index=0),
                EmbeddingsV1ResponseEmbeddingElement(embedding=[0.4, 0.5, 0.6], index=1),
            ],
        )
        result = self.serving.endpoint("emb").embed(["a", "b"])
        self.assertEqual(self.serving_api.query.call_args.kwargs["input"], ["a", "b"])
        self.assertEqual(result.embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# Query result wrapper
# ---------------------------------------------------------------------------


class TestQueryResult(ServingTestCase):
    def _result(self, response: QueryEndpointResponse) -> ServingQueryResult:
        return ServingQueryResult(endpoint=self.serving.endpoint("my-llm"), response=response)

    def test_text_prefers_message_content(self):
        r = self._result(_chat_response("answer"))
        self.assertEqual(r.text, "answer-0")
        self.assertEqual(r.message, {"role": "assistant", "content": "answer-0"})

    def test_texts_multiple_choices(self):
        r = self._result(_chat_response("a", n=3))
        self.assertEqual(r.texts, ["a-0", "a-1", "a-2"])

    def test_predictions_passthrough(self):
        r = self._result(QueryEndpointResponse(predictions=[1, 2, 3]))
        self.assertEqual(r.predictions, [1, 2, 3])

    def test_metadata_and_to_dict(self):
        r = self._result(_chat_response())
        self.assertEqual(r.model, "my-llm")
        self.assertEqual(r.served_model_name, "gpt-4o")
        self.assertEqual(r.id, "q-1")
        self.assertIn("choices", r.to_dict())
