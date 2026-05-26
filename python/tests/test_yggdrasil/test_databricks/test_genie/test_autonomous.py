"""Tests for :class:`yggdrasil.databricks.genie.AutonomousAgent`.

Exercises the autonomous goal-directed loop, self-duplication, plan
execution, parallel dispatch, resource creation tools, and workspace
introspection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieMessage,
    MessageStatus,
    TextAttachment,
)

from yggdrasil.enums.state import State
from yggdrasil.databricks.genie import (
    AgentResponse,
    AgentResult,
    AgentStep,
    AutonomousAgent,
)
from yggdrasil.databricks.genie.profiles import (
    INGESTION_CLUSTER,
    SERVERLESS_WAREHOUSE,
    STARTER_WAREHOUSE,
    ClusterProfile,
    StorageProfile,
    WarehouseProfile,
)
from yggdrasil.databricks.tests import DatabricksTestCase


def _build_completed_message(
    *,
    space_id: str = "space-1",
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    text: str | None = "answer",
) -> GenieMessage:
    attachments = [
        GenieAttachment(
            attachment_id="att-1",
            text=TextAttachment(id="tid", content=text) if text else None,
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


class AutonomousAgentTestCase(DatabricksTestCase):
    """Base — wires the Genie API mock and builds the autonomous agent."""

    def setUp(self):
        super().setUp()
        self.genie_api = self.workspace_client.genie
        self.genie = self.client.genie
        self.genie.defaults = replace(self.genie.defaults, space_id="space-1")
        self.agent = AutonomousAgent(service=self.genie)

    def _start_returns(self, message: GenieMessage) -> None:
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            space_id=message.space_id,
        )

    def _ask_returns(self, text: str) -> None:
        """Configure Genie to return the given text for any ask."""
        msg = _build_completed_message(text=text)
        self._start_returns(msg)
        self.genie_api.create_message.return_value = Wait(
            waiter=lambda **kwargs: msg,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )


# ----------------------------------------------------------------------- #
# AgentStep / AgentResult data structures
# ----------------------------------------------------------------------- #
class TestAgentStep(AutonomousAgentTestCase):
    def test_step_pending_by_default(self):
        step = AgentStep(action="do something")
        self.assertEqual(step.state, State.PENDING)
        self.assertFalse(step.succeeded)
        self.assertFalse(step.failed)

    def test_step_done(self):
        step = AgentStep(action="did it", state=State.SUCCEEDED, result="ok")
        self.assertTrue(step.succeeded)
        self.assertFalse(step.failed)

    def test_step_failed(self):
        step = AgentStep(action="oops", state=State.FAILED, error="boom")
        self.assertFalse(step.succeeded)
        self.assertTrue(step.failed)


class TestAgentResult(AutonomousAgentTestCase):
    def test_result_summary(self):
        r = AgentResult(
            goal="Create tables",
            steps=[
                AgentStep(action="step 1", state=State.SUCCEEDED),
                AgentStep(action="step 2", state=State.FAILED, error="oops"),
            ],
            conclusion="Partially done",
            state=State.FAILED,
        )
        summary = r.summary()
        self.assertIn("Create tables", summary)
        self.assertIn("FAILED", summary)
        self.assertIn("1/2", summary)

    def test_completed_and_failed_steps(self):
        r = AgentResult(
            goal="test",
            steps=[
                AgentStep(action="a", state=State.SUCCEEDED),
                AgentStep(action="b", state=State.FAILED),
                AgentStep(action="c", state=State.SUCCEEDED),
            ],
        )
        self.assertEqual(len(r.completed_steps), 2)
        self.assertEqual(len(r.failed_steps), 1)

    def test_result_uses_state_enum(self):
        r = AgentResult(goal="test")
        self.assertEqual(r.state, State.RUNNING)
        self.assertFalse(r.succeeded)
        r.state = State.SUCCEEDED
        self.assertTrue(r.succeeded)


# ----------------------------------------------------------------------- #
# Identity & repr
# ----------------------------------------------------------------------- #
class TestAutonomousRepr(AutonomousAgentTestCase):
    def test_repr_shows_state(self):
        r = repr(self.agent)
        self.assertIn("AutonomousAgent", r)
        self.assertIn("name='root'", r)
        self.assertIn("history=0", r)
        self.assertIn("children=0", r)

    def test_default_name_is_root(self):
        self.assertEqual(self.agent.name, "root")


# ----------------------------------------------------------------------- #
# Autonomous loop — accomplish()
# ----------------------------------------------------------------------- #
class TestAccomplish(AutonomousAgentTestCase):
    def test_accomplish_with_tool_plan(self):
        """Genie returns a plan with TOOL steps; agent executes them."""
        plan_text = (
            "1. TOOL: create_catalog(energy)\n"
            "2. TOOL: create_schema(energy.entsoe)\n"
            "3. DONE: Storage ready\n"
        )
        self._ask_returns(plan_text)

        result = self.agent.accomplish("Set up energy storage")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.goal, "Set up energy storage")
        self.assertTrue(len(result.steps) >= 2)

        catalog_steps = [s for s in result.steps if s.tool == "create_catalog"]
        schema_steps = [s for s in result.steps if s.tool == "create_schema"]
        self.assertTrue(len(catalog_steps) >= 1 or len(schema_steps) >= 1)

    def test_accomplish_respects_budget(self):
        self._ask_returns(
            "1. ASK: What tables?\n"
            "2. ASK: What schema?\n"
            "3. ASK: What volume?\n"
            "4. ASK: What catalog?\n"
        )
        result = self.agent.accomplish("Big plan", max_steps=2)
        self.assertLessEqual(len(result.steps), 2)
        self.assertIn("budget", result.conclusion.lower())

    def test_accomplish_fallback_when_genie_fails(self):
        """When Genie can't respond, the agent uses a fallback plan."""
        self.genie_api.start_conversation.side_effect = RuntimeError("Genie down")
        result = self.agent.accomplish("Create a table", max_steps=1)
        self.assertIsInstance(result, AgentResult)

    def test_accomplish_handles_step_failure(self):
        plan_text = "1. TOOL: create_catalog(nonexistent_tool_xxx)\n"
        self._ask_returns(plan_text)
        result = self.agent.accomplish("Fail test", max_steps=3)
        self.assertIsInstance(result, AgentResult)


class TestAccomplishDoneStep(AutonomousAgentTestCase):
    def test_done_step_succeeds_immediately(self):
        self._ask_returns("1. DONE: Already complete\n")
        result = self.agent.accomplish("Nothing to do")
        done_steps = [s for s in result.steps if s.tool == "_done"]
        self.assertTrue(len(done_steps) >= 1)
        self.assertTrue(done_steps[0].succeeded)


class TestAccomplishAskStep(AutonomousAgentTestCase):
    def test_ask_step_queries_genie(self):
        self._ask_returns("1. ASK: What tables exist?\n")
        result = self.agent.accomplish("Explore workspace", max_steps=2)
        ask_steps = [s for s in result.steps if s.tool == "ask"]
        self.assertTrue(len(ask_steps) >= 1)


# ----------------------------------------------------------------------- #
# Introspection
# ----------------------------------------------------------------------- #
class TestIntrospect(AutonomousAgentTestCase):
    def test_introspect_returns_context(self):
        self.workspace_client.catalogs.list.return_value = []
        context = self.agent.introspect()
        self.assertIn("agent", context)
        self.assertIn("catalogs", context)
        self.assertIn("warehouses", context)
        self.assertIn("jobs", context)

    def test_introspect_handles_errors_gracefully(self):
        self.workspace_client.catalogs.list.side_effect = RuntimeError("no access")
        context = self.agent.introspect()
        self.assertEqual(context["catalogs"], [])


# ----------------------------------------------------------------------- #
# Plan parsing
# ----------------------------------------------------------------------- #
class TestPlanParsing(AutonomousAgentTestCase):
    def test_parse_tool_step(self):
        steps = self.agent._parse_plan(
            "1. TOOL: create_catalog(energy, comment='test')\n",
            "goal",
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool, "create_catalog")
        self.assertEqual(steps[0].args, ("energy",))
        self.assertEqual(steps[0].kwargs, {"comment": "test"})

    def test_parse_ask_step(self):
        steps = self.agent._parse_plan("1. ASK: What schemas exist?\n", "goal")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool, "ask")

    def test_parse_done_step(self):
        steps = self.agent._parse_plan("1. DONE: All created\n", "goal")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool, "_done")

    def test_parse_mixed_plan(self):
        text = (
            "1. TOOL: create_catalog(data)\n"
            "2. ASK: What tables?\n"
            "3. DONE: Finished\n"
        )
        steps = self.agent._parse_plan(text, "goal")
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].tool, "create_catalog")
        self.assertEqual(steps[1].tool, "ask")
        self.assertEqual(steps[2].tool, "_done")

    def test_parse_ignores_empty_lines(self):
        steps = self.agent._parse_plan("\n\n1. TOOL: ask(hello)\n\n", "goal")
        self.assertEqual(len(steps), 1)


class TestCoerceValue(AutonomousAgentTestCase):
    def test_coerce_int(self):
        self.assertEqual(self.agent._coerce_value("42"), 42)

    def test_coerce_float(self):
        self.assertEqual(self.agent._coerce_value("3.14"), 3.14)

    def test_coerce_bool(self):
        self.assertTrue(self.agent._coerce_value("true"))
        self.assertFalse(self.agent._coerce_value("false"))

    def test_coerce_none(self):
        self.assertIsNone(self.agent._coerce_value("none"))
        self.assertIsNone(self.agent._coerce_value("null"))

    def test_coerce_quoted_string(self):
        self.assertEqual(self.agent._coerce_value("'hello'"), "hello")
        self.assertEqual(self.agent._coerce_value('"world"'), "world")

    def test_coerce_plain_string(self):
        self.assertEqual(self.agent._coerce_value("energy"), "energy")


class TestSplitArgs(AutonomousAgentTestCase):
    def test_simple_args(self):
        self.assertEqual(self.agent._split_args("a, b, c"), ["a", " b", " c"])

    def test_quoted_comma(self):
        parts = self.agent._split_args("'a,b', c")
        self.assertEqual(len(parts), 2)

    def test_nested_parens(self):
        parts = self.agent._split_args("fn(a, b), c")
        self.assertEqual(len(parts), 2)


# ----------------------------------------------------------------------- #
# Fallback planning
# ----------------------------------------------------------------------- #
class TestFallbackPlan(AutonomousAgentTestCase):
    def test_storage_keywords_trigger_ask(self):
        steps = self.agent._fallback_plan("Create a catalog for energy data")
        self.assertTrue(len(steps) >= 2)
        self.assertEqual(steps[0].tool, "introspect")
        self.assertEqual(steps[1].tool, "ask")

    def test_warehouse_keywords_trigger_ask(self):
        steps = self.agent._fallback_plan("Set up a SQL warehouse")
        self.assertTrue(len(steps) >= 2)

    def test_generic_goal_triggers_ask(self):
        steps = self.agent._fallback_plan("Do something vague")
        self.assertTrue(len(steps) >= 2)
        self.assertEqual(steps[0].tool, "introspect")
        self.assertEqual(steps[1].tool, "ask")


# ----------------------------------------------------------------------- #
# Self-duplication
# ----------------------------------------------------------------------- #
class TestFork(AutonomousAgentTestCase):
    def test_fork_creates_child(self):
        child = self.agent.fork(name="raw-layer")
        self.assertIsInstance(child, AutonomousAgent)
        self.assertEqual(child.name, "raw-layer")
        self.assertIs(child.parent, self.agent)
        self.assertIn(child, self.agent.children)

    def test_fork_generates_name_when_omitted(self):
        child = self.agent.fork()
        self.assertEqual(child.name, "root.child-0")

    def test_fork_children_have_isolated_history(self):
        child = self.agent.fork(name="isolated")
        self._start_returns(_build_completed_message(text="parent answer"))
        self.agent.run("parent question")
        self.assertEqual(len(self.agent.history), 1)
        self.assertEqual(len(child.history), 0)

    def test_fork_many_creates_multiple(self):
        children = self.agent.fork_many(["a", "b", "c"])
        self.assertEqual(len(children), 3)
        self.assertEqual(len(self.agent.children), 3)
        self.assertEqual([c.name for c in children], ["a", "b", "c"])

    def test_fork_shares_client(self):
        child = self.agent.fork()
        self.assertIs(child.client, self.agent.client)

    def test_fork_inherits_max_steps(self):
        self.agent.max_steps = 42
        child = self.agent.fork()
        self.assertEqual(child.max_steps, 42)


# ----------------------------------------------------------------------- #
# Service property
# ----------------------------------------------------------------------- #
class TestAutonomousAgentProperty(AutonomousAgentTestCase):
    def test_service_property_is_cached(self):
        first = self.genie.autonomous_agent
        second = self.genie.autonomous_agent
        self.assertIs(first, second)

    def test_service_returns_autonomous_agent(self):
        self.assertIsInstance(self.genie.autonomous_agent, AutonomousAgent)


# ----------------------------------------------------------------------- #
# Plan execution (lower-level)
# ----------------------------------------------------------------------- #
class TestExecutePlan(AutonomousAgentTestCase):
    def test_execute_plan_runs_tools_in_order(self):
        results = []
        self.agent.register_tool("append_a", lambda: results.append("a") or "a")
        self.agent.register_tool("append_b", lambda: results.append("b") or "b")

        plan = [
            {"tool": "append_a"},
            {"tool": "append_b"},
        ]
        out = self.agent.execute_plan(plan)
        self.assertEqual(results, ["a", "b"])
        self.assertEqual(out, ["a", "b"])

    def test_execute_plan_passes_args_and_kwargs(self):
        self.agent.register_tool("add", lambda x, y: x + y)
        out = self.agent.execute_plan(
            [
                {"tool": "add", "args": [3], "kwargs": {"y": 7}},
            ]
        )
        self.assertEqual(out, [10])

    def test_execute_plan_stop_on_error(self):
        self.agent.register_tool(
            "boom", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        self.agent.register_tool("ok", lambda: "ok")

        with self.assertRaises(RuntimeError):
            self.agent.execute_plan(
                [
                    {"tool": "boom"},
                    {"tool": "ok"},
                ]
            )

    def test_execute_plan_continue_on_error(self):
        self.agent.register_tool("boom", lambda: 1 / 0)
        self.agent.register_tool("ok", lambda: "ok")

        out = self.agent.execute_plan(
            [{"tool": "boom"}, {"tool": "ok"}],
            stop_on_error=False,
        )
        self.assertIsInstance(out[0], ZeroDivisionError)
        self.assertEqual(out[1], "ok")

    def test_execute_plan_missing_tool_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.agent.execute_plan([{"args": [1]}])
        self.assertIn("missing 'tool' key", str(ctx.exception))

    def test_step_results_tracks_last_plan(self):
        self.agent.register_tool("one", lambda: 1)
        self.agent.execute_plan([{"tool": "one"}])
        self.assertEqual(self.agent.step_results, [1])


# ----------------------------------------------------------------------- #
# Parallel execution
# ----------------------------------------------------------------------- #
class TestParallel(AutonomousAgentTestCase):
    def test_parallel_runs_concurrently(self):
        results = self.agent.parallel(
            [
                lambda: "a",
                lambda: "b",
                lambda: "c",
            ]
        )
        self.assertEqual(set(results), {"a", "b", "c"})
        self.assertEqual(len(results), 3)

    def test_parallel_preserves_order(self):
        import time

        results = self.agent.parallel(
            [
                lambda: (time.sleep(0.01), 1)[1],
                lambda: 2,
                lambda: 3,
            ]
        )
        self.assertEqual(results, [1, 2, 3])


# ----------------------------------------------------------------------- #
# Tool registry
# ----------------------------------------------------------------------- #
class TestAutonomousTools(AutonomousAgentTestCase):
    def test_autonomous_tools_registered(self):
        for name in (
            "create_catalog",
            "create_schema",
            "create_table",
            "create_volume",
            "create_warehouse",
            "create_cluster",
            "deploy_workflow",
            "setup_storage",
            "setup_genie_space",
            "introspect",
            "describe_catalog",
            "describe_schema",
            "describe_table",
            "accomplish",
            "fork",
            "fork_many",
            "execute_plan",
            "parallel",
            "step_results",
            "children",
            "create_job",
            "run_and_wait",
            "fetch",
            "fetch_text",
            "fetch_json",
            "fetch_many",
            "scrape_links",
            "open_browser",
            "fetch_entsoe_zones",
        ):
            self.assertIn(name, self.agent.tools, msg=name)

    def test_inherits_base_tools(self):
        for name in ("ask", "chat", "inspect", "save", "sql", "history"):
            self.assertIn(name, self.agent.tools, msg=name)


# ----------------------------------------------------------------------- #
# Resource creation (SDK mock boundary tests)
# ----------------------------------------------------------------------- #
class TestCreateCatalog(AutonomousAgentTestCase):
    def test_create_catalog_delegates_to_sdk(self):
        result = self.agent.create_catalog("test_catalog", comment="test")
        self.workspace_client.catalogs.get.assert_called()
        self.assertIsNotNone(result)

    def test_create_catalog_via_tool(self):
        result = self.agent.run_tool("create_catalog", "my_cat")
        self.assertIsNotNone(result)


class TestCreateSchema(AutonomousAgentTestCase):
    def test_create_schema_delegates_to_sdk(self):
        result = self.agent.create_schema("main.sales", comment="sales data")
        self.assertIsNotNone(result)


class TestCreateVolume(AutonomousAgentTestCase):
    def test_create_volume_delegates_to_sdk(self):
        result = self.agent.create_volume("main.sales.uploads", comment="uploads")
        self.assertIsNotNone(result)


class TestCreateWarehouse(AutonomousAgentTestCase):
    def test_create_warehouse_calls_sdk(self):
        result = self.agent.create_warehouse("my-wh")
        self.workspace_client.warehouses.create_and_wait.assert_called_once()
        call_kwargs = self.workspace_client.warehouses.create_and_wait.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("name") or call_kwargs[1].get("name"), "my-wh"
        )

    def test_create_warehouse_with_custom_profile(self):
        self.agent.create_warehouse("wh", profile=STARTER_WAREHOUSE)
        call_kwargs = self.workspace_client.warehouses.create_and_wait.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        self.assertFalse(all_kwargs.get("enable_serverless_compute", True))

    def test_create_warehouse_overrides(self):
        self.agent.create_warehouse("wh", cluster_size="Large")
        call_kwargs = self.workspace_client.warehouses.create_and_wait.call_args
        all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        self.assertEqual(all_kwargs.get("cluster_size"), "Large")


class TestCreateCluster(AutonomousAgentTestCase):
    def _setup_spark_versions(self):
        """Stub the spark versions SDK response so cluster creation can resolve a version."""
        from databricks.sdk.service.compute import SparkVersion

        self.workspace_client.clusters.spark_versions.return_value = MagicMock(
            versions=[SparkVersion(key="14.3.x-scala2.12", name="14.3 LTS")],
        )

    def test_create_cluster_returns_cluster(self):
        self._setup_spark_versions()
        result = self.agent.create_cluster(
            "my-cluster", spark_version="14.3.x-scala2.12"
        )
        self.assertIsNotNone(result)

    def test_create_cluster_with_fixed_workers(self):
        self._setup_spark_versions()
        profile = ClusterProfile(num_workers=3)
        result = self.agent.create_cluster(
            "fixed", spark_version="14.3.x-scala2.12", profile=profile
        )
        self.assertIsNotNone(result)


# ----------------------------------------------------------------------- #
# Storage layout setup
# ----------------------------------------------------------------------- #
class TestSetupStorage(AutonomousAgentTestCase):
    def test_setup_creates_catalog_schema_volume(self):
        profile = StorageProfile(
            catalog="energy",
            source="entsoe",
            raw_entities=("prices", "load"),
            curated_entities=("prices",),
        )
        result = self.agent.setup_storage(profile)

        self.assertIn("catalog", result)
        self.assertIn("schema", result)
        self.assertIn("volume", result)
        self.assertIn("raw_prices", result)
        self.assertIn("raw_load", result)
        self.assertIn("curated_prices", result)

        self.assertEqual(result["raw_prices"], "energy.entsoe.raw_prices")
        self.assertEqual(result["curated_prices"], "energy.entsoe.prices")

    def test_setup_without_volume(self):
        profile = StorageProfile(
            catalog="test",
            source="src",
            create_volume=False,
        )
        result = self.agent.setup_storage(profile)
        self.assertNotIn("volume", result)


# ----------------------------------------------------------------------- #
# Profiles
# ----------------------------------------------------------------------- #
class TestProfiles(AutonomousAgentTestCase):
    def test_warehouse_profile_frozen(self):
        with self.assertRaises(AttributeError):
            SERVERLESS_WAREHOUSE.cluster_size = "Large"

    def test_cluster_profile_frozen(self):
        with self.assertRaises(AttributeError):
            INGESTION_CLUSTER.num_workers = 99

    def test_storage_profile_names(self):
        p = StorageProfile(catalog="cat", source="src", raw_entities=("tbl",))
        self.assertEqual(p.schema_name, "cat.src")
        self.assertEqual(p.raw_table_name("tbl"), "cat.src.raw_tbl")
        self.assertEqual(p.curated_table_name("tbl"), "cat.src.tbl")
        self.assertEqual(p.volume_full_name(), "cat.src.uploads")

    def test_storage_profile_meta_schema(self):
        p = StorageProfile(catalog="cat", source="src")
        self.assertEqual(p.meta_schema_name, "cat.src._meta")

    def test_replace_profile(self):
        custom = replace(SERVERLESS_WAREHOUSE, cluster_size="Large")
        self.assertEqual(custom.cluster_size, "Large")
        self.assertEqual(custom.enable_serverless_compute, True)


# ----------------------------------------------------------------------- #
# Genie space setup
# ----------------------------------------------------------------------- #
class TestSetupGenieSpace(AutonomousAgentTestCase):
    def test_setup_genie_space_creates_and_pins(self):
        space_mock = MagicMock()
        space_mock.space_id = "space-new"
        self.genie_api.create_space.return_value = space_mock

        self.genie.defaults = replace(
            self.genie.defaults,
            warehouse_id="wh-123",
        )
        result = self.agent.setup_genie_space(
            tables=["main.sales.orders"],
            title="Test Space",
        )
        self.genie_api.create_space.assert_called_once()
        self.assertEqual(self.genie.defaults.space_id, "space-new")


# ----------------------------------------------------------------------- #
# Tool description helper
# ----------------------------------------------------------------------- #
class TestDescribeTools(AutonomousAgentTestCase):
    def test_describe_tools_lists_all(self):
        desc = self.agent._describe_tools()
        self.assertIn("create_catalog", desc)
        self.assertIn("accomplish", desc)
        self.assertIn("fork", desc)

    def test_format_context_handles_empty(self):
        ctx = {"agent": "root"}
        formatted = self.agent._format_context(ctx)
        self.assertIn("empty workspace", formatted)

    def test_format_context_lists_resources(self):
        ctx = {
            "agent": "root",
            "catalogs": ["main", "dev"],
            "warehouses": [],
        }
        formatted = self.agent._format_context(ctx)
        self.assertIn("main", formatted)
        self.assertIn("(none)", formatted)


# ----------------------------------------------------------------------- #
# Web / HTTP tools
# ----------------------------------------------------------------------- #
class TestWebTools(AutonomousAgentTestCase):
    def test_get_http_session_cached(self):
        s1 = self.agent._get_http_session()
        s2 = self.agent._get_http_session()
        self.assertIs(s1, s2)

    def test_scrape_links_extracts_hrefs(self):
        from unittest.mock import patch as _patch

        html = '<a href="https://example.com">Link</a> <a href="/page">Other</a>'
        with _patch.object(self.agent, "fetch_text", return_value=html):
            links = self.agent.scrape_links("http://test.com")
        self.assertEqual(links, ["https://example.com", "/page"])

    def test_open_browser_via_tool(self):
        from unittest.mock import patch as _patch

        with _patch("webbrowser.open", return_value=True) as mock_open:
            result = self.agent.run_tool("open_browser", "https://example.com")
        mock_open.assert_called_once_with("https://example.com")
        self.assertTrue(result)

    def test_web_tools_registered(self):
        for name in (
            "fetch",
            "fetch_text",
            "fetch_json",
            "fetch_many",
            "scrape_links",
            "open_browser",
            "fetch_entsoe_zones",
        ):
            self.assertIn(name, self.agent.tools, msg=name)


# ----------------------------------------------------------------------- #
# User-scoped space title
# ----------------------------------------------------------------------- #
class TestUserScopedSpaceTitle(AutonomousAgentTestCase):
    def test_resolve_managed_title_includes_username(self):
        from unittest.mock import patch as _patch

        with _patch.object(self.client, "user_scoped_name", return_value="alice"):
            title = self.genie._resolve_managed_title()
        self.assertIn("alice", title)
        self.assertIn("Yggdrasil Genie", title)

    def test_resolve_managed_title_falls_back_on_error(self):
        from unittest.mock import patch as _patch

        with _patch.object(
            self.client, "user_scoped_name", side_effect=RuntimeError("no user")
        ):
            title = self.genie._resolve_managed_title()
        self.assertEqual(title, "Yggdrasil Genie")


# ----------------------------------------------------------------------- #
# Deep introspection
# ----------------------------------------------------------------------- #
class TestDeepIntrospection(AutonomousAgentTestCase):
    def _setup_catalogs(self, names):
        """Configure the workspace mock to list catalogs with the given names."""
        mocks = []
        for n in names:
            m = MagicMock()
            m.name = n
            mocks.append(m)
        self.workspace_client.catalogs.list.return_value = mocks

    def _setup_schemas(self, names):
        mocks = []
        for n in names:
            m = MagicMock()
            m.name = n
            mocks.append(m)
        self.workspace_client.schemas.list.return_value = mocks

    def _setup_tables(self, names):
        mocks = []
        for n in names:
            m = MagicMock()
            m.name = n
            mocks.append(m)
        self.workspace_client.tables.list.return_value = mocks

    def test_introspect_deep_discovers_schemas(self):
        self._setup_catalogs(["main"])
        self._setup_schemas(["sales"])
        self._setup_tables([])

        context = self.agent.introspect(deep=True)
        self.assertIn("schemas", context)
        self.assertIn("main", context["schemas"])
        self.assertEqual(context["schemas"]["main"], ["sales"])

    def test_introspect_deep_discovers_tables(self):
        self._setup_catalogs(["main"])
        self._setup_schemas(["sales"])
        self._setup_tables(["orders"])

        context = self.agent.introspect(deep=True)
        self.assertIn("tables", context)
        self.assertIn("main.sales", context["tables"])
        self.assertEqual(context["tables"]["main.sales"], ["orders"])

    def test_introspect_shallow_skips_schemas(self):
        self._setup_catalogs(["main"])

        context = self.agent.introspect(deep=False)
        self.assertNotIn("schemas", context)
        self.assertNotIn("tables", context)

    def test_introspect_handles_schema_list_failure(self):
        self._setup_catalogs(["main"])
        self.workspace_client.schemas.list.side_effect = RuntimeError("no access")

        context = self.agent.introspect(deep=True)
        self.assertEqual(context["schemas"]["main"], [])

    def test_introspect_captures_current_user(self):
        self.workspace_client.catalogs.list.return_value = []
        from unittest.mock import patch as _patch

        with _patch.object(self.client, "user_scoped_name", return_value="alice"):
            context = self.agent.introspect(deep=False)
        self.assertEqual(context["current_user"], "alice")


# ----------------------------------------------------------------------- #
# VERIFY step parsing
# ----------------------------------------------------------------------- #
class TestVerifyStepParsing(AutonomousAgentTestCase):
    def test_parse_verify_step(self):
        steps = self.agent._parse_plan("1. VERIFY: describe_catalog(main)\n", "goal")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool, "describe_catalog")
        self.assertTrue(steps[0].kwargs.get("_verify"))

    def test_verify_step_fails_on_none_result(self):
        self.agent.register_tool("returns_none", lambda: None)
        step = AgentStep(
            action="verify nothing",
            tool="returns_none",
            kwargs={"_verify": True},
        )
        result = self.agent._execute_step(step)
        self.assertTrue(result.failed)
        self.assertIn("Verification failed", result.error)

    def test_verify_step_succeeds_on_real_result(self):
        self.agent.register_tool("returns_data", lambda: {"name": "main"})
        step = AgentStep(
            action="verify data",
            tool="returns_data",
            kwargs={"_verify": True},
        )
        result = self.agent._execute_step(step)
        self.assertTrue(result.succeeded)


# ----------------------------------------------------------------------- #
# PARALLEL step parsing
# ----------------------------------------------------------------------- #
class TestParallelStepParsing(AutonomousAgentTestCase):
    def test_parse_parallel_step(self):
        steps = self.agent._parse_plan(
            "1. PARALLEL:\n"
            "2. TOOL: create_catalog(a)\n"
            "3. TOOL: create_catalog(b)\n",
            "goal",
        )
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].tool, "_parallel")
        self.assertEqual(steps[1].tool, "create_catalog")
        self.assertEqual(steps[2].tool, "create_catalog")

    def test_parallel_step_execution(self):
        results = []
        self.agent.register_tool("record_a", lambda: results.append("a") or "a")
        self.agent.register_tool("record_b", lambda: results.append("b") or "b")

        parallel_step = AgentStep(action="parallel", tool="_parallel")
        plan = [
            AgentStep(action="record a", tool="record_a"),
            AgentStep(action="record b", tool="record_b"),
        ]
        executed = self.agent._execute_parallel_block(
            parallel_step,
            plan,
            "test goal",
            10,
        )
        self.assertEqual(len(executed), 2)
        self.assertTrue(all(s.succeeded for s in executed))
        self.assertEqual(set(results), {"a", "b"})

    def test_parallel_single_branch_runs_serially(self):
        self.agent.register_tool("single", lambda: "ok")
        plan = [AgentStep(action="single step", tool="single")]
        executed = self.agent._execute_parallel_block(
            AgentStep(action="par", tool="_parallel"),
            plan,
            "goal",
            10,
        )
        self.assertEqual(len(executed), 1)
        self.assertTrue(executed[0].succeeded)


# ----------------------------------------------------------------------- #
# Re-planning (ADJUST evaluation)
# ----------------------------------------------------------------------- #
class TestReplanning(AutonomousAgentTestCase):
    def test_accomplish_replans_on_adjust(self):
        """When evaluation says ADJUST, new steps get executed."""
        call_count = [0]

        def _ask_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                text = "1. TOOL: create_catalog(main)\n2. DONE: Ready\n"
            elif call_count[0] == 2:
                text = (
                    "ADJUST:\n"
                    "1. TOOL: create_schema(main.sales)\n"
                    "2. DONE: Now ready\n"
                )
            else:
                text = "DONE: All resources created."
            msg = _build_completed_message(text=text)
            return msg

        self.genie_api.start_conversation.return_value = Wait(
            waiter=_ask_side_effect,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )
        self.genie_api.create_message.return_value = Wait(
            waiter=_ask_side_effect,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )

        result = self.agent.accomplish("Set up sales storage", max_steps=10)
        self.assertTrue(result.succeeded)
        tool_names = [s.tool for s in result.steps if s.tool not in ("_done",)]
        self.assertIn("create_catalog", tool_names)


# ----------------------------------------------------------------------- #
# SKIP recovery
# ----------------------------------------------------------------------- #
class TestSkipRecovery(AutonomousAgentTestCase):
    def test_skip_recovery_continues_execution(self):
        """When recovery says SKIP, the agent continues past the failure."""
        call_count = [0]

        def _ask_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                text = (
                    "1. TOOL: boom_tool()\n"
                    "2. TOOL: create_catalog(main)\n"
                    "3. DONE: Ready\n"
                )
            elif call_count[0] == 2:
                text = "SKIP"
            else:
                text = "DONE: Completed."
            return _build_completed_message(text=text)

        self.genie_api.start_conversation.return_value = Wait(
            waiter=_ask_side_effect,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )
        self.genie_api.create_message.return_value = Wait(
            waiter=_ask_side_effect,
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )

        self.agent.register_tool(
            "boom_tool",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        result = self.agent.accomplish("Test skip", max_steps=10)
        catalog_steps = [s for s in result.steps if s.tool == "create_catalog"]
        self.assertTrue(len(catalog_steps) >= 1)


# ----------------------------------------------------------------------- #
# Tool name fuzzy matching
# ----------------------------------------------------------------------- #
class TestToolNameSuggestion(AutonomousAgentTestCase):
    def test_suggest_tool_finds_close_match(self):
        match = self.agent._suggest_tool("creat_catalog")
        self.assertEqual(match, "create_catalog")

    def test_suggest_tool_returns_none_for_garbage(self):
        match = self.agent._suggest_tool("xyzzy_foobar")
        self.assertIsNone(match)

    def test_suggest_tool_returns_none_for_empty(self):
        match = self.agent._suggest_tool("")
        self.assertIsNone(match)

    def test_execute_step_shows_suggestion(self):
        step = AgentStep(action="test", tool="creat_catalog")
        result = self.agent._execute_step(step)
        self.assertTrue(result.failed)
        self.assertIn("did you mean", result.error)
        self.assertIn("create_catalog", result.error)


# ----------------------------------------------------------------------- #
# History context
# ----------------------------------------------------------------------- #
class TestHistoryContext(AutonomousAgentTestCase):
    def test_empty_history_returns_empty(self):
        ctx = self.agent._format_history_context()
        self.assertEqual(ctx, "")

    def test_history_context_includes_recent_answers(self):
        self._start_returns(_build_completed_message(text="test answer"))
        self.agent.run("q1")
        ctx = self.agent._format_history_context()
        self.assertIn("test answer", ctx)
        self.assertIn("Recent conversation context", ctx)


# ----------------------------------------------------------------------- #
# Enhanced context formatting
# ----------------------------------------------------------------------- #
class TestEnhancedContextFormatting(AutonomousAgentTestCase):
    def test_format_context_with_nested_dicts(self):
        ctx = {
            "agent": "root",
            "schemas": {"main": ["sales", "marketing"]},
            "tables": {"main.sales": ["orders", "customers"]},
        }
        formatted = self.agent._format_context(ctx)
        self.assertIn("schemas.main: sales, marketing", formatted)
        self.assertIn("tables.main.sales: orders, customers", formatted)

    def test_format_context_with_empty_nested_dicts(self):
        ctx = {
            "agent": "root",
            "schemas": {"main": []},
        }
        formatted = self.agent._format_context(ctx)
        self.assertIn("schemas.main: (empty)", formatted)

    def test_format_context_with_current_user(self):
        ctx = {
            "agent": "root",
            "current_user": "alice",
            "catalogs": [],
        }
        formatted = self.agent._format_context(ctx)
        self.assertIn("current_user: alice", formatted)

    def test_format_context_skips_none_values(self):
        ctx = {
            "agent": "root",
            "current_user": None,
            "catalogs": ["main"],
        }
        formatted = self.agent._format_context(ctx)
        self.assertNotIn("current_user", formatted)
        self.assertIn("catalogs: main", formatted)


# ----------------------------------------------------------------------- #
# Fallback plan — new keyword coverage
# ----------------------------------------------------------------------- #
class TestFallbackPlanExpanded(AutonomousAgentTestCase):
    def test_fallback_detects_setup_keyword(self):
        steps = self.agent._fallback_plan("set up a data pipeline")
        tool_names = [s.tool for s in steps]
        self.assertIn("introspect", tool_names)
        self.assertIn("ask", tool_names)

    def test_fallback_detects_fetch_keyword(self):
        steps = self.agent._fallback_plan("fetch data from API")
        tool_names = [s.tool for s in steps]
        self.assertIn("introspect", tool_names)
        ask_steps = [s for s in steps if s.tool == "ask"]
        self.assertTrue(
            any("endpoint" in (s.args[0] if s.args else "") for s in ask_steps)
        )

    def test_fallback_detects_job_keyword(self):
        steps = self.agent._fallback_plan("schedule a Spark job")
        self.assertTrue(len(steps) >= 2)

    def test_fallback_always_starts_with_introspect(self):
        for goal in ["do something", "create table", "set up warehouse"]:
            steps = self.agent._fallback_plan(goal)
            self.assertEqual(
                steps[0].tool,
                "introspect",
                msg=f"Fallback for {goal!r} should start with introspect",
            )


# ----------------------------------------------------------------------- #
# Intent classification
# ----------------------------------------------------------------------- #
class TestClassifyIntent(AutonomousAgentTestCase):
    def test_question_with_question_mark(self):
        self.assertEqual(self.agent.classify_intent("How many orders?"), "question")

    def test_question_with_prefix(self):
        for text in [
            "How many orders last month",
            "What is the average revenue",
            "Show me the top 10 customers",
            "Tell me about sales trends",
            "Count the number of orders",
            "List all tables in main.sales",
        ]:
            self.assertEqual(
                self.agent.classify_intent(text),
                "question",
                msg=f"{text!r} should be classified as 'question'",
            )

    def test_goal_with_imperative(self):
        for text in [
            "Create a catalog for energy data",
            "Set up the ingestion pipeline",
            "Deploy the workflow to production",
            "Build a curated layer for prices",
            "Configure a new warehouse",
            "Delete the test schema",
            "Schedule a daily refresh job",
            "Ingest data from the ENTSO-E API",
        ]:
            self.assertEqual(
                self.agent.classify_intent(text),
                "goal",
                msg=f"{text!r} should be classified as 'goal'",
            )

    def test_goal_with_polite_phrase(self):
        for text in [
            "I want to create a new table",
            "I need to set up storage",
            "Can you create a catalog?",
            "Please deploy the pipeline",
            "Let's build the raw layer",
            "Go ahead and create the schema",
        ]:
            self.assertEqual(
                self.agent.classify_intent(text),
                "goal",
                msg=f"{text!r} should be classified as 'goal'",
            )

    def test_tool_with_run_prefix(self):
        self.assertEqual(self.agent.classify_intent("run introspect"), "tool")

    def test_tool_with_parens(self):
        self.assertEqual(
            self.agent.classify_intent("introspect()"),
            "tool",
        )

    def test_tool_with_call_prefix(self):
        self.assertEqual(
            self.agent.classify_intent("call introspect"),
            "tool",
        )

    def test_ambiguous_defaults_to_question(self):
        self.assertEqual(
            self.agent.classify_intent("orders by region"),
            "question",
        )

    def test_resource_noun_without_question_mark_is_goal(self):
        self.assertEqual(
            self.agent.classify_intent("need a new warehouse for analytics"),
            "goal",
        )


# ----------------------------------------------------------------------- #
# Smart respond()
# ----------------------------------------------------------------------- #
class TestRespond(AutonomousAgentTestCase):
    def test_respond_routes_question_to_genie(self):
        self._ask_returns("42 orders last month")
        resp = self.agent.respond("How many orders last month?")
        self.assertIsInstance(resp, AgentResponse)
        self.assertEqual(resp.intent, "question")
        self.assertTrue(resp.succeeded)

    def test_respond_routes_goal_to_accomplish(self):
        self._ask_returns("1. DONE: Already exists\n")
        resp = self.agent.respond("Create a catalog for energy data")
        self.assertIsInstance(resp, AgentResponse)
        self.assertEqual(resp.intent, "goal")
        self.assertIsInstance(resp.result, AgentResult)

    def test_respond_routes_tool_call(self):
        resp = self.agent.respond("run introspect")
        self.assertIsInstance(resp, AgentResponse)
        self.assertEqual(resp.intent, "tool")
        self.assertTrue(resp.succeeded)
        self.assertIsInstance(resp.result, dict)

    def test_respond_tool_with_parens(self):
        self.agent.register_tool("ping", lambda: "pong")
        resp = self.agent.respond("ping()")
        self.assertEqual(resp.intent, "tool")
        self.assertEqual(resp.result, "pong")

    def test_respond_failed_tool(self):
        self.agent.register_tool(
            "boom", lambda: (_ for _ in ()).throw(RuntimeError("kaboom"))
        )
        resp = self.agent.respond("run boom")
        self.assertEqual(resp.intent, "tool")
        self.assertFalse(resp.succeeded)
        self.assertIn("kaboom", resp.text)


# ----------------------------------------------------------------------- #
# Extract tool call
# ----------------------------------------------------------------------- #
class TestExtractToolCall(AutonomousAgentTestCase):
    def test_extract_run_tool(self):
        name, args, kwargs = self.agent._extract_tool_call("run introspect")
        self.assertEqual(name, "introspect")
        self.assertEqual(args, ())

    def test_extract_tool_with_args(self):
        name, args, kwargs = self.agent._extract_tool_call(
            "create_catalog(energy, comment='test')"
        )
        self.assertEqual(name, "create_catalog")
        self.assertEqual(args, ("energy",))
        self.assertEqual(kwargs, {"comment": "test"})

    def test_extract_unknown_tool_returns_none(self):
        name, _, _ = self.agent._extract_tool_call("run nonexistent_tool")
        self.assertIsNone(name)

    def test_extract_plain_text_returns_none(self):
        name, _, _ = self.agent._extract_tool_call("just a question")
        self.assertIsNone(name)
