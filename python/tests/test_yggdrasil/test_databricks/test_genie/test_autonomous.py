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

from yggdrasil.data.enums.state import State
from yggdrasil.databricks.genie import AgentResult, AgentStep, AutonomousAgent
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
            "1. TOOL: create_catalog(energy, comment='test')\n", "goal",
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
        self.assertTrue(len(steps) >= 1)
        self.assertEqual(steps[0].tool, "ask")

    def test_warehouse_keywords_trigger_ask(self):
        steps = self.agent._fallback_plan("Set up a SQL warehouse")
        self.assertTrue(len(steps) >= 1)

    def test_generic_goal_triggers_ask(self):
        steps = self.agent._fallback_plan("Do something vague")
        self.assertTrue(len(steps) >= 1)
        self.assertEqual(steps[0].tool, "ask")


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
        out = self.agent.execute_plan([
            {"tool": "add", "args": [3], "kwargs": {"y": 7}},
        ])
        self.assertEqual(out, [10])

    def test_execute_plan_stop_on_error(self):
        self.agent.register_tool("boom", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        self.agent.register_tool("ok", lambda: "ok")

        with self.assertRaises(RuntimeError):
            self.agent.execute_plan([
                {"tool": "boom"},
                {"tool": "ok"},
            ])

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
        results = self.agent.parallel([
            lambda: "a",
            lambda: "b",
            lambda: "c",
        ])
        self.assertEqual(set(results), {"a", "b", "c"})
        self.assertEqual(len(results), 3)

    def test_parallel_preserves_order(self):
        import time
        results = self.agent.parallel([
            lambda: (time.sleep(0.01), 1)[1],
            lambda: 2,
            lambda: 3,
        ])
        self.assertEqual(results, [1, 2, 3])


# ----------------------------------------------------------------------- #
# Tool registry
# ----------------------------------------------------------------------- #
class TestAutonomousTools(AutonomousAgentTestCase):
    def test_autonomous_tools_registered(self):
        for name in (
            "create_catalog", "create_schema", "create_table",
            "create_volume", "create_warehouse", "create_cluster",
            "deploy_workflow", "setup_storage", "setup_genie_space",
            "introspect", "describe_catalog", "describe_schema", "describe_table",
            "accomplish", "fork", "fork_many", "execute_plan", "parallel",
            "step_results", "children",
            "create_job", "run_and_wait",
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
        self.assertEqual(call_kwargs.kwargs.get("name") or call_kwargs[1].get("name"), "my-wh")

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
        result = self.agent.create_cluster("my-cluster", spark_version="14.3.x-scala2.12")
        self.assertIsNotNone(result)

    def test_create_cluster_with_fixed_workers(self):
        self._setup_spark_versions()
        profile = ClusterProfile(num_workers=3)
        result = self.agent.create_cluster("fixed", spark_version="14.3.x-scala2.12", profile=profile)
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
            self.genie.defaults, warehouse_id="wh-123",
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
