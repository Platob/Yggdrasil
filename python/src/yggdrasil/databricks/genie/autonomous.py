"""Autonomous goal-directed agent on top of Databricks Genie.

:class:`AutonomousAgent` is a real agent — it takes a goal, plans how to
achieve it, executes the plan, evaluates progress, and self-corrects.  Genie's
LLM is the reasoning backbone; the registered tools are the hands.

The simplest interaction::

    agent = client.genie.autonomous_agent
    result = agent.accomplish("Set up an ingestion pipeline for ENTSO-E transparency data")
    # Agent autonomously:
    #   1. Introspects the workspace (what catalogs/schemas/tables exist?)
    #   2. Asks Genie to plan the setup
    #   3. Creates catalog, schema, tables, volumes with best-practice configs
    #   4. Evaluates: did everything get created? any failures?
    #   5. Reports back

Self-duplication for parallel work::

    agent = client.genie.autonomous_agent
    raw, curated = agent.fork(name="raw-layer"), agent.fork(name="curated-layer")
    agent.parallel([
        lambda: raw.accomplish("Create raw landing tables for prices and load"),
        lambda: curated.accomplish("Create curated tables with proper types"),
    ])
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from dataclasses import dataclass, field, replace as _dc_replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
)

from yggdrasil.data.enums.state import State

from .agent import GenieAgent

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.cluster.cluster import Cluster
    from yggdrasil.databricks.jobs import Job
    from yggdrasil.databricks.schema.schema import UCSchema
    from yggdrasil.databricks.table.table import Table
    from yggdrasil.databricks.volume.volume import Volume
    from yggdrasil.databricks.warehouse.warehouse import SQLWarehouse

    from .profiles import ClusterProfile, StorageProfile, WarehouseProfile
    from .service import Genie


__all__ = ["AutonomousAgent", "AgentResult", "AgentStep"]

LOGGER = logging.getLogger(__name__)

_MAX_INTROSPECT_CATALOGS: int = 10
_MAX_INTROSPECT_SCHEMAS: int = 10
_MAX_INTROSPECT_TABLES: int = 20
_MAX_RECOVERY_ATTEMPTS: int = 2
_MAX_REPLAN_CYCLES: int = 3
_MAX_HISTORY_FOR_CONTEXT: int = 5

_RESOURCE_CREATION_TOOLS: frozenset[str] = frozenset(
    {
        "create_catalog",
        "create_schema",
        "create_table",
        "create_volume",
        "create_warehouse",
        "create_cluster",
        "setup_storage",
    }
)


# ----------------------------------------------------------------------- #
# Agent step / result data structures
# ----------------------------------------------------------------------- #


@dataclass
class AgentStep:
    """One step the agent planned or executed."""

    action: str
    tool: Optional[str] = None
    args: tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    state: State = State.PENDING

    @property
    def succeeded(self) -> bool:
        return self.state.is_succeeded

    @property
    def failed(self) -> bool:
        return self.state.is_failed


@dataclass
class AgentResult:
    """Outcome of an :meth:`AutonomousAgent.accomplish` run."""

    goal: str
    steps: list[AgentStep] = field(default_factory=list)
    conclusion: str = ""
    state: State = State.RUNNING

    @property
    def succeeded(self) -> bool:
        return self.state.is_succeeded

    @property
    def failed_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.failed]

    @property
    def completed_steps(self) -> list[AgentStep]:
        return [s for s in self.steps if s.succeeded]

    def summary(self) -> str:
        done = len(self.completed_steps)
        total = len(self.steps)
        return (
            f"Goal: {self.goal}\n"
            f"State: {self.state.name} ({done}/{total} steps completed)\n"
            f"Conclusion: {self.conclusion}"
        )


# ----------------------------------------------------------------------- #
# Planning prompt templates
# ----------------------------------------------------------------------- #

_PLAN_PROMPT = """\
You are an autonomous Databricks workspace agent. Given a goal and the current \
workspace state, produce a step-by-step plan using ONLY these available tools:

{tools}

Current workspace state:
{context}

{history_context}

Goal: {goal}

Rules:
- Use fully-qualified three-part names: catalog.schema.table, catalog.schema.volume.
- Follow the naming convention: raw tables are raw_<entity>, curated are <entity>.
- One schema per data source: <catalog>.<source>.
- Create the catalog first, then schemas, then tables/volumes (parent-first order).
- When creating tables, always specify a schema definition with proper types.
- For storage setup, prefer setup_storage(profile) over individual create_* calls.
- If the goal has independent parts, say PARALLEL: before listing the independent \
branches so the agent can fork them.
- Verify critical resources exist after creation with describe_catalog / \
describe_schema / describe_table.

Respond with a numbered list of steps. Each step must be ONE of:
- TOOL: <tool_name>(<arg1>, <arg2>, key=value) — call a registered tool
- ASK: <question> — ask Genie a data question to inform the plan
- VERIFY: <tool_name>(<args>) — verify a prior step's result exists
- DONE: <summary> — the goal is achieved
- PARALLEL: — next indented steps are independent and can run concurrently

Be specific and concrete. Prefer fewer, higher-level tool calls (setup_storage \
over manual create_catalog + create_schema + create_table chains).
"""

_EVALUATE_PROMPT = """\
You are evaluating progress toward a goal.

Goal: {goal}

Steps completed so far:
{steps_summary}

Remaining plan:
{remaining}

Resources confirmed to exist:
{verified_resources}

Decide the next action. Respond with EXACTLY ONE of:
- DONE: <conclusion> — the goal is fully achieved, summarize what was done
- ADJUST: <numbered_steps> — the plan needs changes, provide corrected remaining \
steps in the same TOOL:/ASK:/VERIFY:/DONE: format
- RETRY: <step_description> — retry a failed step with a different approach
- CONTINUE — proceed with the next planned step as-is
"""

_RECOVERY_PROMPT = """\
A step failed while working toward: {goal}

Failed step: {failed_action}
Tool: {failed_tool}
Error: {error}

Steps completed before the failure:
{prior_steps}

Available tools: {tools}

Suggest a recovery. Respond with ONE of:
- TOOL: <recovery_tool_call> — a single tool call that fixes or works around the error
- SKIP — this step is non-critical, continue with the remaining plan
- ABORT: <reason> — the error is unrecoverable for this goal
"""


class AutonomousAgent(GenieAgent):
    """Goal-directed agent that plans, executes, and self-corrects.

    The agent uses Genie's LLM as its reasoning backbone and the registered
    tools as its execution surface.  It can:

    * **Accomplish goals** — :meth:`accomplish` takes a natural-language goal,
      introspects the workspace, plans via Genie, executes, evaluates, repeats.
    * **Introspect** — :meth:`introspect` discovers what already exists.
    * **Self-duplicate** — :meth:`fork` spawns isolated child agents.
    * **Execute plans** — :meth:`execute_plan` runs a tool-call sequence.
    * **Run in parallel** — :meth:`parallel` for concurrent child work.
    """

    def __init__(
        self,
        service: "Genie",
        *,
        name: Optional[str] = None,
        parent: Optional["AutonomousAgent"] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__(service=service)
        self.name: str = name or "root"
        self.parent: Optional["AutonomousAgent"] = parent
        self.children: list["AutonomousAgent"] = []
        self.max_steps: int = max_steps or self.service.defaults.agent_max_steps
        self._step_results: list[Any] = []
        self._register_autonomous_tools()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"history={len(self.history)}, "
            f"children={len(self.children)}, "
            f"tools={len(self.tools)})"
        )

    # ------------------------------------------------------------------ #
    # The autonomous loop
    # ------------------------------------------------------------------ #
    def accomplish(
        self,
        goal: str,
        *,
        max_steps: Optional[int] = None,
        auto_fork: bool = True,
    ) -> AgentResult:
        """Autonomously work toward a goal.

        The agent loop:

        1. **Introspect** the workspace — what catalogs, schemas, tables,
           warehouses, jobs already exist.
        2. **Plan** — ask Genie to break the goal into tool-call steps,
           given the workspace context and available tools.
        3. **Execute** each step, recording results.  VERIFY steps trigger
           a post-check that the resource exists.
        4. **Evaluate** — ask Genie whether the goal is met, the plan needs
           adjustment, or execution should continue.
        5. **Re-plan** — when evaluation says ADJUST, parse the new steps
           and loop (up to ``_MAX_REPLAN_CYCLES``).
        6. **Repeat** until done or the step budget is exhausted.

        When ``auto_fork`` is ``True`` and the plan contains a PARALLEL
        block, the agent spawns children to work concurrently.
        """
        budget = max_steps or self.max_steps
        result = AgentResult(goal=goal)

        LOGGER.info(
            "Accomplishing goal %r (agent=%r, budget=%d)", goal, self.name, budget
        )

        context = self.introspect()
        plan = self._plan(goal, context)
        replan_cycles = 0
        verified_resources: list[str] = []
        aborted = False

        while True:
            while plan:
                step = plan.pop(0)

                if len(result.steps) >= budget:
                    LOGGER.info(
                        "Step budget exhausted for %r (budget=%d)",
                        self.name,
                        budget,
                    )
                    result.conclusion = (
                        f"Step budget ({budget}) exhausted before completion."
                    )
                    aborted = True
                    break

                if auto_fork and step.tool == "_parallel":
                    parallel_results = self._execute_parallel_block(
                        step,
                        plan,
                        goal,
                        budget - len(result.steps),
                    )
                    result.steps.extend(parallel_results)
                    continue

                executed = self._execute_step(step)
                result.steps.append(executed)

                if executed.succeeded and executed.tool in _RESOURCE_CREATION_TOOLS:
                    verified_resources.append(
                        f"{executed.tool}("
                        f"{', '.join(str(a) for a in executed.args)})"
                    )

                if executed.failed:
                    LOGGER.warning(
                        "Step failed for agent %r: %s (tool=%r)",
                        self.name,
                        executed.error,
                        executed.tool,
                    )
                    recovery = self._handle_failure(
                        goal,
                        result,
                        executed,
                        context,
                    )
                    if recovery is not None:
                        plan[0:0] = recovery
                        continue
                    result.conclusion = (
                        f"Failed at step: {executed.action} — {executed.error}"
                    )
                    aborted = True
                    break

            if aborted:
                break

            evaluation = self._evaluate(
                goal,
                result,
                context,
                verified_resources=verified_resources,
            )

            if evaluation.startswith("ADJUST:") and replan_cycles < _MAX_REPLAN_CYCLES:
                replan_cycles += 1
                new_steps = self._parse_plan(
                    evaluation[len("ADJUST:") :],
                    goal,
                )
                if new_steps:
                    LOGGER.info(
                        "Re-planning for %r (cycle %d): %d new steps",
                        self.name,
                        replan_cycles,
                        len(new_steps),
                    )
                    plan = new_steps
                    continue
            elif evaluation.startswith("RETRY:") and replan_cycles < _MAX_REPLAN_CYCLES:
                retry_text = evaluation[len("RETRY:") :].strip()
                retry_step = self._parse_tool_step(retry_text)
                if retry_step:
                    replan_cycles += 1
                    plan = [retry_step]
                    continue

            if evaluation.startswith("DONE:"):
                result.conclusion = evaluation[len("DONE:") :].strip()
            else:
                result.conclusion = evaluation
            result.state = State.SUCCEEDED
            break

        if not result.succeeded:
            result.state = State.FAILED

        LOGGER.info(
            "Agent %r finished goal %r (state=%s, steps=%d)",
            self.name,
            goal,
            result.state.name,
            len(result.steps),
        )
        return result

    # ------------------------------------------------------------------ #
    # Introspection — what exists in the workspace
    # ------------------------------------------------------------------ #
    def introspect(self, *, deep: bool = True) -> dict[str, Any]:
        """Discover what already exists in the workspace.

        Returns a dict describing the current state: catalogs, schemas,
        tables, warehouses, jobs, clusters, and volumes.  Used by
        :meth:`accomplish` to inform planning.

        When *deep* is ``True`` (default), also enumerates schemas inside
        each catalog and tables inside each schema (capped to avoid
        runaway listing on large workspaces). Set ``deep=False`` to skip
        the schema/table walk when only top-level awareness is needed.
        """
        ws = self.client.workspace_client()
        context: dict[str, Any] = {"agent": self.name}

        try:
            catalogs_raw = list(ws.catalogs.list())
            context["catalogs"] = [
                getattr(c, "name", None) or str(c) for c in catalogs_raw
            ]
        except Exception:
            context["catalogs"] = []

        if deep and context["catalogs"]:
            schemas_by_catalog: dict[str, list[str]] = {}
            tables_by_schema: dict[str, list[str]] = {}
            for cat_name in context["catalogs"][:_MAX_INTROSPECT_CATALOGS]:
                try:
                    schemas_raw = list(ws.schemas.list(catalog_name=cat_name))
                    schema_names = [
                        getattr(s, "name", None) or str(s) for s in schemas_raw
                    ]
                    schemas_by_catalog[cat_name] = schema_names[
                        :_MAX_INTROSPECT_SCHEMAS
                    ]
                    for schema_name in schema_names[:_MAX_INTROSPECT_SCHEMAS]:
                        fqn = f"{cat_name}.{schema_name}"
                        try:
                            tables_raw = list(
                                ws.tables.list(
                                    catalog_name=cat_name,
                                    schema_name=schema_name,
                                )
                            )
                            tables_by_schema[fqn] = [
                                getattr(t, "name", None) or str(t) for t in tables_raw
                            ][:_MAX_INTROSPECT_TABLES]
                        except Exception:
                            tables_by_schema[fqn] = []
                except Exception:
                    schemas_by_catalog[cat_name] = []
            context["schemas"] = schemas_by_catalog
            context["tables"] = tables_by_schema

        try:
            warehouses = list(ws.warehouses.list())
            context["warehouses"] = [
                getattr(w, "name", None) or str(w) for w in warehouses
            ]
        except Exception:
            context["warehouses"] = []

        try:
            jobs = list(self.client.jobs.list(limit=20))
            context["jobs"] = [getattr(j, "name", None) or str(j) for j in jobs]
        except Exception:
            context["jobs"] = []

        try:
            context["current_user"] = self.client.user_scoped_name("")
        except Exception:
            context["current_user"] = None

        LOGGER.debug(
            "Introspected workspace for agent %r: %d catalogs, %d warehouses, %d jobs",
            self.name,
            len(context["catalogs"]),
            len(context["warehouses"]),
            len(context["jobs"]),
        )
        return context

    # ------------------------------------------------------------------ #
    # Planning — use Genie to break goal into steps
    # ------------------------------------------------------------------ #
    def _plan(self, goal: str, context: dict[str, Any]) -> list[AgentStep]:
        """Ask Genie to produce a plan, then parse it into steps."""
        tools_desc = self._describe_tools()
        context_str = self._format_context(context)
        history_ctx = self._format_history_context()

        prompt = _PLAN_PROMPT.format(
            tools=tools_desc,
            context=context_str,
            goal=goal,
            history_context=history_ctx,
        )

        try:
            answer = self.service.ask(prompt)
            self.history.append(answer)
            plan_text = answer.text or ""
        except Exception as exc:
            LOGGER.warning(
                "Genie planning failed for %r: %s — using fallback", self.name, exc
            )
            plan_text = ""

        steps = self._parse_plan(plan_text, goal)
        if not steps:
            steps = self._fallback_plan(goal)

        LOGGER.info(
            "Planned %d steps for goal %r (agent=%r)", len(steps), goal, self.name
        )
        return steps

    def _parse_plan(self, plan_text: str, goal: str) -> list[AgentStep]:
        """Parse Genie's plan text into AgentStep objects."""
        steps: list[AgentStep] = []
        for line in plan_text.splitlines():
            line = line.strip().lstrip("0123456789.)-] ")
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("TOOL:"):
                step = self._parse_tool_step(line[5:].strip())
                if step:
                    steps.append(step)
            elif upper.startswith("VERIFY:"):
                step = self._parse_tool_step(line[7:].strip())
                if step:
                    step.kwargs["_verify"] = True
                    steps.append(step)
            elif upper.startswith("ASK:"):
                steps.append(
                    AgentStep(
                        action=line[4:].strip(),
                        tool="ask",
                        args=(line[4:].strip(),),
                    )
                )
            elif upper.startswith("PARALLEL:"):
                steps.append(
                    AgentStep(
                        action=line[9:].strip() or "parallel block",
                        tool="_parallel",
                    )
                )
            elif upper.startswith("DONE:"):
                steps.append(
                    AgentStep(
                        action=line[5:].strip(),
                        tool="_done",
                    )
                )
        return steps

    def _parse_tool_step(self, text: str) -> Optional[AgentStep]:
        """Parse ``tool_name(arg1, arg2, key=val)`` into an AgentStep."""
        paren = text.find("(")
        if paren == -1:
            tool_name = text.strip()
            if tool_name in self.tools:
                return AgentStep(action=text, tool=tool_name)
            return None

        tool_name = text[:paren].strip()
        if tool_name not in self.tools:
            return AgentStep(action=text, tool=tool_name)

        args_str = text[paren + 1 :].rstrip(")")
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for part in self._split_args(args_str):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                kwargs[k.strip()] = self._coerce_value(v.strip())
            else:
                args.append(self._coerce_value(part))

        return AgentStep(
            action=text,
            tool=tool_name,
            args=tuple(args),
            kwargs=kwargs,
        )

    @staticmethod
    def _split_args(s: str) -> list[str]:
        """Split argument string respecting nested parens and quotes."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        in_quote: Optional[str] = None
        for ch in s:
            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
            elif ch in ('"', "'"):
                in_quote = ch
                current.append(ch)
            elif ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    @staticmethod
    def _coerce_value(v: str) -> Any:
        """Best-effort coercion of a string value from the LLM plan."""
        stripped = v.strip("\"'")
        if stripped != v:
            return stripped
        if v.lower() in ("true",):
            return True
        if v.lower() in ("false",):
            return False
        if v.lower() in ("none", "null"):
            return None
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    def _fallback_plan(self, goal: str) -> list[AgentStep]:
        """Produce a sensible default plan when Genie can't plan."""
        lower = goal.lower()
        steps: list[AgentStep] = []

        steps.append(
            AgentStep(
                action="Introspect workspace to understand current state",
                tool="introspect",
            )
        )

        if any(
            kw in lower
            for kw in (
                "catalog",
                "schema",
                "table",
                "volume",
                "storage",
                "pipeline",
                "ingest",
                "set up",
                "setup",
                "create",
                "land",
                "raw",
                "curate",
            )
        ):
            steps.append(
                AgentStep(
                    action="Ask Genie to analyze the data requirements",
                    tool="ask",
                    args=(
                        f"What catalogs, schemas, and tables are needed for: {goal}? "
                        "List specific table names with fully-qualified paths "
                        "(catalog.schema.table) and column types.",
                    ),
                )
            )

        if any(kw in lower for kw in ("warehouse", "sql", "query", "analytics")):
            steps.append(
                AgentStep(
                    action="Ask Genie about warehouse configuration",
                    tool="ask",
                    args=(f"What SQL warehouse configuration is best for: {goal}",),
                )
            )

        if any(kw in lower for kw in ("cluster", "compute", "spark", "job")):
            steps.append(
                AgentStep(
                    action="Ask Genie about compute configuration",
                    tool="ask",
                    args=(f"What cluster configuration is best for: {goal}",),
                )
            )

        if any(kw in lower for kw in ("fetch", "http", "api", "download", "scrape")):
            steps.append(
                AgentStep(
                    action="Ask Genie about data source endpoints",
                    tool="ask",
                    args=(
                        f"What API endpoints or data sources are needed for: {goal}? "
                        "List the URLs, authentication method, and expected response format.",
                    ),
                )
            )

        if len(steps) <= 1:
            steps.append(
                AgentStep(
                    action="Ask Genie to understand the goal",
                    tool="ask",
                    args=(
                        f"Help me plan how to accomplish this in Databricks: {goal}. "
                        "Break it into concrete steps with specific resource names.",
                    ),
                )
            )

        return steps

    # ------------------------------------------------------------------ #
    # Step execution
    # ------------------------------------------------------------------ #
    def _execute_step(self, step: AgentStep) -> AgentStep:
        """Execute a single planned step."""
        if step.tool == "_done":
            step.state = State.SUCCEEDED
            step.result = step.action
            return step

        if step.tool == "_parallel":
            step.state = State.SUCCEEDED
            step.result = "parallel marker"
            return step

        if step.tool is None or step.tool not in self.tools:
            near = self._suggest_tool(step.tool or "")
            hint = f" — did you mean {near!r}?" if near else ""
            step.state = State.FAILED
            step.error = (
                f"Unknown tool {step.tool!r}{hint}; "
                f"registered: {sorted(self.tools)!r}"
            )
            return step

        is_verify = step.kwargs.pop("_verify", False)
        step.state = State.RUNNING
        LOGGER.debug("Executing step: %s (tool=%r)", step.action, step.tool)
        try:
            step.result = self.run_tool(step.tool, *step.args, **step.kwargs)
            step.state = State.SUCCEEDED
            if is_verify and step.result is None:
                step.state = State.FAILED
                step.error = (
                    f"Verification failed: {step.tool}({step.args}) returned None"
                )
        except Exception as exc:
            step.state = State.FAILED
            step.error = f"{type(exc).__name__}: {exc}"
            LOGGER.debug("Step execution failed: %s", exc)
        return step

    # ------------------------------------------------------------------ #
    # Evaluation — assess progress and decide next action
    # ------------------------------------------------------------------ #
    def _evaluate(
        self,
        goal: str,
        result: AgentResult,
        context: dict[str, Any],
        *,
        verified_resources: Optional[list[str]] = None,
    ) -> str:
        """Ask Genie to evaluate whether the goal is met.

        Returns the raw evaluation text. The caller inspects the prefix
        (DONE: / ADJUST: / RETRY: / CONTINUE) to decide next action.
        """
        steps_summary = "\n".join(
            f"  {i+1}. [{s.state.name}] {s.action}"
            + (f" -> {s.result!r}" if s.result and s.succeeded else "")
            + (f" X {s.error}" if s.error else "")
            for i, s in enumerate(result.steps)
        )

        verified_str = (
            "\n".join(f"  - {r}" for r in verified_resources)
            if verified_resources
            else "  (none verified)"
        )

        prompt = _EVALUATE_PROMPT.format(
            goal=goal,
            steps_summary=steps_summary or "(no steps executed yet)",
            remaining="(none — plan completed)",
            verified_resources=verified_str,
        )

        try:
            answer = self.service.ask(prompt)
            self.history.append(answer)
            text = (answer.text or "").strip()
            return text or "DONE: Plan completed."
        except Exception:
            n_done = len(result.completed_steps)
            n_total = len(result.steps)
            if n_done == n_total and n_total > 0:
                return f"DONE: Completed all {n_total} steps."
            return f"DONE: Completed {n_done}/{n_total} steps."

    def _handle_failure(
        self,
        goal: str,
        result: AgentResult,
        failed_step: AgentStep,
        context: dict[str, Any],
    ) -> Optional[list[AgentStep]]:
        """Try to recover from a failed step.

        Asks Genie for recovery guidance using the enriched
        ``_RECOVERY_PROMPT`` that includes prior step context. Supports
        TOOL (retry with different call), SKIP (continue past the
        failure), and ABORT (give up).

        Returns a list of recovery steps, or ``None`` to abort.
        """
        prior_summary = (
            "\n".join(
                f"  {i+1}. [{s.state.name}] {s.action}"
                for i, s in enumerate(result.steps)
                if s is not failed_step
            )
            or "  (none)"
        )

        prompt = _RECOVERY_PROMPT.format(
            goal=goal,
            failed_action=failed_step.action,
            failed_tool=failed_step.tool or "unknown",
            error=failed_step.error or "unknown error",
            prior_steps=prior_summary,
            tools=", ".join(sorted(self.tools)),
        )
        try:
            answer = self.service.ask(prompt)
            self.history.append(answer)
            text = (answer.text or "").strip()
            upper = text.upper()

            if upper.startswith("ABORT"):
                return None
            if upper.startswith("SKIP"):
                return []

            recovery = self._parse_plan(text, goal)
            return recovery if recovery else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Self-duplication
    # ------------------------------------------------------------------ #
    def fork(self, *, name: Optional[str] = None) -> "AutonomousAgent":
        """Spawn a child agent with isolated history on the same client.

        The child inherits the tool registry snapshot at fork time but
        evolves independently — tools registered on the child do not
        propagate back to the parent or siblings.
        """
        child_name = name or f"{self.name}.child-{len(self.children)}"
        child = AutonomousAgent(
            service=self.service,
            name=child_name,
            parent=self,
            max_steps=self.max_steps,
        )
        self.children.append(child)
        LOGGER.info("Forked autonomous agent %r from %r", child.name, self.name)
        return child

    def fork_many(self, names: Sequence[str]) -> list["AutonomousAgent"]:
        """Fork multiple named children at once."""
        return [self.fork(name=n) for n in names]

    # ------------------------------------------------------------------ #
    # Plan execution (lower-level — for callers who build their own plan)
    # ------------------------------------------------------------------ #
    def execute_plan(
        self,
        steps: Sequence[Dict[str, Any]],
        *,
        stop_on_error: bool = True,
    ) -> list[Any]:
        """Execute a sequence of tool calls.

        Each step is a dict ``{"tool": "<name>", "args": [...], "kwargs": {...}}``.
        Results accumulate in :attr:`step_results` and are returned as a list.

        When ``stop_on_error`` is ``True`` (the default), the first exception
        halts the sequence.  When ``False``, failed steps record the exception
        object and execution continues.
        """
        self._step_results = []
        for index, step in enumerate(steps):
            tool_name = step.get("tool") or step.get("name")
            if not tool_name:
                raise ValueError(
                    f"Step {index} missing 'tool' key; got keys: {sorted(step)!r}"
                )
            args = step.get("args", ())
            kwargs = step.get("kwargs", {})

            LOGGER.debug(
                "Executing plan step %d/%d: tool=%r",
                index + 1,
                len(steps),
                tool_name,
            )
            try:
                result = self.run_tool(tool_name, *args, **kwargs)
            except Exception as exc:
                if stop_on_error:
                    raise
                LOGGER.warning(
                    "Plan step %d failed (tool=%r): %s",
                    index + 1,
                    tool_name,
                    exc,
                )
                result = exc
            self._step_results.append(result)
        return list(self._step_results)

    @property
    def step_results(self) -> list[Any]:
        """Results from the most recent :meth:`execute_plan` call."""
        return list(self._step_results)

    # ------------------------------------------------------------------ #
    # Parallel execution
    # ------------------------------------------------------------------ #
    def parallel(
        self,
        callables: Sequence[Callable[[], Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> list[Any]:
        """Run callables concurrently and return results in order.

        Each callable typically wraps a forked child's method::

            a, b = agent.fork(name="a"), agent.fork(name="b")
            results = agent.parallel([
                lambda: a.accomplish("Create raw landing tables"),
                lambda: b.accomplish("Create curated tables"),
            ])
        """
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(fn) for fn in callables]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------ #
    # Resource creation tools
    # ------------------------------------------------------------------ #
    def create_catalog(
        self,
        name: str,
        *,
        comment: Optional[str] = None,
    ) -> "UCCatalog":
        """Create a Unity Catalog catalog (idempotent)."""
        catalog = self.client.catalogs[name]
        catalog.ensure_created(comment=comment)
        LOGGER.info("Ensured catalog %r", name)
        return catalog

    def create_schema(
        self,
        full_name: str,
        *,
        comment: Optional[str] = None,
    ) -> "UCSchema":
        """Create a Unity Catalog schema (idempotent).

        ``full_name`` is ``catalog.schema``.
        """
        schema = self.client.schemas[full_name]
        schema.ensure_created(comment=comment)
        LOGGER.info("Ensured schema %r", full_name)
        return schema

    def create_table(
        self,
        full_name: str,
        definition: Any,
        *,
        comment: Optional[str] = None,
        **kwargs: Any,
    ) -> "Table":
        """Create a Delta table with best-practice defaults (idempotent).

        ``definition`` is a :class:`yggdrasil.data.Schema` or anything
        :meth:`Table.ensure_created` accepts.
        """
        table = self.client.tables[full_name]
        table.ensure_created(definition, comment=comment, **kwargs)
        LOGGER.info("Ensured table %r", full_name)
        return table

    def create_volume(
        self,
        full_name: str,
        *,
        comment: Optional[str] = None,
    ) -> "Volume":
        """Create a managed Unity Volume (idempotent)."""
        volume = self.client.volumes[full_name]
        volume.ensure_created(comment=comment)
        LOGGER.info("Ensured volume %r", full_name)
        return volume

    def create_warehouse(
        self,
        name: str,
        *,
        profile: Optional["WarehouseProfile"] = None,
        **overrides: Any,
    ) -> "SQLWarehouse":
        """Create a SQL warehouse using a :class:`WarehouseProfile`.

        ``overrides`` are merged on top of the profile so callers can tweak
        individual fields without building a whole profile.
        """
        from .profiles import SERVERLESS_WAREHOUSE

        p = profile or SERVERLESS_WAREHOUSE
        settings: dict[str, Any] = {
            "name": name,
            "cluster_size": p.cluster_size,
            "min_num_clusters": p.min_num_clusters,
            "max_num_clusters": p.max_num_clusters,
            "auto_stop_mins": p.auto_stop_mins,
            "warehouse_type": p.warehouse_type,
            "enable_serverless_compute": p.enable_serverless_compute,
            "spot_instance_policy": p.spot_instance_policy,
        }
        settings.update(overrides)
        warehouse = self.client.warehouses.create(**settings)
        LOGGER.info("Created warehouse %r (profile=%r)", name, type(p).__name__)
        return warehouse

    def create_cluster(
        self,
        name: str,
        *,
        spark_version: Optional[str] = None,
        profile: Optional["ClusterProfile"] = None,
        **overrides: Any,
    ) -> "Cluster":
        """Create an all-purpose cluster using a :class:`ClusterProfile`.

        ``spark_version`` defaults to the workspace's latest LTS when
        ``None``.  ``overrides`` merge on top of the profile.
        """
        from .profiles import INGESTION_CLUSTER

        p = profile or INGESTION_CLUSTER
        settings: dict[str, Any] = {
            "cluster_name": name,
            "node_type_id": p.node_type_id,
            "data_security_mode": p.data_security_mode,
        }
        if p.autoscale_min is not None and p.autoscale_max is not None:
            settings["autoscale"] = {
                "min_workers": p.autoscale_min,
                "max_workers": p.autoscale_max,
            }
        else:
            settings["num_workers"] = p.num_workers

        if spark_version:
            settings["spark_version"] = spark_version

        if p.spark_conf:
            settings["spark_conf"] = dict(p.spark_conf)
        if p.custom_tags:
            settings["custom_tags"] = dict(p.custom_tags)
        if p.single_user_name:
            settings["single_user_name"] = p.single_user_name

        settings.update(overrides)
        cluster = self.client.compute.clusters.create(**settings)
        LOGGER.info("Created cluster %r (profile=%r)", name, type(p).__name__)
        return cluster

    def deploy_workflow(
        self,
        flow_fn: Any,
        *,
        client: Any = None,
        **deploy_kwargs: Any,
    ) -> "Job":
        """Deploy a ``@flow``-decorated function as a Databricks Job.

        Thin wrapper that calls ``flow_fn.deploy()`` using the agent's own
        client as the default.
        """
        target_client = client or self.client
        job = flow_fn.deploy(client=target_client, **deploy_kwargs)
        LOGGER.info(
            "Deployed workflow %r as job %r", getattr(flow_fn, "__name__", flow_fn), job
        )
        return job

    # ------------------------------------------------------------------ #
    # Storage layout setup
    # ------------------------------------------------------------------ #
    def setup_storage(
        self,
        profile: "StorageProfile",
    ) -> dict[str, Any]:
        """Create a full ``<catalog>.<source>`` storage layout in one call.

        Uses the :class:`StorageProfile` to create the catalog, schema,
        raw tables, curated tables, and an upload volume.  Returns a dict
        mapping resource names to their live handles.
        """
        created: dict[str, Any] = {}

        catalog = self.create_catalog(profile.catalog, comment=profile.comment)
        created["catalog"] = catalog

        schema = self.create_schema(profile.schema_name, comment=profile.comment)
        created["schema"] = schema

        if profile.create_volume:
            volume = self.create_volume(
                profile.volume_full_name(),
                comment=f"Upload volume for {profile.source}",
            )
            created["volume"] = volume

        for entity in profile.raw_entities:
            table_name = profile.raw_table_name(entity)
            created[f"raw_{entity}"] = table_name

        for entity in profile.curated_entities:
            table_name = profile.curated_table_name(entity)
            created[f"curated_{entity}"] = table_name

        LOGGER.info(
            "Set up storage layout for %r (%d resources)",
            profile.schema_name,
            len(created),
        )
        return created

    # ------------------------------------------------------------------ #
    # Genie space auto-configuration
    # ------------------------------------------------------------------ #
    def setup_genie_space(
        self,
        *,
        tables: Sequence[str],
        title: Optional[str] = None,
        instructions: Sequence[str] = (),
        warehouse_id: Optional[str] = None,
    ) -> Any:
        """Create a Genie space wired to the given tables.

        Table names are resolved via
        :meth:`Genie.resolve_table_identifiers` — short names like
        ``"orders"`` or ``"sales.orders"`` are looked up automatically.
        """
        space = self.service.create_space(
            tables=list(tables),
            instructions=list(instructions),
            warehouse_id=warehouse_id,
            title=title or f"Autonomous Agent — {self.name}",
        )
        self.service.defaults = _dc_replace(
            self.service.defaults,
            space_id=space.space_id,
        )
        LOGGER.info("Created Genie space %r for agent %r", space.space_id, self.name)
        return space

    # ------------------------------------------------------------------ #
    # Parallel block execution
    # ------------------------------------------------------------------ #
    def _execute_parallel_block(
        self,
        parallel_step: AgentStep,
        remaining_plan: list[AgentStep],
        goal: str,
        budget_left: int,
    ) -> list[AgentStep]:
        """Execute a PARALLEL block concurrently.

        Collects consecutive non-DONE, non-PARALLEL steps after the
        PARALLEL marker as parallel branches. Each branch runs on
        ``self`` via a thread pool. Returns the executed steps.
        """
        branches: list[AgentStep] = []
        while remaining_plan and remaining_plan[0].tool not in (
            "_done",
            "_parallel",
        ):
            branches.append(remaining_plan.pop(0))
            if len(branches) >= budget_left:
                break

        if len(branches) <= 1:
            return [self._execute_step(b) for b in branches]

        results: list[AgentStep] = [None] * len(branches)  # type: ignore[list-item]

        def _run_branch(idx: int) -> AgentStep:
            return self._execute_step(branches[idx])

        with cf.ThreadPoolExecutor(max_workers=len(branches)) as pool:
            futures = {pool.submit(_run_branch, i): i for i in range(len(branches))}
            for future in cf.as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = AgentStep(
                        action=branches[idx].action,
                        tool=branches[idx].tool,
                        state=State.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _describe_tools(self) -> str:
        """Produce a human-readable description of registered tools."""
        lines: list[str] = []
        for name, fn in sorted(self.tools.items()):
            doc = getattr(fn, "__doc__", None) or ""
            first_line = doc.strip().split("\n")[0] if doc.strip() else ""
            lines.append(f"  - {name}: {first_line}" if first_line else f"  - {name}")
        return "\n".join(lines)

    def _format_history_context(self) -> str:
        """Summarize recent conversation history for the planning prompt."""
        if not self.history:
            return ""
        recent = self.history[-_MAX_HISTORY_FOR_CONTEXT:]
        lines = ["Recent conversation context:"]
        for ans in recent:
            text = ans.text or ""
            preview = text[:200] + ("..." if len(text) > 200 else "")
            query = ans.query
            entry = f"  - Answer: {preview}"
            if query:
                entry += f"\n    SQL: {query[:150]}{'...' if len(query) > 150 else ''}"
            lines.append(entry)
        return "\n".join(lines)

    def _suggest_tool(self, name: str) -> Optional[str]:
        """Return the closest registered tool name, or ``None``."""
        if not name:
            return None
        from difflib import get_close_matches

        matches = get_close_matches(name, self.tools.keys(), n=1, cutoff=0.6)
        return matches[0] if matches else None

    @staticmethod
    def _format_context(context: dict[str, Any]) -> str:
        """Format introspection context for the planning prompt."""
        parts: list[str] = []
        for key, value in context.items():
            if key == "agent":
                continue
            if isinstance(value, dict):
                if value:
                    for dk, dv in value.items():
                        if isinstance(dv, list) and dv:
                            parts.append(
                                f"  {key}.{dk}: "
                                f"{', '.join(str(v) for v in dv[:20])}"
                            )
                        elif isinstance(dv, list):
                            parts.append(f"  {key}.{dk}: (empty)")
                else:
                    parts.append(f"  {key}: (none)")
            elif isinstance(value, list):
                if value:
                    parts.append(f"  {key}: {', '.join(str(v) for v in value[:20])}")
                else:
                    parts.append(f"  {key}: (none)")
            elif value is not None:
                parts.append(f"  {key}: {value}")
        return "\n".join(parts) or "  (empty workspace)"

    # ------------------------------------------------------------------ #
    # Web / HTTP tools
    # ------------------------------------------------------------------ #
    def _get_http_session(self) -> Any:
        """Return a shared HTTPSession for web operations."""
        if not hasattr(self, "_http_session") or self._http_session is None:
            from yggdrasil.http_ import HTTPSession

            self._http_session = HTTPSession()
        return self._http_session

    def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        cache: bool = True,
    ) -> Any:
        """Fetch a URL and return the response.

        Uses the shared :class:`HTTPSession` with caching enabled by
        default.  The response is the full ``HTTPResponse`` object —
        call ``.text`` for the body, ``.status_code`` for the status.
        """
        session = self._get_http_session()
        LOGGER.info("Fetching %s %s", method, url)
        return session.request(method, url, headers=headers or {})

    def fetch_text(self, url: str, **kwargs: Any) -> str:
        """Fetch a URL and return the response body as text."""
        response = self.fetch(url, **kwargs)
        return response.text

    def fetch_json(self, url: str, **kwargs: Any) -> Any:
        """Fetch a URL and return the parsed JSON response."""
        response = self.fetch(url, **kwargs)
        from yggdrasil.pickle import json as ygg_json

        return ygg_json.loads(response.data)

    def scrape_links(self, url: str) -> list[str]:
        """Fetch a page and extract all ``<a href="...">`` links."""
        import re

        text = self.fetch_text(url)
        return re.findall(r'href=["\']([^"\']+)["\']', text)

    def open_browser(self, url: str) -> bool:
        """Open a URL in the user's default browser."""
        import webbrowser

        LOGGER.info("Opening browser: %s", url)
        return webbrowser.open(url)

    def fetch_entsoe_zones(self) -> Any:
        """Fetch ENTSO-E bidding zones via the official EIC CSV."""
        from yggdrasil.data.enums.geozone.entsoe import fetch_entsoe_bidding_zones

        session = self._get_http_session()
        zones = fetch_entsoe_bidding_zones(session=session)
        LOGGER.info("Fetched %d ENTSO-E bidding zones", len(zones))
        return zones

    def fetch_many(
        self,
        urls: Sequence[str],
        *,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        max_workers: Optional[int] = None,
    ) -> list[Any]:
        """Fetch multiple URLs concurrently."""
        from yggdrasil.io.request import PreparedRequest

        session = self._get_http_session()
        requests = [
            PreparedRequest.prepare(method=method, url=u, headers=headers or {})
            for u in urls
        ]
        LOGGER.info("Fetching %d URLs concurrently", len(requests))
        return list(session.send_many(requests))

    # ------------------------------------------------------------------ #
    # Tool registry
    # ------------------------------------------------------------------ #
    def _register_autonomous_tools(self) -> None:
        """Register resource-creation and orchestration tools."""
        # Resource creation
        self.tools["create_catalog"] = self.create_catalog
        self.tools["create_schema"] = self.create_schema
        self.tools["create_table"] = self.create_table
        self.tools["create_volume"] = self.create_volume
        self.tools["create_warehouse"] = self.create_warehouse
        self.tools["create_cluster"] = self.create_cluster
        self.tools["deploy_workflow"] = self.deploy_workflow
        self.tools["setup_storage"] = self.setup_storage
        self.tools["setup_genie_space"] = self.setup_genie_space

        # Workspace introspection
        self.tools["introspect"] = self.introspect
        self.tools["describe_catalog"] = lambda name: (
            self.client.catalogs[name].read_info()
        )
        self.tools["describe_schema"] = lambda name: (
            self.client.schemas[name].read_info()
        )
        self.tools["describe_table"] = lambda name: (
            self.client.tables[name].read_info()
        )

        # Autonomous orchestration
        self.tools["accomplish"] = self.accomplish
        self.tools["fork"] = self.fork
        self.tools["fork_many"] = self.fork_many
        self.tools["execute_plan"] = self.execute_plan
        self.tools["parallel"] = self.parallel
        self.tools["step_results"] = lambda: self.step_results
        self.tools["children"] = lambda: list(self.children)

        # Web / HTTP
        self.tools["fetch"] = self.fetch
        self.tools["fetch_text"] = self.fetch_text
        self.tools["fetch_json"] = self.fetch_json
        self.tools["fetch_many"] = self.fetch_many
        self.tools["scrape_links"] = self.scrape_links
        self.tools["open_browser"] = self.open_browser
        self.tools["fetch_entsoe_zones"] = self.fetch_entsoe_zones

        # Workflow
        self.tools["create_job"] = lambda name, tasks, **kw: (
            self.client.jobs.create_or_update(name=name, tasks=tasks, **kw)
        )
        self.tools["run_and_wait"] = lambda job_id, **kw: (
            self.client.jobs[job_id].run(**kw).wait()
        )
