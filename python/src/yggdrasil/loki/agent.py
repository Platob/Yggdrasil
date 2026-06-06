"""Loki — the global yggdrasil agent.

Loki is one agent that adapts to wherever it runs. It detects the backends
it can reach (:mod:`yggdrasil.loki.capability`), acts as a **token /
credential provider** for them (chiefly Databricks — when a session is
present Loki hands its authenticated client to whatever it drives), and
dispatches :class:`~yggdrasil.loki.behavior.LokiBehavior` actions. The CLI
(`ygg loki`) is a thin shell over this object.

    from yggdrasil.loki import Loki

    loki = Loki.current()
    loki.card()                      # who am I + what can I reach
    loki.databricks                  # the live DatabricksClient, or None
    loki.run("genie", space="01ef…", question="revenue by region")
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from . import behavior as _behavior
from .capability import Backend, detect

if TYPE_CHECKING:
    from yggdrasil.databricks import DatabricksClient

    from .engine import TokenEngine
    from .tools import Toolbox

__all__ = ["Loki", "ROUTES"]

#: Problem-category signals for :meth:`Loki.route`. Data over code — extend the
#: keyword lists, not the routing branches.
ROUTES: dict[str, tuple[str, ...]] = {
    "web": (
        "http://", "https://", "www.", "fetch ", "download ", "browse",
        "look up", "on the internet", "search the web", "scrape", "this url",
    ),
    "aws": (
        "aws", " s3", "ec2", "lambda", "dynamodb", "cloudwatch", "sqs ",
        "sns ", " ecr", " ecs", "step function", "secrets manager", "boto3",
        "iam role", "rds ", "glue ",
    ),
    "databricks": (
        "databricks", "genie", "unity catalog", "warehouse", "cluster",
        "dbfs", "delta", "spark", "serving endpoint", "notebook", "job run",
        "dbu", "catalog.", "uc ",
    ),
    "sql": (
        "select ", "insert ", "update ", " join ", "query", "sql", "schema",
        "group by", "where ",
    ),
    "files": (
        "fix ", "refactor", "edit ", "create file", "write a", "implement",
        "modify", "rename", "add a ", "scaffold", "build a", "generate code",
        ".py", ".ts", ".md", "patch", "the codebase",
    ),
}

#: Signals that a request is about *data* (worth fetching as a tabular frame).
DATA_SIGNALS: tuple[str, ...] = (
    "dataset", "data set", " csv", "parquet", "table of", "rows", "columns",
    "records", "metrics", "statistics", "prices", "price of", "rate", "rates",
    "exchange", "stock", "ticker", "quotes", "ohlc", "candles", "fx ",
)
#: Signals that a request is a *time series* (history over a period).
TIMESERIES_SIGNALS: tuple[str, ...] = (
    "time series", "timeseries", "over the last", "over the past", "since ",
    "history", "historical", "trend", "daily", "weekly", "monthly", "yearly",
    "change over", "evolution", "past two weeks", "last two weeks",
)


class Loki:
    """The global yggdrasil agent — capability-aware, token-providing."""

    name = "loki"

    #: Order in which Loki picks a reasoning engine when none is named.
    ENGINE_PREFERENCE: tuple[str, ...] = ("claude", "openai", "databricks")

    _CURRENT: "Optional[Loki]" = None

    def __init__(self) -> None:
        import getpass
        import socket

        self.user = _safe(getpass.getuser)
        self.host = _safe(socket.gethostname)
        self._backends: "Optional[list[Backend]]" = None

    # -- singleton ---------------------------------------------------------

    @classmethod
    def current(cls) -> "Loki":
        """The process-global Loki (created on first use)."""
        if cls._CURRENT is None:
            cls._CURRENT = cls()
        return cls._CURRENT

    # -- identity ----------------------------------------------------------

    @property
    def agent_id(self) -> int:
        """Stable int64 id derived from ``user@host`` (xxhash, not crypto)."""
        import xxhash

        return xxhash.xxh64_intdigest(f"{self.user}@{self.host}") & 0x7FFFFFFFFFFFFFFF

    # -- backends / capabilities ------------------------------------------

    def backends(self, *, refresh: bool = False) -> list[Backend]:
        """Detected backends (cached; pass ``refresh=True`` to re-sniff)."""
        if self._backends is None or refresh:
            self._backends = detect()
        return self._backends

    def backend(self, name: str, *, refresh: bool = False) -> "Optional[Backend]":
        for b in self.backends(refresh=refresh):
            if b.name == name:
                return b
        return None

    def has(self, name: str) -> bool:
        b = self.backend(name)
        return bool(b and b.available)

    # -- Databricks token provider ----------------------------------------

    @property
    def databricks(self) -> "Optional[DatabricksClient]":
        """The authenticated Databricks client when a session is present.

        This is Loki acting as a token provider: behaviors and downstream
        code take this client to reach Databricks service endpoints (SQL,
        Genie, jobs, serving, …) under the agent's resolved credentials.
        Returns ``None`` when no Databricks session is detected.
        """
        if not self.has("databricks"):
            return None
        from yggdrasil.databricks import DatabricksClient

        return DatabricksClient.current()

    @property
    def aws(self) -> Any:
        """The configured :class:`~yggdrasil.aws.AWSClient` when AWS is reachable.

        Loki as an AWS token provider — the AWS behavior fleet rides this
        client (its resolved credentials / region / role). ``None`` when no
        AWS session is detected.
        """
        if not self.has("aws"):
            return None
        from yggdrasil.aws import AWSClient

        return AWSClient.current()

    def load_specialists(self) -> list[str]:
        """Import the specialized behavior fleets for every reachable backend.

        Databricks problems get the ``databricks-*`` skills, AWS problems the
        ``aws-*`` skills — registered only when their backend is detected, so
        ``ygg loki behaviors`` shows the fleet that actually applies here.
        Returns the backends whose fleet loaded.
        """
        loaded: list[str] = []
        for name, module in (("databricks", "yggdrasil.databricks.loki"),
                             ("aws", "yggdrasil.aws.loki")):
            if self.has(name):
                try:
                    __import__(module)
                    loaded.append(name)
                except Exception:
                    pass
        return loaded

    def token_info(self) -> dict[str, Any]:
        """Non-secret summary of the Databricks credentials Loki provides."""
        b = self.backend("databricks")
        if not b or not b.available:
            return {"backend": "databricks", "available": False}
        return {
            "backend": "databricks",
            "available": True,
            "host": b.detail.get("host"),
            "auth_type": b.detail.get("auth_type"),
            "catalog": b.detail.get("catalog"),
            "schema": b.detail.get("schema"),
        }

    def whoami(self, *, probe: bool = False) -> "Optional[str]":
        """The Databricks user, if reachable. ``probe`` allows one network call."""
        client = self.databricks
        if client is None or not probe:
            return None
        try:
            return client.workspace_client().current_user.me().user_name
        except Exception:
            return None

    # -- reasoning engines -------------------------------------------------

    def _engine_instances(self) -> "dict[str, TokenEngine]":
        """One instance of every known engine (Databricks bound to our client)."""
        from .engines import ClaudeEngine, DatabricksServingEngine, OpenAIEngine

        return {
            "claude": ClaudeEngine(),
            "openai": OpenAIEngine(),
            "databricks": DatabricksServingEngine(client=self.databricks),
        }

    def engines(self) -> "list[TokenEngine]":
        """Every known reasoning engine (call ``.available()`` to filter)."""
        return list(self._engine_instances().values())

    def engine(self, name: "Optional[str]" = None) -> "Optional[TokenEngine]":
        """Resolve a reasoning engine by name, or the best available one."""
        insts = self._engine_instances()
        if name is not None:
            if name not in insts:
                raise KeyError(f"unknown engine {name!r}; known: {', '.join(insts)}")
            return insts[name]
        for n in self.ENGINE_PREFERENCE:
            eng = insts.get(n)
            if eng is not None and eng.available():
                return eng
        return None

    def reason(
        self,
        prompt: str,
        *,
        system: "Optional[str]" = None,
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        **options: Any,
    ) -> str:
        """Reason about *prompt* with the best (or named) engine → reply text.

        ``tier`` (``"fast"`` / ``"deep"``) forces the model tier; the default
        (``None``) lets the engine pick adaptively from the prompt.
        """
        eng = self.engine(engine)
        if eng is None or not eng.available():
            raise RuntimeError(
                "no reasoning engine available — log into Claude Code, or set "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a "
                "Databricks session"
            )
        return eng.generate(prompt, system=system, tier=tier, **options)

    def reason_stream(
        self,
        prompt: str,
        *,
        system: "Optional[str]" = None,
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        **options: Any,
    ) -> "Iterator[str]":
        """Stream a reply to *prompt* — yields text chunks as they arrive.

        Same engine/tier resolution as :meth:`reason`, but live: the chosen
        engine streams token deltas so the terminal prints them as they come.
        """
        eng = self.engine(engine)
        if eng is None or not eng.available():
            raise RuntimeError(
                "no reasoning engine available — log into Claude Code, or set "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a "
                "Databricks session"
            )
        yield from eng.generate_stream(prompt, system=system, tier=tier, **options)

    # -- autonomous action loop -------------------------------------------

    def act(
        self,
        task: str,
        *,
        root: str = ".",
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        max_steps: int = 12,
        read_only: bool = False,
        allow_shell: bool = False,
        allow_web: bool = False,
        confirm: "Optional[Callable[[str], bool]]" = None,
        toolbox: "Optional[Toolbox]" = None,
        on_step: "Optional[Callable[[dict[str, Any]], None]]" = None,
    ) -> dict[str, Any]:
        """Pursue *task* autonomously: discover, decide, and modify files.

        This is Loki acting on its own — the reason→act→observe loop. The
        agent's engine plans against a tool catalog (filesystem discovery +
        edits, optionally a shell), emits **one JSON tool call per turn**,
        and Loki runs it and feeds the observation back, until the engine
        declares it's ``done`` or *max_steps* is hit. The tools are confined
        to *root* (the working tree the agent was pointed at).

        ``tier`` pins the model tier for every turn; left ``None`` the engine
        adapts per turn — cheap early scouting turns, the capable model once
        the transcript (and the reasoning) grows.

        Returns a transcript: the resolved ``engine``, every ``step`` (its
        ``thought``/``tool``/``args``/``observation``), the final ``answer``,
        whether it ``completed``, and the ``files_changed`` list. Pass
        ``on_step`` to stream progress (the CLI uses it to print each turn).
        """
        from .tools import filesystem_toolbox

        eng = self.engine(engine)
        if eng is None or not eng.available():
            raise RuntimeError(
                "no reasoning engine available — log into Claude Code, or set "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a "
                "Databricks session"
            )
        box = toolbox or filesystem_toolbox(
            root, read_only=read_only, allow_shell=allow_shell, allow_web=allow_web,
            confirm=confirm,
        )
        system = (
            "You are Loki, an autonomous engineering agent working inside a "
            "file tree. Pursue the user's GOAL by taking one action at a time, "
            "inspecting before you modify.\n\n"
            f"Tools:\n{box.spec()}\n\n"
            "Reply with EXACTLY ONE JSON object per turn and nothing else — "
            "no prose, no markdown fences:\n"
            '  to use a tool:  {"thought": "...", "tool": "<name>", "args": {...}}\n'
            '  when finished:  {"thought": "...", "done": true, "answer": "<summary>"}\n\n'
            "Rules: take the smallest useful step; read a file before editing "
            "it; `edit_file` needs `old` to be unique, else write the whole "
            "file; stop as soon as the goal is met and summarize what changed."
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"GOAL: {task}"}]
        steps: list[dict[str, Any]] = []
        answer, completed = "", False

        for n in range(1, max_steps + 1):
            reply = eng.complete(messages, system=system, max_tokens=4000, tier=tier).text
            decision = _parse_decision(reply)
            if decision is None:
                # Nudge the model back onto the protocol rather than aborting.
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content":
                                 "That was not a single JSON object. Reply with "
                                 "one JSON tool call or a done object."})
                continue
            if decision.get("done"):
                answer = str(decision.get("answer", "")).strip()
                completed = True
                if on_step:
                    on_step({"n": n, "thought": decision.get("thought", ""), "done": True,
                             "answer": answer})
                break

            name = decision.get("tool", "")
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            observation = box.call(name, args)
            record = {"n": n, "thought": decision.get("thought", ""),
                      "tool": name, "args": args, "observation": observation}
            steps.append(record)
            if on_step:
                on_step(record)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             f"Observation from {name}:\n{observation}"})
        else:
            answer = "stopped: reached max_steps without finishing"

        return {
            "task": task,
            "engine": eng.name,
            "root": str(root),
            "steps": steps,
            "answer": answer,
            "completed": completed,
            "files_changed": list(box.changed),
        }

    # -- reasoning router --------------------------------------------------

    def route(self, text: str) -> dict[str, Any]:
        """Categorize a request and decide how to solve it.

        Loki reasoning, optimized: rather than throwing every prompt at one
        engine, it first *categorizes the problem* (data-driven, see
        :data:`ROUTES`) and picks a *solution path* — answer (``reason``),
        act on files (``act``), or ask Genie (``genie``) — and whether to
        **isolate** the work to a specialized agent. Databricks problems are
        handed to :class:`~yggdrasil.databricks.loki.DatabricksLoki` (the
        "databricks on databricks" scheme: a workspace-resident brain answers
        workspace questions) when one is reachable.

        Returns ``{"category", "action", "specialist", "why"}``.
        """
        low = text.lower()
        url_match = re.search(r"https?://\S+", text)
        if url_match or any(s in low for s in ROUTES["web"]):
            url = url_match.group(0).rstrip(").,") if url_match else None
            info = self.classify_data(text)
            # A data/timeseries request with a source → fetch it as a cached
            # tabular frame (the data path), not a plain page fetch.
            if url and info["data"]:
                return {"category": "data", "action": "tabular", "specialist": None,
                        "url": url, "data": True, "timeseries": info["timeseries"],
                        "why": "data/timeseries source → fetch as a cached tabular frame"}
            return {"category": "web", "action": "web", "specialist": None, "url": url,
                    "why": "a URL / web-fetch request — uses the HTTP session + io handlers"}
        if any(s in low for s in ROUTES["databricks"]):
            action = "genie" if "genie" in low else "reason"
            return {"category": "databricks", "action": action,
                    "specialist": "databricks",
                    "why": "matched a Databricks/Unity/warehouse signal"}
        if any(s in low for s in ROUTES["aws"]):
            return {"category": "aws", "action": "reason", "specialist": None,
                    "why": "matched an AWS service signal — see the aws-* behaviors"}
        if self.has("databricks") and any(s in low for s in ROUTES["sql"]):
            return {"category": "databricks", "action": "reason",
                    "specialist": "databricks", "why": "SQL with a live workspace"}
        if any(s in low for s in ROUTES["files"]):
            return {"category": "files", "action": "act", "specialist": None,
                    "why": "matched a file/code-change signal"}
        return {"category": "chat", "action": "reason", "specialist": None,
                "why": "no specialized signal — plain reasoning"}

    def classify_data(self, text: str) -> dict[str, Any]:
        """Global context: is this request data- or time-series-shaped?

        Drives the *data path* — a positive classification routes a sourced
        request to tabular fetching + caching (:class:`TabularBehavior`) instead
        of a plain page fetch. Returns ``{"data", "timeseries", "why"}``.
        """
        low = text.lower()
        ts = (any(s in low for s in TIMESERIES_SIGNALS)
              or bool(re.search(r"\b(last|past|since|over)\b.{0,20}\b(day|week|month|year)s?\b", low)))
        data = ts or any(s in low for s in DATA_SIGNALS)
        return {
            "data": data,
            "timeseries": ts,
            "why": "time-series signal" if ts else ("data/tabular signal" if data else "no data signal"),
        }

    def specialist(self, name: str) -> "Optional[Loki]":
        """A specialized agent to isolate a category of work, or ``None``.

        ``"databricks"`` resolves the workspace-bound
        :class:`~yggdrasil.databricks.loki.DatabricksLoki` when the SDK and a
        session are present; otherwise falls back to ``self``.
        """
        if name == "databricks":
            try:
                from yggdrasil.databricks.loki import DatabricksLoki

                agent = DatabricksLoki.current()
            except Exception:
                return None
            return agent if agent.has("databricks") else None
        return None

    # -- behaviors ---------------------------------------------------------

    def behaviors(self) -> list["_behavior.LokiBehavior"]:
        return _behavior.registry()

    def behavior(self, name: str) -> "Optional[_behavior.LokiBehavior]":
        return _behavior.get(name)

    def run(self, name: str, **kwargs: Any) -> Any:
        """Dispatch behavior *name* with *kwargs*, using self as the provider."""
        b = self.behavior(name)
        if b is None:
            known = ", ".join(x.name for x in self.behaviors()) or "(none)"
            raise KeyError(f"unknown behavior {name!r}; registered: {known}")
        if not b.available(self):
            raise RuntimeError(
                f"behavior {name!r} needs backend {b.requires!r}, "
                f"which is not available here"
            )
        return b.run(self, **kwargs)

    # -- self-description --------------------------------------------------

    def card(self, *, refresh: bool = False) -> dict[str, Any]:
        """Everything Loki knows about itself — identity, reach, behaviors."""
        return {
            "agent": self.name,
            "agent_id": self.agent_id,
            "user": self.user,
            "host": self.host,
            "backends": [b.to_dict() for b in self.backends(refresh=refresh)],
            "token": self.token_info(),
            "engines": [
                {"name": e.name, "model": e.model_label, "available": e.available()}
                for e in self.engines()
            ],
            "behaviors": [b.to_dict() for b in self.behaviors()],
        }

    def __repr__(self) -> str:
        reach = ",".join(b.name for b in self.backends() if b.available)
        return f"Loki(user={self.user!r}, host={self.host!r}, reach=[{reach}])"


def _parse_decision(reply: str) -> "Optional[dict[str, Any]]":
    """Extract the agent's single JSON decision from an engine reply.

    Tolerates the slop a model wraps JSON in — markdown fences, a stray
    sentence either side — by falling back to the outermost ``{...}`` span.
    Returns ``None`` when nothing parses (the loop then re-prompts).
    """
    text = reply.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _safe(fn) -> str:
    try:
        return fn()
    except Exception:
        return "unknown"
