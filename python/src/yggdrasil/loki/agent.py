"""Loki — the global yggdrasil agent.

Loki is one agent that adapts to wherever it runs. It detects the backends
it can reach (:mod:`yggdrasil.loki.capability`), acts as a **token /
credential provider** for them (chiefly Databricks — when a session is
present Loki hands its authenticated client to whatever it drives), and
dispatches :class:`~yggdrasil.loki.skill.LokiSkill` actions. The CLI
(`ygg loki`) is a thin shell over this object.

    from yggdrasil.loki import Loki

    loki = Loki.current()
    loki.card()                      # who am I + what can I reach
    loki.databricks                  # the live DatabricksClient, or None
    loki.run("genie", space="01ef…", question="revenue by region")
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

from . import skill as _skill
from .capability import Backend, detect
from .result import DictResult

#: Engine fallback / skip notices ride this logger; ``ygg loki`` routes it to
#: the terminal (``style.install_logging``) so a skipped engine is visible.
_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from yggdrasil.databricks import DatabricksClient

    from .engine import TokenEngine
    from .planning import AgentPlan
    from .tools import Toolbox

__all__ = ["Loki", "ROUTES", "ActStep", "ActResult"]


@dataclass(slots=True)
class ActStep(DictResult):
    """One turn of the autonomous loop — a tool call (or the final ``done``)."""

    n: int
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    done: bool = False
    answer: str = ""


@dataclass(slots=True)
class ActResult(DictResult):
    """The transcript of an :meth:`Loki.act` run (typed, mapping-compatible)."""

    task: str
    engine: str
    root: str
    steps: list[ActStep] = field(default_factory=list)
    answer: str = ""
    completed: bool = False
    files_changed: list[str] = field(default_factory=list)

#: Raised when no reasoning engine is reachable (or every one failed in turn).
_NO_ENGINE = (
    "no reasoning engine available — log into Claude Code, or set "
    "ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a Databricks session"
)

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

#: "Make me a new project" phrasing → the ``scaffold`` action (no slash command
#: needed — Loki routes it itself).
SCAFFOLD_SIGNALS: tuple[str, ...] = (
    "scaffold", "new project", "create a project", "create a repo", "new repo",
    "bootstrap a project", "from scratch", "starter project", "boilerplate",
    "project template", "set up a repo", "set up a project", "ready to push",
    "new app", "spin up a project",
    # full-app phrasings (pair with the scaffold presets)
    "full-stack", "fullstack", "full stack", "web app", "webapp",
    "realtime app", "real-time app", "streaming app", "dashboard app",
)

#: "Run these in parallel" phrasing → the ``delegate`` action (a monitored fleet
#: of background process agents).
DELEGATE_SIGNALS: tuple[str, ...] = (
    "in parallel", "delegate", "swarm", "spawn agents", "concurrently",
    "at the same time", "simultaneously", "fan out", "parallel agents",
    "multiple agents", "several agents", "background agents",
)

#: Power-market phrasing → the ``entsoe`` skill (autonomous energy-data path).
ENERGY_SIGNALS: tuple[str, ...] = (
    "entso", "electricity price", "power price", "power prices", "electricity",
    "day-ahead", "day ahead", "spot price", "power market", "energy market",
    "grid load", "power generation", "power demand", "megawatt", "mwh", "bidding zone",
)

#: Meta/advice phrasing that asks *how to build* something rather than to do it
#: now — routed to the ``guide`` skill when paired with a yggdrasil mention.
GUIDE_SIGNALS: tuple[str, ...] = (
    "best way", "how do i", "how should i", "how to", "idiomatic", "best practice",
    "most optimized", "most efficient", "recommended way", "which abstraction",
    "optimi",
)

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

    #: Order in which Loki picks a reasoning engine when none is named —
    #: capable remote APIs first, then free local engines as a fallback.
    #: :meth:`select` overrides this for simple work on a capable workstation.
    ENGINE_PREFERENCE: tuple[str, ...] = (
        "claude", "openai", "databricks", "ollama", "openvino", "transformers",
    )

    _CURRENT: "Optional[Loki]" = None

    def __init__(self) -> None:
        import getpass
        import socket

        self.user = _safe(getpass.getuser)
        self.host = _safe(socket.gethostname)
        self._backends: "Optional[list[Backend]]" = None
        self._engines: "Optional[dict[str, TokenEngine]]" = None
        self._capable: "Optional[bool]" = None
        self._agent_id: "Optional[int]" = None
        self._specialists: "dict[str, Optional[Loki]]" = {}
        #: ``(engine, brief-error)`` pairs already logged at WARNING this session
        #: — a wedged engine (e.g. a model that can't download) fails on *every*
        #: turn, so we warn once and stay quiet after, instead of re-dumping the
        #: same failure each time it's skipped in the fallback chain.
        self._warned_failures: "set[tuple[str, str]]" = set()

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
        """Stable int64 id derived from ``user@host`` (xxhash, not crypto); cached."""
        if self._agent_id is None:
            import xxhash

            self._agent_id = xxhash.xxh64_intdigest(f"{self.user}@{self.host}") & 0x7FFFFFFFFFFFFFFF
        return self._agent_id

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

        This is Loki acting as a token provider: skills and downstream
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

        Loki as an AWS token provider — the AWS skill fleet rides this
        client (its resolved credentials / region / role). ``None`` when no
        AWS session is detected.
        """
        if not self.has("aws"):
            return None
        from yggdrasil.aws import AWSClient

        return AWSClient.current()

    def load_specialists(self) -> list[str]:
        """Import the specialized skill fleets for every reachable backend.

        Databricks problems get the ``databricks-*`` skills, AWS problems the
        ``aws-*`` skills — registered only when their backend is detected, so
        ``ygg loki skills`` shows the fleet that actually applies here.
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

    def _engine_instances(self, *, refresh: bool = False) -> "dict[str, TokenEngine]":
        """One instance of every known engine — remote APIs plus the free
        local ones (Databricks bound to our client).

        Cached for the process: ``engine()``, ``engines()``, and ``select()``
        all resolve engines, and rebuilding them each time would re-run every
        ``available()`` check — including the Ollama liveness *network* probe —
        several times per ``ygg loki`` command. Reusing the instances keeps the
        startup path to a single probe (each engine memoizes its own check).
        ``refresh=True`` re-detects (e.g. after a Databricks session appears).
        """
        if self._engines is None or refresh:
            from .engines import (
                ClaudeEngine,
                DatabricksServingEngine,
                OllamaEngine,
                OpenAIEngine,
                OpenVINOEngine,
                TransformersEngine,
            )

            self._engines = {
                "claude": ClaudeEngine(),
                "openai": OpenAIEngine(),
                # Lazy client + a cheap availability signal from the detected
                # backend — so listing/selecting engines never imports the SDK;
                # the heavy load is deferred to an actual serving completion.
                "databricks": DatabricksServingEngine(available=self.has("databricks")),
                "ollama": OllamaEngine(),
                # Intel NPU (AI Boost) via OpenVINO/optimum-intel, else GPU/CPU.
                "openvino": OpenVINOEngine(),
                "transformers": TransformersEngine(),
            }
        return self._engines

    def engines(self) -> "list[TokenEngine]":
        """Every known reasoning engine (call ``.available()`` to filter)."""
        return list(self._engine_instances().values())

    def available_engines(self, *, refresh: bool = False) -> "dict[str, TokenEngine]":
        """Reachable engines (name → instance), availability probed **in parallel**.

        Several engines gate on a *network* round-trip — the Ollama liveness
        probe, the Databricks backend check — so probing them one after another
        stacks up their latencies on the startup path. Fanning the
        ``available()`` checks across a small thread pool collapses that to the
        slowest single probe. Each engine memoizes its own result, so this also
        warms the caches that later serial ``available()`` calls (the status
        line, :meth:`engine`, :meth:`select`) reuse for free.
        """
        insts = self._engine_instances(refresh=refresh)
        if not insts:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(insts)) as pool:
            oks = list(pool.map(_is_available, insts.values()))
        return {n: e for (n, e), ok in zip(insts.items(), oks) if ok}

    def engine(self, name: "Optional[str]" = None) -> "Optional[TokenEngine]":
        """Resolve a reasoning engine by name, or the best available one."""
        insts = self._engine_instances()
        if name is not None:
            if name not in insts:
                raise KeyError(f"unknown engine {name!r}; known: {', '.join(insts)}")
            return insts[name]
        available = self.available_engines()
        for n in self.ENGINE_PREFERENCE:
            if n in available:
                return available[n]
        return None

    def select(
        self,
        text: "Optional[str]" = None,
        *,
        tier: "Optional[str]" = None,
        base: "Optional[str]" = None,
        confirm: "Optional[Callable[[TokenEngine, Optional[str]], bool]]" = None,
    ) -> "Optional[TokenEngine]":
        """Resource-aware engine choice: keep light work cheap, escalate heavy
        work to a capable remote — asking first.

        Two axes drive the pick:

        - **Complexity** — an explicit ``tier`` (``deep`` → heavy), else the
          prompt itself (long or reasoning-heavy text → heavy).
        - **Resources** — a CUDA GPU, or enough CPU + RAM (≥ 4 cores, ≥ 8 GB),
          decides whether this box can comfortably run a local model.

        A session pins a **base** provider, but complexity moves the choice
        **both ways**:

        - *remote → local* (demote): light work on a capable box drops to a
          free local model even when the base is a remote API — saving money,
          silently (no downside to ask about).
        - *local → remote* (escalate): heavy work climbs to the most capable
          remote (the base remote if it is one). When this means switching
          **from a free local model up to a paid remote** one,
          ``confirm(engine, model)`` is asked first; a falsy answer keeps the
          work on the cheap/local path.

        With no local engine the base simply stands; with no remote the local
        engine carries even heavy work. Returns an available engine, or
        ``None`` when nothing is reachable.
        """
        available = self.available_engines()
        if not available:
            return None

        if tier in ("fast", "deep"):
            heavy = tier == "deep"
        else:
            from .engine import ADAPTIVE_DEEP_CHARS, ADAPTIVE_DEEP_SIGNALS

            blob = (text or "").lower()
            heavy = (len(blob) >= ADAPTIVE_DEEP_CHARS
                     or any(s in blob for s in ADAPTIVE_DEEP_SIGNALS))

        order = [n for n in self.ENGINE_PREFERENCE if n in available]
        locals_ = [available[n] for n in order if available[n].local]
        remotes = [available[n] for n in order if not available[n].local]
        base_eng = available.get(base or "")

        # Can this workstation comfortably host a local model? Only worth asking
        # when a local engine is actually reachable — the probe imports torch
        # (slow first call), so skip it entirely on a remote-only box.
        capable = bool(locals_) and self.can_run_local()

        # The cheap/home choice for ordinary work: a capable local model if we
        # have one (free, private), else the session base, else the best engine.
        home = (locals_[0] if (locals_ and capable)
                else base_eng or (remotes[0] if remotes else locals_[0]))
        if not heavy:
            return home

        # Heavy work wants the most capable remote — the base if it's remote,
        # otherwise the preferred remote. No remote reachable → stay on home.
        target = (base_eng if (base_eng and not base_eng.local)
                  else (remotes[0] if remotes else home))
        # Escalating from a free local home up to a paid remote → confirm first.
        if confirm is not None and target is not home and not target.local and home.local:
            model = target.resolve_model(
                messages=[{"role": "user", "content": text or ""}], tier="deep"
            )
            if not confirm(target, model):
                return home
        return target

    def can_run_local(self) -> bool:
        """Whether this box can comfortably host a local model — cached.

        Wraps :func:`yggdrasil.loki.resources.can_run_local` but memoizes the
        result for the process: the probe imports ``torch`` to check for a CUDA
        GPU (slow on the first call, on a box that has it), and engine selection
        asks this on every turn. Hardware doesn't change within a session.
        """
        if self._capable is None:
            from .resources import can_run_local

            self._capable = can_run_local()
        return self._capable

    def _engine_chain(
        self,
        text: "Optional[str]" = None,
        *,
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        base: "Optional[str]" = None,
        confirm: "Optional[Callable[[TokenEngine, Optional[str]], bool]]" = None,
    ) -> "list[TokenEngine]":
        """Ordered, available engines to try for a request — the chosen one
        first, then the rest in preference order.

        Powers log-and-skip fallback: a pinned/selected engine that errors
        (bad token, 403, missing dep) is logged and the next available engine
        is tried instead of failing the whole request.
        """
        available = {n: e for n, e in self._engine_instances().items() if e.available()}
        chain: "list[TokenEngine]" = []
        first = (self.engine(engine) if engine
                 else self.select(text, tier=tier, base=base, confirm=confirm))
        if first is not None and first.available():
            chain.append(first)
        for n in self.ENGINE_PREFERENCE:
            eng = available.get(n)
            if eng is not None and all(eng is not c for c in chain):
                chain.append(eng)
        return chain

    def bootstrap_local(
        self,
        *,
        model: "Optional[str]" = None,
        pull: bool = True,
        on_progress: "Optional[Callable[[dict[str, Any]], None]]" = None,
    ) -> dict[str, Any]:
        """Ready a free **local** reasoning engine, lazily installing on demand.

        Prefers a reachable Ollama server — ensures the model **sized to this
        workstation** (the more RAM/GPU, the larger the default) is pulled, only
        if missing. Falls back to the HF ``transformers`` engine (weights
        lazy-download on first use). When neither is present, returns what to
        install. This is the "free local brain" entry point — sized to the box,
        smart enough for basic setup/config, and able to hand heavier work up to
        a remote model.
        """
        from .engines import OllamaEngine, TransformersEngine

        oll = OllamaEngine()
        if oll.available():
            target = model or oll.bootstrap_model
            receipt = oll.ensure(target, on_progress=on_progress) if pull else {
                "model": target, "was_present": oll.has_model(target),
                "status": "skipped (pull=False)",
            }
            return {"engine": "ollama", "ready": True, **receipt}
        tf = TransformersEngine()
        if tf.available():
            return {"engine": "transformers", "ready": True,
                    "model": model or tf.bootstrap_model, "was_present": False,
                    "status": "ready (weights lazy-download on first use)"}
        # A box with an Intel GPU should install the GPU (XPU) torch build, not
        # the stock CPU wheel — so the local model runs on the GPU from the start.
        from .resources import XPU_TORCH_INDEX, intel_gpu_present

        torch_hint = (f"pip install transformers && pip install --index-url {XPU_TORCH_INDEX} torch"
                      "  (HF local engine, on the Intel GPU)"
                      if intel_gpu_present() else
                      "or: pip install transformers torch  (HF local engine)")
        return {
            "engine": None, "ready": False,
            "install": [
                "install Ollama (https://ollama.com), then it auto-pulls "
                f"{OllamaEngine().bootstrap_model!r} (sized to this box)",
                torch_hint,
            ],
        }

    def reason(
        self,
        prompt: str,
        *,
        system: "Optional[str]" = None,
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        base: "Optional[str]" = None,
        confirm: "Optional[Callable[[TokenEngine, Optional[str]], bool]]" = None,
        **options: Any,
    ) -> str:
        """Reason about *prompt* with the best (or named) engine → reply text.

        ``tier`` (``"fast"`` / ``"deep"``) forces the model tier; the default
        (``None``) lets the engine pick adaptively from the prompt. A pinned
        ``engine`` is used as-is; otherwise the choice is resource-aware
        (:meth:`select`), sticking to the session ``base`` and asking
        ``confirm`` before escalating to a paid remote model.
        """
        chain = self._engine_chain(prompt, engine=engine, tier=tier, base=base, confirm=confirm)
        last: "Optional[Exception]" = None
        for eng in chain:
            try:
                return eng.generate(prompt, system=system, tier=tier, **options)
            except Exception as exc:
                last = exc
                self._warn_engine_failure(eng, exc)
        raise last if last is not None else RuntimeError(_NO_ENGINE)

    def reason_stream(
        self,
        prompt: str,
        *,
        system: "Optional[str]" = None,
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        base: "Optional[str]" = None,
        confirm: "Optional[Callable[[TokenEngine, Optional[str]], bool]]" = None,
        **options: Any,
    ) -> "Iterator[str]":
        """Stream a reply to *prompt* — yields text chunks as they arrive.

        Same engine/tier resolution as :meth:`reason`, but live: the chosen
        engine streams token deltas so the terminal prints them as they come.
        """
        chain = self._engine_chain(prompt, engine=engine, tier=tier, base=base, confirm=confirm)
        last: "Optional[Exception]" = None
        for eng in chain:
            started = False
            try:
                for chunk in eng.generate_stream(prompt, system=system, tier=tier, **options):
                    started = True
                    yield chunk
                return
            except Exception as exc:
                if started:
                    raise            # already emitted output — can't silently switch
                last = exc
                self._warn_engine_failure(eng, exc)
        raise last if last is not None else RuntimeError(_NO_ENGINE)

    def _warn_engine_failure(self, eng: "TokenEngine", exc: Exception) -> None:
        """Warn that *eng* was skipped — but only the **first** time this exact
        failure is seen. A wedged engine fails identically every turn; logging it
        once keeps the session readable instead of re-dumping the same error."""
        key = (eng.name, _short_err(exc))
        if key in self._warned_failures:
            _log.debug("engine '%s' failed again (suppressed): %s", *key)
            return
        self._warned_failures.add(key)
        _log.warning("engine '%s' failed: %s — skipping to next (further repeats "
                     "of this error are suppressed)", *key)

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
        on_think: "Optional[Callable[[int], None]]" = None,
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
        ``on_step`` to stream each completed turn, and ``on_think(n)`` to learn
        when turn *n* is about to call the (slow) model — the CLI uses the pair
        to keep a live spinner + step-budget bar running through the otherwise
        silent reasoning between tool calls.
        """
        from .tools import filesystem_toolbox

        eng = self.engine(engine) if engine else self.select(task, tier=tier)
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
            "it; `edit_file` needs `old` to be unique, else write the whole file.\n"
            "Validation discipline (checkpoints): after each meaningful change, run a "
            "quick `smoke` test to confirm it still works before moving on — never "
            "stack unverified changes. At a bigger milestone (a feature done, a module "
            "reshaped) run `bench` to validate performance, not just correctness. Do "
            "NOT declare done until a `smoke` test passes; in your final answer say "
            "what you validated. Stop as soon as the goal is met and verified, and "
            "summarize what changed and what you ran."
            + ("\nMesh: you are one of several agents sharing a workspace. FIRST "
               "`mesh` (action='list') to see peers' published results and avoid "
               "redundant work; when you produce something reusable (a file path, an "
               "API, a decision) publish it with `mesh` (action='put', key, value)."
               if "mesh" in box.tools else "")
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"GOAL: {task}"}]
        steps: list[ActStep] = []
        answer, completed = "", False

        from .usage import METER

        for n in range(1, max_steps + 1):
            # Cost cap (set via the meter, e.g. `ygg loki do --budget`) — stop
            # before the next paid turn rather than blowing through it.
            if METER.over_budget():
                answer = f"stopped: cost budget reached (${METER.total_cost:.4f})"
                break
            if on_think:
                on_think(n)
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
                    on_step(ActStep(n=n, thought=decision.get("thought", ""),
                                    done=True, answer=answer))
                break

            name = decision.get("tool", "")
            args = decision.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            observation = box.call(name, args)
            record = ActStep(n=n, thought=decision.get("thought", ""),
                             tool=name, args=args, observation=observation)
            steps.append(record)
            if on_step:
                on_step(record)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content":
                             f"Observation from {name}:\n{observation}"})
        else:
            answer = "stopped: reached max_steps without finishing"

        return ActResult(
            task=task,
            engine=eng.name,
            root=str(root),
            steps=steps,
            answer=answer,
            completed=completed,
            files_changed=list(box.changed),
        )

    def delegate(
        self,
        tasks: list[str],
        *,
        root: str = ".",
        engine: "Optional[str]" = None,
        tier: "Optional[str]" = None,
        max_steps: int = 8,
        allow_web: bool = True,
        allow_shell: bool = False,
        read_only: bool = False,
        timeout: "Optional[float]" = None,
        on_update: "Optional[Callable[[list[Any]], None]]" = None,
    ) -> list[dict[str, Any]]:
        """Fan *tasks* out to background process agents and wait for them.

        Each task runs as its own ``ygg loki do`` subprocess (an isolated act
        loop), so independent work proceeds in parallel while Loki monitors them
        — its autonomy multiplier. Returns one summary row per agent (status,
        elapsed, answer, files_changed). ``on_update(agents)`` streams live
        progress (the CLI renders the dashboard from it).
        """
        from .fleet import Fleet

        fleet = Fleet()
        fleet.spawn_all(tasks, root=root, engine=engine, tier=tier, max_steps=max_steps,
                        allow_web=allow_web, allow_shell=allow_shell, read_only=read_only)
        fleet.monitor(on_update, timeout=timeout)
        return fleet.summary()

    def decompose(self, goal: str, *, engine: "Optional[str]" = None,
                  tier: "Optional[str]" = None, max_tasks: int = 6) -> list[str]:
        """Break *goal* into independent subtasks an agent fleet can run in parallel.

        Asks the reasoning engine for a JSON array of self-contained tasks
        (parallel-safe — no ordering between them). Returns the parsed list,
        capped at *max_tasks*; falls back to ``[goal]`` if nothing parses.
        """
        prompt = (
            f"Break this goal into at most {max_tasks} INDEPENDENT subtasks that can "
            f"run in parallel (no subtask depends on another's output). Each must be "
            f"a single concrete instruction for an autonomous coding agent.\n\n"
            f"GOAL: {goal}\n\n"
            f"Reply with ONLY a JSON array of strings, nothing else."
        )
        reply = self.reason(prompt, engine=engine, tier=tier,
                            system="You decompose goals into parallel-safe subtasks. Output JSON only.")
        start, end = reply.find("["), reply.rfind("]")
        if start != -1 and end > start:
            try:
                tasks = json.loads(reply[start : end + 1])
            except json.JSONDecodeError:
                tasks = []
            cleaned = [str(t).strip() for t in tasks if str(t).strip()]
            if cleaned:
                return cleaned[:max_tasks]
        return [goal]

    # -- reasoning planner -------------------------------------------------

    def plan(self, text: str) -> "AgentPlan":
        """Classify a request into a structured :class:`AgentPlan`.

        Loki reasoning, optimized: rather than throwing every prompt at one
        engine, it *classifies the problem* — a **category** and solution
        **action** (answer / act on files / fetch tabular / ask Genie), the
        **persona** to embody (data engineer, analyst, software engineer,
        trader, confessor, companion, …), the **skills** likely required, and
        whether to **isolate** the work to a specialist (the "databricks on
        databricks" scheme). Returns an :class:`AgentPlan` (mapping-compatible).
        """
        from .planning import DEFAULT_PERSONA, AgentPlan, classify_persona, skills_for

        low = text.lower()
        signalled = classify_persona(text)
        dataf = self.classify_data(text)

        def made(category, action, *, specialist=None, url=None, why=""):
            # A signalled persona wins; otherwise fall back to the category's
            # default (data → analyst, databricks → engineer, …).
            persona = signalled if signalled != "assistant" else DEFAULT_PERSONA.get(category, "assistant")
            return AgentPlan(
                text=text, category=category, action=action, specialist=specialist,
                url=url, persona=persona, data=dataf["data"], timeseries=dataf["timeseries"],
                required_skills=skills_for(category, data=dataf["data"]), why=why)

        # "How do I build X the yggdrasil way?" → guidance, not execution.
        if ("yggdrasil" in low or "ygg " in low) and any(s in low for s in GUIDE_SIGNALS):
            return made("guide", "guide",
                        why="how-to: the optimized yggdrasil implementation path")

        # Autonomy: route "make a new project" / "do these in parallel" to the
        # scaffold / delegate actions directly — no slash command needed.
        if any(s in low for s in DELEGATE_SIGNALS):
            return made("files", "delegate",
                        why="delegate independent tasks to a monitored parallel agent fleet")
        # Scaffold: an explicit signal, or a "<create-verb> … <project-noun>" phrase
        # (so "create a new python project" routes even with words in between).
        if any(s in low for s in SCAFFOLD_SIGNALS) or (
            re.search(r"\b(new|create|start|bootstrap|scaffold|generate|spin up|set up|build|make)\b", low)
            and re.search(r"\b(project|repo|repository|app|package|library|service|microservice|cli|tool)\b", low)
        ):
            return made("files", "scaffold",
                        why="scaffold a ready-to-push project from scratch")
        # Power-market data → the entsoe skill, with series/zone inferred. Gate on
        # a data word (or an explicit ENTSO-E mention) so "what is electricity"
        # stays a plain question.
        if (not re.search(r"https?://", text) and any(s in low for s in ENERGY_SIGNALS)
                and ("entso" in low or any(w in low for w in (
                    "price", "load", "demand", "consumption", "generation",
                    "production", "spot", "market")))):
            from .entsoe import infer_query

            p = made("data", "skill", why="ENTSO-E power-market data → entsoe skill")
            p.skill = "entsoe"
            p.skill_kwargs = infer_query(text)
            return p

        # With a live Databricks session, a precise NL request ("list catalogs",
        # "tables in cat.sch", "describe cat.sch.tbl", "who am i") dispatches the
        # specialized databricks-* skill directly rather than just reasoning.
        if self.has("databricks") and not re.search(r"https?://", text):
            try:
                from yggdrasil.databricks.loki.router import route as _dbx_route

                hit = _dbx_route(text)
            except Exception:
                hit = None
            if hit is not None:
                skill, kw = hit
                p = made("databricks", "skill", specialist="databricks",
                         why=f"databricks request → {skill}")
                p.skill = skill
                p.skill_kwargs = {k: v for k, v in kw.items() if v is not None}
                return p

        # Same for AWS: a precise NL read request ("list my s3 buckets", "show
        # ec2 instances", "who am i on aws") dispatches the right aws-* skill.
        if self.has("aws") and not re.search(r"https?://", text):
            try:
                from yggdrasil.aws.loki.router import route as _aws_route

                hit = _aws_route(text)
            except Exception:
                hit = None
            if hit is not None:
                skill, kw = hit
                p = made("aws", "skill", why=f"aws request → {skill}")
                p.skill = skill
                p.skill_kwargs = {k: v for k, v in kw.items() if v is not None}
                return p

        url_match = re.search(r"https?://\S+", text)
        if url_match or any(s in low for s in ROUTES["web"]):
            url = url_match.group(0).rstrip(").,") if url_match else None
            if url and dataf["data"]:
                return made("data", "tabular", url=url,
                            why="data/timeseries source → fetch as a cached tabular frame")
            return made("web", "web", url=url,
                        why="a URL / web-fetch request — uses the HTTP session + io handlers")
        # A local (or s3/dbfs) tabular file → the same data path, read through
        # the io handlers (IO.from_) and cached, no fetch needed.
        file_match = re.search(
            r"(?:[\w./~-]*/)?[\w.-]+\.(?:csv|tsv|parquet|pq|arrow|feather|xlsx|xls)\b",
            text, re.I,
        )
        if file_match:
            return made("data", "tabular", url=file_match.group(0),
                        why="a local/columnar data file → io handlers → cached frame")
        if any(s in low for s in ROUTES["databricks"]):
            return made("databricks", "genie" if "genie" in low else "reason",
                        specialist="databricks",
                        why="matched a Databricks/Unity/warehouse signal")
        if any(s in low for s in ROUTES["aws"]):
            return made("aws", "reason", why="matched an AWS service signal — see the aws-* skills")
        if self.has("databricks") and any(s in low for s in ROUTES["sql"]):
            return made("databricks", "reason", specialist="databricks",
                        why="SQL with a live workspace")
        if any(s in low for s in ROUTES["files"]):
            return made("files", "act", why="matched a file/code-change signal")
        return made("chat", "reason", why=f"no specialized signal — plain reasoning ({signalled})")

    #: Back-compat alias — ``route`` returns the (mapping-compatible) plan.
    route = plan

    def classify_data(self, text: str) -> dict[str, Any]:
        """Global context: is this request data- or time-series-shaped?

        Drives the *data path* — a positive classification routes a sourced
        request to tabular fetching + caching (:class:`TabularSkill`) instead
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

        Cached per name: the REPL asks for the same specialist on every
        databricks turn, and the resolution (import + singleton lookup +
        backend check) is stable for the process.
        """
        if name in self._specialists:
            return self._specialists[name]
        resolved: "Optional[Loki]" = None
        if name == "databricks":
            try:
                from yggdrasil.databricks.loki import DatabricksLoki

                agent = DatabricksLoki.current()
                resolved = agent if agent.has("databricks") else None
            except Exception:
                resolved = None
        self._specialists[name] = resolved
        return resolved

    # -- skills ------------------------------------------------------------

    def skills(self) -> list["_skill.LokiSkill"]:
        return _skill.registry()

    def skill(self, name: str) -> "Optional[_skill.LokiSkill]":
        return _skill.get(name)


    def run(self, skill_name: str, **kwargs: Any) -> Any:
        """Dispatch skill *skill_name* with *kwargs*, using self as the provider.

        The first parameter is ``skill_name`` (not ``name``) so a skill that
        itself takes a ``name=`` kwarg — e.g. ``scaffold(name=…)`` — can be
        dispatched as ``loki.run("scaffold", name="acme")`` without a clash.
        """
        s = self.skill(skill_name)
        if s is None:
            known = ", ".join(x.name for x in self.skills()) or "(none)"
            raise KeyError(f"unknown skill {skill_name!r}; registered: {known}")
        if not s.available(self):
            raise RuntimeError(
                f"skill {skill_name!r} needs backend {s.requires!r}, "
                f"which is not available here"
            )
        return s.run(self, **kwargs)

    # -- self-description --------------------------------------------------

    def card(self, *, refresh: bool = False) -> dict[str, Any]:
        """Everything Loki knows about itself — identity, reach, skills."""
        return {
            "agent": self.name,
            "agent_id": self.agent_id,
            "user": self.user,
            "host": self.host,
            "backends": [b.to_dict() for b in self.backends(refresh=refresh)],
            "token": self.token_info(),
            "engines": [
                {"name": e.name, "model": e.model_label,
                 "local": e.local, "available": e.available()}
                for e in self.engines()
            ],
            "skills": [s.to_dict() for s in self.skills()],
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


def _is_available(eng) -> bool:
    """An engine's ``available()``, guarded — capability detection must never
    raise (offline is a normal answer), and this runs across threads where an
    escaped exception would sink the whole parallel probe."""
    try:
        return bool(eng.available())
    except Exception:
        return False


def _short_err(exc: Exception) -> str:
    """A one-line, truncated rendering of an engine error for a log line."""
    msg = (exc.args[0] if exc.args else str(exc)) if exc.args else str(exc)
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= 200 else msg[:199] + "…"
