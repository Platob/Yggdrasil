"""DatabricksLoki — the specialized Databricks agent.

A :class:`~yggdrasil.loki.Loki` that lives on Databricks at most: it detects
its workspace **only** from the session ``ygg databricks configure`` wrote
(not env vars / hard parameters), reasons through a Databricks **serving
endpoint** by default, and can deploy itself to run on Databricks compute.

    from yggdrasil.databricks.loki import DatabricksLoki

    loki = DatabricksLoki.current()
    loki.databricks          # client resolved from the configure profile
    loki.reason("summarize today's failed jobs")   # via a serving endpoint
    loki.deploy(behavior="reason", prompt="...")    # run it on Databricks
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend, detect_local

from .session import read_session

if TYPE_CHECKING:
    from yggdrasil.databricks import DatabricksClient
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.loki.engine import TokenEngine

__all__ = ["DatabricksLoki"]

#: Default serving endpoint the agent reasons through — the **lowest**
#: (smallest / cheapest) broadly-available Foundation Model endpoint, so the
#: agent is cheap by default (override with ``YGG_LOKI_SERVING_ENDPOINT`` or the
#: ``serving_endpoint=`` arg).
DEFAULT_ENDPOINT = "databricks-meta-llama-3-1-8b-instruct"


class DatabricksLoki(Loki):
    """Loki specialized to a single Databricks workspace."""

    name = "databricks-loki"
    # Prefer the Databricks brain; fall back to Claude/OpenAI if creds exist.
    ENGINE_PREFERENCE = ("databricks", "claude", "openai")

    _CURRENT: "Optional[DatabricksLoki]" = None

    def __init__(self, *, serving_endpoint: Optional[str] = None) -> None:
        super().__init__()
        self.serving_endpoint = (
            serving_endpoint or os.getenv("YGG_LOKI_SERVING_ENDPOINT") or DEFAULT_ENDPOINT
        )
        self._client: "Optional[DatabricksClient]" = None

    # -- detection: configure session only ---------------------------------

    def backends(self, *, refresh: bool = False) -> list[Backend]:
        if self._backends is None or refresh:
            self._backends = [self._detect_session(), detect_local()]
        return self._backends

    def _detect_session(self) -> Backend:
        """Detect Databricks strictly from the configure-written session."""
        sess = read_session()
        if not sess:
            return Backend(
                "databricks",
                available=False,
                detail={"reason": "no `ygg databricks configure` session found"},
            )
        return Backend(
            "databricks",
            available=bool(sess.get("host")),
            detail={
                "host": sess.get("host"),
                "profile": sess.get("profile"),
                "user": sess.get("user"),
                "auth_type": sess.get("auth_type"),
                "source": "ygg databricks configure",
            },
        )

    @property
    def databricks(self) -> "Optional[DatabricksClient]":
        """Client built from the configure profile (cached). ``None`` if unset."""
        if self._client is not None:
            return self._client
        backend = self.backend("databricks")
        if not backend or not backend.available:
            return None
        from yggdrasil.databricks import DatabricksClient

        profile = backend.detail.get("profile") or "DEFAULT"
        self._client = DatabricksClient(profile=profile)
        return self._client

    # -- Databricks-first engine wiring ------------------------------------

    def _engine_instances(self) -> "dict[str, TokenEngine]":
        from yggdrasil.loki.engines import (
            ClaudeEngine,
            DatabricksServingEngine,
            OpenAIEngine,
        )

        engines: "dict[str, TokenEngine]" = {"claude": ClaudeEngine(), "openai": OpenAIEngine()}
        # Bind the Databricks brain only to the configure-session client — never
        # an env-resolved one — so detection stays strictly configure-driven.
        client = self.databricks
        if client is not None:
            engines = {
                "databricks": DatabricksServingEngine(
                    client=client, endpoint=self.serving_endpoint,
                ),
                **engines,
            }
        return engines

    # -- deploy to Databricks ----------------------------------------------

    def deploy(
        self,
        *,
        name: Optional[str] = None,
        behavior: str = "reason",
        job: bool = True,
        **params: Any,
    ) -> "Job":
        """Deploy this agent to run on Databricks compute.

        Creates (or updates) a Databricks **Job** that runs the agent on
        **serverless** against the seeded ygg image through the single ``ygg``
        wheel entry point — ``ygg loki reason ...`` for the default ``reason``
        behavior, ``ygg loki run <behavior> --kwarg k=v ...`` otherwise (on the
        runtime ``ygg loki`` resolves to this DatabricksLoki). The job is
        upserted by name and returned; trigger it with ``job.run()``.

        Requires a workspace seeded with ``ygg databricks deploy`` (for the
        serverless ygg environment) and a ``ygg databricks configure`` session.
        """
        client = self.databricks
        if client is None:
            raise RuntimeError(
                "no Databricks session — run `ygg databricks configure` first"
            )
        if not job:
            raise NotImplementedError(
                "only job deployment is implemented; pass job=True"
            )

        import json

        from databricks.sdk.service.jobs import PythonWheelTask, Task

        # The single ``ygg`` entry point with a ``loki`` subcommand. ``reason``
        # takes the prompt positionally; every other behavior goes through
        # ``run`` with JSON-encoded ``--kwarg`` pairs (the CLI JSON-decodes each).
        if behavior == "reason":
            parameters = ["loki", "reason", str(params.pop("prompt", ""))]
            if params.get("system"):
                parameters += ["--system", str(params["system"])]
            if params.get("engine"):
                parameters += ["--engine", str(params["engine"])]
        else:
            parameters = ["loki", "run", behavior]
            for key, value in params.items():
                parameters += ["--kwarg", f"{key}={json.dumps(value)}"]

        # Serverless environment carrying the pre-built ygg wheel image.
        environment = client.environments.find("ygg").job_environment(environment_key="default")
        task = Task(
            task_key="loki",
            environment_key="default",
            python_wheel_task=PythonWheelTask(
                package_name="ygg",
                entry_point="ygg",
                parameters=parameters,
            ),
        )
        return client.jobs.create_or_update(
            name=name or f"loki-{self.user}",
            tasks=[task],
            environments=[environment],
            tags={"ygg": "loki", "ygg_kind": "loki-agent", "ygg_behavior": behavior},
        )
