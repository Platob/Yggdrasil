"""Deployable schedule for the :class:`RepoOptimizer`.

:class:`RepoOptimizerFlow` is a class-based :class:`~yggdrasil.databricks.job.skeleton.Flow`
that deploys as a serverless Databricks job whose periodic trigger runs
``ygg databricks optimizer run ...`` on a schedule — turning the propose-only
optimizer into a *continuous* one. The job body re-enters the ygg CLI (the
flow's wheel ships ``ygg[databricks]``), so the cluster runs exactly this code.

    from yggdrasil.databricks import DatabricksClient
    from yggdrasil.databricks.ai.optimizer import RepoOptimizerFlow

    client = DatabricksClient.current()
    RepoOptimizerFlow(repo_path="/Workspace/Shared/monteleq", interval=6).deploy(client)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from yggdrasil.databricks.job.skeleton import Flow

from .optimizer import OptimizerConfig

__all__ = ["RepoOptimizerFlow"]


class RepoOptimizerFlow(Flow):
    """Serverless, periodically-triggered deployment of :class:`RepoOptimizer`."""

    def __init__(
        self,
        *,
        repo_path: str = "/Workspace/Shared/monteleq",
        endpoint_name: Optional[str] = None,
        proposals_path: Optional[str] = None,
        max_files: Optional[int] = None,
        max_tokens: Optional[int] = None,
        interval: int = 6,
        unit: str = "HOURS",
        paused: bool = False,
        name: Optional[str] = None,
        **flow_kwargs: Any,
    ) -> None:
        self.repo_path = repo_path
        self.endpoint_name = endpoint_name or OptimizerConfig.endpoint_name
        self.proposals_path = proposals_path
        self.max_files = max_files
        self.max_tokens = max_tokens
        self.interval = interval
        self.unit = unit
        self.paused = paused
        slug = re.sub(r"[^0-9A-Za-z._-]+", "_", repo_path.strip("/")).strip("_") or "repo"
        super().__init__(name=name or f"optimize-{slug}", **flow_kwargs)

    def parameters(self) -> list:
        """The ``ygg`` CLI invocation the deployed wheel task runs each cycle."""
        params = [
            "databricks", "optimizer", "run",
            "--repo", self.repo_path,
            "--endpoint", self.endpoint_name,
        ]
        if self.proposals_path:
            params += ["--proposals", self.proposals_path]
        if self.max_files is not None:
            params += ["--max-files", str(self.max_files)]
        if self.max_tokens is not None:
            params += ["--max-tokens", str(self.max_tokens)]
        return params

    def trigger(self) -> Any:
        """A periodic :class:`TriggerSettings` so the job runs continuously."""
        from databricks.sdk.service.jobs import (
            PauseStatus,
            PeriodicTriggerConfiguration,
            PeriodicTriggerConfigurationTimeUnit,
            TriggerSettings,
        )

        return TriggerSettings(
            periodic=PeriodicTriggerConfiguration(
                interval=self.interval,
                unit=PeriodicTriggerConfigurationTimeUnit(self.unit.upper()),
            ),
            pause_status=PauseStatus.PAUSED if self.paused else PauseStatus.UNPAUSED,
        )
