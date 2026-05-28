from __future__ import annotations

import logging
from typing import Any

import httpx

from ...config import Settings
from ..schemas.pyenv import PyEnvCreate
from ..schemas.pyfunc import PyFuncCreate
from ..schemas.replicate import NodeSnapshot, ReplicateRequest, ReplicateStatus
from .dag import DAGService
from .pyenv import PyEnvService
from .pyfunc import PyFuncService

LOGGER = logging.getLogger(__name__)


class ReplicateService:
    """Replicate node assets (envs, funcs, dags) to/from other nodes."""

    def __init__(
        self,
        settings: Settings,
        pyenv: PyEnvService,
        pyfunc: PyFuncService,
        dag: DAGService,
    ) -> None:
        self.settings = settings
        self._pyenv = pyenv
        self._pyfunc = pyfunc
        self._dag = dag
        self._client = httpx.AsyncClient(timeout=120.0)

    async def export_snapshot(self) -> NodeSnapshot:
        envs = await self._pyenv.list()
        funcs = await self._pyfunc.list()
        dags = await self._dag.list()

        return NodeSnapshot(
            node_id=self.settings.node_id,
            envs=[e.model_dump() for e in envs.envs],
            funcs=[f.model_dump() for f in funcs.funcs],
            dags=[d.model_dump() for d in dags.dags],
        )

    async def import_snapshot(self, snapshot: NodeSnapshot) -> ReplicateStatus:
        status = ReplicateStatus(
            source_node_id=snapshot.node_id,
            target_node_id=self.settings.node_id,
            status="running",
        )
        try:
            for env_data in snapshot.envs:
                await self._pyenv.create(PyEnvCreate(
                    name=env_data["name"],
                    python_version=env_data.get("python_version", "3.11"),
                    dependencies=env_data.get("dependencies", []),
                ))
                status.envs_synced += 1

            for func_data in snapshot.funcs:
                await self._pyfunc.create(PyFuncCreate(
                    name=func_data["name"],
                    code=func_data["code"],
                    description=func_data.get("description", ""),
                    python_version=func_data.get("python_version"),
                    dependencies=func_data.get("dependencies", []),
                    env_id=None,
                ))
                status.funcs_synced += 1

            for dag_data in snapshot.dags:
                from ..schemas.dag import DAGCreate, DAGEdge, DAGNodeRef, DAGStep
                steps = [
                    DAGStep(
                        id=s["id"],
                        ref=DAGNodeRef(**s["ref"]),
                        depends_on=s.get("depends_on", []),
                    )
                    for s in dag_data.get("steps", [])
                ]
                edges = [DAGEdge(**e) for e in dag_data.get("edges", [])]
                req = DAGCreate(
                    name=dag_data["name"],
                    description=dag_data.get("description", ""),
                    steps=steps,
                    edges=edges,
                )
                await self._dag.create(req)
                status.dags_synced += 1

            status.status = "completed"
        except Exception as exc:
            status.status = "failed"
            status.error = str(exc)
            LOGGER.error("Import failed: %s", exc)

        return status

    async def replicate_to(self, req: ReplicateRequest) -> ReplicateStatus:
        """Push this node's assets to a target node."""
        snapshot = await self.export_snapshot()

        if not req.include_envs:
            snapshot.envs = []
        if not req.include_funcs:
            snapshot.funcs = []
        if not req.include_dags:
            snapshot.dags = []

        try:
            resp = await self._client.post(
                f"{req.target_node_url}/api/v2/replicate/import",
                json=snapshot.model_dump(),
            )
            resp.raise_for_status()
            data = resp.json()
            return ReplicateStatus(**data)
        except Exception as exc:
            LOGGER.error("Replicate to %s failed: %s", req.target_node_url, exc)
            return ReplicateStatus(
                source_node_id=self.settings.node_id,
                target_node_id="unknown",
                status="failed",
                error=str(exc),
            )

    async def replicate_from(self, source_node_url: str) -> ReplicateStatus:
        """Pull assets from a source node into this node."""
        try:
            resp = await self._client.get(
                f"{source_node_url}/api/v2/replicate/export",
            )
            resp.raise_for_status()
            data = resp.json()
            snapshot = NodeSnapshot(**data)
            return await self.import_snapshot(snapshot)
        except Exception as exc:
            LOGGER.error("Replicate from %s failed: %s", source_node_url, exc)
            return ReplicateStatus(
                source_node_id="unknown",
                target_node_id=self.settings.node_id,
                status="failed",
                error=str(exc),
            )
