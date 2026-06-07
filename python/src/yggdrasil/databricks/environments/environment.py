"""``Environment`` — a deployed base environment in the workspace.

A handle over a project's reusable image under
``/Workspace/Shared/environment/<proj>/``: the serverless ``<stem>.yml`` and the
classic-cluster ``<stem>.requirements.txt``. Built by the
:class:`~yggdrasil.databricks.environments.service.Environments` service.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from yggdrasil.version import VersionInfo

from ..resource import DatabricksResource

if TYPE_CHECKING:
    from ..fs.workspace_path import WorkspacePath


class Environment(DatabricksResource):
    """A deployed base environment — ``<proj>-<version>-py3XX`` and its files."""

    def __init__(
        self,
        service=None,
        *,
        name: str,
        project: Optional[str] = None,
        version: Optional[VersionInfo] = None,
        python: Optional[str] = None,
        env_dir: Optional[str] = None,
        serverless: Optional[str] = None,
        cluster: Optional[str] = None,
        dependencies: Sequence[str] = (),
    ) -> None:
        super().__init__(service=service)
        #: Version-tagged file stem (``ygg-0.8.57-py311``).
        self.name = name
        #: Project / distribution name (``ygg``).
        self.project = project
        #: :class:`VersionInfo` of the image, when known.
        self.version = version
        self.python = python
        self.env_dir = env_dir
        #: Serverless ``.yml`` workspace path (a job's ``base_environment``).
        self.serverless = serverless
        #: Classic-cluster ``.requirements.txt`` path (the cluster install layer).
        self.cluster = cluster
        self.dependencies = list(dependencies)

    def _path(self, dest: Optional[str]) -> "Optional[WorkspacePath]":
        if dest is None:
            return None
        from ..path import DatabricksPath

        return DatabricksPath.from_(dest, client=self.client)

    @property
    def serverless_path(self) -> "Optional[WorkspacePath]":
        return self._path(self.serverless)

    @property
    def requirements_path(self) -> "Optional[WorkspacePath]":
        return self._path(self.cluster)

    def exists(self) -> bool:
        p = self.serverless_path
        return bool(p and p.exists())

    def job_environment(self, environment_key: str = "default") -> "Any":
        """The serverless ``JobEnvironment`` referencing this image by path —
        drop into a job's ``environments=[…]``."""
        from databricks.sdk.service.compute import Environment as _Env
        from databricks.sdk.service.jobs import JobEnvironment

        return JobEnvironment(environment_key=environment_key,
                              spec=_Env(base_environment=self.serverless))

    def delete(self) -> "Environment":
        for p in (self.serverless_path, self.requirements_path):
            if p is not None:
                p.unlink(missing_ok=True)
        return self

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Environment({self.name!r})"
