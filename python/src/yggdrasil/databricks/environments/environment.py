"""``Environment`` — a deployed base environment in the workspace.

A handle over a project's reusable image under
``/Workspace/Shared/environment/<proj>/``: the serverless ``<stem>.yml``
(``environment_version`` + wheel dependency paths) and the classic-cluster
``<stem>.requirements.txt`` that sit side by side. Built by the
:class:`~yggdrasil.databricks.environments.service.Environments` service; not
constructed directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

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
        env_dir: Optional[str] = None,
        serverless: Optional[str] = None,
        cluster: Optional[str] = None,
        dependencies: Sequence[str] = (),
        version: Optional[str] = None,
        python: Optional[str] = None,
    ) -> None:
        super().__init__(service=service)
        #: Version-tagged file stem (``ygg-0.8.57-py311``).
        self.name = name
        #: The project / distribution name (``ygg``), when known.
        self.project = project
        #: The project folder holding the env files.
        self.env_dir = env_dir
        #: Serverless ``.yml`` workspace path.
        self.serverless = serverless
        #: Classic-cluster ``.requirements.txt`` workspace path — the cluster
        #: install layer (``Library(requirements=…)``).
        self.cluster = cluster
        self.dependencies = list(dependencies)
        self.version = version
        self.python = python

    def _path(self, dest: Optional[str]) -> "Optional[WorkspacePath]":
        if dest is None:
            return None
        from ..path import DatabricksPath

        return DatabricksPath.from_(dest, client=self.client)

    @property
    def serverless_path(self) -> "Optional[WorkspacePath]":
        """The serverless ``.yml`` :class:`WorkspacePath` (job ``base_environment``)."""
        return self._path(self.serverless)

    @property
    def requirements_path(self) -> "Optional[WorkspacePath]":
        """The cluster ``.requirements.txt`` :class:`WorkspacePath`."""
        return self._path(self.cluster)

    def exists(self) -> bool:
        p = self.serverless_path
        return bool(p and p.exists())

    def delete(self) -> "Environment":
        for p in (self.serverless_path, self.requirements_path):
            if p is not None:
                p.unlink(missing_ok=True)
        return self

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Environment({self.name!r})"
