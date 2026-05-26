"""
Workspace-hosted PyPI-style index for non-public Python distributions.

Thin specialization of :class:`yggdrasil.io.pypi.PyPIPath` that pins
the root to a :class:`~yggdrasil.databricks.fs.workspace_path.WorkspacePath`
under ``/Workspace/Shared/.ygg/pypi/simple`` (configurable) and binds
the publisher to a :class:`DatabricksClient` so it inherits the
workspace's auth.

All the wheel-build, versioning, subfolder layout and
``index.html`` maintenance lives on :class:`PyPIPath`; this subclass
exists so the Jobs surface can spell the workspace-side index as a
first-class object (``WorkspacePyPI(client)``) without each caller
re-deriving the standard root.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

from yggdrasil.path.pypi import PyPIPath

if TYPE_CHECKING:
    from ..client import DatabricksClient
    from ..fs.workspace_path import WorkspacePath


__all__ = [
    "DEFAULT_WORKSPACE_PYPI_ROOT",
    "WorkspacePyPI",
]

LOGGER = logging.getLogger(__name__)

#: Default root for the workspace-side simple index. Lands under
#: ``/Workspace/Shared`` so anyone with workspace read can pip install
#: published wheels; restrict via the workspace ACL when stricter
#: visibility is required.
DEFAULT_WORKSPACE_PYPI_ROOT = "/Workspace/Shared/.ygg/pypi/simple"


class WorkspacePyPI(PyPIPath):
    """Publish local / editable Python distributions into the workspace.

    Parameters
    ----------
    client
        Bound :class:`DatabricksClient`. Determines the workspace the
        wheels land in.
    root
        Workspace path that anchors the simple index. Defaults to
        :data:`DEFAULT_WORKSPACE_PYPI_ROOT`. Strings resolve through
        :class:`WorkspacePath` so ``<me>`` placeholders work.

    Examples
    --------
    Used implicitly by :meth:`JobTask.from_callable` when
    ``workspace_pypi=True``::

        @job.pytask(workspace_pypi=True)
        def step(): ...

    Or explicitly to publish a single package::

        pypi = WorkspacePyPI(client)
        target = pypi.publish("my_local_pkg")
        # target.full_path() → /Workspace/Shared/.ygg/pypi/simple/...
    """

    def __init__(
        self,
        client: "DatabricksClient",
        *,
        root: Union[str, "WorkspacePath"] = DEFAULT_WORKSPACE_PYPI_ROOT,
    ) -> None:
        from ..fs.workspace_path import WorkspacePath

        if isinstance(root, str):
            root = WorkspacePath(root, service=client.workspaces)
        super().__init__(root)
        self.client = client
