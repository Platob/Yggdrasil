"""
Cluster management helpers for Databricks compute.

This module provides a lightweight ``Cluster`` helper that wraps the
Databricks SDK to simplify common CRUD operations and metadata handling
for clusters. Metadata is stored in custom tags prefixed with
``yggdrasil:``.
"""

from __future__ import annotations

import base64
import datetime as dt
import importlib
import inspect
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union, List

from ..workspaces.workspace import WorkspaceObject, Workspace
from ...libs.databrickslib import databricks_sdk

if databricks_sdk is None:  # pragma: no cover - import guard
    ResourceDoesNotExist = Exception  # type: ignore
    Language = Any  # type: ignore
    ResultType = Any  # type: ignore
    SparkVersion = Any  # type: ignore
else:  # pragma: no cover - runtime fallback when SDK is missing
    from databricks.sdk import WorkspaceClient, ClustersAPI
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.compute import (
        ClusterDetails, Language, ResultType, Kind, State, DataSecurityMode, Library, PythonPyPiLibrary
    )

    _CREATE_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.create).parameters.keys()}
    _EDIT_ARG_NAMES = {_ for _ in inspect.signature(ClustersAPI.edit).parameters.keys()}

__all__ = ["Cluster"]


# module-level mapping Databricks Runtime -> (major, minor) Python version
_PYTHON_BY_DBR: dict[str, tuple[int, int]] = {
    "10.4": (3, 8),
    "11.3": (3, 9),
    "12.2": (3, 9),
    "13.3": (3, 10),
    "14.3": (3, 10),
    "15.4": (3, 11),
    "16.4": (3, 12),
    "17.0": (3, 12),
    "17.1": (3, 12),
    "17.2": (3, 12),
    "17.3": (3, 12),
    "18.0": (3, 12),
}




@dataclass
class Cluster(WorkspaceObject):
    """Helper for creating, retrieving, updating, and deleting clusters.

    Parameters
    ----------
    workspace:
        Optional :class:`Workspace` (or config-compatible object) used to
        build the underlying :class:`databricks.sdk.WorkspaceClient`.
        Defaults to a new :class:`Workspace`.
    cluster_id:
        Optional existing cluster identifier. Methods that operate on a
        cluster will use this value when ``cluster_id`` is omitted.
    """
    workspace: Workspace = field(default_factory=Workspace)
    cluster_id: Optional[str] = None
    _details: Optional["ClusterDetails"] = None

    def connect(self):
        super().connect()

        self._details = self.clusters_client().get(cluster_id=self.cluster_id)

        return self

    @property
    def details(self):
        if self._details is None:
            self._details = self.clusters_client().get(cluster_id=self.cluster_id)
        return self._details

    @details.setter
    def details(self, value: "ClusterDetails"):
        self._details = value
        self.cluster_id = value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _client(self) -> "WorkspaceClient":
        """Return a connected WorkspaceClient instance."""
        return self.workspace.sdk()

    def clusters_client(self) -> "ClustersAPI":
        return self._client().clusters

    def spark_versions(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
    ):
        all_versions = self.clusters_client().spark_versions().versions

        if not all_versions:
            raise ValueError("No databricks spark versions found")

        versions = all_versions

        # --- filter by Photon / non-Photon ---
        if photon is not None:
            if photon:
                versions = [v for v in versions if "photon" in v.key.lower()]
            else:
                versions = [v for v in versions if "photon" not in v.key.lower()]

        # --- filter by Python version (Databricks Runtime mapping) ---
        if python_version is not None:
            # normalize input python_version to (major, minor)
            if isinstance(python_version, str):
                if python_version.lower() == "current":
                    py_filter = (sys.version_info.major, sys.version_info.minor)
                else:
                    parts = python_version.split(".")
                    py_filter = tuple(int(p) for p in parts[:2])
            else:
                py_filter = tuple(python_version[:2])

            def dbr_from_key(key: str) -> Optional[str]:
                # "17.3.x-gpu-ml-scala2.13" -> "17.3"
                parts = key.split(".")
                if len(parts) < 2:
                    return None
                return ".".join(parts[:2])

            def py_for_key(key: str) -> Optional[tuple[int, int]]:
                dbr = dbr_from_key(key)
                if dbr is None:
                    return None
                return _PYTHON_BY_DBR.get(dbr)

            versions = [v for v in versions if py_for_key(v.key) == py_filter]

        return versions

    def latest_spark_version(
        self,
        photon: Optional[bool] = None,
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
    ):
        versions = self.spark_versions(photon=photon, python_version=python_version)

        max_version = None

        for version in versions:
            if max_version is None or version.key > max_version.key:
                max_version = version

        return max_version

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #
    def _check_details(
        self,
        details: "ClusterDetails",
        python_version: Optional[Union[str, tuple[int, ...]]] = None,
        **kwargs
    ):
        if kwargs:
            details = ClusterDetails(**{
                **details.as_shallow_dict(),
                **kwargs
            })

        if details.cluster_name is None:
            details.cluster_name = self.workspace.current_user.user_name

        if details.spark_version is None or python_version:
            details.spark_version = self.latest_spark_version(photon=False, python_version=python_version).key

        if details.single_user_name:
            if not details.data_security_mode:
                details.data_security_mode = DataSecurityMode.DATA_SECURITY_MODE_DEDICATED

        if not details.node_type_id:
            details.node_type_id = "rd-fleet.xlarge"

        if getattr(details, "virtual_cluster_size", None) is None and details.num_workers is None and details.autoscale is None:
            if details.is_single_node is None:
                details.is_single_node = True

        if details.is_single_node is not None and details.kind is None:
            details.kind = Kind.CLASSIC_PREVIEW

        return details

    def create_or_update(
        self,
        cluster_name: Optional[str] = None,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        upload_local_lib: bool | None = None,
        **cluster_spec: Any
    ):
        found = self.find_cluster(
            cluster_id=self.cluster_id,
            cluster_name=cluster_name,
            raise_error=False
        )

        if found is not None:
            return found.update(
                cluster_name=cluster_name,
                libraries=libraries,
                upload_local_lib=upload_local_lib,
                **cluster_spec
            )

        return self.create(
            cluster_name=cluster_name,
            libraries=libraries,
            upload_local_lib=upload_local_lib,
            **cluster_spec
        )

    def create(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        upload_local_lib: bool | None = None,
        **cluster_spec: Any
    ) -> str:
        cluster_spec["autotermination_minutes"] = int(cluster_spec.get("autotermination_minutes", 30))
        update_details = self._check_details(details=ClusterDetails(), **cluster_spec)

        self.details = self.clusters_client().create_and_wait(**{
            k: v
            for k, v in update_details.as_shallow_dict().items()
            if k not in _CREATE_ARG_NAMES
        })

        self.install_libraries(libraries=libraries, upload_local_lib=upload_local_lib)

        return self

    def update(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        upload_local_lib: bool | None = None,
        **cluster_spec: Any
    ) -> "Cluster":
        self.install_libraries(libraries=libraries, upload_local_lib=upload_local_lib)

        existing_details = {
            k: v
            for k, v in self.details.as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        update_details = {
            k: v
            for k, v in self._check_details(details=self.details, **cluster_spec).as_shallow_dict().items()
            if k in _EDIT_ARG_NAMES
        }

        if update_details != existing_details:
            self.details = self.clusters_client().edit_and_wait(**update_details)

        return self

    def list_clusters(self) -> Iterator["Cluster"]:
        """Iterate clusters, yielding helpers annotated with metadata."""

        for details in self.clusters_client().list():
            details: ClusterDetails = details

            yield Cluster(
                workspace=self.workspace,
                cluster_id=details.cluster_id,
                _details=details
            )

    def find_cluster(
        self,
        cluster_id: Optional[str] = None,
        *,
        cluster_name: Optional[str] = None,
        raise_error: Optional[bool] = None
    ) -> Optional["Cluster"]:
        """Find a cluster by name or id and return a populated helper."""
        if not cluster_name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        if cluster_id:
            try:
                details = self.clusters_client().get(cluster_id=cluster_id)
            except ResourceDoesNotExist:
                if raise_error:
                    raise ValueError(f"Cannot find databricks cluster {cluster_id!r}")
                return None

            return Cluster(
                workspace=self.workspace, cluster_id=details.cluster_id, _details=details
            )

        cluster_name_cf = cluster_name.casefold()

        for cluster in self.list_clusters():
            if cluster_name_cf == cluster.details.cluster_name.casefold():
                return cluster

        if raise_error:
            raise ValueError(f"Cannot find databricks cluster {cluster_name!r}")
        return None

    def ensure_running(
        self,
    ):
        self.details = self.clusters_client().wait_get_cluster_running(cluster_id=self.cluster_id)

        if self._details.state != State.RUNNING:
            return self.start()

        return self

    def start(
        self,
        timeout: Optional[dt.timedelta] = dt.timedelta(minutes=20)
    ):
        r = self.clusters_client().start(cluster_id=self.cluster_id)

        if timeout:
            self.details = r.result(timeout=timeout)
            return self

        return r

    def restart(
        self,
        timeout: Optional[dt.timedelta] = dt.timedelta(minutes=20)
    ):
        r = self.clusters_client().restart(cluster_id=self.cluster_id)

        if timeout:
            self.details = r.result(timeout=timeout)
            return self

        return r

    def delete(
        self
    ) -> None:
        self.clusters_client().delete(cluster_id=self.cluster_id)

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #
    def execute_command(
        self,
        command: str,
        *,
        language: Optional["Language"] = None,
        timeout: Optional[dt.timedelta] = None,
        result_tag: Optional[str] = None,
    ) -> str:
        """Execute a command string on the cluster and return its output.

        Raises a :class:`RuntimeError` with the remote traceback attached
        when the command fails.

        Parameters
        ----------
        command:
            The code to execute remotely.
        language:
            The language to execute (defaults to Python).
        timeout:
            Optional override for the execution timeout.
        result_tag:
            When provided, looks for two occurrences of this marker in the
            command output. If found, anything before the first marker is
            printed for visibility and the text between the markers is
            returned. If the marker is not found twice, the raw output is
            returned.
        """
        lang = language or Language.PYTHON
        cid = self.cluster_id
        timeout = timeout or dt.timedelta(minutes=20)

        client = self._client()
        context = client.command_execution.create_and_wait(
            cluster_id=cid, language=lang
        )
        context_id = getattr(context, "id", None) or getattr(context, "context_id", None)
        if not context_id:
            raise RuntimeError("Failed to create command execution context")

        try:
            result = client.command_execution.execute_and_wait(
                cluster_id=cid,
                context_id=context_id,
                language=lang,
                command=command,
                timeout=timeout,
            )
        finally:
            try:
                client.command_execution.destroy(
                    cluster_id=cid, context_id=context_id
                )
            except Exception:
                pass

        if not getattr(result, "results", None):
            raise RuntimeError("Command execution returned no results")

        res = result.results
        if res.result_type == ResultType.ERROR:
            message = res.cause or "Command execution failed"
            remote_tb = (
                getattr(res, "data", None)
                or getattr(res, "stack_trace", None)
                or getattr(res, "traceback", None)
            )
            if remote_tb:
                message = f"{message}\n\nRemote traceback:\n{remote_tb}"
            raise RuntimeError(message)

        if res.result_type == ResultType.TEXT:
            output = getattr(res, "data", "") or ""
        elif getattr(res, "data", None) is not None:
            output = str(res.data)
        else:
            output = ""

        if result_tag:
            start = output.find(result_tag)
            if start != -1:
                content_start = start + len(result_tag)
                end = output.find(result_tag, content_start)
                if end != -1:
                    before = output[:start]
                    if before:
                        print(before)
                    return output[content_start:end]

        return output

    def upload_driver_file(
        self,
        remote_path: str,
        content: bytes | str,
        *,
        overwrite: bool = True,
    ) -> str:
        """Upload content directly to the cluster driver filesystem.

        Parameters
        ----------
        remote_path:
            Destination path on the cluster driver node.
        content:
            Data to write. ``str`` values are encoded as UTF-8 before upload;
            ``bytes`` are written as-is.
        overwrite:
            Whether to replace an existing file. When ``False`` and the file
            exists remotely, a ``FileExistsError`` is raised by the remote
            command.

        Returns
        -------
        str
            The ``remote_path`` after a successful write.
        """
        payload = (
            content.encode("utf-8") if isinstance(content, str) else bytes(content)
        )
        encoded = base64.b64encode(payload).decode("utf-8")

        command = f"""
import base64
import os

path = {remote_path!r}
data = base64.b64decode({encoded!r})

if os.path.exists(path) and not {overwrite!r}:
    raise FileExistsError(f"{remote_path} already exists")

os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "wb") as fp:
    fp.write(data)
print(path)
"""

        return self.execute_command(command)

    def install_libraries(
        self,
        libraries: Optional[List[Union[str, "Library"]]] = None,
        upload_local_lib: bool | None = None,
    ):
        if not libraries:
            return self

        wsdk = self.workspace.sdk()

        wsdk.libraries.install(
            cluster_id=self.cluster_id,
            libraries=[
                self._check_library(_, upload_local_lib=upload_local_lib)
                for _ in libraries if _
            ]
        )

        return self

    def _check_library(
        self,
        value,
        upload_local_lib: bool | None = None,
    ) -> Library:
        if isinstance(value, Library):
            return value

        if isinstance(value, str):
            original_value = value

            # 1) If it's an existing path (file or dir), upload as-is
            if os.path.exists(value):
                value = self._upload_local_path(value)

            # 2) If no path exists, but copy_local_lib=True, try to resolve from current env
            elif upload_local_lib:
                pkg_path = _resolve_local_package_path(original_value)
                if pkg_path is not None and os.path.exists(pkg_path):
                    value = self._upload_local_path(pkg_path)
                # if we can't resolve it locally, we just fall through to PyPI handling

            # 3) Now `value` is either:
            #    - a workspace/DBFS path (dbfs:/.../something.whl/.zip/.jar/requirements.txt)
            #    - or still the original string (e.g. "pandas") for PyPI install

            # workspace / file-based libs
            if value.endswith(".jar"):
                return Library(jar=value)
            elif value.endswith("requirements.txt"):
                return Library(requirements=value)
            elif value.endswith(".whl"):
                return Library(whl=value)
            elif value.endswith(".zip"):
                # Databricks can install zip as Python lib via `whl` field (zip or whl path)
                return Library(whl=value)

            # 4) Fallback: treat as PyPI package name
            return Library(
                pypi=PythonPyPiLibrary(
                    package=value,
                    repo=os.getenv("PIP_EXTRA_INDEX_URL"),
                )
            )

        raise ValueError(f"Cannot build Library object from {type(value)}")

    # ----------------- helpers ----------------
    def _upload_local_path(self, local_path: str) -> str:
        """
        Given a local file/folder, optionally zip, upload to workspace cache, and
        return the workspace/DBFS target path.
        """
        local_path = Path(local_path)

        # If folder -> zip it first
        if local_path.is_dir():
            tmp_dir = Path(tempfile.mkdtemp())
            zip_base = tmp_dir / local_path.name
            archive_path = shutil.make_archive(
                base_name=str(zip_base),
                format="zip",
                root_dir=str(local_path),
            )
            upload_source = Path(archive_path)
        else:
            upload_source = local_path

        target_path = (
            self.workspace.cache_user_folder(
                suffix=f"/clusters/{self.cluster_id}/libraries/{upload_source.name}"
            )
        )

        self.workspace.upload_local_file(
            local_path=str(upload_source),
            target_path=target_path,
        )

        return target_path


def _resolve_local_package_path(pkg_name: str) -> str | None:
    """Try to import a package/module from current env and return its folder/file path."""
    try:
        mod = importlib.import_module(pkg_name)
    except ModuleNotFoundError:
        return None

    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        return None

    p = Path(mod_file)

    # package: /.../pkg/__init__.py -> use folder
    if p.name == "__init__.py":
        return str(p.parent)

    # single module: /.../pkg.py -> use file directly
    return str(p)


def _zip_folder(folder_path: str) -> str:
    folder_path = str(folder_path)
    folder_name = os.path.basename(os.path.normpath(folder_path))

    # temp dir to hold the zip
    tmp_dir = tempfile.mkdtemp()

    # base path *without* extension, make_archive adds `.zip`
    zip_base = os.path.join(tmp_dir, folder_name)

    # create zip archive
    archive_path = shutil.make_archive(
        base_name=zip_base,
        format="zip",
        root_dir=folder_path,
    )

    return archive_path  # e.g. /tmp/tmpabc123/my-lib.zip