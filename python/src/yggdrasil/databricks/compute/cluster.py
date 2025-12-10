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
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Optional

from ...libs.databrickslib import databricks_sdk, require_databricks_sdk
from ..workspaces.workspace import WorkspaceObject, Workspace

if databricks_sdk is not None:  # pragma: no cover - import guard
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import ResourceDoesNotExist
    from databricks.sdk.service.compute import ClusterDetails, Language, ResultType, SparkVersion
else:  # pragma: no cover - runtime fallback when SDK is missing
    WorkspaceClient = Any  # type: ignore
    ResourceDoesNotExist = Exception  # type: ignore
    ClusterDetails = Any  # type: ignore
    Language = Any  # type: ignore
    ResultType = Any  # type: ignore
    SparkVersion = Any  # type: ignore

__all__ = ["Cluster", "ClusterInfo"]


_METADATA_PREFIX = "yggdrasil:"


@dataclass
class ClusterInfo:
    """Representation of a cluster with parsed metadata."""

    cluster_id: str
    name: str
    metadata: Dict[str, str]
    state: Optional[str] = None
    details: Optional["ClusterDetails"] = None


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
    metadata:
        Optional initial metadata dictionary to write to custom tags
        after creation.
    """

    workspace: Workspace = field(default_factory=Workspace)
    cluster_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    info: Optional[ClusterInfo] = field(default=None, init=False, repr=False)
    _cluster_details_cache: Dict[str, "ClusterDetails"] = field(
        default_factory=dict, init=False, repr=False
    )
    _spark_versions_cache: Optional[list["SparkVersion"]] = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _client(self) -> "WorkspaceClient":
        """Return a connected WorkspaceClient instance."""

        require_databricks_sdk()
        return self.workspace.sdk()

    def _resolve_cluster_id(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> str:
        cid = cluster_id or self.cluster_id
        if not cid and cluster_name:
            match = self.find_cluster(name=cluster_name)
            if match:
                cid = match.cluster_id or (match.info.cluster_id if match.info else None)
                if cid:
                    self.cluster_id = self.cluster_id or cid
                    self._cluster_details_cache.update(match._cluster_details_cache)
                    if match.info:
                        self.info = match.info

        if not cid:
            raise ValueError("cluster_id or cluster_name is required for this operation")
        return cid

    def _load_cluster(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> "ClusterDetails":
        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        if cid in self._cluster_details_cache:
            return self._cluster_details_cache[cid]

        details = self._client().clusters.get(cluster_id=cid)
        self._cluster_details_cache[cid] = details
        return details

    def _metadata_from_tags(self, tags: Dict[str, str] | None) -> Dict[str, str]:
        tags = tags or {}
        metadata: Dict[str, str] = {}
        for key, value in tags.items():
            if key.startswith(_METADATA_PREFIX):
                metadata[key[len(_METADATA_PREFIX) :]] = value
        return metadata

    def _with_metadata_tags(self, details: "ClusterDetails", metadata: Dict[str, str]) -> Dict[str, Any]:
        """Return a cluster edit payload with updated metadata tags."""

        current_tags = getattr(details, "custom_tags", None) or {}
        # Remove existing metadata tags, then add new ones.
        base_tags = {k: v for k, v in current_tags.items() if not k.startswith(_METADATA_PREFIX)}
        merged_tags = {**base_tags, **{f"{_METADATA_PREFIX}{k}": v for k, v in metadata.items()}}

        # dataclasses from SDK typically expose ``as_dict`` for serialization
        if hasattr(details, "as_dict"):
            payload = details.as_dict()
        else:
            payload = details.__dict__.copy()

        payload["custom_tags"] = merged_tags
        payload["cluster_id"] = self._resolve_cluster_id(details.cluster_id)
        return payload

    def _payload_from_details(self, details: "ClusterDetails") -> Dict[str, Any]:
        """Return a mutable payload from cluster details preserving cluster_id."""

        if hasattr(details, "as_dict"):
            payload = details.as_dict()
        else:
            payload = details.__dict__.copy()

        payload["cluster_id"] = self._resolve_cluster_id(details.cluster_id)
        self._cluster_details_cache[payload["cluster_id"]] = details
        return payload

    def _library_payload(self, packages: Iterable[str]) -> list[Dict[str, Any]]:
        """Build a list of library payloads for PyPI packages."""

        return [{"pypi": {"package": pkg}} for pkg in packages]

    def _runtime_key(self, version: "SparkVersion") -> str:
        """Extract the canonical runtime key from a spark version."""

        return getattr(version, "key", None) or getattr(version, "spark_version", "") or ""

    def _available_spark_versions(self) -> list["SparkVersion"]:
        """Return cached spark/runtime versions, fetching if needed."""

        if self._spark_versions_cache is None:
            self._spark_versions_cache = list(self._client().clusters.list_spark_versions())
        return self._spark_versions_cache

    def _select_spark_version_for_python(self, python_version: str) -> str:
        """Find a spark version that matches a requested Python version.

        Prefers a runtime that matches the ``DATABRICKS_RUNTIME_VERSION``
        environment (when running inside a Databricks cluster) before
        falling back to name-based matching on the advertised Python
        version.
        """

        normalized = python_version.strip().lower()
        versions = self._available_spark_versions()

        env_runtime = os.getenv("DATABRICKS_RUNTIME_VERSION")
        if env_runtime:
            normalized_runtime = env_runtime.lower()
            for version in versions:
                key = self._runtime_key(version)
                if not key:
                    continue
                if normalized_runtime in key.lower():
                    return key

        for version in versions:
            name = (getattr(version, "name", "") or getattr(version, "description", "")).lower()
            key = (getattr(version, "key", "") or "").lower()
            if normalized in name or f"python {normalized}" in name or normalized in key:
                return self._runtime_key(version)

        raise ValueError(
            f"No spark version found that advertises Python {python_version}."
        )

    # ------------------------------------------------------------------ #
    # CRUD operations
    # ------------------------------------------------------------------ #
    def create(self, **cluster_spec: Any) -> str:
        """Create a cluster and optionally apply initial metadata tags.

        Returns the created ``cluster_id`` and stores it on the instance.
        """

        client = self._client()
        response = client.clusters.create_and_wait(**cluster_spec)
        self.cluster_id = getattr(response, "cluster_id", None)
        if self.cluster_id:
            self._cluster_details_cache[self.cluster_id] = response

        if self.metadata:
            self.update_metadata(self.metadata, cluster_id=self.cluster_id)

        return self.cluster_id or ""

    def get(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> "ClusterDetails":
        """Retrieve a cluster's details."""

        return self._load_cluster(cluster_id, cluster_name)

    def list_clusters(self) -> Iterator["Cluster"]:
        """Iterate clusters, yielding helpers annotated with metadata."""

        for details in self._client().clusters.list():
            cid = getattr(details, "cluster_id", "") or ""
            name = getattr(details, "cluster_name", "") or getattr(details, "name", "") or ""
            metadata = self._metadata_from_tags(getattr(details, "custom_tags", None))

            cluster = Cluster(
                workspace=self.workspace, cluster_id=cid or None, metadata=metadata
            )
            cluster.info = ClusterInfo(
                cluster_id=cid,
                name=name,
                metadata=metadata,
                state=getattr(details, "state", None),
                details=details,
            )
            if cid:
                cluster._cluster_details_cache[cid] = details
                self._cluster_details_cache[cid] = details

            yield cluster

    def find_cluster(
        self,
        *,
        name: Optional[str] = None,
        cluster_name: Optional[str] = None,
        cluster_id: Optional[str] = None,
    ) -> Optional["Cluster"]:
        """Find a cluster by name or id and return a populated helper."""

        name = name or cluster_name
        if not name and not cluster_id:
            raise ValueError("Either name or cluster_id must be provided")

        if cluster_id:
            try:
                details = self._load_cluster(cluster_id)
            except ResourceDoesNotExist:
                return None

            metadata = self._metadata_from_tags(getattr(details, "custom_tags", None))
            cluster = Cluster(
                workspace=self.workspace, cluster_id=cluster_id, metadata=metadata
            )
            cluster.info = ClusterInfo(
                cluster_id=cluster_id,
                name=getattr(details, "cluster_name", "")
                or getattr(details, "name", "")
                or "",
                metadata=metadata,
                state=getattr(details, "state", None),
                details=details,
            )
            cluster._cluster_details_cache[cluster_id] = details
            return cluster

        assert name is not None
        target = name.strip().lower()
        for cluster in self.list_clusters():
            cluster_name = cluster.info.name if cluster.info else None
            if cluster_name and cluster_name.strip().lower() == target:
                return cluster

        return None

    def delete(
        self,
        cluster_id: Optional[str] = None,
        *,
        cluster_name: Optional[str] = None,
        permanent: bool = False,
    ) -> None:
        """Delete (or permanently delete) a cluster."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        self._client().clusters.delete(cluster_id=cid, permanent=permanent)
        self._cluster_details_cache.pop(cid, None)

    def edit(
        self,
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        **cluster_spec: Any,
    ) -> "ClusterDetails":
        """Edit an existing cluster with the provided specification."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        cluster_spec.setdefault("cluster_id", cid)
        details = self._client().clusters.edit_and_wait(**cluster_spec)
        self._cluster_details_cache[cid] = details
        return details

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def get_metadata(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> Dict[str, str]:
        """Return metadata stored in the cluster's custom tags."""

        details = self._load_cluster(cluster_id, cluster_name)
        tags = getattr(details, "custom_tags", None) or {}
        return self._metadata_from_tags(tags)

    def update_metadata(
        self,
        metadata: Dict[str, str],
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "ClusterDetails":
        """Merge new metadata into the cluster's custom tags."""

        existing = self.get_metadata(cluster_id, cluster_name)
        merged = {**existing, **metadata}

        details = self._load_cluster(cluster_id, cluster_name)
        payload = self._with_metadata_tags(details, merged)
        updated = self._client().clusters.edit_and_wait(**payload)
        self._cluster_details_cache[
            self._resolve_cluster_id(cluster_id, cluster_name)
        ] = updated
        return updated

    def replace_metadata(
        self,
        metadata: Dict[str, str],
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "ClusterDetails":
        """Replace all stored metadata with the provided mapping."""

        details = self._load_cluster(cluster_id, cluster_name)
        payload = self._with_metadata_tags(details, metadata)
        updated = self._client().clusters.edit_and_wait(**payload)
        self._cluster_details_cache[
            self._resolve_cluster_id(cluster_id, cluster_name)
        ] = updated
        return updated

    def remove_metadata_keys(
        self,
        keys: Dict[str, Any] | list[str],
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "ClusterDetails":
        """Remove specified metadata keys from the cluster tags."""

        if isinstance(keys, dict):
            keys = list(keys.keys())

        existing = self.get_metadata(cluster_id, cluster_name)
        for key in keys:
            existing.pop(key, None)

        details = self._load_cluster(cluster_id, cluster_name)
        payload = self._with_metadata_tags(details, existing)
        updated = self._client().clusters.edit_and_wait(**payload)
        self._cluster_details_cache[
            self._resolve_cluster_id(cluster_id, cluster_name)
        ] = updated
        return updated

    # ------------------------------------------------------------------ #
    # Library management
    # ------------------------------------------------------------------ #
    def install_python_libraries(
        self,
        packages: Iterable[str],
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> None:
        """Install PyPI libraries on the cluster."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        payload = self._library_payload(packages)
        self._client().libraries.install(cluster_id=cid, libraries=payload)

    def uninstall_python_libraries(
        self,
        packages: Iterable[str],
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> None:
        """Uninstall PyPI libraries from the cluster."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        payload = self._library_payload(packages)
        self._client().libraries.uninstall(cluster_id=cid, libraries=payload)

    # ------------------------------------------------------------------ #
    # Runtime helpers
    # ------------------------------------------------------------------ #
    def update_runtime_version(
        self,
        runtime_version: str | "SparkVersion",
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "ClusterDetails":
        """Update the cluster to a specific Databricks runtime version."""

        details = self._load_cluster(cluster_id, cluster_name)
        payload = self._payload_from_details(details)
        payload["spark_version"] = (
            self._runtime_key(runtime_version)
            if not isinstance(runtime_version, str)
            else runtime_version
        )
        updated = self._client().clusters.edit_and_wait(**payload)
        self._cluster_details_cache[
            self._resolve_cluster_id(cluster_id, cluster_name)
        ] = updated
        return updated

    def update_runtime_by_python_version(
        self,
        python_version: str,
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
    ) -> "ClusterDetails":
        """Select and apply a runtime matching the requested Python version."""

        spark_version = self._select_spark_version_for_python(python_version)
        return self.update_runtime_version(
            spark_version, cluster_id=cluster_id, cluster_name=cluster_name
        )

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def exists(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> bool:
        """Check whether a cluster exists."""

        try:
            self._load_cluster(cluster_id, cluster_name)
            return True
        except ResourceDoesNotExist:
            return False

    def start(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> "ClusterDetails":
        """Start a cluster and update cached details."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        started = self._client().clusters.start_and_wait(cluster_id=cid)
        self._cluster_details_cache[cid] = started

        if self.info and self.info.cluster_id == cid:
            self.info.state = getattr(started, "state", None)
            self.info.details = started

        return started

    def terminate(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> None:
        """Terminate a cluster without permanent deletion."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        self._client().clusters.delete(cluster_id=cid, permanent=False)
        self._cluster_details_cache.pop(cid, None)

        if self.info and self.info.cluster_id == cid:
            self.info.state = "TERMINATED"
            self.info.details = None

    def restart(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> "ClusterDetails":
        """Restart a cluster and refresh cached details."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        restarted = self._client().clusters.restart_and_wait(cluster_id=cid)
        self._cluster_details_cache[cid] = restarted

        if self.info and self.info.cluster_id == cid:
            self.info.state = getattr(restarted, "state", None)
            self.info.details = restarted

        return restarted

    def check_started(
        self, cluster_id: Optional[str] = None, cluster_name: Optional[str] = None
    ) -> "ClusterDetails":
        """Ensure the cluster is running, starting it when needed."""

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
        details = self._load_cluster(cid)
        state = (getattr(details, "state", None) or "").upper()
        if state == "RUNNING":
            return details

        started = self._client().clusters.start_and_wait(cluster_id=cid)
        self._cluster_details_cache[cid] = started

        if self.info and self.info.cluster_id == cid:
            self.info.state = getattr(started, "state", None)
            self.info.details = started

        return started

    def reset_cached_clusters(self) -> None:
        """Clear any cached cluster details."""

        self._cluster_details_cache.clear()

    def reset_cached_spark_versions(self) -> None:
        """Clear any cached spark version information."""

        self._spark_versions_cache = None

    def reset_cache(self) -> None:
        """Clear all internal caches maintained by the helper."""

        self.reset_cached_clusters()
        self.reset_cached_spark_versions()

    # ------------------------------------------------------------------ #
    # Command execution
    # ------------------------------------------------------------------ #
    def execute_command(
        self,
        command: str,
        *,
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
        language: "Language" | None = None,
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
        cluster_id:
            Optional override for the target cluster identifier.
        cluster_name:
            Optional cluster name to resolve the target when ``cluster_id``
            is not provided.
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
        cid = self._resolve_cluster_id(cluster_id, cluster_name)
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
        cluster_id: Optional[str] = None,
        cluster_name: Optional[str] = None,
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
        cluster_id:
            Optional override for the target cluster identifier.
        cluster_name:
            Optional cluster name to resolve the target when ``cluster_id``
            is not provided.
        overwrite:
            Whether to replace an existing file. When ``False`` and the file
            exists remotely, a ``FileExistsError`` is raised by the remote
            command.

        Returns
        -------
        str
            The ``remote_path`` after a successful write.
        """

        cid = self._resolve_cluster_id(cluster_id, cluster_name)
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

        return self.execute_command(command, cluster_id=cid)
