from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class PyEnvCreate(StrictModel):
    name: str
    python_version: str = "3.11"
    dependencies: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)


class PyEnvUpdate(StrictModel):
    name: str | None = None
    dependencies: list[str] | None = None
    env_vars: dict[str, str] | None = None


class PyEnvEntry(StrictModel):
    id: int
    name: str
    python_version: str
    dependencies: list[str] = Field(default_factory=list)
    # Environment variables applied to every run in this env (merged
    # over the node environment). Config, not part of the content hash.
    env_vars: dict[str, str] = Field(default_factory=dict)
    path: str
    status: str = "pending"
    created_at: str
    updated_at: str
    error: str | None = None
    last_used_at: str | None = None
    content_hash: str = ""
    replicated_at: str | None = None
    replicated_from: str | None = None


class PyEnvEnvVarsUpdate(StrictModel):
    """Set environment variables on a PyEnv. ``replace`` swaps the whole
    map; otherwise the given keys are merged over the existing ones."""
    env_vars: dict[str, str] = Field(default_factory=dict)
    replace: bool = False


class PyEnvEnvVarsResponse(StrictModel):
    env_id: int
    name: str
    env_vars: dict[str, str] = Field(default_factory=dict)


class PyEnvResponse(StrictModel):
    env: PyEnvEntry


class PyEnvListResponse(StrictModel):
    node_id: str
    envs: list[PyEnvEntry]


class PyEnvPackage(StrictModel):
    """A single library installed in a PyEnv's venv."""
    name: str
    version: str


class PyEnvPackagesResponse(StrictModel):
    """Resolved interpreter version + the libraries actually installed in
    a PyEnv's venv (``uv pip list`` / ``pip list``). Distinct from the
    declared ``dependencies`` on :class:`PyEnvEntry`: this is the full
    transitive set as it exists on disk. Served from a TTL cache so
    repeated UI polls don't re-spawn the listing subprocess."""
    env_id: int
    name: str
    python_version: str
    package_count: int
    packages: list[PyEnvPackage] = Field(default_factory=list)
    cached_at: str
    error: str | None = None
