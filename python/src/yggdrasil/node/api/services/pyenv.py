from __future__ import annotations

import datetime as dt
import hashlib
import logging
import shutil
import subprocess
from functools import partial
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import httpx
from fastapi.concurrency import run_in_threadpool

from yggdrasil.dataclasses.expiring import ExpiringDict

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.pyenv import (
    PyEnvCreate,
    PyEnvEntry,
    PyEnvListResponse,
    PyEnvPackage,
    PyEnvPackagesResponse,
    PyEnvResponse,
    PyEnvUpdate,
)

if TYPE_CHECKING:
    from .audit import AuditLog

LOGGER = logging.getLogger(__name__)


class PyEnvService:
    def __init__(self, settings: Settings, *, audit: AuditLog | None = None) -> None:
        self.settings = settings
        self._envs: ExpiringDict[int, PyEnvEntry] = ExpiringDict(default_ttl=None, max_size=settings.max_environments)
        self._name_to_id: dict[str, int] = {}
        self._lock = Lock()
        self._envs_root = settings.node_home / "envs"
        self._envs_root.mkdir(parents=True, exist_ok=True)
        self._audit = audit
        # Resolved python version + installed-library listing per env,
        # TTL-cached so repeated UI polls don't re-spawn ``pip list``.
        self._packages_cache: ExpiringDict[int, PyEnvPackagesResponse] = ExpiringDict(
            default_ttl=settings.pyenv_packages_cache_ttl,
            max_size=settings.max_environments,
        )

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: PyEnvCreate) -> PyEnvResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._lock:
            existing_id = self._name_to_id.get(req.name)
            existing = self._envs.get(existing_id) if existing_id is not None else None

        if existing:
            new_deps = [d for d in req.dependencies if d not in existing.dependencies]
            updates: dict = {"updated_at": now}
            if new_deps:
                updates["dependencies"] = list(existing.dependencies) + new_deps
            merged_deps = updates.get("dependencies", list(existing.dependencies))
            updates["content_hash"] = hashlib.sha256(
                (existing.name + existing.python_version + "".join(sorted(merged_deps))).encode()
            ).hexdigest()
            updated = existing.model_copy(update=updates)
            with self._lock:
                self._envs[existing.id] = updated
            if new_deps:
                await self._run(self._install_packages, existing.id, new_deps)
                with self._lock:
                    updated = self._envs.get(existing.id, updated)
            if self._audit is not None:
                self._audit.log("update", "pyenv", existing.id, detail=f"name={req.name}")
            return PyEnvResponse(env=updated)

        env_id = make_id(req.name)
        env_path = self._envs_root / str(env_id)
        content_hash = hashlib.sha256(
            (req.name + req.python_version + "".join(sorted(req.dependencies))).encode()
        ).hexdigest()
        entry = PyEnvEntry(
            id=env_id,
            name=req.name,
            python_version=req.python_version,
            dependencies=list(req.dependencies),
            path=str(env_path),
            status="creating",
            created_at=now,
            updated_at=now,
            content_hash=content_hash,
        )
        with self._lock:
            self._envs.set(env_id, entry)
            self._name_to_id[req.name] = env_id

        await self._run(
            self._build_env, env_id, req.python_version,
            list(req.dependencies), env_path,
        )
        with self._lock:
            entry = self._envs.get(env_id, entry)
        if self._audit is not None:
            self._audit.log("create", "pyenv", env_id, detail=f"name={req.name}")
        return PyEnvResponse(env=entry)

    async def get(self, env_id: int) -> PyEnvEntry:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"PyEnv {env_id!r} not found")
        return entry

    async def list(self) -> PyEnvListResponse:
        with self._lock:
            items = list(self._envs.values())
        return PyEnvListResponse(node_id=self.settings.node_id, envs=items)

    async def update(self, env_id: int, req: PyEnvUpdate) -> PyEnvResponse:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"PyEnv {env_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        updates: dict = {"updated_at": now}
        if req.name is not None:
            updates["name"] = req.name
        hash_name = updates.get("name", entry.name)
        hash_deps = list(entry.dependencies)
        updates["content_hash"] = hashlib.sha256(
            (hash_name + entry.python_version + "".join(sorted(hash_deps))).encode()
        ).hexdigest()
        updated = entry.model_copy(update=updates)
        with self._lock:
            self._envs[env_id] = updated
            if req.name is not None and req.name != entry.name:
                self._name_to_id.pop(entry.name, None)
                self._name_to_id[req.name] = env_id

        if req.dependencies is not None:
            await self._run(self._install_packages, env_id, list(req.dependencies))
            with self._lock:
                updated = self._envs.get(env_id, updated)
        return PyEnvResponse(env=updated)

    async def delete(self, env_id: int) -> PyEnvResponse:
        with self._lock:
            entry = self._envs.pop(env_id, None)
            if entry is not None:
                self._name_to_id.pop(entry.name, None)
        if entry is None:
            raise NotFoundError(f"PyEnv {env_id!r} not found")
        self._packages_cache.pop(env_id, None)
        env_path = self._envs_root / str(env_id)
        if env_path.exists():
            shutil.rmtree(env_path, ignore_errors=True)
        if self._audit is not None:
            self._audit.log("delete", "pyenv", env_id, detail=f"name={entry.name}")
        return PyEnvResponse(env=entry)

    async def packages(self, env_id: int, *, refresh: bool = False) -> PyEnvPackagesResponse:
        """Resolved interpreter version + libraries installed in the env.

        Served from a TTL cache (``pyenv_packages_cache_ttl``): the
        expensive ``uv pip list`` / ``pip list`` subprocess only re-runs
        once the cached listing expires, so a dashboard polling every few
        seconds doesn't spawn a process per request. Pass ``refresh=True``
        to force a fresh read (e.g. right after an install).
        """
        entry = await self.get(env_id)  # raises NotFoundError if missing
        if not refresh:
            cached = self._packages_cache.get(env_id)
            if cached is not None:
                return cached
        result = await self._run(self._collect_packages, env_id, entry)
        self._packages_cache.set(env_id, result)
        return result

    async def packages_by_name(self, name: str, *, refresh: bool = False) -> PyEnvPackagesResponse:
        """Name-keyed variant of :meth:`packages`.

        PyEnvs are upserted by name, so the name is a stable string
        identifier — handy for clients (the web UI) that can't carry the
        int64 ``env_id`` losslessly through JSON.
        """
        with self._lock:
            env_id = self._name_to_id.get(name)
        if env_id is None:
            raise NotFoundError(f"PyEnv {name!r} not found")
        return await self.packages(env_id, refresh=refresh)

    def _collect_packages(self, env_id: int, entry: PyEnvEntry) -> PyEnvPackagesResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        env_path = Path(entry.path)
        python_bin = env_path / "bin" / "python"
        python_version = entry.python_version
        packages: list[PyEnvPackage] = []
        error: str | None = None

        if not python_bin.exists():
            return PyEnvPackagesResponse(
                env_id=env_id, name=entry.name, python_version=python_version,
                package_count=0, packages=[], cached_at=now,
                error=f"interpreter not found at {python_bin}",
            )

        # Resolve the real interpreter version (e.g. ``3.11.9``) rather
        # than the declared major.minor request (e.g. ``3.11``).
        try:
            proc = subprocess.run(
                [str(python_bin), "-c",
                 "import sys;print('.'.join(map(str, sys.version_info[:3])))"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            python_version = proc.stdout.strip() or python_version
        except (subprocess.SubprocessError, OSError):
            pass

        uv = shutil.which("uv")
        try:
            if uv:
                cmd = [uv, "pip", "list", "--python", str(python_bin), "--format", "json"]
            else:
                cmd = [str(python_bin), "-m", "pip", "list", "--format", "json"]
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
            import json
            for item in json.loads(proc.stdout or "[]"):
                packages.append(PyEnvPackage(name=item["name"], version=item["version"]))
            packages.sort(key=lambda p: p.name.lower())
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or str(exc)).strip()
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            error = str(exc)

        return PyEnvPackagesResponse(
            env_id=env_id, name=entry.name, python_version=python_version,
            package_count=len(packages), packages=packages, cached_at=now,
            error=error,
        )

    def get_python_path(self, env_id: int) -> str | None:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None or entry.status != "ready":
            return None
        python = Path(entry.path) / "bin" / "python"
        return str(python) if python.exists() else None

    # -- replication --------------------------------------------------------

    async def replicate_to(self, env_id: int, target_url: str) -> dict:
        """Replicate a PyEnv to a remote node by POSTing its data."""
        entry = await self.get(env_id)
        payload = {
            "name": entry.name,
            "python_version": entry.python_version,
            "dependencies": list(entry.dependencies),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{target_url.rstrip('/')}/api/v2/pyenv",
                json=payload,
                headers={"X-YGG-Source-Node": self.settings.node_id},
            )
            resp.raise_for_status()
            return resp.json()

    # -- internals ----------------------------------------------------------

    def _build_env(
        self, env_id: int, python_version: str,
        dependencies: list[str], env_path: Path,
    ) -> None:
        try:
            uv = shutil.which("uv")
            if uv:
                subprocess.run(
                    [uv, "venv", "--python", python_version, str(env_path)],
                    check=True, capture_output=True, text=True,
                )
            else:
                import sys
                subprocess.run(
                    [sys.executable, "-m", "venv", str(env_path)],
                    check=True, capture_output=True, text=True,
                )

            # Install yggdrasil from project root if available (mirrors v1 EnvironmentService)
            ygg_root = Path(__file__).resolve().parents[5]
            base_packages: list[str] = []
            if (ygg_root / "pyproject.toml").exists():
                base_packages.append(str(ygg_root))
            if base_packages:
                try:
                    self._pip_install(env_path, base_packages, uv=uv)
                except subprocess.CalledProcessError:
                    # Non-fatal: base package install failure should not block env creation
                    LOGGER.warning("Base package install failed for env %r, continuing", env_id)

            if dependencies:
                self._pip_install(env_path, dependencies, uv=uv)
            self._update_entry(env_id, status="ready")
            # Env is now on disk with its packages — drop any stale
            # (empty / interpreter-not-found) cached listing.
            self._packages_cache.pop(env_id, None)
        except subprocess.CalledProcessError as exc:
            self._update_entry(env_id, status="failed", error=exc.stderr or str(exc))
        except Exception as exc:
            self._update_entry(env_id, status="failed", error=str(exc))

    def _install_packages(self, env_id: int, packages: list[str]) -> None:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            return
        env_path = Path(entry.path)
        uv = shutil.which("uv")
        # The installed set is about to change either way — invalidate the
        # cached library listing so the next ``packages()`` re-collects.
        self._packages_cache.pop(env_id, None)
        try:
            self._pip_install(env_path, packages, uv=uv)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            with self._lock:
                current = self._envs.get(env_id)
                if current is not None:
                    merged = list(current.dependencies) + [
                        p for p in packages if p not in current.dependencies
                    ]
                    self._envs[env_id] = current.model_copy(
                        update={"dependencies": merged, "updated_at": now}
                    )
        except subprocess.CalledProcessError as exc:
            self._update_entry(env_id, error=exc.stderr or str(exc))

    def _pip_install(self, env_path: Path, packages: list[str], *, uv: str | None) -> None:
        python_bin = str(env_path / "bin" / "python")
        if uv:
            cmd = [uv, "pip", "install", "--python", python_bin] + packages
        else:
            cmd = [python_bin, "-m", "pip", "install"] + packages
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _update_entry(self, env_id: int, **updates) -> None:
        updates["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            entry = self._envs.get(env_id)
            if entry is not None:
                self._envs[env_id] = entry.model_copy(update=updates)

    def _touch(self, env_id: int) -> None:
        self._update_entry(env_id, last_used_at=dt.datetime.now(dt.timezone.utc).isoformat())
