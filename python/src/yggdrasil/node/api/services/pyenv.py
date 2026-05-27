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
        self._lock = Lock()
        self._envs_root = settings.node_home / "envs"
        self._envs_root.mkdir(parents=True, exist_ok=True)
        self._audit = audit

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: PyEnvCreate) -> PyEnvResponse:
        now = self._now()

        with self._lock:
            existing = next(
                (e for e in self._envs.values() if e.name == req.name), None
            )

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

        now = self._now()
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

        if req.dependencies is not None:
            await self._run(self._install_packages, env_id, list(req.dependencies))
            with self._lock:
                updated = self._envs.get(env_id, updated)
        return PyEnvResponse(env=updated)

    async def delete(self, env_id: int) -> PyEnvResponse:
        with self._lock:
            entry = self._envs.pop(env_id, None)
        if entry is None:
            raise NotFoundError(f"PyEnv {env_id!r} not found")
        env_path = self._envs_root / str(env_id)
        if env_path.exists():
            shutil.rmtree(env_path, ignore_errors=True)
        if self._audit is not None:
            self._audit.log("delete", "pyenv", env_id, detail=f"name={entry.name}")
        return PyEnvResponse(env=entry)

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
        try:
            self._pip_install(env_path, packages, uv=uv)
            now = self._now()
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
        updates["updated_at"] = self._now()
        with self._lock:
            entry = self._envs.get(env_id)
            if entry is not None:
                self._envs[env_id] = entry.model_copy(update=updates)

    def _touch(self, env_id: int) -> None:
        self._update_entry(env_id, last_used_at=self._now())

    @staticmethod
    def _now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()
