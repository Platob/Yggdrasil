from __future__ import annotations

import datetime as dt
import logging
import shutil
import subprocess
from collections import OrderedDict
from functools import partial
from pathlib import Path
from threading import Lock

from fastapi.concurrency import run_in_threadpool

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ...schemas.environment import (
    EnvironmentCreate,
    EnvironmentEntry,
    EnvironmentListResponse,
    EnvironmentResponse,
    EnvironmentUpdate,
    InstallRequest,
)

LOGGER = logging.getLogger(__name__)


class EnvironmentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._envs: OrderedDict[int, EnvironmentEntry] = OrderedDict()
        self._lock = Lock()
        self._envs_root = settings.node_home / "envs"
        self._envs_root.mkdir(parents=True, exist_ok=True)

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: EnvironmentCreate) -> EnvironmentResponse:
        return await self.create_or_update(req)

    async def create_or_update(self, req: EnvironmentCreate) -> EnvironmentResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        with self._lock:
            existing = next(
                (e for e in self._envs.values() if e.name == req.name), None
            )

        if existing:
            new_deps = [d for d in req.dependencies if d not in existing.dependencies]
            updates: dict = {"updated_at": now}
            if new_deps:
                updates["dependencies"] = list(existing.dependencies) + new_deps

            updated = existing.model_copy(update=updates)
            with self._lock:
                self._envs[existing.id] = updated

            if new_deps:
                await self._run(self._install_packages, existing.id, new_deps)
                with self._lock:
                    updated = self._envs.get(existing.id, updated)

            LOGGER.info("Upserted environment %r (name=%r, mode=update)", existing.id, req.name)
            return EnvironmentResponse(environment=updated)
        else:
            env_id = make_id(req.name)
            env_path = self._envs_root / str(env_id)

            entry = EnvironmentEntry(
                id=env_id,
                name=req.name,
                python_version=req.python_version,
                dependencies=list(req.dependencies),
                path=str(env_path),
                status="creating",
                created_at=now,
                updated_at=now,
            )

            with self._lock:
                self._envs[env_id] = entry
                self._evict()

            LOGGER.info("Upserted environment %r (name=%r, mode=create, python=%s)", env_id, req.name, req.python_version)

            await self._run(self._build_env, env_id, req.python_version, list(req.dependencies), env_path)

            with self._lock:
                entry = self._envs.get(env_id, entry)
            return EnvironmentResponse(environment=entry)

    async def get(self, env_id: int) -> EnvironmentEntry:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"Environment {env_id!r} not found")
        return entry

    async def list(self) -> EnvironmentListResponse:
        with self._lock:
            items = list(self._envs.values())
        return EnvironmentListResponse(
            node_id=self.settings.node_id,
            environments=items,
        )

    async def update(self, env_id: int, req: EnvironmentUpdate) -> EnvironmentResponse:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"Environment {env_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        updates: dict = {"updated_at": now}
        if req.name is not None:
            updates["name"] = req.name

        updated = entry.model_copy(update=updates)
        with self._lock:
            self._envs[env_id] = updated

        if req.dependencies is not None:
            await self._run(self._install_packages, env_id, list(req.dependencies))
            with self._lock:
                updated = self._envs.get(env_id, updated)

        LOGGER.info("Updated environment %r", env_id)
        return EnvironmentResponse(environment=updated)

    async def delete(self, env_id: int) -> EnvironmentResponse:
        with self._lock:
            entry = self._envs.pop(env_id, None)
        if entry is None:
            raise NotFoundError(f"Environment {env_id!r} not found")

        env_path = self._envs_root / str(env_id)
        if env_path.exists():
            shutil.rmtree(env_path, ignore_errors=True)

        LOGGER.info("Deleted environment %r", env_id)
        return EnvironmentResponse(environment=entry)

    async def install(self, env_id: int, req: InstallRequest) -> EnvironmentResponse:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"Environment {env_id!r} not found")

        await self._run(self._install_packages, env_id, list(req.packages))

        with self._lock:
            entry = self._envs.get(env_id, entry)
        return EnvironmentResponse(environment=entry)

    async def clone(self, env_id: int, new_name: str | None = None) -> EnvironmentResponse:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            raise NotFoundError(f"Environment {env_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        clone_name = new_name or f"{entry.name}_clone"
        clone_id = make_id(clone_name)
        clone_path = self._envs_root / str(clone_id)

        clone_entry = EnvironmentEntry(
            id=clone_id,
            name=clone_name,
            python_version=entry.python_version,
            dependencies=list(entry.dependencies),
            path=str(clone_path),
            status="creating",
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._envs[clone_id] = clone_entry
            self._evict()

        LOGGER.info("Cloning environment %r -> %r (name=%r)", env_id, clone_id, clone_name)

        await self._run(self._clone_venv, env_id, clone_id, clone_path)

        with self._lock:
            clone_entry = self._envs.get(clone_id, clone_entry)
        return EnvironmentResponse(environment=clone_entry)

    def _clone_venv(self, source_id: int, dest_id: int, dest_path) -> None:
        try:
            source_path = self._envs_root / str(source_id)
            if source_path.exists():
                shutil.copytree(str(source_path), str(dest_path))
            else:
                with self._lock:
                    entry = self._envs.get(dest_id)
                if entry is not None:
                    self._build_env(dest_id, entry.python_version, list(entry.dependencies), Path(entry.path))
                return

            self._update_entry(dest_id, status="ready")
            LOGGER.info("Cloned environment venv %r ready", dest_id)
        except Exception as exc:
            self._update_entry(dest_id, status="failed", error=str(exc))
            LOGGER.error("Environment clone %r failed: %s", dest_id, exc)

    def get_python_path(self, env_id: int) -> str | None:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            return None
        if entry.status != "ready":
            return None
        from . import venv_python
        python = venv_python(entry.path)
        if Path(python).exists():
            return python
        return None

    # -- internals ----------------------------------------------------------

    def _build_env(self, env_id: int, python_version: str, dependencies: list[str], env_path) -> None:
        try:
            uv = shutil.which("uv")
            if uv:
                subprocess.run(
                    [uv, "venv", "--python", python_version, str(env_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                import sys
                subprocess.run(
                    [sys.executable, "-m", "venv", str(env_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            from pathlib import Path as _Path
            base_packages: list[str] = ["uv"]
            ygg_root = _Path(__file__).resolve().parents[5]
            if (ygg_root / "pyproject.toml").exists():
                base_packages.append(str(ygg_root))
            try:
                self._pip_install(env_id, env_path, base_packages, uv=uv)
            except subprocess.CalledProcessError:
                LOGGER.warning("Base package install failed for env %r, continuing", env_id)

            if dependencies:
                self._pip_install(env_id, env_path, dependencies, uv=uv)

            self._update_entry(env_id, status="ready")
            LOGGER.info("Environment %r ready", env_id)

        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or str(exc)
            self._update_entry(env_id, status="failed", error=error_msg)
            LOGGER.error("Environment %r creation failed: %s", env_id, error_msg)
        except Exception as exc:
            self._update_entry(env_id, status="failed", error=str(exc))
            LOGGER.error("Environment %r creation failed: %s", env_id, exc)

    def _install_packages(self, env_id: int, packages: list[str]) -> None:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None:
            return

        env_path = Path(entry.path)
        uv = shutil.which("uv")

        try:
            self._pip_install(env_id, env_path, packages, uv=uv)
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            with self._lock:
                current = self._envs.get(env_id)
                if current is not None:
                    merged = list(current.dependencies) + [p for p in packages if p not in current.dependencies]
                    self._envs[env_id] = current.model_copy(
                        update={"dependencies": merged, "updated_at": now}
                    )
            LOGGER.info("Installed %d packages into environment %r", len(packages), env_id)
        except subprocess.CalledProcessError as exc:
            error_msg = exc.stderr or str(exc)
            self._update_entry(env_id, error=error_msg)
            LOGGER.error("Package install failed for env %r: %s", env_id, error_msg)

    def _pip_install(self, env_id: int, env_path, packages: list[str], *, uv: str | None) -> None:
        from . import venv_python
        python_bin = venv_python(env_path)
        if uv:
            cmd = [uv, "pip", "install", "--python", python_bin] + packages
        else:
            cmd = [python_bin, "-m", "pip", "install"] + packages

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _update_entry(self, env_id: int, **updates) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        updates["updated_at"] = now
        with self._lock:
            entry = self._envs.get(env_id)
            if entry is not None:
                self._envs[env_id] = entry.model_copy(update=updates)

    def _evict(self) -> None:
        while len(self._envs) > self.settings.max_environments:
            evicted_id, evicted = self._envs.popitem(last=False)
            evicted_path = self._envs_root / str(evicted_id)
            if evicted_path.exists():
                shutil.rmtree(evicted_path, ignore_errors=True)
