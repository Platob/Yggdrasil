from __future__ import annotations

import datetime as dt
import hashlib
import logging
import shutil
import subprocess
from functools import partial
from pathlib import Path
from threading import Lock

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

LOGGER = logging.getLogger(__name__)


class PyEnvService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._envs: ExpiringDict[int, PyEnvEntry] = ExpiringDict(default_ttl=None, max_size=settings.max_environments)
        self._lock = Lock()
        self._envs_root = settings.node_home / "envs"
        self._envs_root.mkdir(parents=True, exist_ok=True)

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
        return PyEnvResponse(env=entry)

    # -- execute dispatch ---------------------------------------------------

    async def execute_pyfunc(
        self,
        env_id: int,
        func_code: str,
        args: list,
        kwargs: dict,
        *,
        timeout: float | None = None,
        max_memory_mb: int | None = None,
    ) -> dict:
        """Execute a PyFunc within this environment. Inner dispatch."""
        entry = await self.get(env_id)
        python_bin = self.get_python_path(env_id)
        if python_bin is None:
            raise NotFoundError(f"PyEnv {env_id!r} not ready (status={entry.status})")

        self._touch(env_id)
        return await self._run(
            self._exec_in_env, python_bin, func_code, args, kwargs,
            timeout or self.settings.max_python_timeout, max_memory_mb,
        )

    def get_python_path(self, env_id: int) -> str | None:
        with self._lock:
            entry = self._envs.get(env_id)
        if entry is None or entry.status != "ready":
            return None
        python = Path(entry.path) / "bin" / "python"
        return str(python) if python.exists() else None

    # -- internals ----------------------------------------------------------

    def _exec_in_env(
        self,
        python_bin: str,
        func_code: str,
        args: list,
        kwargs: dict,
        timeout: float,
        max_memory_mb: int | None,
    ) -> dict:
        import json
        import os
        import platform
        import resource as resource_mod
        import tempfile
        import time

        t0 = time.monotonic()
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        try:
            preamble = (
                "import json as _json\n"
                f"_args = _json.loads({json.dumps(json.dumps(args))!r})\n"
                f"_kwargs = _json.loads({json.dumps(json.dumps(kwargs))!r})\n"
            )
            tmp.write(preamble + func_code)
            tmp.flush()
            tmp.close()

            preexec = None
            if max_memory_mb and platform.system() == "Linux":
                mem_bytes = max_memory_mb * 1024 * 1024

                def _limit():
                    resource_mod.setrlimit(
                        resource_mod.RLIMIT_AS, (mem_bytes, mem_bytes)
                    )

                preexec = _limit

            proc = subprocess.run(
                [python_bin, tmp.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                preexec_fn=preexec,
                env={**os.environ, "YGG_RUNTIME_VERSION": self.settings.app_version},
            )
            duration = round(time.monotonic() - t0, 3)
            return {
                "status": "completed" if proc.returncode == 0 else "failed",
                "returncode": proc.returncode,
                "stdout": proc.stdout or None,
                "stderr": proc.stderr or None,
                "duration": duration,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "returncode": -1,
                "stderr": f"Timed out after {timeout:.0f}s",
                "duration": round(time.monotonic() - t0, 3),
            }
        finally:
            Path(tmp.name).unlink(missing_ok=True)

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
