from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from functools import partial
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool

from yggdrasil.environ.environment import PyEnv

from ..config import Settings
from ..exceptions import APIError, ConflictError, NotFoundError
from ..schemas.python import (
    CommandResult,
    CreateEnvRequest,
    DataFramePayload,
    DeleteEnvRequest,
    DeleteEnvResponse,
    EnvInfo,
    EnvListResponse,
    EnvRefRequest,
    EnvResponse,
    ExcelExecuteRequest,
    ExcelExecuteResponse,
    ExcelPrepareRequest,
    ExcelPrepareResponse,
    ExecuteCodeRequest,
    ExecutionResponse,
    MutationResponse,
    PackageRequest,
    RequirementItem,
    RequirementsResponse,
)


class PythonService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    async def current_env(self, prefer_uv: bool = True) -> EnvResponse:
        env = await self._run(PyEnv.current, prefer_uv=prefer_uv)
        return EnvResponse(env=self._serialize_env(env, identifier="current"))

    async def list_envs(self, prefer_uv: bool = True) -> EnvListResponse:
        current = await self._run(PyEnv.current, prefer_uv=prefer_uv)
        items: list[EnvInfo] = [self._serialize_env(current, identifier="current")]

        for root in await self._run(self._discover_env_roots):
            env = await self._run(self._env_from_root, root, prefer_uv=prefer_uv)
            if env is None:
                continue
            if env.python_path.resolve() == current.python_path.resolve():
                continue
            items.append(self._serialize_env(env, identifier=self._identifier_from_root(root)))

        return EnvListResponse(items=items)

    async def resolve_env(self, req: EnvRefRequest) -> EnvResponse:
        env = await self._run(
            self._resolve_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            create_if_missing=req.create_if_missing,
            version=req.version,
            packages=req.packages,
            seed=req.seed,
        )
        return EnvResponse(env=self._serialize_env(env, identifier=req.identifier))

    async def create_env(self, req: CreateEnvRequest) -> EnvResponse:
        anchor = await self._run(PyEnv.current, prefer_uv=req.prefer_uv)
        env = await self._run(
            anchor.create,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            seed=req.seed,
            version=req.version,
            packages=req.packages,
            linked=req.linked,
            native_tls=req.native_tls,
            clear=req.clear,
        )
        return EnvResponse(env=self._serialize_env(env, identifier=req.identifier))

    async def delete_env(self, req: DeleteEnvRequest) -> DeleteEnvResponse:
        env = await self._run(
            self._resolve_existing_env,
            req.identifier,
            cwd=None,
            prefer_uv=req.prefer_uv,
        )

        if env.python_path.resolve() == Path(sys.executable).resolve():
            raise ConflictError("Refusing to delete the current interpreter environment")

        await self._run(env.delete, req.raise_error)

        return DeleteEnvResponse(
            deleted=True,
            identifier=req.identifier,
            root_path=str(env.root_path),
        )

    async def requirements(
        self,
        *,
        identifier: str | None,
        cwd: str | None,
        prefer_uv: bool,
        with_system: bool,
    ) -> RequirementsResponse:
        env = await self._run(
            self._resolve_existing_env,
            identifier,
            cwd=self._path_or_none(cwd),
            prefer_uv=prefer_uv,
        )
        items = await self._run(
            env.requirements,
            prefer_uv,
            with_system=with_system,
        )
        return RequirementsResponse(
            env=self._serialize_env(env, identifier=identifier),
            requirements=[RequirementItem(name=name, version=version) for name, version in items],
        )

    async def install_packages(self, req: PackageRequest) -> MutationResponse:
        env = await self._run(
            self._resolve_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            create_if_missing=req.create_if_missing,
            version=req.version,
            seed=req.seed,
        )
        result = await self._run(
            env.install,
            *req.packages,
            requirements=req.requirements,
            extra_args=req.extra_args,
            wait=True,
            raise_error=True,
            prefer_uv=req.prefer_uv,
            target=self._path_or_none(req.target),
            break_system_packages=req.break_system_packages,
        )
        return MutationResponse(
            ok=True,
            env=self._serialize_env(env, identifier=req.identifier),
            result=self._serialize_command(result) if result is not None else None,
        )

    async def update_packages(self, req: PackageRequest) -> MutationResponse:
        env = await self._run(
            self._resolve_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            create_if_missing=req.create_if_missing,
            version=req.version,
            seed=req.seed,
        )
        result = await self._run(
            env.update,
            *req.packages,
            extra_args=req.extra_args,
            wait=True,
            prefer_uv=req.prefer_uv,
        )
        return MutationResponse(
            ok=True,
            env=self._serialize_env(env, identifier=req.identifier),
            result=self._serialize_command(result) if result is not None else None,
        )

    async def uninstall_packages(self, req: PackageRequest) -> MutationResponse:
        env = await self._run(
            self._resolve_existing_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
        )
        result = await self._run(
            env.uninstall,
            *req.packages,
            extra_args=req.extra_args,
            wait=True,
            prefer_uv=req.prefer_uv,
        )
        return MutationResponse(
            ok=True,
            env=self._serialize_env(env, identifier=req.identifier),
            result=self._serialize_command(result) if result is not None else None,
        )

    async def execute_code(self, req: ExecuteCodeRequest) -> ExecutionResponse:
        env = await self._run(
            self._resolve_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            create_if_missing=req.create_if_missing,
            version=req.version,
            packages=None,
            seed=req.seed,
        )

        result = await self._run(
            env.run_python_code,
            req.code,
            cwd=self._path_or_none(req.cwd),
            env=req.env,
            wait=True,
            raise_error=req.raise_error,
            stdin=req.stdin,
            python=env,
            packages=req.packages,
            prefer_uv=req.prefer_uv,
            globs=req.globs,
            auto_install=req.auto_install,
        )

        return ExecutionResponse(
            ok=True,
            env=self._serialize_env(env, identifier=req.identifier),
            result=self._serialize_command(result),
        )

    async def execute_excel(self, req: ExcelExecuteRequest) -> ExcelExecuteResponse:
        env = await self._run(
            self._resolve_env,
            req.identifier,
            cwd=self._path_or_none(req.cwd),
            prefer_uv=req.prefer_uv,
            create_if_missing=req.create_if_missing,
            version=req.version,
            packages=None,
            seed=req.seed,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            output_path = Path(tmp.name)

        packages = list(dict.fromkeys([*req.packages, "pandas"]))
        wrapped_code = self._build_excel_capture_code(
            user_code=req.code,
            output_path=output_path,
            df_name=req.df_name,
            max_rows=req.max_rows,
        )

        result = None
        payload: dict[str, Any] = {}

        try:
            result = await self._run(
                env.run_python_code,
                wrapped_code,
                cwd=self._path_or_none(req.cwd),
                env=req.env,
                wait=True,
                raise_error=req.raise_error,
                stdin=req.stdin,
                python=env,
                packages=packages,
                prefer_uv=req.prefer_uv,
                globs=req.globs,
                auto_install=req.auto_install,
            )
            payload = await self._run(self._read_excel_payload, output_path)
        finally:
            try:
                output_path.unlink(missing_ok=True)
            except Exception:
                pass

        if not payload.get("ok"):
            error = payload.get("error") or {}
            error_type = error.get("type", "ExecutionError")
            message = error.get("message", "DataFrame export failed")
            raise APIError(f"{error_type}: {message}", status_code=400)

        data = DataFramePayload.model_validate(payload["data"])
        return ExcelExecuteResponse(
            ok=True,
            env=self._serialize_env(env, identifier=req.identifier),
            data=data,
            result=self._serialize_command(result) if req.include_result else None,
        )

    # ------------------------------------------------------------------
    # Excel prepare (cached parquet execution)
    # ------------------------------------------------------------------

    RUNNER_PATH: Path = Path(__file__).with_name("excel_exec_runner.py")

    async def prepare_excel(self, req: ExcelPrepareRequest) -> ExcelPrepareResponse:
        packages = self._normalize_packages(req.packages)
        identifier = self._safe_name(req.identifier)

        request_fingerprint = {
            "app_version": self.settings.excel_app_version,
            "identifier": identifier,
            "packages": packages,
            "df_name": req.df_name,
            "max_rows": req.max_rows,
            "code": req.code,
        }
        cache_key = self._stable_hash(request_fingerprint)

        run_root = self.settings.excel_run_root
        run_dir = run_root / identifier / cache_key[:2] / cache_key
        result_path = run_dir / "result.parquet"
        manifest_path = run_dir / "manifest.json"
        request_json_path = run_dir / "request.json"
        request_gz_path = run_dir / "request.json.gz"
        stdout_gz_path = run_dir / "stdout.txt.gz"
        stderr_gz_path = run_dir / "stderr.txt.gz"

        # Cache hit
        if not req.force_refresh and result_path.exists() and manifest_path.exists():
            manifest = await self._run(
                lambda p: json.loads(p.read_text(encoding="utf-8")), manifest_path
            )
            return ExcelPrepareResponse(
                ok=True,
                data={**manifest, "cache_hit": True, "cache_key": cache_key},
            )

        await self._run(lambda d: d.mkdir(parents=True, exist_ok=True), run_dir)

        python_exe = await self._run(
            self._ensure_isolated_env, packages, req.prefer_uv
        )

        runner_payload = {
            "code": req.code,
            "df_name": req.df_name,
            "max_rows": req.max_rows,
        }

        payload_bytes = json.dumps(runner_payload, ensure_ascii=False).encode("utf-8")
        await self._run(request_json_path.write_bytes, payload_bytes)
        await self._run(
            request_gz_path.write_bytes,
            gzip.compress(payload_bytes, compresslevel=5),
        )

        proc = await self._run(
            subprocess.run,
            [
                str(python_exe),
                str(self.RUNNER_PATH),
                str(request_json_path),
                str(result_path),
                str(manifest_path),
            ],
            capture_output=True,
            text=True,
        )

        await self._run(
            stdout_gz_path.write_bytes,
            gzip.compress(proc.stdout.encode("utf-8"), compresslevel=5),
        )
        await self._run(
            stderr_gz_path.write_bytes,
            gzip.compress(proc.stderr.encode("utf-8"), compresslevel=5),
        )

        if proc.returncode != 0:
            raise APIError(
                detail=(
                    f"Python execution failed (exit {proc.returncode}). "
                    f"stderr: {str(stderr_gz_path)}"
                ),
                status_code=500,
            )

        manifest = await self._run(
            lambda p: json.loads(p.read_text(encoding="utf-8")), manifest_path
        )
        return ExcelPrepareResponse(
            ok=True,
            data={**manifest, "cache_hit": False, "cache_key": cache_key},
        )

    # ------------------------------------------------------------------
    # Excel-prepare helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_packages(values: list[str]) -> list[str]:
        cleaned = {v.strip() for v in values if v and v.strip()}
        cleaned.update({"pandas", "pyarrow"})
        return sorted(cleaned)

    @staticmethod
    def _stable_hash(obj: Any) -> str:
        raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _safe_name(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "default"

    @staticmethod
    def _python_exe_from_env(env_dir: Path) -> Path:
        if os.name == "nt":
            return env_dir / "Scripts" / "python.exe"
        return env_dir / "bin" / "python"

    def _ensure_isolated_env(self, packages: list[str], prefer_uv: bool) -> Path:
        env_key = self._stable_hash({"packages": packages})
        env_dir = self.settings.excel_env_root / env_key
        python_exe = self._python_exe_from_env(env_dir)

        if python_exe.exists():
            return python_exe

        env_dir.parent.mkdir(parents=True, exist_ok=True)

        if prefer_uv and shutil.which("uv"):
            subprocess.run(["uv", "venv", str(env_dir)], check=True)
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python_exe), *packages],
                check=True,
            )
            return python_exe

        venv.EnvBuilder(with_pip=True).create(env_dir)
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [str(python_exe), "-m", "pip", "install", *packages],
            check=True,
        )
        return python_exe

    def _path_or_none(self, value: str | None) -> Path | None:
        if not value:
            return None
        return Path(value).expanduser().resolve()

    def _version_to_str(self, version_info: Any) -> str | None:
        if version_info is None:
            return None
        major = getattr(version_info, "major", None)
        minor = getattr(version_info, "minor", None)
        patch = getattr(version_info, "patch", None)
        if major is not None and minor is not None and patch is not None:
            return f"{major}.{minor}.{patch}"
        return str(version_info)

    def _jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, bytearray):
            return bytes(value).decode("utf-8", errors="replace")
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonable(v) for v in value]
        return str(value)

    def _serialize_command(self, cmd: Any) -> CommandResult:
        data: dict[str, Any] = {"type": type(cmd).__name__}

        for key in (
            "returncode",
            "exit_code",
            "stdout",
            "stderr",
            "output",
            "cwd",
            "duration",
            "cmd",
            "args",
            "command",
        ):
            value = getattr(cmd, key, None)
            if value is not None:
                data[key] = self._jsonable(value)

        popen = getattr(cmd, "popen", None)
        if popen is not None:
            data.setdefault("returncode", getattr(popen, "returncode", None))
            data.setdefault("pid", getattr(popen, "pid", None))

        result = getattr(cmd, "result", None)
        if result is not None:
            for key in ("returncode", "exit_code", "stdout", "stderr", "output"):
                if key not in data:
                    value = getattr(result, key, None)
                    if value is not None:
                        data[key] = self._jsonable(value)

        return CommandResult.model_validate(data)

    def _identifier_from_root(self, root: Path) -> str:
        try:
            return str(root.relative_to(self.settings.env_home))
        except Exception:
            return str(root)

    def _env_label(self, identifier: str | None, env: PyEnv) -> str:
        if not identifier or identifier.strip().lower() in {"current", "sys", "system"}:
            return "current"
        try:
            return str(env.root_path.relative_to(self.settings.env_home))
        except Exception:
            return str(identifier)

    def _serialize_env(self, env: PyEnv, *, identifier: str | None = None) -> EnvInfo:
        return EnvInfo(
            identifier=self._env_label(identifier, env),
            python_path=str(env.python_path),
            cwd=str(env.cwd),
            bin_path=str(env.bin_path),
            root_path=str(env.root_path),
            prefer_uv=env.prefer_uv,
            has_uv=env.has_uv(),
            is_current=env.python_path.resolve() == Path(sys.executable).resolve(),
            version=self._version_to_str(env.version_info),
        )

    def _discover_env_roots(self) -> list[Path]:
        if not self.settings.env_home.exists():
            return []

        roots: list[Path] = []
        for pyvenv_cfg in self.settings.env_home.rglob("pyvenv.cfg"):
            root = pyvenv_cfg.parent.resolve()
            if root not in roots:
                roots.append(root)
        return sorted(roots)

    def _env_from_root(self, root: Path, *, prefer_uv: bool = True) -> PyEnv | None:
        try:
            python_path = PyEnv._venv_python_from_dir(root)  # noqa: SLF001
            return PyEnv.instance(python_path, prefer_uv=prefer_uv)
        except Exception:
            return None

    def _looks_like_named_env(self, identifier: str) -> bool:
        if not identifier:
            return False
        if identifier.strip().lower() in {"current", "sys", "system"}:
            return False
        return not PyEnv._looks_like_path(identifier)  # noqa: SLF001

    def _named_env_path(self, identifier: str) -> Path:
        return (self.settings.env_home / identifier).expanduser().resolve()

    def _resolve_existing_env(
        self,
        identifier: str | None,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
    ) -> PyEnv:
        if not identifier or identifier.strip().lower() in {"current", "sys", "system"}:
            return PyEnv.current(prefer_uv=prefer_uv)

        anchor = PyEnv.current(prefer_uv=prefer_uv)

        if self._looks_like_named_env(identifier):
            path = self._named_env_path(identifier)
            if path.exists():
                return anchor.venv(path, cwd=cwd, prefer_uv=prefer_uv)

        raw = Path(identifier).expanduser()
        if raw.exists():
            return anchor.venv(raw, cwd=cwd, prefer_uv=prefer_uv)

        try:
            python_path = PyEnv.resolve_python_executable(identifier)
            return PyEnv.instance(python_path, cwd=cwd, prefer_uv=prefer_uv)
        except Exception as exc:
            raise NotFoundError(f"Environment not found for identifier={identifier!r}") from exc

    def _resolve_env(
        self,
        identifier: str | None,
        *,
        cwd: Path | None = None,
        prefer_uv: bool = True,
        create_if_missing: bool = False,
        version: str | None = None,
        packages: list[str] | None = None,
        seed: bool = True,
    ) -> PyEnv:
        if create_if_missing:
            try:
                return PyEnv.get_or_create(
                    identifier=identifier,
                    cwd=cwd,
                    version=version,
                    packages=packages,
                    prefer_uv=prefer_uv,
                    seed=seed,
                )
            except Exception as exc:
                raise APIError(str(exc), status_code=400) from exc

        return self._resolve_existing_env(
            identifier,
            cwd=cwd,
            prefer_uv=prefer_uv,
        )

    def _read_excel_payload(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise APIError(
                "The execution finished without producing a dataframe payload. "
                "Make sure your code defines locals()['df'].",
                status_code=400,
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_excel_capture_code(
        self,
        *,
        user_code: str,
        output_path: Path,
        df_name: str,
        max_rows: int | None,
    ) -> str:
        output_literal = repr(str(output_path))
        df_name_literal = json.dumps(df_name)
        user_code_literal = json.dumps(user_code)
        max_rows_literal = "None" if max_rows is None else str(int(max_rows))
        return f'''
import json
import traceback
from pathlib import Path

OUTPUT_PATH = Path({output_literal})
USER_CODE = {user_code_literal}
DF_NAME = {df_name_literal}
MAX_ROWS = {max_rows_literal}


def _write_payload(payload):
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


try:
    import pandas as pd
except Exception as exc:
    _write_payload({{
        "ok": False,
        "error": {{
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }},
    }})
    raise


globals_dict = {{"__name__": "__main__", "__file__": "<powerquery-execute>"}}
locals_dict = {{}}

try:
    exec(compile(USER_CODE, "<user-code>", "exec"), globals_dict, locals_dict)
    namespace = {{**globals_dict, **locals_dict}}

    if DF_NAME not in namespace:
        raise KeyError(f"Expected dataframe '{{DF_NAME}}' in locals()")

    df = namespace[DF_NAME]
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"locals()['{{DF_NAME}}'] must be a pandas.DataFrame, got {{type(df).__name__}}"
        )

    total_rows = int(len(df.index))
    export_df = df if MAX_ROWS is None else df.head(MAX_ROWS)
    rows = json.loads(export_df.to_json(orient="records", date_format="iso"))
    columns = [str(column) for column in export_df.columns]
    schema = [
        {{"name": str(column), "dtype": str(dtype)}}
        for column, dtype in zip(export_df.columns, export_df.dtypes)
    ]

    _write_payload({{
        "ok": True,
        "data": {{
            "df_name": DF_NAME,
            "columns": columns,
            "schema": schema,
            "rows": rows,
            "row_count": total_rows,
            "returned_rows": int(len(export_df.index)),
            "truncated": bool(MAX_ROWS is not None and total_rows > len(export_df.index)),
        }},
    }})
except Exception as exc:
    _write_payload({{
        "ok": False,
        "error": {{
            "type": exc.__class__.__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }},
    }})
    raise
'''
