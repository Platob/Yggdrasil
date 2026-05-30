from __future__ import annotations

import datetime as dt
import hashlib
import logging
from threading import Lock
from typing import TYPE_CHECKING

import httpx

from yggdrasil.dataclasses.expiring import ExpiringDict

from ...config import Settings
from ...exceptions import NotFoundError
from ...ids import make_id
from ..schemas.pyfunc import (
    PyFuncCreate,
    PyFuncEntry,
    PyFuncListResponse,
    PyFuncResponse,
    PyFuncUpdate,
)

if TYPE_CHECKING:
    from .audit import AuditLog

LOGGER = logging.getLogger(__name__)


# Python builtin annotation → yggdrasil.data dtype name.
_PY_DTYPE = {
    "int": "int64", "float": "float64", "str": "string", "bool": "bool",
    "bytes": "binary", "complex": "float64", "list": "list", "tuple": "list",
    "set": "list", "dict": "struct", "datetime": "timestamp[us]",
    "date": "date32", "Any": "string", "None": "null", "NoneType": "null",
}


def _ann_to_dtype(ann: str) -> str:
    """Map a source annotation to a yggdrasil.data dtype (best effort)."""
    if not ann:
        return ""
    base = ann.split("[")[0].strip()  # list[int] -> list, dict[str,int] -> dict
    return _PY_DTYPE.get(base.rsplit(".", 1)[-1], "")


def infer_function(code: str, name: str | None, *, pin_versions: bool, default_py: str):
    """AST-scan code for the function's name, typed signature and imports.

    Reuses the @function decorator's dependency inference; maps annotations to
    yggdrasil.data dtypes; pins inferred deps to the installed version when
    available. Returns a ``PyFuncInferResult``.
    """
    import ast

    from yggdrasil.exceptions.api import BadRequestError

    from ...fn import _infer_dependencies
    from ..schemas.pyfunc import PyFuncInferResult, PyFuncParam

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise BadRequestError(f"cannot parse code: {exc}")

    fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (name is None or node.name == name):
            fn = node
            break
    if fn is None:
        raise BadRequestError("no function definition found in code")

    def _src(n) -> str:
        try:
            return ast.unparse(n) if n is not None else ""
        except Exception:
            return ""

    args = fn.args
    pos = args.posonlyargs + args.args
    pad = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
    params: list[PyFuncParam] = []
    sig_parts: list[str] = []
    for a, dflt in zip(pos, pad):
        ann = _src(a.annotation)
        dval = _src(dflt) if dflt is not None else None
        params.append(PyFuncParam(name=a.arg, annotation=ann, dtype=_ann_to_dtype(ann),
                                  default=dval, has_default=dflt is not None))
        sig_parts.append(f"{a.arg}{f': {ann}' if ann else ''}{f' = {dval}' if dval is not None else ''}")
    ret_ann = _src(fn.returns)
    sig = f"{fn.name}({', '.join(sig_parts)}){f' -> {ret_ann}' if ret_ann else ''}"

    deps = _infer_dependencies(code)
    if pin_versions:
        from importlib.metadata import version
        pinned = []
        for d in deps:
            try:
                pinned.append(f"{d}>={version(d)}")
            except Exception:
                pinned.append(d)
        deps = pinned

    return PyFuncInferResult(
        name=fn.name, signature=sig, params=params,
        return_annotation=ret_ann, return_dtype=_ann_to_dtype(ret_ann),
        dependencies=deps, python_version=default_py,
        docstring=(ast.get_docstring(fn) or ""),
    )


class PyFuncService:
    def __init__(self, settings: Settings, *, audit: AuditLog | None = None) -> None:
        self.settings = settings
        self._funcs: ExpiringDict[int, PyFuncEntry] = ExpiringDict(default_ttl=None, max_size=settings.max_functions)
        self._name_to_id: dict[str, int] = {}
        self._lock = Lock()
        self._audit = audit

    # -- CRUD ---------------------------------------------------------------

    async def create(self, req: PyFuncCreate) -> PyFuncResponse:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            existing_id = self._name_to_id.get(req.name)
            existing = self._funcs.get(existing_id) if existing_id is not None else None
            if existing:
                updates: dict = {
                    "updated_at": now,
                    "code": req.code,
                }
                if req.description:
                    updates["description"] = req.description
                if req.python_version is not None:
                    updates["python_version"] = req.python_version
                if req.dependencies:
                    updates["dependencies"] = list(req.dependencies)
                if req.env_id is not None:
                    updates["env_id"] = req.env_id
                hash_code = updates.get("code", existing.code)
                hash_name = existing.name
                hash_deps = updates.get("dependencies", list(existing.dependencies))
                updates["content_hash"] = hashlib.sha256(
                    (hash_code + hash_name + "".join(sorted(hash_deps))).encode()
                ).hexdigest()
                updated = existing.model_copy(update=updates)
                self._funcs[existing.id] = updated
                if self._audit is not None:
                    self._audit.log("update", "pyfunc", existing.id, detail=f"name={req.name}")
                return PyFuncResponse(func=updated)

            func_id = make_id(req.name)
            content_hash = hashlib.sha256(
                (req.code + req.name + "".join(sorted(req.dependencies))).encode()
            ).hexdigest()
            entry = PyFuncEntry(
                id=func_id,
                name=req.name,
                code=req.code,
                description=req.description,
                python_version=req.python_version,
                dependencies=list(req.dependencies),
                env_id=req.env_id,
                created_at=now,
                updated_at=now,
                content_hash=content_hash,
            )
            self._funcs.set(func_id, entry)
            self._name_to_id[req.name] = func_id
            if self._audit is not None:
                self._audit.log("create", "pyfunc", func_id, detail=f"name={req.name}")
            return PyFuncResponse(func=entry)

    async def get(self, func_id: int) -> PyFuncEntry:
        with self._lock:
            entry = self._funcs.get(func_id)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")
        return entry

    async def list(self) -> PyFuncListResponse:
        with self._lock:
            items = list(self._funcs.values())
        return PyFuncListResponse(node_id=self.settings.node_id, funcs=items)

    async def update(self, func_id: int, req: PyFuncUpdate) -> PyFuncResponse:
        with self._lock:
            entry = self._funcs.get(func_id)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        updates: dict = {"updated_at": now}
        for field in ("name", "code", "description", "python_version", "dependencies", "env_id"):
            val = getattr(req, field)
            if val is not None:
                updates[field] = list(val) if field == "dependencies" else val

        hash_code = updates.get("code", entry.code)
        hash_name = updates.get("name", entry.name)
        hash_deps = updates.get("dependencies", list(entry.dependencies))
        updates["content_hash"] = hashlib.sha256(
            (hash_code + hash_name + "".join(sorted(hash_deps))).encode()
        ).hexdigest()

        updated = entry.model_copy(update=updates)
        with self._lock:
            self._funcs[func_id] = updated
            if "name" in updates and updates["name"] != entry.name:
                self._name_to_id.pop(entry.name, None)
                self._name_to_id[updates["name"]] = func_id
        return PyFuncResponse(func=updated)

    async def delete(self, func_id: int) -> PyFuncResponse:
        with self._lock:
            entry = self._funcs.pop(func_id, None)
            if entry is not None:
                self._name_to_id.pop(entry.name, None)
        if entry is None:
            raise NotFoundError(f"PyFunc {func_id!r} not found")
        if self._audit is not None:
            self._audit.log("delete", "pyfunc", func_id, detail=f"name={entry.name}")
        return PyFuncResponse(func=entry)

    def increment_run_count(self, func_id: int) -> None:
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._lock:
            entry = self._funcs.get(func_id)
            if entry is not None:
                self._funcs[func_id] = entry.model_copy(
                    update={"run_count": entry.run_count + 1, "last_run_at": now}
                )

    def record_run_completion(self, func_id: int, duration_ms: float, success: bool) -> None:
        """Update perf metrics after a run completes."""
        with self._lock:
            entry = self._funcs.get(func_id)
            if entry is None:
                return
            new_success = entry.success_count + (1 if success else 0)
            new_failure = entry.failure_count + (0 if success else 1)
            total_runs = new_success + new_failure
            new_avg = (
                (entry.avg_duration_ms * (total_runs - 1) + duration_ms) / total_runs
                if total_runs > 0 else 0
            )
            self._funcs[func_id] = entry.model_copy(update={
                "last_duration_ms": duration_ms,
                "avg_duration_ms": round(new_avg, 2),
                "success_count": new_success,
                "failure_count": new_failure,
            })

    # -- replication --------------------------------------------------------

    async def replicate_to(self, func_id: int, target_url: str) -> dict:
        """Replicate a PyFunc to a remote node by POSTing its data."""
        entry = await self.get(func_id)
        payload = {
            "name": entry.name,
            "code": entry.code,
            "description": entry.description,
            "python_version": entry.python_version,
            "dependencies": list(entry.dependencies),
        }
        if entry.env_id is not None:
            payload["env_id"] = entry.env_id
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{target_url.rstrip('/')}/api/v2/pyfunc",
                json=payload,
                headers={"X-YGG-Source-Node": self.settings.node_id},
            )
            resp.raise_for_status()
            return resp.json()

    # -- internals ----------------------------------------------------------

    async def get_by_name(self, name: str) -> PyFuncEntry:
        """Resolve a function by name. O(1) via name index."""
        with self._lock:
            func_id = self._name_to_id.get(name)
            if func_id is not None:
                entry = self._funcs.get(func_id)
                if entry is not None:
                    return entry
        raise NotFoundError(
            f"PyFunc with name {name!r} not found. "
            f"Check spelling or create it first with POST /api/v2/pyfunc."
        )
