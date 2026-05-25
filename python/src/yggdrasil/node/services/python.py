from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from functools import partial
from threading import Lock
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError, TimeoutError
from ..schemas.python import (
    PythonEntry,
    PythonListResponse,
    PythonRequest,
    PythonResponse,
)

LOGGER = logging.getLogger(__name__)

def _build_capture_wrapper(code: str) -> str:
    code_repr = repr(code)
    return (
        "import sys, json, traceback\n"
        "_stdout_lines = []\n"
        "_stderr_lines = []\n"
        "class _Capture:\n"
        "    def __init__(self, buf): self._buf = buf\n"
        "    def write(self, s): self._buf.append(s); return len(s)\n"
        "    def flush(self): pass\n"
        "_old_stdout, _old_stderr = sys.stdout, sys.stderr\n"
        "sys.stdout = _Capture(_stdout_lines)\n"
        "sys.stderr = _Capture(_stderr_lines)\n"
        "_result = None\n"
        "_error = None\n"
        "try:\n"
        "    _ns = {'__name__': '__main__'}\n"
        f"    exec(compile({code_repr}, '<bot-python>', 'exec'), _ns)\n"
        "    if '__result__' in _ns: _result = _ns['__result__']\n"
        "except Exception:\n"
        "    _error = traceback.format_exc()\n"
        "finally:\n"
        "    sys.stdout = _old_stdout\n"
        "    sys.stderr = _old_stderr\n"
        "_payload = {'stdout': ''.join(_stdout_lines), 'stderr': ''.join(_stderr_lines), 'result': _result, 'error': _error}\n"
        "print(json.dumps(_payload, default=str))\n"
    )


class PythonExecService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: OrderedDict[str, PythonEntry] = OrderedDict()
        self._results: dict[str, PythonResponse] = {}
        self._lock = Lock()

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    async def execute(self, req: PythonRequest) -> PythonResponse:
        exec_id = uuid.uuid4().hex[:12]
        timeout = req.timeout
        if timeout is not None:
            timeout = min(timeout, self.settings.max_python_timeout)
        else:
            timeout = self.settings.max_python_timeout

        LOGGER.info("Executing Python code (id=%s, timeout=%.0fs)", exec_id, timeout)

        entry = PythonEntry(
            id=exec_id,
            status="running",
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        self._store_entry(entry)

        try:
            response = await self._run(
                self._exec_sync, exec_id, req, timeout
            )
        except Exception:
            entry = entry.model_copy(update={"status": "failed"})
            self._store_entry(entry)
            raise

        entry = entry.model_copy(update={
            "status": response.status,
            "returncode": response.returncode,
            "duration": response.duration,
        })
        self._store_entry(entry)
        return response

    def _exec_sync(
        self, exec_id: str, req: PythonRequest, timeout: float
    ) -> PythonResponse:
        wrapped = _build_capture_wrapper(req.code)
        env = dict(req.env) if req.env else None
        t0 = time.monotonic()

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapped],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Python execution timed out after {timeout:.0f}s"
            )

        duration = time.monotonic() - t0
        status = "completed" if proc.returncode == 0 else "failed"

        result: Any = None
        stdout = proc.stdout
        stderr = proc.stderr

        if proc.returncode == 0 and stdout:
            try:
                from yggdrasil.pickle.json import loads
                payload = loads(stdout.rstrip().split("\n")[-1])
                stdout = payload.get("stdout", "")
                stderr = payload.get("stderr", "") or stderr
                result = payload.get("result")
                error = payload.get("error")
                if error:
                    stderr = (stderr or "") + "\n" + error
                    status = "failed"
            except Exception:
                pass

        LOGGER.info("Completed Python execution (id=%s, rc=%s, %.2fs)", exec_id, proc.returncode, duration)

        response = PythonResponse(
            id=exec_id,
            node_id=self.settings.node_id,
            returncode=proc.returncode,
            stdout=stdout or None,
            stderr=stderr or None,
            result=result,
            duration=round(duration, 3),
            status=status,
        )
        with self._lock:
            self._results[exec_id] = response
        return response

    async def get(self, exec_id: str) -> PythonResponse:
        with self._lock:
            result = self._results.get(exec_id)
        if result is None:
            raise NotFoundError(f"Python execution {exec_id!r} not found")
        return result

    async def list_history(self) -> PythonListResponse:
        with self._lock:
            items = list(self._history.values())
        return PythonListResponse(
            node_id=self.settings.node_id,
            items=items,
        )

    async def delete(self, exec_id: str) -> PythonResponse:
        with self._lock:
            result = self._results.pop(exec_id, None)
            self._history.pop(exec_id, None)
        if result is None:
            raise NotFoundError(f"Python execution {exec_id!r} not found")
        return result

    def _store_entry(self, entry: PythonEntry) -> None:
        with self._lock:
            self._history[entry.id] = entry
            while len(self._history) > self.settings.job_max_history:
                oldest_id, _ = self._history.popitem(last=False)
                self._results.pop(oldest_id, None)

    @staticmethod
    def result_to_arrow_ipc(result: Any) -> bytes:
        if isinstance(result, pa.Table):
            table = result
        elif isinstance(result, pa.RecordBatch):
            table = pa.Table.from_batches([result])
        elif isinstance(result, list):
            table = pa.table({"value": pa.array(result)})
        elif isinstance(result, dict):
            table = pa.table(result)
        else:
            table = pa.table({"value": pa.array([result])})

        sink = pa.BufferOutputStream()
        with ipc.RecordBatchFileWriter(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        return sink.getvalue().to_pybytes()
