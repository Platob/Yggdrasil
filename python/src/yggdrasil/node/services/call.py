from __future__ import annotations

import collections.abc
import logging
import time
import uuid
from functools import partial

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import NotFoundError
from ..remote import ensure_modules, get_registered, list_registered
from ..transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    is_tabular,
    serialize_pickle,
    to_arrow_table,
    write_arrow_stream_bytes,
    write_arrow_stream_chunked,
)

LOGGER = logging.getLogger(__name__)


class CallService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def _run(self, fn, /, *args, **kwargs):
        return await run_in_threadpool(partial(fn, *args, **kwargs))

    async def execute_call(
        self,
        body: bytes,
        accept: str | None = None,
    ) -> tuple[bytes | collections.abc.Iterator, str, dict[str, str]]:
        payload = deserialize_pickle(body)
        func_key = payload["func"]
        args = tuple(payload.get("args", ()))
        kwargs = dict(payload.get("kwargs", {}))
        want_stream = payload.get("stream", False)
        extra_modules = payload.get("modules")

        spec = get_registered(func_key)
        if spec is None:
            raise NotFoundError(
                f"Function {func_key!r} is not registered on this node. "
                f"Available: {list(list_registered())}"
            )

        if spec.modules or extra_modules:
            await self._run(self._ensure_deps, spec, extra_modules)

        args, kwargs = self._coerce_args(spec.func, args, kwargs)

        call_id = uuid.uuid4().hex[:12]
        LOGGER.info("Executing call %r (id=%s)", func_key, call_id)

        t0 = time.monotonic()
        result = await self._run(spec.func, *args, **kwargs)
        duration = time.monotonic() - t0

        LOGGER.info("Completed call %r (id=%s, %.2fs)", func_key, call_id, duration)

        headers: dict[str, str] = {
            "X-Bot-Call-Id": call_id,
            "X-Bot-Call-Func": func_key,
            "X-Bot-Call-Duration": f"{duration:.3f}",
            "X-Bot-Node-Id": self.settings.node_id,
        }

        prefer_stream = (
            want_stream
            or (accept and CONTENT_TYPE_ARROW_STREAM in accept)
        )

        if is_tabular(result):
            table = to_arrow_table(result)
            headers["X-Arrow-Num-Rows"] = str(table.num_rows)
            headers["X-Arrow-Num-Columns"] = str(table.num_columns)
            headers["X-Arrow-Schema"] = ",".join(
                f"{f.name}:{f.type}" for f in table.schema
            )

            if prefer_stream:
                return (
                    write_arrow_stream_chunked(table),
                    CONTENT_TYPE_ARROW_STREAM,
                    headers,
                )

            return (
                write_arrow_stream_bytes(table),
                CONTENT_TYPE_ARROW_STREAM,
                headers,
            )

        return serialize_pickle(result), CONTENT_TYPE_PICKLE, headers

    @staticmethod
    def _coerce_args(func, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        try:
            from yggdrasil.dataclasses.safe_function import check_function_args
            return check_function_args(func, args, kwargs)
        except Exception:
            return args, kwargs

    @staticmethod
    def _ensure_deps(spec, extra_modules: list[str] | None) -> None:
        ensure_modules(spec)
        if extra_modules:
            from yggdrasil.node.remote import _RemoteSpec
            tmp = _RemoteSpec(func=spec.func, key=spec.key, timeout=spec.timeout, modules=extra_modules)
            ensure_modules(tmp)

    def get_registry(self) -> dict[str, str]:
        return list_registered()
