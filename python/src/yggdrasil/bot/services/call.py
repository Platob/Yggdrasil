from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from functools import partial
from typing import Any, Iterator

import pyarrow as pa

from fastapi.concurrency import run_in_threadpool

from ..config import Settings
from ..exceptions import BotError, NotFoundError
from ..remote import get_registered, list_registered
from ..transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_pickle,
    is_tabular,
    serialize_pickle,
    serialize_result,
    to_arrow_table,
    write_arrow_stream,
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
    ) -> tuple[bytes | Iterator[bytes], str, dict[str, str]]:
        """Deserialize call payload, execute, serialize result.

        Returns (body_bytes_or_iterator, content_type, extra_headers).
        """
        payload = deserialize_pickle(body)
        func_key = payload["func"]
        args = tuple(payload.get("args", ()))
        kwargs = dict(payload.get("kwargs", {}))
        want_stream = payload.get("stream", False)

        func = get_registered(func_key)
        if func is None:
            raise NotFoundError(
                f"Function {func_key!r} is not registered on this node. "
                f"Available: {list(list_registered())}"
            )

        call_id = uuid.uuid4().hex[:12]
        LOGGER.info("Executing call %r (id=%s)", func_key, call_id)

        t0 = time.monotonic()
        result = await self._run(func, *args, **kwargs)
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
            field_info = ",".join(
                f"{f.name}:{f.type}" for f in table.schema
            )
            headers["X-Arrow-Schema"] = field_info

            if prefer_stream:
                return (
                    write_arrow_stream_chunked(table),
                    CONTENT_TYPE_ARROW_STREAM,
                    headers,
                )

            chunks = list(write_arrow_stream(table))
            return b"".join(chunks), CONTENT_TYPE_ARROW_STREAM, headers

        result_bytes = serialize_pickle(result)
        return result_bytes, CONTENT_TYPE_PICKLE, headers

    def get_registry(self) -> dict[str, str]:
        return list_registered()
