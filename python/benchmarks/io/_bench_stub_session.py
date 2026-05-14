"""In-process Session stub for the ``spark_send`` benchmark.

Kept in its own module so :class:`_StubBenchSession` resolves to a
stable ``benchmarks.io._bench_stub_session._StubBenchSession`` qualname.
Spark workers unpickle the broadcast session by importing that path;
defining the class inline in ``bench_spark_send.py`` would land it
under ``__main__`` and break the worker-side import.
"""
from __future__ import annotations

import datetime as dt

from yggdrasil.io.memory import Memory
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.send_config import SendConfig
from yggdrasil.io.session import Session


class _StubBenchSession(Session):
    """Session double whose ``_local_send`` returns a stock 200 :class:`Response`.

    The benchmark measures the surrounding pipeline (chunking, plan
    build, Arrow encode, Spark IPC) without any real network cost.
    Pickle survives the broadcast — no transient state outside what
    :class:`Session` already declares.
    """

    def _local_send(self, request: PreparedRequest, config: SendConfig) -> Response:
        return Response(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=Memory(binary=b'{"ok":true}'),
            received_at=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        )
