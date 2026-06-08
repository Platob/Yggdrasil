"""Regression: warehouse external-link chunks download *concurrently*.

``WarehouseStatementResult._read_arrow_batches`` fans the per-chunk
downloads out across a worker pool. An earlier shape handed the pool a
*generator function* whose body (the HTTP GET + IPC decode) only ran when
the consumer iterated it — so every chunk was fetched serially in the
consumer thread and the pool did no real work.

This test pins the fix deterministically (no timing assertions): a fake
HTTP session blocks each fetch on a :class:`threading.Barrier` and records
the peak number of simultaneous fetches. If the downloads were serial only
one fetch would ever be in flight, the barrier would never trip, and the
fetch would raise :class:`threading.BrokenBarrierError`. Concurrent
downloads trip the barrier and push the peak above one.
"""
from __future__ import annotations

import io
import threading
from unittest.mock import MagicMock

import pyarrow as pa
from databricks.sdk.service.sql import (
    ColumnInfo,
    ColumnInfoTypeName,
    Disposition,
    ExternalLink,
    ResultData,
    ResultManifest,
    ResultSchema,
    StatementResponse,
    StatementState,
    StatementStatus,
)

# Import the SQL package first to sidestep the warehouse<->sql.engine
# circular import (see test_warehouse_polars_lazy.py).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult
from yggdrasil.data.options import CastOptions


_N_CHUNKS = 4
# Require only 2 concurrent fetches to "trip" — robust to the pool's worker
# count while still failing hard if the downloads run one at a time.
_PARTIES = 2

_COLUMNS = [
    ColumnInfo(name="id", position=0, type_name=ColumnInfoTypeName.LONG, type_text="bigint"),
]


def _arrow_ipc_stream(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


class _FakeResponse(io.BytesIO):
    status = 200

    def drain_conn(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _BarrierHTTP:
    """Fake external-link session that proves fetches overlap.

    Each ``fetch`` enters a barrier; the call can't return until
    ``_PARTIES`` fetches are inside it at once. Tracks the observed peak
    concurrency so the test can assert real overlap.
    """

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.barrier = threading.Barrier(_PARTIES, timeout=10)
        self._live = 0
        self._lock = threading.Lock()
        self.max_concurrent = 0

    def fetch(self, method: str, url: str, **kwargs):
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            self.barrier.wait()
        finally:
            with self._lock:
                self._live -= 1
        return _FakeResponse(self.payloads[url])

    def clear_connections(self) -> None:
        pass


def _multi_chunk_result() -> tuple[WarehouseStatementResult, _BarrierHTTP]:
    service = MagicMock(name="Warehouses")
    wh = SQLWarehouse(service=service, warehouse_id="wh-1", warehouse_name="wh")

    payloads: dict[str, bytes] = {}
    links: list[ExternalLink] = []
    for i in range(_N_CHUNKS):
        url = f"https://fake-cloud/chunk-{i}"
        table = pa.table({"id": pa.array([i], type=pa.int64())})
        payloads[url] = _arrow_ipc_stream(table)
        links.append(ExternalLink(external_link=url, chunk_index=i, row_count=1))

    http = _BarrierHTTP(payloads)
    wh._external_link_pool_instance = http

    response = StatementResponse(
        statement_id="stmt-multi",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        manifest=ResultManifest(
            schema=ResultSchema(column_count=len(_COLUMNS), columns=_COLUMNS),
            total_row_count=_N_CHUNKS,
            total_chunk_count=_N_CHUNKS,
        ),
        result=ResultData(chunk_index=0, external_links=links),
    )
    stmt = WarehousePreparedStatement("SELECT * FROM t", disposition=Disposition.EXTERNAL_LINKS)
    result = WarehouseStatementResult(
        executor=wh,
        statement=stmt,
        statement_id=response.statement_id,
        _response=response,
    )
    return result, http


class TestWarehouseChunkConcurrency:
    def test_chunks_download_concurrently(self) -> None:
        result, http = _multi_chunk_result()

        # Drain every chunk. A serial fetch loop would deadlock on the
        # barrier (only one fetch ever in flight) and raise BrokenBarrierError.
        rows = [
            row
            for batch in result._read_arrow_batches(CastOptions())
            for row in batch.column("id").to_pylist()
        ]

        assert http.max_concurrent >= _PARTIES
        # ``ordered=True`` keeps chunk order despite the concurrent fetch.
        assert rows == list(range(_N_CHUNKS))

    def test_read_arrow_table_fetches_concurrently_and_casts_once(self) -> None:
        # The optimized whole-result path fans the *same* parallel chunk fetch
        # out across the pool, then assembles one table cast in a single pass —
        # so a serial fetch would still deadlock on the barrier, and every
        # chunk's row lands in order.
        result, http = _multi_chunk_result()

        table = result.read_arrow_table()

        assert http.max_concurrent >= _PARTIES
        assert table.num_rows == _N_CHUNKS
        assert table.column("id").to_pylist() == list(range(_N_CHUNKS))

    def test_read_arrow_tabular_uses_cast_once_table(self) -> None:
        # ``read_arrow_tabular`` wraps the cast-once table in an in-memory
        # ArrowTabular — same concurrent fetch, all rows present.
        result, http = _multi_chunk_result()

        tabular = result.read_arrow_tabular()

        assert http.max_concurrent >= _PARTIES
        assert tabular.read_arrow_table().column("id").to_pylist() == list(range(_N_CHUNKS))
