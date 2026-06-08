"""Read-option coverage for the optimized whole-result table read.

``WarehouseStatementResult._read_arrow_table`` fetches every external-link
chunk in parallel and casts the lot once. These tests pin that the
``CastOptions`` read pipeline still applies on that path — chiefly
``row_limit`` (both its result correctness and its fetch *pushdown*, where a
small limit stops pulling chunks instead of downloading them all) and
column projection.

A non-blocking fake HTTP session (no barrier) records which chunk URLs were
fetched so the pushdown can be asserted deterministically.
"""
from __future__ import annotations

import io
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
# circular import (see test_warehouse_chunk_concurrency.py).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult


def _ipc(table: pa.Table) -> bytes:
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


class _CountingHTTP:
    """Non-blocking fake session that records every fetched chunk URL."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.fetched: list[str] = []

    def fetch(self, method: str, url: str, **kwargs):
        self.fetched.append(url)
        return _FakeResponse(self.payloads[url])

    def clear_connections(self) -> None:
        pass


_COLUMNS = [
    ColumnInfo(name="id", position=0, type_name=ColumnInfoTypeName.LONG, type_text="bigint"),
    ColumnInfo(name="label", position=1, type_name=ColumnInfoTypeName.STRING, type_text="string"),
]


def _result(n_chunks: int) -> tuple[WarehouseStatementResult, _CountingHTTP]:
    """One row per chunk so ``row_limit`` maps directly to a chunk count."""
    service = MagicMock(name="Warehouses")
    wh = SQLWarehouse(service=service, warehouse_id="wh-1", warehouse_name="wh")

    payloads: dict[str, bytes] = {}
    links: list[ExternalLink] = []
    for i in range(n_chunks):
        url = f"https://fake-cloud/chunk-{i}"
        table = pa.table({"id": pa.array([i], type=pa.int64()),
                          "label": pa.array([f"r{i}"], type=pa.string())})
        payloads[url] = _ipc(table)
        links.append(ExternalLink(external_link=url, chunk_index=i, row_count=1))

    http = _CountingHTTP(payloads)
    wh._external_link_pool_instance = http

    response = StatementResponse(
        statement_id="stmt-rl",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        manifest=ResultManifest(
            schema=ResultSchema(column_count=len(_COLUMNS), columns=_COLUMNS),
            total_row_count=n_chunks,
            total_chunk_count=n_chunks,
        ),
        result=ResultData(chunk_index=0, external_links=links),
    )
    stmt = WarehousePreparedStatement("SELECT * FROM t", disposition=Disposition.EXTERNAL_LINKS)
    result = WarehouseStatementResult(
        executor=wh, statement=stmt, statement_id="stmt-rl", _response=response,
    )
    return result, http


class TestWarehouseReadOptions:
    def test_row_limit_caps_the_result(self) -> None:
        result, _ = _result(20)
        table = result.read_arrow_table(row_limit=5)
        assert table.num_rows == 5
        assert table.column("id").to_pylist() == [0, 1, 2, 3, 4]

    def test_row_limit_pushes_down_to_the_fetch(self) -> None:
        # A small limit must not download all 50 chunks: the generator is
        # closed once enough rows are in hand, cancelling the rest.
        result, http = _result(50)
        table = result.read_arrow_table(row_limit=3)
        assert table.num_rows == 3
        assert len(http.fetched) < 50

    def test_no_row_limit_reads_every_chunk(self) -> None:
        result, http = _result(8)
        table = result.read_arrow_table()
        assert table.num_rows == 8
        assert len(http.fetched) == 8
        assert table.column("id").to_pylist() == list(range(8))

    def test_row_limit_on_read_arrow_tabular(self) -> None:
        result, _ = _result(20)
        tabular = result.read_arrow_tabular(row_limit=4)
        assert tabular.read_arrow_table().num_rows == 4

    def test_column_projection_selects_subset(self) -> None:
        result, _ = _result(4)
        table = result.read_arrow_table(columns=["id"])
        assert table.column_names == ["id"]
        assert table.column("id").to_pylist() == list(range(4))


class TestWarehousePolarsPandas:
    """polars / pandas reads funnel through ``_read_arrow_table``, so they
    inherit the cast-once parallel fetch and the same read options."""

    def test_polars_frame_reads_every_chunk(self) -> None:
        result, http = _result(8)
        frame = result.read_polars_frame()
        assert frame.shape == (8, 2)
        assert frame["id"].to_list() == list(range(8))
        assert len(http.fetched) == 8

    def test_polars_frame_honors_row_limit_pushdown(self) -> None:
        result, http = _result(50)
        frame = result.read_polars_frame(row_limit=4)
        assert frame.shape == (4, 2)
        assert frame["id"].to_list() == [0, 1, 2, 3]
        assert len(http.fetched) < 50

    def test_polars_frame_column_projection(self) -> None:
        result, _ = _result(5)
        frame = result.read_polars_frame(columns=["id"])
        assert frame.columns == ["id"]
        assert frame["id"].to_list() == list(range(5))

    def test_pandas_frame_reads_every_chunk(self) -> None:
        result, http = _result(8)
        df = result.read_pandas_frame()
        assert df.shape == (8, 2)
        assert list(df["id"]) == list(range(8))
        assert len(http.fetched) == 8

    def test_pandas_frame_honors_row_limit_pushdown(self) -> None:
        result, http = _result(50)
        df = result.read_pandas_frame(row_limit=3)
        assert df.shape == (3, 2)
        assert list(df["id"]) == [0, 1, 2]
        assert len(http.fetched) < 50
