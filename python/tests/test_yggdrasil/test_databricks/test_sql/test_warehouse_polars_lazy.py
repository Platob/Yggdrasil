"""Tests for :meth:`WarehouseStatementResult._scan_polars_frame` — the
autonomous polars :class:`~polars.LazyFrame` over a warehouse result.

The lazy frame must be *self-contained*: re-collectable (each ``collect``
re-streams the external-link Arrow chunks), and honour polars' projection
/ predicate / row-count pushdown.

These tests stub the Statement Execution API response plus the warehouse's
external-link HTTP session so chunk reads return canned Arrow IPC bytes,
without needing a live warehouse.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import polars as pl
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
# circular import (see test_warehouse_empty_result.py).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _arrow_ipc_stream(table: pa.Table) -> bytes:
    """Serialize *table* as an Arrow IPC *stream* (what the warehouse
    external links serve and ``pyarrow.ipc.open_stream`` reads)."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


class _FakeResponse(io.BytesIO):
    """A urllib3-ish response over canned bytes.

    ``WarehouseStatementResult._read_arrow_batches`` feeds this straight
    into ``pa.input_stream`` (needs ``read``) and then drains / releases
    the connection — so we add the three attributes it touches.
    """

    status = 200

    def drain_conn(self) -> None:  # noqa: D401 - urllib3 shim
        pass

    def release_conn(self) -> None:  # noqa: D401 - urllib3 shim
        pass


class _FakeHTTP:
    """Stand-in for the warehouse external-link :class:`HTTPSession`.

    Maps URL -> Arrow IPC bytes; counts fetches so a test can assert the
    lazy frame really re-streams on each collect.
    """

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.fetch_count = 0

    def fetch(self, method: str, url: str, **kwargs):
        self.fetch_count += 1
        return _FakeResponse(self.payloads[url])

    def clear_connections(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _warehouse() -> SQLWarehouse:
    service = MagicMock(name="Warehouses")
    return SQLWarehouse(service=service, warehouse_id="wh-1", warehouse_name="wh")


_COLUMNS = [
    ColumnInfo(name="id", position=0, type_name=ColumnInfoTypeName.LONG, type_text="bigint"),
    ColumnInfo(name="name", position=1, type_name=ColumnInfoTypeName.STRING, type_text="string"),
]


def _data_table() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "name": pa.array(["a", "b", "c", "d"], type=pa.string()),
        }
    )


def _result_with_data(wh: SQLWarehouse, table: pa.Table) -> WarehouseStatementResult:
    """A terminal result whose single external-link chunk serves *table*."""
    url = "https://fake-cloud/chunk-0"
    response = StatementResponse(
        statement_id="stmt-data",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        manifest=ResultManifest(
            schema=ResultSchema(column_count=len(_COLUMNS), columns=_COLUMNS),
            total_row_count=table.num_rows,
            total_chunk_count=1,
        ),
        result=ResultData(
            external_links=[
                ExternalLink(
                    external_link=url,
                    row_count=table.num_rows,
                    byte_count=table.nbytes,
                ),
            ],
        ),
    )
    wh._external_link_pool_instance = _FakeHTTP({url: _arrow_ipc_stream(table)})

    stmt = WarehousePreparedStatement("SELECT * FROM t", disposition=Disposition.EXTERNAL_LINKS)
    return WarehouseStatementResult(
        executor=wh,
        statement=stmt,
        statement_id=response.statement_id,
        _response=response,
    )


def _empty_result(wh: SQLWarehouse) -> WarehouseStatementResult:
    response = StatementResponse(
        statement_id="stmt-empty",
        status=StatementStatus(state=StatementState.SUCCEEDED),
        manifest=ResultManifest(
            schema=ResultSchema(column_count=len(_COLUMNS), columns=_COLUMNS),
            total_row_count=0,
            total_chunk_count=0,
        ),
        result=ResultData(external_links=[]),
    )
    stmt = WarehousePreparedStatement("SELECT * FROM t LIMIT 0", disposition=Disposition.EXTERNAL_LINKS)
    return WarehouseStatementResult(
        executor=wh,
        statement=stmt,
        statement_id=response.statement_id,
        _response=response,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScanPolarsFrame:
    def test_returns_lazyframe_with_schema_without_fetching(self) -> None:
        """Constructing the scan is lazy — no chunk fetched until collect."""
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        lf = result.scan_polars_frame()

        assert isinstance(lf, pl.LazyFrame)
        assert lf.collect_schema().names() == ["id", "name"]
        assert wh._external_link_pool_instance.fetch_count == 0

    def test_collect_materializes_rows(self) -> None:
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = result.scan_polars_frame().collect()

        assert df.shape == (4, 2)
        assert df["id"].to_list() == [1, 2, 3, 4]
        assert df["name"].to_list() == ["a", "b", "c", "d"]

    def test_lazyframe_is_re_collectable(self) -> None:
        """The base ``scan_pyarrow_dataset`` path drains on first collect;
        the autonomous frame re-streams on every collect."""
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())
        lf = result.scan_polars_frame()

        first = lf.collect()
        second = lf.collect()

        assert first.equals(second)
        # Two collects -> two independent chunk fetches.
        assert wh._external_link_pool_instance.fetch_count == 2

    def test_projection_pushdown(self) -> None:
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = result.scan_polars_frame().select("id").collect()

        assert df.columns == ["id"]
        assert df["id"].to_list() == [1, 2, 3, 4]

    def test_predicate_pushdown(self) -> None:
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = result.scan_polars_frame().filter(pl.col("id") >= 3).collect()

        assert df["id"].to_list() == [3, 4]

    def test_n_rows_pushdown(self) -> None:
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = result.scan_polars_frame().head(2).collect()

        assert df["id"].to_list() == [1, 2]

    def test_composes_in_a_larger_plan(self) -> None:
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = (
            result.scan_polars_frame()
            .filter(pl.col("id") % 2 == 0)
            .select("name")
            .collect()
        )

        assert df["name"].to_list() == ["b", "d"]

    def test_empty_result_yields_schema_bearing_empty_frame(self) -> None:
        wh = _warehouse()
        result = _empty_result(wh)

        df = result.scan_polars_frame().collect()

        assert df.height == 0
        assert df.columns == ["id", "name"]

    def test_to_polars_alias_eager_read_still_works(self) -> None:
        """The eager ``to_polars`` surface is unaffected by the lazy override."""
        wh = _warehouse()
        result = _result_with_data(wh, _data_table())

        df = result.read_polars_frame()

        assert df.shape == (4, 2)
        assert df["id"].to_list() == [1, 2, 3, 4]
