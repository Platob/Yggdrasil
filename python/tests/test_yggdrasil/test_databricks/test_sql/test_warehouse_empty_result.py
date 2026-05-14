"""Tests for :class:`WarehouseStatementResult` schema preservation on
empty result sets.

A warehouse statement that returns zero rows must still surface its
column schema to callers — ``to_arrow_table`` / ``read_arrow_batches`` /
``read_arrow_batch_reader`` should produce a schema-bearing empty table
rather than collapsing to ``Schema.empty()`` / a schema-less iterator.

These tests stub the Statement Execution API response so the result
short-circuits to a terminal state with no external links, without
needing a live warehouse.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk.service.sql import (
    ColumnInfo,
    ColumnInfoTypeName,
    Disposition,
    ResultData,
    ResultManifest,
    ResultSchema,
    StatementResponse,
    StatementState,
    StatementStatus,
)

# Importing the SQL package first sidesteps the circular import between
# ``warehouse`` and ``sql.engine`` (engine pulls SQLWarehouse from the
# warehouse package, warehouse.statement pulls SQLError from the sql
# package).
from yggdrasil.databricks.sql import SQLEngine  # noqa: F401  -- import-order fix
from yggdrasil.databricks.warehouse import SQLWarehouse, WarehousePreparedStatement
from yggdrasil.databricks.warehouse.statement import WarehouseStatementResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warehouse() -> SQLWarehouse:
    """Build an :class:`SQLWarehouse` with a mocked service so the
    constructor skips the live ``find_warehouse`` lookup.
    """
    service = MagicMock(name="Warehouses")
    return SQLWarehouse(
        service=service,
        warehouse_id="wh-1",
        warehouse_name="wh",
    )


def _empty_response(
    *,
    statement_id: str = "stmt-empty",
    columns: list[ColumnInfo] | None = None,
) -> StatementResponse:
    """A ``SUCCEEDED`` :class:`StatementResponse` with a known schema
    but no external-link chunks (zero rows)."""
    return StatementResponse(
        statement_id=statement_id,
        status=StatementStatus(state=StatementState.SUCCEEDED),
        manifest=ResultManifest(
            schema=ResultSchema(
                column_count=len(columns) if columns is not None else 0,
                columns=columns,
            ),
            total_row_count=0,
            total_chunk_count=0,
        ),
        result=ResultData(external_links=[]),
    )


def _result(
    wh: SQLWarehouse,
    response: StatementResponse,
    *,
    disposition: Disposition = Disposition.EXTERNAL_LINKS,
) -> WarehouseStatementResult:
    """Build a :class:`WarehouseStatementResult` already seeded with
    *response* so ``wait()`` returns immediately without hitting the API.
    """
    stmt = WarehousePreparedStatement("SELECT 1 LIMIT 0", disposition=disposition)
    return WarehouseStatementResult(
        executor=wh,
        statement=stmt,
        statement_id=response.statement_id,
        _response=response,
    )


def _two_column_schema() -> list[ColumnInfo]:
    return [
        ColumnInfo(
            name="a",
            position=0,
            type_name=ColumnInfoTypeName.LONG,
            type_text="bigint",
        ),
        ColumnInfo(
            name="b",
            position=1,
            type_name=ColumnInfoTypeName.STRING,
            type_text="string",
        ),
    ]


# ---------------------------------------------------------------------------
# _collect_schema
# ---------------------------------------------------------------------------


class TestCollectSchemaEmptyResult:
    """Schema collection must read columns off the manifest even when
    the result has zero rows."""

    def test_returns_named_columns_when_row_count_is_zero(self) -> None:
        wh = _warehouse()
        result = _result(wh, _empty_response(columns=_two_column_schema()))

        sch = result.collect_schema()

        assert sch.names == ["a", "b"]
        assert result.collect_schema() is sch

    def test_returns_empty_schema_when_manifest_is_missing(self) -> None:
        """A response without any manifest (e.g. some DDL paths) must
        not raise — collect_schema should return an empty schema and
        cache it."""
        wh = _warehouse()
        resp = StatementResponse(
            statement_id="stmt-no-manifest",
            status=StatementStatus(state=StatementState.SUCCEEDED),
            manifest=None,
            result=None,
        )
        result = _result(wh, resp)

        sch = result.collect_schema()

        assert sch.names == []

    def test_returns_empty_schema_when_manifest_schema_is_missing(self) -> None:
        """``ResultManifest.schema`` is ``Optional`` in the SDK — guard
        against the None branch so we don't AttributeError."""
        wh = _warehouse()
        resp = StatementResponse(
            statement_id="stmt-no-schema",
            status=StatementStatus(state=StatementState.SUCCEEDED),
            manifest=ResultManifest(schema=None, total_row_count=0),
            result=None,
        )
        result = _result(wh, resp)

        sch = result.collect_schema()

        assert sch.names == []


# ---------------------------------------------------------------------------
# _read_arrow_batches / read_arrow_table
# ---------------------------------------------------------------------------


class TestReadArrowEmptyResult:
    """Arrow read hooks must surface the warehouse-known schema even
    when the result iterator has no data batches."""

    def test_read_arrow_batches_yields_empty_batch_with_schema(self) -> None:
        wh = _warehouse()
        result = _result(wh, _empty_response(columns=_two_column_schema()))

        batches = list(result.read_arrow_batches())

        # Exactly one zero-row batch carrying the warehouse schema.
        assert len(batches) == 1
        assert batches[0].num_rows == 0
        assert batches[0].schema.names == ["a", "b"]

    def test_read_arrow_table_preserves_schema_on_empty_result(self) -> None:
        wh = _warehouse()
        result = _result(wh, _empty_response(columns=_two_column_schema()))

        table = result.read_arrow_table()

        assert table.num_rows == 0
        assert table.schema.names == ["a", "b"]

    def test_read_arrow_batch_reader_preserves_schema_on_empty_result(self) -> None:
        wh = _warehouse()
        result = _result(wh, _empty_response(columns=_two_column_schema()))

        reader = result.read_arrow_batch_reader()

        assert reader.schema.names == ["a", "b"]
        batches = list(reader)
        # Reader yields the single zero-row batch coming out of
        # _read_arrow_batches; the reader's own schema is already correct.
        assert all(b.num_rows == 0 for b in batches)
