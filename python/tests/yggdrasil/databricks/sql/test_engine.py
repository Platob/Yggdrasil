import types
from unittest import mock

import pyarrow as pa
import pyarrow.ipc as pipc
import pytest
from databricks.sdk.service.sql import Format, StatementState

from yggdrasil.databricks.sql.engine import DBXStatementResult


def _make_arrow_stream_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _make_statement_response(*, result, format=Format.ARROW_STREAM):
    status = types.SimpleNamespace(
        state=StatementState.SUCCEEDED,
        error=types.SimpleNamespace(message=None, error_code=None),
    )
    manifest = types.SimpleNamespace(format=format, schema=types.SimpleNamespace(columns=[]))
    return types.SimpleNamespace(
        status=status,
        manifest=manifest,
        result=result,
        statement_id="stmt-123",
    )


def test_result_chunks_fetches_paginated_results():
    first_chunk = types.SimpleNamespace(external_links=[], next_chunk_index=1)
    second_chunk = types.SimpleNamespace(external_links=[], next_chunk_index=None)

    base = _make_statement_response(result=first_chunk)

    mock_execution = mock.Mock()
    mock_execution.statement_execution.get_statement_result_chunk_n.return_value = second_chunk

    workspace = mock.Mock()
    workspace.sdk.return_value = mock_execution

    result = DBXStatementResult(base=base, workspace=workspace)

    assert list(result.result_chunks()) == [first_chunk, second_chunk]
    mock_execution.statement_execution.get_statement_result_chunk_n.assert_called_once_with(
        statement_id="stmt-123", chunk_index=1
    )


def test_arrow_batches_fetches_links_across_chunks(monkeypatch):
    table = pa.table({"col": [1, 2, 3]})
    arrow_bytes = _make_arrow_stream_bytes(table)

    first_chunk = types.SimpleNamespace(
        external_links=[types.SimpleNamespace(external_link="https://example.com/chunk1")],
        next_chunk_index=1,
    )
    second_chunk = types.SimpleNamespace(
        external_links=[types.SimpleNamespace(external_link="https://example.com/chunk2")],
        next_chunk_index=None,
    )

    base = _make_statement_response(result=first_chunk)

    mock_execution = mock.Mock()
    mock_execution.statement_execution.get_statement_result_chunk_n.return_value = second_chunk

    workspace = mock.Mock()
    workspace.sdk.return_value = mock_execution

    # Fake HTTP session that returns the same arrow payload for each link
    class FakeResponse:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.requests = []

        def get(self, url, verify=None, timeout=None):  # pragma: no cover - trivial wrapper
            self.requests.append((url, verify, timeout))
            return FakeResponse(arrow_bytes)

    fake_session = FakeSession()
    monkeypatch.setattr("yggdrasil.databricks.sql.engine.YGGSession", lambda: fake_session)

    result = DBXStatementResult(base=base, workspace=workspace)

    batches = list(result.arrow_batches())
    assert len(batches) == 2  # one batch per link
    assert fake_session.requests == [
        ("https://example.com/chunk1", False, 10),
        ("https://example.com/chunk2", False, 10),
    ]

    # Validate batch content matches the source table
    reconstructed = pa.Table.from_batches(batches)
    assert reconstructed.to_pydict() == {"col": [1, 2, 3, 1, 2, 3]}
    mock_execution.statement_execution.get_statement_result_chunk_n.assert_called_once_with(
        statement_id="stmt-123", chunk_index=1
    )


def test_fetch_chunk_requires_workspace():
    base = _make_statement_response(result=types.SimpleNamespace(external_links=[], next_chunk_index=None))
    result = DBXStatementResult(base=base, workspace=None)

    with pytest.raises(ValueError):
        result._fetch_chunk(1)
