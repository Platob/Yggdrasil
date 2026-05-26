"""Tests for :class:`yggdrasil.spark.sql_statement.SparkSQLStatement`."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from yggdrasil.dataclasses.awaitable import Awaitable
from yggdrasil.enums import State
from yggdrasil.io.tabular.base import Tabular
from yggdrasil.spark.sql_statement import SparkSQLStatement


def _mock_session(table: pa.Table | None = None):
    if table is None:
        table = pa.table({"x": [1, 2, 3]})
    df = MagicMock()
    df.toArrow.return_value = table
    df.limit.return_value = df
    df.schema = table.schema
    session = MagicMock()
    session.sql.return_value = df
    return session, df


class TestInheritance:

    def test_is_tabular(self):
        assert issubclass(SparkSQLStatement, Tabular)

    def test_is_awaitable(self):
        assert issubclass(SparkSQLStatement, Awaitable)


class TestConstruction:

    def test_default_state_is_idle(self):
        s = SparkSQLStatement("SELECT 1")
        assert s.state is State.IDLE

    def test_text_stored(self):
        s = SparkSQLStatement("SELECT 42")
        assert s.text == "SELECT 42"

    def test_dataframe_none_before_start(self):
        assert SparkSQLStatement("SELECT 1").dataframe is None

    def test_not_started(self):
        assert not SparkSQLStatement("SELECT 1").started

    def test_zero_attempts(self):
        assert SparkSQLStatement("SELECT 1").attempts == 0

    def test_row_limit(self):
        s = SparkSQLStatement("SELECT 1", row_limit=100)
        assert s.row_limit == 100


class TestStart:

    def test_start_executes_sql(self):
        session, df = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        session.sql.assert_called_once_with("SELECT 1")

    def test_start_succeeds(self):
        session, df = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        assert s.is_succeeded
        assert s.dataframe is df

    def test_start_increments_attempts(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        assert s.attempts == 1

    def test_start_with_row_limit(self):
        session, df = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session, row_limit=10)
        s.start(wait=False)
        df.limit.assert_called_once_with(10)

    def test_start_no_row_limit_skips_limit(self):
        session, df = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        df.limit.assert_not_called()

    def test_start_idempotent(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        s.start(wait=False)
        session.sql.assert_called_once()

    def test_start_reset_reruns(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        s.start(reset=True, wait=False)
        assert session.sql.call_count == 2
        assert s.attempts == 2

    def test_start_failure_sets_failed(self):
        session = MagicMock()
        session.sql.side_effect = RuntimeError("boom")
        s = SparkSQLStatement("BAD SQL", spark_session=session)
        s.start(wait=False, raise_error=False)
        assert s.is_failed
        assert s.dataframe is None

    def test_start_failure_raise_error(self):
        session = MagicMock()
        session.sql.side_effect = RuntimeError("boom")
        s = SparkSQLStatement("BAD SQL", spark_session=session)
        with pytest.raises(RuntimeError, match="boom"):
            s.start(wait=True, raise_error=True)

    def test_start_returns_self(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        assert s.start(wait=False) is s


class TestError:

    def test_error_none_when_succeeded(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        assert s.error is None

    def test_error_returns_exception(self):
        session = MagicMock()
        session.sql.side_effect = ValueError("bad")
        s = SparkSQLStatement("BAD", spark_session=session)
        s.start(wait=False, raise_error=False)
        assert isinstance(s.error, ValueError)
        assert "bad" in str(s.error)

    def test_raise_for_status_raises_original(self):
        session = MagicMock()
        session.sql.side_effect = ValueError("bad")
        s = SparkSQLStatement("BAD", spark_session=session)
        s.start(wait=False, raise_error=False)
        with pytest.raises(ValueError, match="bad"):
            s.raise_for_status()

    def test_raise_for_status_noop_when_ok(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        s.raise_for_status()


class TestCancel:

    def test_cancel_is_noop(self):
        session, _ = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        result = s.cancel(wait=False)
        assert result is s
        assert s.is_succeeded


class TestReadArrow:

    def test_read_arrow_batches(self):
        import sys
        table = pa.table({"a": [1, 2], "b": ["x", "y"]})
        session, df = _mock_session(table)
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        cast_mock = MagicMock()
        cast_mock.spark_dataframe_to_arrow = MagicMock(return_value=table)
        saved = sys.modules.get("yggdrasil.spark.cast")
        sys.modules["yggdrasil.spark.cast"] = cast_mock
        try:
            batches = list(s._read_arrow_batches(CastOptions()))
        finally:
            if saved is not None:
                sys.modules["yggdrasil.spark.cast"] = saved
            else:
                sys.modules.pop("yggdrasil.spark.cast", None)
        total = sum(b.num_rows for b in batches)
        assert total == 2
        cast_mock.spark_dataframe_to_arrow.assert_called_once_with(df)

    def test_read_before_start_raises(self):
        s = SparkSQLStatement("SELECT 1")
        with pytest.raises(RuntimeError, match="before start"):
            list(s._read_arrow_batches(CastOptions()))

    def test_write_raises(self):
        s = SparkSQLStatement("SELECT 1")
        with pytest.raises(NotImplementedError):
            s._write_arrow_batches([], CastOptions())


class TestNativeSparkFrame:

    def test_returns_dataframe(self):
        session, df = _mock_session()
        s = SparkSQLStatement("SELECT 1", spark_session=session)
        s.start(wait=False)
        assert s._native_spark_frame() is df

    def test_returns_none_before_start(self):
        s = SparkSQLStatement("SELECT 1")
        assert s._native_spark_frame() is None


class TestRepr:

    def test_repr_short_sql(self):
        s = SparkSQLStatement("SELECT 1")
        r = repr(s)
        assert "SparkSQLStatement" in r
        assert "SELECT 1" in r
        assert "idle" in r

    def test_repr_long_sql_truncated(self):
        sql = "SELECT " + ", ".join(f"col_{i}" for i in range(100))
        s = SparkSQLStatement(sql)
        r = repr(s)
        assert "..." in r
        assert len(r) < len(sql)


class TestRetryable:

    def test_not_retryable_by_default(self):
        assert not SparkSQLStatement("SELECT 1").retryable


from yggdrasil.data.options import CastOptions
