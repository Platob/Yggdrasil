"""Tests for the Spark fast-path anti-join helper.

:func:`_spark_filter_existing_keys` is the optimized side of keyed
APPEND on Spark: it filters incoming rows whose ``match_by`` tuple
already exists in the target via a DataFrame left-anti-join. The
SQL engine then runs a plain INSERT — dramatically cheaper than the
``NOT EXISTS`` subquery shape used on the warehouse path.

Tests use plain :class:`unittest.mock.Mock` to stand in for the
PySpark session / DataFrame surface; we verify the right method
chain fires (``session.table → select → distinct → join``) without
needing a real Spark cluster.
"""
from __future__ import annotations

from unittest.mock import MagicMock


from yggdrasil.databricks.table.table import _spark_filter_existing_keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(target_table: MagicMock) -> MagicMock:
    """Mock SparkSession whose ``.table(...)`` returns *target_table*."""
    session = MagicMock()
    session.table.return_value = target_table
    return session


def _make_df() -> MagicMock:
    """Mock DataFrame with chainable ``select`` / ``distinct`` / ``join``."""
    df = MagicMock(name="DataFrame")
    df.select.return_value = df
    df.distinct.return_value = df
    df.join.return_value = MagicMock(name="JoinedDataFrame")
    return df


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestAntiJoinSuccess:

    def test_calls_session_table_with_target_location(self) -> None:
        target_df = _make_df()
        session = _make_session(target_df)
        data_df = _make_df()

        _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["id"],
        )
        session.table.assert_called_once_with("cat.sch.t")

    def test_selects_only_match_by_columns(self) -> None:
        target_df = _make_df()
        session = _make_session(target_df)

        _spark_filter_existing_keys(
            session=session,
            data_df=_make_df(),
            target_location="cat.sch.t",
            match_by=["id", "region"],
        )
        # Reads ONLY the key columns from the target — Catalyst
        # pushes that down to the Delta files, so we never read
        # the value columns just to dedup.
        target_df.select.assert_called_once_with("id", "region")
        target_df.select.return_value.distinct.assert_called_once_with()

    def test_left_anti_join_against_data_df(self) -> None:
        target_df = _make_df()
        session = _make_session(target_df)
        data_df = _make_df()

        out, ok = _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["id"],
        )
        assert ok is True
        # ``data_df.join(key_df, ["id"], "left_anti")`` is the call
        # we want.
        data_df.join.assert_called_once()
        args, kwargs = data_df.join.call_args
        # key_df is target.select.distinct()
        assert args[0] is target_df.select.return_value.distinct.return_value
        # Match-by list passed as the second arg.
        assert args[1] == ["id"]
        # ``left_anti`` join type.
        assert args[2] == "left_anti"
        # Returned the joined DataFrame.
        assert out is data_df.join.return_value

    def test_composite_key_anti_join(self) -> None:
        target_df = _make_df()
        session = _make_session(target_df)
        data_df = _make_df()

        out, ok = _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["a", "b", "c"],
        )
        assert ok is True
        target_df.select.assert_called_once_with("a", "b", "c")
        args, _ = data_df.join.call_args
        assert args[1] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Fallback — target doesn't exist yet
# ---------------------------------------------------------------------------


class TestAntiJoinFallback:

    def test_target_not_found_returns_unchanged(self) -> None:
        session = MagicMock()
        # Mimic ``AnalysisException: Table or view not found``.
        session.table.side_effect = Exception("TABLE_OR_VIEW_NOT_FOUND")
        data_df = _make_df()

        out, ok = _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["id"],
        )
        # No anti-join fired; original DataFrame returned unchanged.
        assert ok is False
        assert out is data_df
        # No join attempt either — caller falls through to the SQL
        # ``NOT EXISTS`` path, which handles the empty-target case
        # without an extra Spark hop.
        data_df.join.assert_not_called()

    def test_join_failure_falls_back(self) -> None:
        target_df = _make_df()
        session = _make_session(target_df)
        data_df = _make_df()
        data_df.join.side_effect = RuntimeError("schema mismatch")

        out, ok = _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["id"],
        )
        # The helper swallows the failure and signals "use SQL path."
        assert ok is False
        assert out is data_df


# ---------------------------------------------------------------------------
# match_by gets normalized to a fresh list
# ---------------------------------------------------------------------------


class TestMatchByNormalization:

    def test_match_by_passed_as_list_to_join(self) -> None:
        """The third arg to ``df.join`` must be a *list*, not a tuple.

        PySpark accepts both, but the type hint we expose is
        ``list[str]`` and tests downstream rely on the list shape.
        """
        target_df = _make_df()
        session = _make_session(target_df)
        data_df = _make_df()

        _spark_filter_existing_keys(
            session=session,
            data_df=data_df,
            target_location="cat.sch.t",
            match_by=["id", "region"],
        )
        args, _ = data_df.join.call_args
        assert isinstance(args[1], list)
