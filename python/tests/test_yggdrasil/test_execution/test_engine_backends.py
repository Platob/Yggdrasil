"""Engine backend tests — Arrow / Polars compilation, the filter
dispatch surface, and cross-backend agreement on one fixture.

The AST's core promise is that every backend implements the same
predicate semantics — the agreement test pins that down for the
backends that run without a JVM.
"""

from __future__ import annotations

import pyarrow as pa
import polars as pl
import pytest

from yggdrasil.execution.expr import Expression, col


@pytest.fixture()
def trades() -> "dict[str, list]":
    return {
        "price": [50, 150, 200, None],
        "side": ["buy", "sell", "hold", "buy"],
    }


def test_to_arrow_filters_table(trades):
    pred = (col("price") >= 100) & col("side").is_in(["buy", "sell"])
    table = pa.table(trades)
    assert table.filter(pred.to_arrow()).to_pydict() == {
        "price": [150],
        "side": ["sell"],
    }


def test_to_polars_filters_frame(trades):
    pred = (col("price") >= 100) & col("side").is_in(["buy", "sell"])
    frame = pl.DataFrame(trades)
    assert frame.filter(pred.to_polars()).to_dicts() == [
        {"price": 150, "side": "sell"},
    ]


def test_backends_agree_on_semantics(trades):
    predicates = [
        col("price") > 100,
        col("price").between(50, 150),
        col("side").is_in(["buy"]),
        col("price").is_null(),
        col("price").is_not_null() & (col("side") != "hold"),
        ~(col("side") == "buy"),
        col("side").like("b%"),
    ]
    rows = [
        {"price": p, "side": s}
        for p, s in zip(trades["price"], trades["side"])
    ]
    table = pa.table(trades)
    frame = pl.DataFrame(trades)
    for pred in predicates:
        fn = pred.to_python()
        expected = [r["side"] for r in rows if fn(r) is True]
        got_arrow = table.filter(pred.to_arrow()).column("side").to_pylist()
        got_polars = frame.filter(pred.to_polars())["side"].to_list()
        assert got_arrow == expected, pred.to_sql()
        assert got_polars == expected, pred.to_sql()


def test_filter_arrow_table_inlist_shortcut(trades):
    # InList / AND-of-InList takes the hashset path — assert it
    # matches the generic kernel result.
    table = pa.table(trades)
    pred = col("side").is_in(["buy"])
    assert pred.filter_arrow_table(table).to_pydict() == \
        table.filter(pred.to_arrow()).to_pydict()


def test_filter_arrow_batch_small_path():
    # ≤ 4 rows rides the per-row probe; an all-drop result must be a
    # zero-row slice with the source schema preserved.
    batch = pa.record_batch({"k": [1, 2]})
    out = col("k").is_in([99]).filter_arrow_batch(batch)
    assert out.num_rows == 0
    assert out.schema == batch.schema


def test_filter_arrow_inlist_null_semantics():
    table = pa.table({"k": [1, None, 2]})
    keep_null = col("k").is_in([1, None])
    drop_null = col("k").is_in([1])
    assert keep_null.filter_arrow_table(table).column("k").to_pylist() == [1, None]
    assert drop_null.filter_arrow_table(table).column("k").to_pylist() == [1]


def test_filter_arrow_batches_streams_survivors():
    pred = col("k") > 1
    batches = [
        pa.record_batch({"k": [0, 1]}),
        pa.record_batch({"k": pa.array([], type=pa.int64())}),
        pa.record_batch({"k": [2, 3]}),
    ]
    out = list(pred.filter_arrow_batches(batches))
    assert len(out) == 1
    assert out[0].column("k").to_pylist() == [2, 3]


def test_filter_pydict_round_trip(trades):
    out = (col("price") >= 150).filter_pydict(trades)
    assert out == {"price": [150, 200], "side": ["sell", "hold"]}
    assert (col("price") > 999).filter_pydict(trades) == {
        "price": [],
        "side": [],
    }
    assert (col("price") > 0).filter_pydict({}) == {}


def test_generic_filter_dispatch(trades):
    pred = col("side") == "buy"
    table = pa.table(trades)
    frame = pl.DataFrame(trades)
    rows = [{"side": s} for s in trades["side"]]

    assert isinstance(pred.filter(table), pa.Table)
    assert isinstance(pred.filter(table.to_batches()[0]), pa.RecordBatch)
    assert isinstance(pred.filter(frame), pl.DataFrame)
    assert pred.filter(rows) == [{"side": "buy"}, {"side": "buy"}]
    assert pred.filter(dict(trades))["side"] == ["buy", "buy"]


def test_from_arrow_compound_lift_is_unsupported():
    # pyarrow ≥ 16 exposes no structural API over compiled
    # expressions — the lifter documents NotImplementedError for
    # compound shapes. Pinned so a future pyarrow that re-opens
    # introspection shows up as a test failure, not silence.
    pred = (col("price") >= 100) & (col("side") == "buy")
    with pytest.raises(NotImplementedError):
        Expression.from_arrow(pred.to_arrow())


def test_from_polars_round_trip():
    pred = (col("price") >= 100) & (col("side") == "buy")
    lifted = Expression.from_polars(pred.to_polars())
    assert lifted.equals(pred)
    frame = pl.DataFrame({"price": [50, 150], "side": ["buy", "buy"]})
    assert frame.filter(lifted.to_polars()).height == 1


def test_from_polars_lifts_predicate_shapes():
    import datetime as dt

    shapes = [
        col("x").is_in([1, 2]),
        col("x").is_null(),
        col("x").is_not_null(),
        ~(col("x") > 1),
        col("d") == dt.date(2026, 1, 2),
        col("t") == dt.datetime(2026, 1, 2, 3, 4, 5),
        col("s") == "it's",
        col("b") == True,  # noqa: E712
        col("f") > 1.5,
    ]
    for pred in shapes:
        lifted = Expression.from_polars(pred.to_polars())
        assert lifted.equals(pred), pred.to_sql()

    # Between emits to polars as ``(x >= lo) & (x <= hi)``, so it
    # round-trips to that AND shape — semantically identical, not
    # structurally. A native polars ``is_between`` lifts to Between.
    lifted = Expression.from_polars(col("x").between(1, 5).to_polars())
    assert lifted.equals((col("x") >= 1) & (col("x") <= 5))
    native = Expression.from_polars(pl.col("x").is_between(1, 5))
    assert native.equals(col("x").between(1, 5))


def test_generic_from_sniffs_polars():
    pred = col("x") > 1
    lifted = Expression.from_(pred.to_polars())
    assert lifted.equals(pred)


def test_literal_coercion_against_target_schema():
    # String literal vs int64 column — the target-schema rewrite
    # safe-casts the literal so the kernel compares natively.
    table = pa.table({"k": [1, 2, 3]})
    assert (col("k") == "2").filter_arrow_table(table).column("k").to_pylist() == [2]


def test_timezone_pushdown_against_target_schema():
    import datetime as dt
    import zoneinfo

    utc = zoneinfo.ZoneInfo("UTC")
    paris = zoneinfo.ZoneInfo("Europe/Paris")
    times = [
        dt.datetime(2026, 1, 1, 10, 0, tzinfo=utc),
        dt.datetime(2026, 1, 1, 12, 0, tzinfo=utc),
    ]
    table = pa.table({"ts": pa.array(times, type=pa.timestamp("us", tz="UTC"))})
    # 13:00 Paris == 12:00 UTC in January.
    pred = col("ts") == dt.datetime(2026, 1, 1, 13, 0, tzinfo=paris)
    out = pred.filter_arrow_table(table)
    assert out.num_rows == 1
    assert out.column("ts").to_pylist()[0] == times[1]
