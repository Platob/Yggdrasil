from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.node.api.schemas.analysis import (
    AggMeasure,
    AggregateRequest,
    ForecastRequest,
    OhlcRequest,
    PivotRequest,
    SeriesRequest,
)
from yggdrasil.node.api.services.analysis import AnalysisService
from yggdrasil.node.api.services.audit import AuditLog
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.config import Settings
from yggdrasil.node.schemas.function import FunctionCreate
from yggdrasil.node.schemas.messenger import MessageSend
from yggdrasil.node.services.function import FunctionService
from yggdrasil.node.services.messenger import MessengerService
from yggdrasil.node.transport import (
    CONTENT_TYPE_ARROW_STREAM,
    CONTENT_TYPE_PICKLE,
    deserialize_result,
    is_tabular,
    read_arrow_stream,
    serialize_result,
    write_arrow_stream,
)


def _settings(home: Path) -> Settings:
    return Settings(node_id="test", node_home=home, front_home=home, allow_remote=True)


# -- transport --------------------------------------------------------------

def test_serialize_result_dispatches_tabular_to_arrow():
    table = pa.table({"a": [1, 2, 3]})
    data, ct = serialize_result(table)
    assert ct == CONTENT_TYPE_ARROW_STREAM
    assert read_arrow_stream(data).equals(table)
    assert deserialize_result(data, ct).equals(table)


def test_serialize_result_dispatches_scalar_to_pickle():
    data, ct = serialize_result({"k": "v"})
    assert ct == CONTENT_TYPE_PICKLE
    assert deserialize_result(data, ct) == {"k": "v"}


def test_is_tabular():
    assert is_tabular(pa.table({"a": [1]}))
    assert not is_tabular({"a": 1})


def test_write_arrow_stream_roundtrip():
    table = pa.table({"x": list(range(100))})
    data = b"".join(write_arrow_stream(table))
    assert read_arrow_stream(data).equals(table)


# -- function service -------------------------------------------------------

@pytest.mark.asyncio
async def test_function_upsert_by_name():
    with tempfile.TemporaryDirectory() as d:
        svc = FunctionService(_settings(Path(d)))
        await svc.create(FunctionCreate(name="f", code="1"))
        r2 = await svc.create(FunctionCreate(name="f", code="2"))
        # Upsert: exactly one record under the name, carrying the latest code.
        listed = await svc.list()
        assert listed == [r2.function]
        assert r2.function.code == "2"


@pytest.mark.asyncio
async def test_function_delete_missing_raises():
    with tempfile.TemporaryDirectory() as d:
        svc = FunctionService(_settings(Path(d)))
        with pytest.raises(Exception):
            await svc.delete(999)


# -- messenger --------------------------------------------------------------

@pytest.mark.asyncio
async def test_messenger_send_and_read():
    with tempfile.TemporaryDirectory() as d:
        svc = MessengerService(_settings(Path(d)))
        await svc.send_message(MessageSend(text="hi", channel="general"))
        msgs = await svc.get_messages("general")
        assert [m.text for m in msgs] == ["hi"]


@pytest.mark.asyncio
async def test_messenger_unknown_channel_raises():
    with tempfile.TemporaryDirectory() as d:
        svc = MessengerService(_settings(Path(d)))
        with pytest.raises(Exception):
            await svc.get_messages("nope")


@pytest.mark.asyncio
async def test_messenger_get_messages_limit():
    with tempfile.TemporaryDirectory() as d:
        svc = MessengerService(_settings(Path(d)))
        for i in range(10):
            await svc.send_message(MessageSend(text=str(i), channel="general"))
        assert len(await svc.get_messages("general", limit=3)) == 3


# -- audit ------------------------------------------------------------------

def test_audit_recent_returns_tail():
    with tempfile.TemporaryDirectory() as d:
        audit = AuditLog(_settings(Path(d)))
        for i in range(5):
            audit.log("create", "pyfunc", i)
        recent = audit.recent(2)
        assert [e.resource_id for e in recent] == [3, 4]


# -- fs ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fs_ls_sorts_dirs_first():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        (home / "b.txt").write_text("x")
        (home / "adir").mkdir()
        res = await FsService(_settings(home)).ls("")
        assert [e.name for e in res.entries] == ["adir", "b.txt"]


@pytest.mark.asyncio
async def test_fs_escape_is_forbidden():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(Exception):
            await FsService(_settings(Path(d))).ls("../../etc")


# -- analysis ---------------------------------------------------------------

def _write_wide(home: Path) -> None:
    n = 2000
    cols = {
        "sector": [["Tech", "Energy"][i % 2] for i in range(n)],
        "region": [["NA", "EU"][i % 2] for i in range(n)],
        "price": [100.0 + i * 0.1 for i in range(n)],
        "pad": [float(i % 5) for i in range(n)],
    }
    pq.write_table(pa.table(cols), str(home / "wide.parquet"))


@pytest.mark.asyncio
async def test_aggregate_projection():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        _write_wide(home)
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        res = await svc.aggregate(AggregateRequest(
            path="wide.parquet", group_by=["sector"],
            measures=[AggMeasure(column="price", agg="mean")],
        ))
        assert res.group_count == 2
        assert "price_mean" in res.columns


@pytest.mark.asyncio
async def test_aggregate_unknown_agg_raises():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        _write_wide(home)
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        with pytest.raises(Exception):
            await svc.aggregate(AggregateRequest(
                path="wide.parquet", group_by=["sector"],
                measures=[AggMeasure(column="price", agg="median")],
            ))


@pytest.mark.asyncio
async def test_series_downsample():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        _write_wide(home)
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        res = await svc.series(SeriesRequest(path="wide.parquet", column="price", points=100))
        assert len(res.x) == len(res.y) <= 110


@pytest.mark.asyncio
async def test_ohlc_bars():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        _write_wide(home)
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        res = await svc.ohlc(OhlcRequest(path="wide.parquet", column="price", buckets=10))
        assert res.bars >= 10
        assert res.data[0].high >= res.data[0].low


@pytest.mark.asyncio
async def test_pivot_grid():
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        _write_wide(home)
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        res = await svc.pivot(PivotRequest(
            path="wide.parquet", rows=["sector"], columns=["region"],
            measures=[AggMeasure(column="price", agg="sum")],
        ))
        assert res.row_count == 2


@pytest.mark.asyncio
async def test_forecast_ridge_always_available():
    import math

    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        m = 2000
        pq.write_table(pa.table({
            "ts": list(range(m)),
            "value": [100.0 + 0.01 * i + 5 * math.sin(2 * math.pi * i / 24) for i in range(m)],
        }), str(home / "ts.parquet"))
        svc = AnalysisService(_settings(home), fs=FsService(_settings(home)))
        res = await svc.forecast(ForecastRequest(
            path="ts.parquet", column="value", x="ts",
            horizon=12, model="ridge", period=24,
        ))
        assert res.model_used == "ridge"
        assert len(res.series[0].y_pred) == 12
