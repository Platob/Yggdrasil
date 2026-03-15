from __future__ import annotations

import pandas as pd
import pytest

from yggdrasil.pickle.ser.pandas import (
    PandasDataFrameSerialized,
    PandasIndexSerialized,
    PandasSeriesSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


def _roundtrip(obj):
    ser = Serialized.from_python_object(obj)
    assert ser is not None
    out = ser.as_python()
    return ser, out


def test_pandas_dataframe_roundtrip() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [1.5, None, 3.5],
        },
        index=pd.Index(["r1", "r2", "r3"], name="row_id"),
    )

    ser, out = _roundtrip(df)

    assert isinstance(ser, PandasDataFrameSerialized)
    assert ser.tag == Tags.PANDAS_DATAFRAME

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"arrow"

    pd.testing.assert_frame_equal(out, df)


def test_pandas_series_roundtrip() -> None:
    s = pd.Series(
        [10, 20, None],
        index=pd.Index(["a", "b", "c"], name="label"),
        name="values",
        dtype="float64",
    )

    ser, out = _roundtrip(s)

    assert isinstance(ser, PandasSeriesSerialized)
    assert ser.tag == Tags.PANDAS_SERIES

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"arrow"

    pd.testing.assert_series_equal(out, s)


def test_pandas_index_roundtrip() -> None:
    idx = pd.Index(["aa", "bb", "cc"], name="code")

    ser, out = _roundtrip(idx)

    assert isinstance(ser, PandasIndexSerialized)
    assert ser.tag == Tags.PANDAS_INDEX

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"arrow"

    pd.testing.assert_index_equal(out, idx)


def test_pandas_tag_category_and_resolution() -> None:
    assert Tags.get_category(Tags.PANDAS_DATAFRAME) == Tags.CATEGORY_PANDAS
    assert Tags.get_category(Tags.PANDAS_SERIES) == Tags.CATEGORY_PANDAS
    assert Tags.get_category(Tags.PANDAS_INDEX) == Tags.CATEGORY_PANDAS

    assert Tags.is_pandas(Tags.PANDAS_DATAFRAME)
    assert Tags.is_pandas(Tags.PANDAS_SERIES)
    assert Tags.is_pandas(Tags.PANDAS_INDEX)

    assert Tags.get_name(Tags.PANDAS_DATAFRAME) == "PANDAS_DATAFRAME"
    assert Tags.get_name(Tags.PANDAS_SERIES) == "PANDAS_SERIES"
    assert Tags.get_name(Tags.PANDAS_INDEX) == "PANDAS_INDEX"

    assert Tags.get_class(Tags.PANDAS_DATAFRAME) is PandasDataFrameSerialized
    assert Tags.get_class(Tags.PANDAS_SERIES) is PandasSeriesSerialized
    assert Tags.get_class(Tags.PANDAS_INDEX) is PandasIndexSerialized


def test_pandas_dataframe_falls_back_to_python_serialized_for_unarrowable_object_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadThing:
        def __init__(self, x: int) -> None:
            self.x = x

        def __eq__(self, other: object) -> bool:
            return self.x == other.x

        def __repr__(self) -> str:
            return f"BadThing({self.x})"

    df = pd.DataFrame(
        {
            "ok": [1, 2],
            "bad": [BadThing(1), BadThing(2)],
        }
    )

    from yggdrasil.pickle.ser import pandas as pandas_ser

    def boom(_df):
        raise TypeError("kaboom")

    monkeypatch.setattr(pandas_ser, "_dataframe_to_arrow_table", boom)

    ser = Serialized.from_python_object(df)
    assert ser is not None

    assert isinstance(ser, PandasDataFrameSerialized)
    assert ser.tag == Tags.PANDAS_DATAFRAME

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"python_serialized"

    out = ser.as_python()

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["ok", "bad"]
    assert out.index.tolist() == [0, 1]
    assert out["ok"].tolist() == [1, 2]
    assert out["bad"].tolist() == [BadThing(1), BadThing(2)]


def test_pandas_series_falls_back_to_python_serialized_for_unarrowable_object_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = pd.Series([object(), object()], name="bad")

    from yggdrasil.pickle.ser import pandas as pandas_ser

    def boom(_series):
        raise TypeError("kaboom")

    monkeypatch.setattr(pandas_ser, "_series_to_arrow_table", boom)

    ser = Serialized.from_python_object(s)
    assert ser is not None

    assert isinstance(ser, PandasSeriesSerialized)
    assert ser.tag == Tags.PANDAS_SERIES

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"python_serialized"

    out = ser.as_python()

    assert isinstance(out, pd.Series)
    assert out.name == "bad"
    assert len(out) == 2
    assert out.index.tolist() == [0, 1]


def test_pandas_index_falls_back_to_python_serialized_when_arrow_conversion_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = pd.Index([1, 2, 3], name="id")

    from yggdrasil.pickle.ser import pandas as pandas_ser

    def boom(_index):
        raise TypeError("kaboom")

    monkeypatch.setattr(pandas_ser, "_index_to_arrow_table", boom)

    ser = Serialized.from_python_object(idx)
    assert ser is not None

    assert isinstance(ser, PandasIndexSerialized)
    assert ser.tag == Tags.PANDAS_INDEX

    metadata = ser.metadata or {}
    assert metadata.get(b"serialization_format") == b"python_serialized"

    out = ser.as_python()

    assert isinstance(out, pd.Index)
    pd.testing.assert_index_equal(out, idx)
