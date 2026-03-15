# yggdrasil.pickle.ser.pandas
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Mapping, TypeVar

from yggdrasil.pandas.lib import pandas as pd
from yggdrasil.arrow.lib import pyarrow as pa

from yggdrasil.io import BytesIO
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags

from .pyarrow import (
    _merge_metadata,
    _table_from_ipc_file_buffer,
    _table_to_ipc_file_buffer,
)

__all__ = [
    "TPandas",
    "PandasSerialized",
    "PandasDataFrameSerialized",
    "PandasSeriesSerialized",
    "PandasIndexSerialized",
]

TPandas = TypeVar("TPandas", bound=object)

_SENTINEL_FIELD_NAME = "__ygg_value__"


def _dataframe_to_arrow_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=True)


def _series_to_arrow_table(series: pd.Series) -> pa.Table:
    name = series.name if series.name is not None else _SENTINEL_FIELD_NAME
    return pa.Table.from_pandas(series.to_frame(name=name), preserve_index=True)


def _index_to_arrow_table(index: pd.Index) -> pa.Table:
    name = index.name if index.name is not None else _SENTINEL_FIELD_NAME
    return pa.Table.from_pandas(index.to_frame(index=False, name=name), preserve_index=False)


def _arrow_table_to_dataframe(table: pa.Table) -> pd.DataFrame:
    return table.to_pandas()


def _arrow_table_to_series(table: pa.Table) -> pd.Series:
    df = table.to_pandas()
    if df.shape[1] != 1:
        raise ValueError(
            f"PANDAS_SERIES payload must contain exactly 1 data column, got {df.shape[1]}"
        )
    return df.iloc[:, 0]


def _arrow_table_to_index(table: pa.Table) -> pd.Index:
    df = table.to_pandas()
    if df.shape[1] != 1:
        raise ValueError(
            f"PANDAS_INDEX payload must contain exactly 1 column, got {df.shape[1]}"
        )
    series = df.iloc[:, 0]
    return pd.Index(series.array, name=series.name)


def _dataframe_to_python_payload(df: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [
            {
                "name": col,
                "values": df.iloc[:, i].tolist(),
            }
            for i, col in enumerate(df.columns)
        ],
        "index": df.index.tolist(),
        "index_name": df.index.name,
        "column_names": list(df.columns.names)
        if getattr(df.columns, "names", None) is not None
        else None,
    }


def _dataframe_from_python_payload(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for DataFrame, got {type(payload)!r}")

    column_entries = payload.get("columns")
    index = payload.get("index")
    index_name = payload.get("index_name")
    column_names = payload.get("column_names")

    if not isinstance(column_entries, list):
        raise TypeError("DataFrame fallback payload missing list 'columns'")
    if not isinstance(index, list):
        raise TypeError("DataFrame fallback payload missing list 'index'")

    data: dict[object, list[object]] = {}
    column_order: list[object] = []

    for entry in column_entries:
        if not isinstance(entry, dict):
            raise TypeError("DataFrame fallback payload column entry must be a dict")
        if "name" not in entry or "values" not in entry:
            raise TypeError(
                "DataFrame fallback payload column entry must contain 'name' and 'values'"
            )

        name = entry["name"]
        values = entry["values"]
        if not isinstance(values, list):
            raise TypeError("DataFrame fallback payload column 'values' must be a list")

        data[name] = values
        column_order.append(name)

    df = pd.DataFrame(data)
    df = df.loc[:, column_order]
    df.index = pd.Index(index, name=index_name)

    if isinstance(column_names, list):
        try:
            df.columns = df.columns.set_names(column_names)
        except Exception:
            pass

    return df


def _series_to_python_payload(series: pd.Series) -> dict[str, object]:
    return {
        "name": series.name,
        "values": series.tolist(),
        "index": series.index.tolist(),
        "index_name": series.index.name,
    }


def _series_from_python_payload(payload: object) -> pd.Series:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for Series, got {type(payload)!r}")

    values = payload.get("values")
    index = payload.get("index")
    name = payload.get("name")
    index_name = payload.get("index_name")

    if not isinstance(values, list):
        raise TypeError("Series fallback payload missing list 'values'")
    if not isinstance(index, list):
        raise TypeError("Series fallback payload missing list 'index'")

    return pd.Series(
        values,
        index=pd.Index(index, name=index_name),
        name=name,
    )


def _index_to_python_payload(index: pd.Index) -> dict[str, object]:
    return {
        "name": index.name,
        "values": index.tolist(),
    }


def _index_from_python_payload(payload: object) -> pd.Index:
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict fallback payload for Index, got {type(payload)!r}")

    values = payload.get("values")
    name = payload.get("name")

    if not isinstance(values, list):
        raise TypeError("Index fallback payload missing list 'values'")

    return pd.Index(values, name=name)


@dataclass(frozen=True, slots=True)
class PandasSerialized(Serialized[TPandas], Generic[TPandas]):
    TAG: ClassVar[int]

    @property
    def value(self) -> TPandas:
        raise NotImplementedError

    def as_python(self) -> TPandas:
        return self.value

    def decode_arrow_buffer(self) -> pa.Buffer:
        return pa.py_buffer(self.decode())

    @staticmethod
    def _deserialize_nested_payload(data: bytes) -> object:
        nested = Serialized.read_from(BytesIO(data), pos=0)
        return nested.as_python()

    @staticmethod
    def _serialize_nested_payload(
        payload: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        return Serialized.from_python_object(
            payload,
            metadata=metadata,
            codec=codec,
        )

    @staticmethod
    def _nested_serialized_bytes(nested: Serialized[object]) -> bytes:
        buf = BytesIO()
        nested.write_to(buf)
        return buf.to_bytes()

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, pd.DataFrame):
            return PandasDataFrameSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pd.Series):
            return PandasSeriesSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        if isinstance(obj, pd.Index):
            return PandasIndexSerialized.from_value(
                obj,
                metadata=metadata,
                codec=codec,
            )

        return None


@dataclass(frozen=True, slots=True)
class PandasDataFrameSerialized(PandasSerialized[pd.DataFrame]):
    TAG: ClassVar[int] = Tags.PANDAS_DATAFRAME

    @property
    def value(self) -> pd.DataFrame:
        metadata = self.metadata or {}
        serialization_format = metadata.get(b"serialization_format", b"arrow")

        if serialization_format == b"python_serialized":
            payload = self._deserialize_nested_payload(self.decode())
            return _dataframe_from_python_payload(payload)

        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        return _arrow_table_to_dataframe(table)

    @classmethod
    def from_value(
        cls,
        df: pd.DataFrame,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        try:
            table = _dataframe_to_arrow_table(df)
            buf = _table_to_ipc_file_buffer(table, metadata=None)
            return cls.build(
                tag=cls.TAG,
                data=buf,
                metadata=_merge_metadata(metadata, {b"serialization_format": b"arrow"}),
                codec=codec,
            )
        except Exception:
            payload = _dataframe_to_python_payload(df)
            nested = cls._serialize_nested_payload(
                payload,
                metadata=None,
                codec=codec,
            )
            return cls.build(
                tag=cls.TAG,
                data=cls._nested_serialized_bytes(nested),
                metadata=_merge_metadata(
                    metadata, {b"serialization_format": b"python_serialized"}
                ),
                codec=codec,
            )


@dataclass(frozen=True, slots=True)
class PandasSeriesSerialized(PandasSerialized[pd.Series]):
    TAG: ClassVar[int] = Tags.PANDAS_SERIES

    @property
    def value(self) -> pd.Series:
        metadata = self.metadata or {}
        serialization_format = metadata.get(b"serialization_format", b"arrow")

        if serialization_format == b"python_serialized":
            payload = self._deserialize_nested_payload(self.decode())
            return _series_from_python_payload(payload)

        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        return _arrow_table_to_series(table)

    @classmethod
    def from_value(
        cls,
        series: pd.Series,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        try:
            table = _series_to_arrow_table(series)
            buf = _table_to_ipc_file_buffer(table, metadata=None)
            return cls.build(
                tag=cls.TAG,
                data=buf,
                metadata=_merge_metadata(metadata, {b"serialization_format": b"arrow"}),
                codec=codec,
            )
        except Exception:
            payload = _series_to_python_payload(series)
            nested = cls._serialize_nested_payload(
                payload,
                metadata=None,
                codec=codec,
            )
            return cls.build(
                tag=cls.TAG,
                data=cls._nested_serialized_bytes(nested),
                metadata=_merge_metadata(
                    metadata, {b"serialization_format": b"python_serialized"}
                ),
                codec=codec,
            )


@dataclass(frozen=True, slots=True)
class PandasIndexSerialized(PandasSerialized[pd.Index]):
    TAG: ClassVar[int] = Tags.PANDAS_INDEX

    @property
    def value(self) -> pd.Index:
        metadata = self.metadata or {}
        serialization_format = metadata.get(b"serialization_format", b"arrow")

        if serialization_format == b"python_serialized":
            payload = self._deserialize_nested_payload(self.decode())
            return _index_from_python_payload(payload)

        table = _table_from_ipc_file_buffer(self.decode_arrow_buffer())
        return _arrow_table_to_index(table)

    @classmethod
    def from_value(
        cls,
        index: pd.Index,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        try:
            table = _index_to_arrow_table(index)
            buf = _table_to_ipc_file_buffer(table, metadata=None)
            return cls.build(
                tag=cls.TAG,
                data=buf,
                metadata=_merge_metadata(metadata, {b"serialization_format": b"arrow"}),
                codec=codec,
            )
        except Exception:
            payload = _index_to_python_payload(index)
            nested = cls._serialize_nested_payload(
                payload,
                metadata=None,
                codec=codec,
            )
            return cls.build(
                tag=cls.TAG,
                data=cls._nested_serialized_bytes(nested),
                metadata=_merge_metadata(
                    metadata, {b"serialization_format": b"python_serialized"}
                ),
                codec=codec,
            )


for cls in PandasSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)