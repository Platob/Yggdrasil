from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pyarrow as pa
import pyarrow.compute as pc

from ...libs import (
    polars,
    pyspark,
)

__all__ = [
    "AbstractField",
    "PythonField",
    "PandasField",
    "PolarsField",
    "SparkField",
    "ArrowField"
]


class AbstractField(ABC):
    @classmethod
    def parse(cls, obj: Any):
        if isinstance(obj, cls):
            return obj
        return cls._parse(obj)

    @classmethod
    @abstractmethod
    def _parse(cls, dtype: Any):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def type(self):
        ...

    @property
    @abstractmethod
    def nullable(self) -> bool:
        ...

    @property
    @abstractmethod
    def metadata(self) -> Optional[Dict[str, Any]]:
        ...

    @property
    @abstractmethod
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        ...

    @property
    @abstractmethod
    def metadata_str(self) -> Optional[Dict[str, str]]:
        ...

    @abstractmethod
    def to_python(self) -> "PythonField":
        ...

    @abstractmethod
    def to_arrow(self) -> "ArrowField":
        ...

    @abstractmethod
    def to_spark(self) -> "SparkField":
        ...

    @abstractmethod
    def to_polars(self) -> "PolarsField":
        ...

    @abstractmethod
    def to_pandas(self) -> "PandasField":
        ...


class PythonField(AbstractField, ABC):
    def __init__(
        self,
        name: str,
        hint: Union[type, Any],
        nullable: bool,
        metadata: Optional[Dict[str, Any]],
    ):
        self._name = name
        self._hint = hint
        self._nullable = nullable
        self._metadata = metadata

    @staticmethod
    def _encode_metadata_value(value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, bool):
            return b"true" if value else b"false"
        return str(value).encode()

    @classmethod
    def validate_type(cls, dtype: pa.DataType):
        return isinstance(dtype, pa.DataType)

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError("Parsing is not supported for PythonField")

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self):
        return self._hint

    @property
    def nullable(self):
        return self._nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self._metadata:
            return None
        return {
            k.encode() if isinstance(k, str) else str(k).encode(): self._encode_metadata_value(v)
            for k, v in self._metadata.items()
        }

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self._metadata:
            return None

        def _to_str(value: Any) -> str:
            if isinstance(value, bytes):
                try:
                    return value.decode()
                except Exception:
                    return str(value)
            return str(value)

        return {str(k): _to_str(v) for k, v in self._metadata.items()}

    def to_python(self) -> "PythonField":
        return self

    def to_arrow(self) -> "ArrowField":
        raise NotImplementedError("Conversion to ArrowField is not available from PythonField")

    def to_spark(self) -> "SparkField":
        raise NotImplementedError("Conversion to SparkField is not available from PythonField")

    def to_polars(self) -> "PolarsField":
        raise NotImplementedError("Conversion to PolarsField is not available from PythonField")

    def to_pandas(self) -> "PandasField":
        raise NotImplementedError("Conversion to PandasField is not available from PythonField")


class PandasField(AbstractField, ABC):
    def __init__(self, name: str, dtype: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        self._name = name
        self._dtype = dtype
        self._nullable = nullable
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self):
        return self._dtype

    @property
    def nullable(self):
        return self._nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self._metadata:
            return None
        return {
            k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v
            for k, v in self._metadata.items()
        }

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self._metadata:
            return None
        return {str(k): str(v) for k, v in self._metadata.items()}

    def to_pandas(self) -> "PandasField":
        return self

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError("Parsing is not supported for PandasField")

    def to_python(self) -> "PythonField":
        raise NotImplementedError("Conversion to PythonField is not available from PandasField")

    def to_arrow(self) -> "ArrowField":
        raise NotImplementedError("Conversion to ArrowField is not available from PandasField")

    def to_spark(self) -> "SparkField":
        raise NotImplementedError("Conversion to SparkField is not available from PandasField")

    def to_polars(self) -> "PolarsField":
        raise NotImplementedError("Conversion to PolarsField is not available from PandasField")


class PolarsField(AbstractField, ABC):
    def __init__(self, inner: "polars.Field", nullable: bool, metadata: Optional[Dict[str, Any]]):
        self.inner = inner
        self._nullable = nullable
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def type(self) -> "polars.DataType":
        return self.inner.dtype

    @property
    def nullable(self):
        return self._nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self._metadata

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self._metadata:
            return None
        return {
            k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v
            for k, v in self._metadata.items()
        }

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self._metadata:
            return None
        return {str(k): str(v) for k, v in self._metadata.items()}

    def to_polars(self) -> "PolarsField":
        return self

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError("Parsing is not supported for PolarsField")

    def to_python(self) -> "PythonField":
        raise NotImplementedError("Conversion to PythonField is not available from PolarsField")

    def to_arrow(self) -> "ArrowField":
        raise NotImplementedError("Conversion to ArrowField is not available from PolarsField")

    def to_spark(self) -> "SparkField":
        raise NotImplementedError("Conversion to SparkField is not available from PolarsField")

    def to_pandas(self) -> "PandasField":
        raise NotImplementedError("Conversion to PandasField is not available from PolarsField")

    def cast_series(
        self,
        data: "polars.Series",
        safe: Optional[bool] = None,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: Optional[bool] = None,
        allow_add_columns: Optional[bool] = None,
        *,
        wrap_numerical: Optional[bool] = None
    ):
        safe = False if safe is None else safe
        wrap_numerical = False if wrap_numerical is None else wrap_numerical
        add_missing_columns = True if add_missing_columns is None else add_missing_columns
        strict_match_names = False if strict_match_names is None else strict_match_names
        allow_add_columns = False if allow_add_columns is None else allow_add_columns

        return data.cast(
            dtype=self.type,
            strict=safe,
            wrap_numerical=wrap_numerical
        )


class ArrowField(AbstractField, ABC):
    @classmethod
    def validate_type(cls, dtype: pa.DataType):
        return isinstance(dtype, pa.DataType)

    def __init__(self, inner: pa.Field):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def type(self) -> pa.DataType:
        return self.inner.type

    @property
    def nullable(self):
        return self.inner.nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self.metadata_str

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        return self.inner.metadata

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.inner.metadata:
            return None

        def _decode(value: Any) -> str:
            if isinstance(value, (bytes, bytearray)):
                try:
                    return value.decode()
                except Exception:
                    return str(value)
            return str(value)

        return {_decode(k): _decode(v) for k, v in self.inner.metadata.items()}

    def to_arrow(self) -> "ArrowField":
        return self

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError("Parsing is not supported for ArrowField")

    def to_python(self) -> "PythonField":
        raise NotImplementedError("Conversion to PythonField is not available from ArrowField")

    def to_spark(self) -> "SparkField":
        raise NotImplementedError("Conversion to SparkField is not available from ArrowField")

    def to_polars(self) -> "PolarsField":
        raise NotImplementedError("Conversion to PolarsField is not available from ArrowField")

    def to_pandas(self) -> "PandasField":
        raise NotImplementedError("Conversion to PandasField is not available from ArrowField")

    def cast_array(
        self,
        data: Union[pa.ChunkedArray, pa.Array],
        safe: Optional[bool] = None,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: Optional[bool] = None,
        allow_add_columns: Optional[bool] = None,
        *,
        memory_pool: Optional[pa.MemoryPool] = None
    ):
        safe = False if safe is None else safe
        add_missing_columns = True if add_missing_columns is None else add_missing_columns
        strict_match_names = False if strict_match_names is None else strict_match_names
        allow_add_columns = False if allow_add_columns is None else allow_add_columns

        return pc.cast(
            data,
            target_type=self.type,
            safe=safe,
            memory_pool=memory_pool
        )


class SparkField(AbstractField, ABC):
    def __init__(self, inner: "pyspark.sql.types.StructField"):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def type(self) -> "pyspark.sql.types.DataType":
        return self.inner.dataType

    @property
    def nullable(self):
        return self.inner.nullable

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self.inner.metadata

    @property
    def metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self.inner.metadata:
            return None

        return {
            k.encode() if isinstance(k, str) else str(k).encode(): v.encode() if isinstance(v, str) else str(v).encode()
            for k, v in self.inner.metadata.items()
        }

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.inner.metadata:
            return None

        return {
            str(k): v if isinstance(v, str) else str(v)
            for k, v in self.inner.metadata.items()
        }

    def to_spark(self) -> "SparkField":
        return self

    @classmethod
    def _parse(cls, dtype: Any):
        raise NotImplementedError("Parsing is not supported for SparkField")

    def to_python(self) -> "PythonField":
        raise NotImplementedError("Conversion to PythonField is not available from SparkField")

    def to_arrow(self) -> "ArrowField":
        raise NotImplementedError("Conversion to ArrowField is not available from SparkField")

    def to_polars(self) -> "PolarsField":
        raise NotImplementedError("Conversion to PolarsField is not available from SparkField")

    def to_pandas(self) -> "PandasField":
        raise NotImplementedError("Conversion to PandasField is not available from SparkField")

    def cast_column(
        self,
        data: "pyspark.sql.Column",
        safe: Optional[bool] = None,
        add_missing_columns: Optional[bool] = None,
        strict_match_names: Optional[bool] = None,
        allow_add_columns: Optional[bool] = None,
    ):
        safe = False if safe is None else safe
        add_missing_columns = True if add_missing_columns is None else add_missing_columns
        strict_match_names = False if strict_match_names is None else strict_match_names
        allow_add_columns = False if allow_add_columns is None else allow_add_columns

        return data.cast(dataType=self.type)