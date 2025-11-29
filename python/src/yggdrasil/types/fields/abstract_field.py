from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import pyarrow as pa

from ...libs import (
    pandas,
    polars,
    pyspark,
    require_pandas,
    require_polars,
    require_pyspark,
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

    @classmethod
    @abstractmethod
    def validate_type(cls, dtype: Any):
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


class PythonField(ABC):
    def __init__(
        self,
        name: str,
        hint: Union[type, Any],
        nullable: bool,
        metadata: Optional[Dict[str, Any]]
    ):
        self.name = name
        self.hint = hint
        self.nullable = nullable
        self.metadata = metadata

    @classmethod
    def validate_type(cls, dtype: pa.DataType):
        return isinstance(dtype, pa.DataType)

    @property
    def field_name(self) -> str:
        return self.name

    @property
    def field_type(self):
        return self.hint

    @property
    def field_nullable(self):
        return self.nullable

    @property
    def field_metadata(self) -> Optional[Dict[str, Any]]:
        return self.field_metadata_str

    @property
    def field_metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        return self.metadata

    @property
    def field_metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.metadata:
            return None

        return {
            k.encode(): v.encode()
            for k, v in self.metadata
        }

    def to_python(self) -> "PythonField":
        return self


class PandasField(ABC):
    @classmethod
    @require_pandas
    def validate_type(cls, dtype: Any):
        return True

    def __init__(self, name: str, dtype: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
        self.metadata = metadata

    @property
    def field_name(self) -> str:
        return self.name

    @property
    def field_type(self):
        return self.dtype

    @property
    def field_nullable(self):
        return self.nullable

    @property
    def field_metadata(self) -> Optional[Dict[str, Any]]:
        return self.metadata

    @property
    def field_metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self.metadata:
            return None
        return {k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v for k, v in self.metadata.items()}

    @property
    def field_metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.metadata:
            return None
        return {str(k): str(v) for k, v in self.metadata.items()}

    def to_pandas(self) -> "PandasField":
        return self


class PolarsField(ABC):
    @classmethod
    @require_polars
    def validate_type(cls, dtype: Any):
        return True

    def __init__(self, name: str, dtype: Any, nullable: bool, metadata: Optional[Dict[str, Any]]):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
        self.metadata = metadata

    @property
    def field_name(self) -> str:
        return self.name

    @property
    def field_type(self):
        return self.dtype

    @property
    def field_nullable(self):
        return self.nullable

    @property
    def field_metadata(self) -> Optional[Dict[str, Any]]:
        return self.metadata

    @property
    def field_metadata_bytes(self) -> Optional[Dict[bytes, bytes]]:
        if not self.metadata:
            return None
        return {k.encode() if isinstance(k, str) else k: v.encode() if isinstance(v, str) else v for k, v in self.metadata.items()}

    @property
    def field_metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.metadata:
            return None
        return {str(k): str(v) for k, v in self.metadata.items()}

    def to_polars(self) -> "PolarsField":
        return self


class ArrowField(ABC):
    @classmethod
    def validate_type(cls, dtype: pa.DataType):
        return isinstance(dtype, pa.DataType)

    def __init__(self, inner: pa.Field):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def type(self):
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

        return {
            k.encode(): v.encode()
            for k, v in self.inner.metadata
        }

    def to_arrow(self) -> "ArrowField":
        return self


class SparkField(ABC):
    @classmethod
    @require_pyspark
    def validate_type(cls, dtype: "pyspark.sql.types.DataType"):
        return isinstance(dtype, pyspark.sql.types.DataType)

    def __init__(self, inner: "pyspark.sql.types.StructField"):
        self.inner = inner

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def type(self):
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
            k.encode(): v.encode() if isinstance(v, str) else str(v).encode()
            for k, v in self.inner.metadata
        }

    @property
    def metadata_str(self) -> Optional[Dict[str, str]]:
        if not self.inner.metadata:
            return None

        return {
            k: v.encode() if isinstance(v, str) else str(v)
            for k, v in self.inner.metadata
        }

    def to_spark(self) -> "SparkField":
        return self
