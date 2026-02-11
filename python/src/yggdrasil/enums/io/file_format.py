# yggdrasil.enums.io.file_format.py

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Optional


class FileFormat(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    AVRO = "avro"
    ORC = "orc"
    ARROW_IPC = "ipc"
    EXCEL = "xlsx"
    BINARY = "bin"

    DEFAULT: ClassVar["FileFormat"] = PARQUET

    @property
    def is_default(self) -> bool:
        return self is self.DEFAULT

    @classmethod
    def parse_any(cls, value: Any, default: Optional["FileFormat"] = None) -> "FileFormat":
        if value is None:
            return default or cls.DEFAULT
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse_str(value)

        if default:
            return default
        raise ValueError(f"Cannot make {cls.__name__} from {type(value).__name__}")

    @classmethod
    def parse_str(cls, value: str) -> "FileFormat":
        if value is None:
            return cls.DEFAULT

        v = value.strip().lower()
        if not v:
            return cls.DEFAULT

        mapped = STR_MAPPING.get(v)
        if mapped is not None:
            return mapped

        # accept member name (case-insensitive), e.g. "PARQUET"
        try:
            return cls[v.upper()]
        except KeyError:
            pass

        # accept value, e.g. "parquet"
        try:
            return cls(v)
        except ValueError:
            pass

        raise ValueError(f"Unknown {cls.__name__}: {value!r}")


STR_MAPPING = {
    "pq": FileFormat.PARQUET,
    "parq": FileFormat.PARQUET,
    "parquet": FileFormat.PARQUET,

    "json": FileFormat.JSON,
    "jsonl": FileFormat.JSON,
    "ndjson": FileFormat.JSON,

    "arrow": FileFormat.ARROW_IPC,
    "ipc": FileFormat.ARROW_IPC,
    "feather": FileFormat.ARROW_IPC,

    "xls": FileFormat.EXCEL,
    "xlsx": FileFormat.EXCEL,
    "xlsm": FileFormat.EXCEL,
    "excel": FileFormat.EXCEL,

    "csv": FileFormat.CSV,

    "orc": FileFormat.ORC,
}

__all__ = ["FileFormat"]
