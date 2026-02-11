from enum import Enum

__all__ = ["SaveMode"]

from typing import Optional


class SaveMode(str, Enum):
    AUTO = "auto"
    OVERWRITE = "overwrite"
    APPEND = "append"
    IGNORE = "ignore"
    ERROR_IF_EXISTS = "error_if_exists"

    @classmethod
    def from_any(cls, value: object, default: Optional["SaveMode"] = None) -> "SaveMode":
        """
        Normalize user input into a SaveMode.

        Accepts:
          - SaveMode
          - strings like "overwrite", "OVERWRITE", "error-if-exists", "error_if_exists"
          - None -> default
        """
        if isinstance(value, cls):
            return value

        if value is None:
            return default or SaveMode.AUTO

        s = str(value).strip().lower().replace("-", "_")
        return cls(s)
