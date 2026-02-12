from enum import Enum
from typing import Optional

__all__ = ["SaveMode", "STR_MAPPING"]


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

        # first check common shorthands / aliases
        existing = STR_MAPPING.get(value)

        if existing is not None:
            return existing

        s = str(value).strip().lower().replace("-", "_")

        existing = STR_MAPPING.get(value)

        if existing is not None:
            return existing

        return cls(s)


STR_MAPPING = {
    # overwrite
    "w": SaveMode.OVERWRITE,
    "wb": SaveMode.OVERWRITE,
    "write": SaveMode.OVERWRITE,
    "overwrite": SaveMode.OVERWRITE,
    "replace": SaveMode.OVERWRITE,
    "clobber": SaveMode.OVERWRITE,
    "truncate": SaveMode.OVERWRITE,

    # append
    "a": SaveMode.APPEND,
    "ab": SaveMode.APPEND,
    "append": SaveMode.APPEND,
    "add": SaveMode.APPEND,

    # ignore (no-op if exists)
    "i": SaveMode.IGNORE,
    "ignore": SaveMode.IGNORE,
    "skip": SaveMode.IGNORE,

    # error if exists
    "error": SaveMode.ERROR_IF_EXISTS,
    "fail": SaveMode.ERROR_IF_EXISTS,
    "raise": SaveMode.ERROR_IF_EXISTS,
    "error_if_exists": SaveMode.ERROR_IF_EXISTS,

    # auto (let implementation decide)
    "": SaveMode.AUTO,
    "auto": SaveMode.AUTO,
    "default": SaveMode.AUTO,
}
