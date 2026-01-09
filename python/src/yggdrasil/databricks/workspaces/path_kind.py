from enum import Enum


__all__ = ["DatabricksPathKind"]


class DatabricksPathKind(str, Enum):
    WORKSPACE = "workspace"
    VOLUME = "volume"
    DBFS = "dbfs"
