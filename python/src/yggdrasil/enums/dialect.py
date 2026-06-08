from __future__ import annotations

from enum import Enum

__all__ = ["Dialect"]


class Dialect(str, Enum):
    ANSI = "ansi"
    DATABRICKS = "databricks"
    POSTGRES = "postgres"
    SQLITE = "sqlite"
    MYSQL = "mysql"
