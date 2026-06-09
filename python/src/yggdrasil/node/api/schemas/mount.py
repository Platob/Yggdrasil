"""Saga mount contracts — named aliases over a base path/URL/DB connection.

A mount lets the SQL engine reach file trees (local, npfs://, Databricks
volume, S3) and live databases (postgres/mysql/sqlite/mssql) through a stable
alias that the UI and SQL editor reference as ``'alias/sub'``.

Counter-part: GET /api/v2/saga/mount   POST /api/v2/saga/mount
              GET /api/v2/saga/mount/{alias}/ls
              GET /api/v2/fs/nodes  (mounts key)
"""
from __future__ import annotations

from pydantic import BaseModel


class MountCreate(BaseModel):
    alias: str
    target: str          # path | URL | DB URI
    kind: str = "local"  # "local" | "s3" | "volume" | "npfs" | "database"
    read_only: bool = True
    comment: str = ""


class MountInfo(BaseModel):
    alias: str
    target: str
    kind: str
    read_only: bool
    comment: str = ""


class MountEntry(BaseModel):
    name: str
    path: str            # alias-relative path
    is_dir: bool
    is_tabular: bool = False
    size: int = 0


class MountLsResult(BaseModel):
    alias: str
    path: str
    entries: list[MountEntry]
