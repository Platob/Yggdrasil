"""Bounded filesystem reads.

:class:`FsService.read` previews a file under ``node_home`` reading at most
``settings.max_read_bytes`` no matter how large the file is — so previewing
a 1 GB log never pulls 1 GB into memory. Content is decoded as UTF-8 text
when possible and returned as raw bytes otherwise.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from yggdrasil.exceptions.node import NodeNotFoundError

from ...config import Settings


class ReadResult(BaseModel):
    path: str
    content: str | bytes
    size: int
    bytes_read: int
    truncated: bool
    encoding: str

    model_config = {"arbitrary_types_allowed": True}


class FsService:
    """File previews rooted at ``settings.node_home``, capped per read."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def read(self, name: str) -> ReadResult:
        local = self.settings.node_home / name
        if not local.is_file():
            raise NodeNotFoundError(
                f"No file {name!r} under {self.settings.node_home}. "
                "Path is resolved relative to the node home."
            )

        size = local.stat().st_size
        cap = self.settings.max_read_bytes
        with open(local, "rb") as fh:
            raw = fh.read(cap)
        truncated = size > len(raw)

        try:
            content: str | bytes = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = raw
            encoding = "binary"

        return ReadResult(
            path=name,
            content=content,
            size=size,
            bytes_read=len(raw),
            truncated=truncated,
            encoding=encoding,
        )
