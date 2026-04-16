from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yggdrasil.databricks.fs.path import DatabricksPath
from yggdrasil.io.enums import MediaType, MimeType, MimeTypes

from .bytes_io import BytesIO
from .path_io import PathIO, _SUPPORTED_MIME_TYPES

__all__ = ["DatabricksPathIO"]


@dataclass(slots=True)
class DatabricksPathIO(PathIO):
    path: DatabricksPath

    def __post_init__(self) -> None:
        if not isinstance(self.path, DatabricksPath) and not hasattr(self.path, "read_bytes"):
            self.path = DatabricksPath.parse(self.path)
        PathIO.__post_init__(self)

    @classmethod
    def make(
        cls,
        path: str | Path | DatabricksPath | Any,
        media: MediaType | MimeType | str | None = None,
    ) -> "DatabricksPathIO":
        resolved_path = (
            path
            if isinstance(path, DatabricksPath) or hasattr(path, "read_bytes")
            else DatabricksPath.parse(str(path))
        )
        resolved_media = (
            MediaType.parse(media)
            if media is not None
            else MediaType.parse(str(resolved_path), default=MediaType(MimeTypes.PARQUET))
        )
        return cls(
            media_type=resolved_media,
            buffer=BytesIO(),
            path=resolved_path,
        )

    def iter_files(
        self,
        recursive: bool = True,
        *,
        include_hidden: bool = False,
        supported_only: bool = True,
        mime_type: MimeType | str | None = None,
    ) -> Iterator["DatabricksPathIO"]:
        resolved_mime = MimeType.parse(mime_type) if mime_type is not None else None

        def keep(file_path: DatabricksPath) -> bool:
            if not file_path.is_file():
                return False
            if not include_hidden and any(part.startswith(".") for part in file_path.parts):
                return False

            file_mime = MimeType.parse(str(file_path))
            if resolved_mime is not None:
                return file_mime is resolved_mime
            if not supported_only:
                return True
            return file_mime in _SUPPORTED_MIME_TYPES

        if self.path.is_dir():
            walker = self.path.rglob("*") if recursive else self.path.iterdir()
            for file_path in walker:
                if keep(file_path):
                    yield type(self).make(file_path)
            return

        if keep(self.path):
            yield self
