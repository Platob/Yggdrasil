"""Databricks-path-backed :class:`PathIO`.

Walks :class:`DatabricksPath` (DBFS, Volumes, UC paths, etc.) the same
way :class:`LocalPathIO` walks :mod:`pathlib`. Delegates all read logic
— dataset/fallback dispatch, filtering, cast, partition extraction —
to the :class:`PathIO` base class.

The only responsibilities of this subclass are:

1. Coerce string / ``pathlib.Path`` inputs into :class:`DatabricksPath`.
2. Iterate files under a directory via :meth:`DatabricksPath.rglob` or
   :meth:`DatabricksPath.iterdir`.
3. Build child :class:`DatabricksPathIO` instances for each yielded file.

Everything else — schema inference, batch iteration, filter pushdown —
happens in the base class.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yggdrasil.databricks.fs.path import DatabricksPath
from yggdrasil.io.enums import MediaType, MimeType

from .bytes_io import BytesIO
from .path_io import PathIO, _SUPPORTED_MIME_TYPES

__all__ = ["DatabricksPathIO"]


@dataclass(slots=True)
class DatabricksPathIO(PathIO):
    """PathIO reading from a :class:`DatabricksPath`."""

    path: DatabricksPath = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Coerce anything non-DatabricksPath into one — strings, pathlib
        # Paths, etc. A DatabricksPath duck-type check (`read_bytes`) is
        # kept for objects that already implement the protocol without
        # being a subclass (e.g. fsspec paths wrapped in a custom type).
        if self.path is None:
            raise ValueError("DatabricksPathIO requires a non-None path")

        if not isinstance(self.path, DatabricksPath) and not hasattr(self.path, "read_bytes"):
            self.path = DatabricksPath.parse(str(self.path))

        # Let the parent handle media_type inference via iter_files when
        # media_type is None. We explicitly do NOT preempt it by calling
        # MediaType.parse(str(self.path)) here, which would wrongly
        # resolve to OCTET_STREAM for directories and short-circuit
        # the parent's file-based inference.
        PathIO.__post_init__(self)

    @classmethod
    def make(
        cls,
        path: str | Path | DatabricksPath | Any,
        media: MediaType | MimeType | str | None = None,
    ) -> "DatabricksPathIO":
        """Build a :class:`DatabricksPathIO`.

        The holder :class:`BytesIO` is a placeholder — PathIO reads
        directly from the filesystem and never uses it, but the base
        :class:`MediaIO` dataclass requires the field.
        """
        # Coerce path into a DatabricksPath. We keep an object that
        # duck-types as having `read_bytes` (some wrappers) unmodified.
        resolved_path = (
            path
            if isinstance(path, DatabricksPath) or hasattr(path, "read_bytes")
            else DatabricksPath.parse(str(path))
        )

        # Coerce media. For a *file*, fall back to extension parsing.
        # For a *directory* with no explicit media, leave media_type as
        # None so the parent infers from the first file on first use.
        resolved_media: MediaType | None
        if media is None:
            is_file = (
                resolved_path.is_file()
                if hasattr(resolved_path, "is_file")
                else False
            )
            if is_file:
                resolved_media = MediaType.parse(str(resolved_path), default=None)
            else:
                resolved_media = None
        elif isinstance(media, MediaType):
            resolved_media = media
        elif isinstance(media, MimeType):
            resolved_media = MediaType(media)
        else:  # str
            resolved_media = MediaType.parse(str(media), default=None)

        return cls(
            media_type=resolved_media,
            holder=BytesIO(),
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
        """Yield child :class:`DatabricksPathIO` for each matching file.

        Filter rules (applied in order):

        1. Must be a file, not a directory.
        2. Unless ``include_hidden`` is True, skip files whose *name*
           starts with ``.`` or ``_`` (matches ``PathOptions``' default
           ``ignore_prefixes``).
        3. When ``mime_type`` is given, only yield matching files.
        4. When ``supported_only`` is True and no explicit ``mime_type``
           was requested, only yield files whose mime is in the set of
           formats PathIO knows how to read.
        """
        resolved_mime = (
            MimeType.parse(mime_type, default=None)
            if isinstance(mime_type, str)
            else mime_type
        )

        if self.path.is_file():
            if self._keep(
                self.path,
                resolved_mime=resolved_mime,
                include_hidden=include_hidden,
                supported_only=supported_only,
            ):
                yield self
            return

        if not self.path.is_dir():
            return

        walker = self.path.rglob("*") if recursive else self.path.iterdir()
        for file_path in walker:
            if self._keep(
                file_path,
                resolved_mime=resolved_mime,
                include_hidden=include_hidden,
                supported_only=supported_only,
            ):
                yield self._child(file_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keep(
        file_path: DatabricksPath,
        *,
        resolved_mime: MimeType | None,
        include_hidden: bool,
        supported_only: bool,
    ) -> bool:
        """Filter rule applied to each candidate file."""
        if not file_path.is_file():
            return False

        if not include_hidden:
            name = file_path.name
            if name.startswith(".") or name.startswith("_"):
                return False

        file_mime = MimeType.parse(str(file_path), default=None)

        if resolved_mime is not None:
            return file_mime is resolved_mime

        if not supported_only:
            return True

        return file_mime in _SUPPORTED_MIME_TYPES

    def _child(self, file_path: DatabricksPath) -> "DatabricksPathIO":
        """Build a child IO for a discovered file.

        Constructs directly rather than going through :meth:`make` —
        we already have a :class:`DatabricksPath`, and the child's
        media type is always file-based (not directory), so extension
        parsing is both correct and redundant with what the caller
        will probably re-check inside the base class.
        """
        return type(self)(
            media_type=MediaType.parse(str(file_path), default=None),
            holder=BytesIO(),
            path=file_path,
        )