"""Local-filesystem :class:`PathIO`.

Walks :mod:`pathlib.Path` the same way :class:`DatabricksPathIO` walks
:class:`~yggdrasil.databricks.fs.path.DatabricksPath`. Delegates all
read logic — dataset/fallback dispatch, filtering, cast, partition
extraction — to the :class:`PathIO` base class.

Kept deliberately parallel to :class:`DatabricksPathIO` so changes to
the walk/filter contract propagate symmetrically.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from yggdrasil.io.enums import MediaType, MimeType

from .bytes_io import BytesIO
from .path_io import PathIO, _SUPPORTED_MIME_TYPES

__all__ = ["LocalPathIO"]


@dataclass(slots=True)
class LocalPathIO(PathIO):
    """PathIO reading from the local filesystem."""

    path: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.path is None:
            raise ValueError("LocalPathIO requires a non-None path")
        self.path = Path(self.path)

        # Let the parent handle media_type inference via iter_files when
        # media_type is None. We explicitly do NOT preempt it by calling
        # MediaType.parse(self.path) here, which would wrongly resolve
        # to OCTET_STREAM for directories and short-circuit the
        # parent's file-based inference.
        PathIO.__post_init__(self)

    @classmethod
    def make(
        cls,
        path: str | Path,
        media: MediaType | MimeType | str | None = None,
    ) -> "LocalPathIO":
        """Build a :class:`LocalPathIO`.

        The holder :class:`BytesIO` is a placeholder — PathIO reads
        directly from the filesystem and never uses it, but the base
        :class:`MediaIO` dataclass requires the field.
        """
        resolved_path = Path(path)

        # Coerce media. For a *file*, fall back to extension parsing.
        # For a *directory* with no explicit media, leave media_type as
        # None so the parent infers from the first file on first use.
        resolved_media: MediaType | None
        if media is None:
            if resolved_path.is_file():
                resolved_media = MediaType.parse(resolved_path, default=None)
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
    ) -> Iterator["LocalPathIO"]:
        """Yield child :class:`LocalPathIO` for each matching file.

        Filter rules (applied in order):

        1. Must be a file, not a directory.
        2. Unless ``include_hidden`` is True, skip files whose *name*
           (not path) starts with ``.`` or ``_`` — matches
           ``PathOptions``' default ``ignore_prefixes``. A file at
           ``/home/user/.venv/data.parquet`` is NOT skipped because its
           filename ``data.parquet`` has no hidden prefix.
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

        walker = self.path.rglob("*") if recursive else self.path.glob("*")
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
        file_path: Path,
        *,
        resolved_mime: MimeType | None,
        include_hidden: bool,
        supported_only: bool,
    ) -> bool:
        """Filter predicate applied to each candidate file."""
        if not file_path.is_file():
            return False

        if not include_hidden:
            name = file_path.name
            if name.startswith(".") or name.startswith("_"):
                return False

        file_mime = MimeType.parse(file_path, default=None)

        if resolved_mime is not None:
            return file_mime is resolved_mime

        if not supported_only:
            return True

        return file_mime in _SUPPORTED_MIME_TYPES

    def _child(self, file_path: Path) -> "LocalPathIO":
        """Build a child IO for a discovered file.

        Constructs directly rather than going through :meth:`make` —
        we already have a :class:`Path`, and the child's media type is
        always file-based (not directory), so extension parsing is both
        correct and cheaper than re-running the full ``make()`` flow
        for every file in a directory walk.
        """
        return type(self)(
            media_type=MediaType.parse(file_path, default=None),
            holder=BytesIO(),
            path=file_path,
        )