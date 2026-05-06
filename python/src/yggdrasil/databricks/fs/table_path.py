""":class:`TablePath` — ``/Tables/catalog/schema/table`` UC table reference.

Tables have no filesystem representation — file-system operations
are stubs; data access goes through SQL. This class exists so SQL
helpers can accept a single :class:`DatabricksPath` parameter type
without branching on scheme.

Opening a table as bytes is nonsensical, so :meth:`_open` and the
SDK transport hooks (:meth:`_remote_download` / :meth:`_remote_upload`)
all raise a clear :class:`OSError` pointing the caller at the SQL
APIs.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple, Union

from yggdrasil.io.io_stats import IOStats, IOKind

from .path import DatabricksPath
from .path_kind import DatabricksPathKind


__all__ = ["TablePath"]


class TablePath(DatabricksPath):
    """UC table reference. No filesystem operations — SQL only."""

    scheme: ClassVar[str] = "dbfs+tables"
    _NAMESPACE_PREFIX: ClassVar[str] = "/Tables/"

    @property
    def kind(self) -> DatabricksPathKind:
        return DatabricksPathKind.TABLE

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = self.url.path.lstrip("/")
        return "/Tables/" + p if p else "/Tables"

    # ==================================================================
    # UC decomposition
    # ==================================================================

    def sql_volume_or_table_parts(self) -> Tuple[
        Optional[str], Optional[str], Optional[str], list,
    ]:
        parts = self.url.parts
        return (
            parts[0] if len(parts) > 0 else None,
            parts[1] if len(parts) > 1 else None,
            parts[2] if len(parts) > 2 else None,
            list(parts[3:]),
        )

    # ==================================================================
    # SDK hooks — all stubs
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        # Treat as an extant directory — SQL helpers are the only
        # meaningful operations and they don't go through FS.
        # Reporting MISSING here would surprise SQL callers that
        # probe ``exists()`` before issuing DML.
        return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=0.0)

    def _ls(self, recursive=False, allow_not_found=True):
        return iter([])

    def _mkdir(self, parents=True, exist_ok=True):
        # No-op. ``CREATE TABLE`` is the SQL counterpart.
        pass

    def _remove_file(self, allow_not_found=True):
        pass

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        pass

    # ==================================================================
    # IO surface — all rejected
    # ==================================================================

    def _open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ):
        raise OSError(
            f"Cannot open {self!r} as a file — UC tables are not "
            "byte-streams. Use the SQL APIs (sql_engine, dio, ...) "
            "instead."
        )

    def _remote_download(self, allow_not_found: bool = False) -> bytes:
        raise OSError(
            f"Cannot download {self!r} as bytes — UC tables are not "
            "byte-streams. Use the SQL APIs."
        )

    def _remote_upload(self, payload: bytes) -> None:
        raise OSError(
            f"Cannot upload to {self!r} as bytes — UC tables are not "
            "byte-streams. Use the SQL APIs."
        )

    def _pread(self):
        raise OSError(
            f"Cannot _pread {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )

    def _pwrite(self, data) -> int:
        raise OSError(
            f"Cannot _pwrite {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )

    # The base class has concrete pread/pwrite that go through
    # download+slice / read-modify-write. Override to fail fast with
    # the same SQL-pointer message rather than letting the base try
    # and bottom out at our raising _remote_download.

    def pread(self, n: int, pos: int, *, default=...) -> bytes:
        raise OSError(
            f"Cannot pread {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        raise OSError(
            f"Cannot pwrite {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )

    def read_bytes(self, *, raise_error: bool = True) -> bytes:
        raise OSError(
            f"Cannot read_bytes {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )

    def write_bytes(
        self,
        data,
        *,
        mode: str = "wb",
        parents: bool = True,
    ) -> int:
        raise OSError(
            f"Cannot write_bytes {self!r} — UC tables are not byte-streams. "
            "Use the SQL APIs."
        )