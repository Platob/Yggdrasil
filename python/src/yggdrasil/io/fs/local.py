"""Local-filesystem :class:`Path` and :class:`FileSystem`.

:class:`LocalPath` is a ``pathlib.Path``-backed concrete :class:`Path`. It
delegates the heavy lifting to :mod:`pathlib` / :mod:`shutil` / :mod:`os`
while exposing the backend-agnostic API used elsewhere in the codebase.

:class:`LocalFileSystem` is the :class:`FileSystem` counterpart — a
singleton is fine (there's only one local FS per process), but the class
stays instantiable so downstream code can pass a fresh handle around.
"""

from __future__ import annotations

import mmap
import os
import pathlib
import shutil
import stat as stat_mod
from dataclasses import dataclass
from typing import IO, Any, ClassVar, Iterator, Optional, Union

from .filesystem import FileSystem
from .path import Path, _split

__all__ = ["LocalPath", "LocalFileSystem"]


# ---------------------------------------------------------------------------
# LocalPath
# ---------------------------------------------------------------------------


@dataclass(eq=False, init=False)
class LocalPath(Path):
    """A local-filesystem path, backed by :class:`pathlib.Path`.

    Accepts strings, other :class:`Path`-likes, or :class:`pathlib.Path`
    objects through :meth:`parse`. Absolute vs. relative is carried via
    the ``anchor`` field. :class:`LocalPath` is also the fallback target
    for :meth:`Path.__new__` dispatch — any input that no other backend
    claims lands here.
    """

    scheme: ClassVar[str] = "file"

    # ---- Factory ------------------------------------------------------

    @classmethod
    def _match(cls, obj: Any) -> bool:
        # LocalPath is the fallback — dispatch should never preferentially
        # pick it. The explicit "file://…" form is handled by ``parse`` when
        # the dispatcher ends up here anyway.
        return False

    @classmethod
    def parse(cls, obj: Any) -> "LocalPath":
        """Build a :class:`LocalPath` from a string, list, or Path-like.

        An absolute POSIX or Windows input keeps its anchor; a relative
        input stays relative. Backslashes are normalized to forward
        slashes so Windows-style strings don't trip the parser.
        """
        if isinstance(obj, LocalPath):
            return obj
        if isinstance(obj, pathlib.PurePath):
            return cls.from_pathlib(obj)
        if isinstance(obj, Path):
            # Cross-backend conversion — copy the segments but re-anchor
            # against the local FS so the rendering stays correct.
            return cls(parts=list(obj.parts), anchor=obj.anchor or "")
        return cls.from_str(
            os.fspath(obj) if isinstance(obj, os.PathLike) else str(obj or "")
        )

    @classmethod
    def from_str(cls, value: str) -> "LocalPath":
        """Parse a string (POSIX or Windows-style) into a :class:`LocalPath`."""
        if not isinstance(value, str):
            raise TypeError(
                f"LocalPath.from_str expected str, got {type(value).__name__!r}. "
                "Use LocalPath.from_any() for permissive input."
            )
        if not value:
            return cls(parts=[], anchor="")
        # Strip a "file://" scheme if the caller passed one through dispatch.
        if value.startswith("file://"):
            value = value[len("file://") :]
        elif value.startswith("file:/"):
            value = value[len("file:") :]
        normalized = value.replace("\\", "/")
        anchor = "/" if normalized.startswith("/") else ""
        return cls(parts=_split(normalized), anchor=anchor)

    @classmethod
    def from_pathlib(cls, value: pathlib.PurePath) -> "LocalPath":
        """Convert a :class:`pathlib.PurePath` into a :class:`LocalPath`.

        Preserves the anchor (posix ``/``, Windows ``C:\\`` → ``C:/``)
        and the segments, no string round-trip.
        """
        if not isinstance(value, pathlib.PurePath):
            raise TypeError(
                f"LocalPath.from_pathlib expected pathlib.PurePath, got "
                f"{type(value).__name__!r}."
            )
        anchor = value.anchor or ""
        # pathlib.PurePath.parts includes the anchor at index 0; strip it
        # so our ``parts`` stays root-free.
        raw_parts = value.parts
        if anchor and raw_parts and raw_parts[0] == anchor:
            raw_parts = raw_parts[1:]
        return cls(parts=list(raw_parts), anchor=_normalize_anchor(anchor))

    # ---- Backing pathlib.Path -----------------------------------------

    def _pl(self) -> pathlib.Path:
        """Return the equivalent :class:`pathlib.Path`.

        Kept as a method (not a cached property) so post-mutation calls
        always reflect the current segments.
        """
        return pathlib.Path(self.full_path() or ".")

    def full_path(self) -> str:
        body = "/".join(self.parts)
        if self.anchor:
            return self.anchor + body if body else self.anchor
        return body

    # ---- Metadata -----------------------------------------------------

    def _refresh_metadata(self) -> None:
        pl = self._pl()
        try:
            st = pl.stat()
        except (FileNotFoundError, NotADirectoryError, OSError):
            self.reset_metadata(is_file=False, is_dir=False, size=0, mtime=None)
            return
        mode = st.st_mode
        self.reset_metadata(
            is_file=stat_mod.S_ISREG(mode),
            is_dir=stat_mod.S_ISDIR(mode),
            size=st.st_size,
            mtime=st.st_mtime,
        )

    def is_symlink(self) -> bool:
        try:
            return self._pl().is_symlink()
        except OSError:
            return False

    # ---- Listing ------------------------------------------------------

    def _ls_impl(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["LocalPath"]:
        pl = self._pl()
        if not pl.exists():
            if allow_not_found:
                return
            raise FileNotFoundError(str(pl))
        if pl.is_file():
            # Mirrors DatabricksPath: listing a file yields itself.
            yield self
            return
        try:
            children = list(pl.iterdir())
        except (FileNotFoundError, PermissionError):
            if allow_not_found:
                return
            raise
        for child in children:
            yield self._child(child)
            if recursive and child.is_dir():
                yield from self._child(child)._ls_impl(
                    recursive=True, allow_not_found=allow_not_found
                )

    def _child(self, child_pl: pathlib.Path) -> "LocalPath":
        """Build a :class:`LocalPath` for a discovered child.

        Constructs directly (rather than through :meth:`parse`) — we
        already have a resolved :class:`pathlib.Path`, so splitting a
        string would just lose information.
        """
        anchor = _normalize_anchor(child_pl.anchor)
        raw_parts = child_pl.parts
        if anchor and raw_parts and raw_parts[0] == child_pl.anchor:
            raw_parts = raw_parts[1:]
        return LocalPath(parts=list(raw_parts), anchor=anchor, _fs=self._fs)

    # ---- Mutations ----------------------------------------------------

    def _mkdir_impl(self, parents: bool = True, exist_ok: bool = True) -> None:
        self._pl().mkdir(parents=parents, exist_ok=exist_ok)
        # Skip the remote round-trip — we just created this.
        self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=None)

    def _remove_file_impl(self, allow_not_found: bool = True) -> None:
        try:
            self._pl().unlink()
        except FileNotFoundError:
            if not allow_not_found:
                raise
        finally:
            self.reset_metadata()

    def _remove_dir_impl(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        pl = self._pl()
        try:
            if recursive:
                shutil.rmtree(pl)
            else:
                pl.rmdir()
        except FileNotFoundError:
            if not allow_not_found:
                raise
        except OSError:
            if allow_not_found and not pl.exists():
                pass
            else:
                raise
        if not with_root:
            # Caller wanted contents-only; recreate the empty directory.
            pl.mkdir(parents=True, exist_ok=True)
            self.reset_metadata(is_file=False, is_dir=True, size=0, mtime=None)
        else:
            self.reset_metadata()

    def rename(self, target: Any) -> "LocalPath":
        """Native rename — avoids the copy-then-delete default path."""
        target_path = self._coerce(target)
        self._pl().rename(target_path._pl())
        self.reset_metadata()
        target_path._is_file = self._is_file
        target_path._is_dir = self._is_dir
        return target_path

    def copy_to(self, dest: Any, allow_not_found: bool = True) -> "LocalPath":
        """Native copy — falls back to ``shutil`` for speed."""
        dest_path = self._coerce(dest)
        src_pl = self._pl()
        if src_pl.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_pl, dest_path._pl())
        elif src_pl.is_dir():
            dest_path._pl().mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_pl, dest_path._pl(), dirs_exist_ok=True)
        elif not allow_not_found:
            raise FileNotFoundError(str(src_pl))
        return dest_path

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """Native touch — update mtime without rewriting the file."""
        try:
            self._pl().touch(mode=mode, exist_ok=exist_ok)
        except FileExistsError:
            # Preserve the default error message wording for consistency.
            raise FileExistsError(f"Path already exists: {self.full_path()!r}")
        self.reset_metadata()

    # ---- File IO -----------------------------------------------------

    def open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> IO:
        return self._pl().open(
            mode=mode, encoding=encoding, errors=errors, newline=newline
        )

    # ---- Optimized memory access -------------------------------------

    @property
    def local_os_path(self) -> str:
        """Return this path as a local-filesystem string (always available)."""
        return self.full_path() or "."

    @property
    def is_local_fs(self) -> bool:
        return True

    def pread(self, n: int, pos: int) -> bytes:
        """Positional read via :func:`os.pread` when available."""
        if n <= 0:
            return b""
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        if not hasattr(os, "pread"):
            return super().pread(n, pos)
        with self.open("rb") as fh:
            return os.pread(fh.fileno(), n, pos)

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
    ) -> int:
        """Positional write via :func:`os.pwrite` when available."""
        mv = memoryview(data)
        if len(mv) == 0:
            return 0
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")
        if not hasattr(os, "pwrite"):
            return super().pwrite(data, pos)

        pl = self._pl()
        if not pl.exists():
            pl.parent.mkdir(parents=True, exist_ok=True)
            pl.touch()
        with self.open("r+b") as fh:
            return int(os.pwrite(fh.fileno(), mv, pos))

    def open_mmap(self, mode: str = "r") -> Optional["mmap.mmap"]:
        """Return a real :class:`mmap.mmap` over this file.

        Returns ``None`` for empty or missing files because ``mmap.mmap``
        rejects length=0 on most platforms — cheaper and clearer than
        letting the caller handle the OSError.
        """
        pl = self._pl()
        if not pl.exists():
            return None
        try:
            size = pl.stat().st_size
        except OSError:
            return None
        if size == 0:
            return None
        access = mmap.ACCESS_READ if mode == "r" else mmap.ACCESS_WRITE
        open_mode = "rb" if mode == "r" else "r+b"
        with pl.open(open_mode) as fh:
            return mmap.mmap(fh.fileno(), length=0, access=access)

    def memoryview(
        self,
        *,
        offset: int = 0,
        size: Optional[int] = None,
    ) -> "memoryview":
        """Zero-copy memoryview backed by an OS mmap when possible."""
        if offset < 0:
            raise ValueError("memoryview offset must be >= 0")
        mm = self.open_mmap("r")
        if mm is None:
            return memoryview(b"")
        mv = memoryview(mm)
        if offset == 0 and size is None:
            return mv
        end = len(mv) if size is None else offset + size
        return mv[offset:end]

    # ---- pathlib-style convenience factories -------------------------

    @classmethod
    def cwd(cls) -> "LocalPath":
        return cls.parse(os.getcwd())

    @classmethod
    def home(cls) -> "LocalPath":
        return cls.parse(os.path.expanduser("~"))


def _normalize_anchor(anchor: str) -> str:
    """Normalize a pathlib anchor to our forward-slash convention.

    Windows anchors like ``'C:\\'`` are kept as-is; they still render
    cleanly through ``full_path()`` because we only join ``parts`` with
    ``'/'`` after them. Empty stays empty.
    """
    if not anchor:
        return ""
    return anchor.replace("\\", "/")


# ---------------------------------------------------------------------------
# LocalFileSystem
# ---------------------------------------------------------------------------


class LocalFileSystem(FileSystem):
    """Local-filesystem backend. Produces :class:`LocalPath` instances.

    Stateless — you can share one instance across the process or build a
    fresh one per call site, both work.
    """

    scheme: ClassVar[str] = "file"
    path_class: ClassVar[type] = LocalPath

    @classmethod
    def default(cls) -> "LocalFileSystem":
        """Return a shared :class:`LocalFileSystem` singleton.

        Cheap to build, but giving callers a canonical instance makes
        equality checks and caching easier downstream.
        """
        inst = getattr(cls, "_default", None)
        if inst is None:
            inst = cls()
            cls._default = inst  # type: ignore[attr-defined]
        return inst
