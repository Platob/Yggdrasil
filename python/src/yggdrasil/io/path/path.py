"""Abstract filesystem path — :class:`Holder` over a :class:`URL`.

A :class:`Path` is a byte holder addressed by a URL. The holder
contract (:meth:`read_mv` / :meth:`write_mv` / :meth:`reserve` /
:meth:`truncate` / :meth:`clear` / :attr:`size`) routes through two
whole-file primitives — :meth:`_bread` / :meth:`_bwrite` — that
subclasses override. There is no buffering at this layer; callers
that want to coalesce wrap the path in
:class:`yggdrasil.io.buffer.BytesIO`.

Subclasses implement seven hooks:

- :meth:`full_path`         — string form of the URL on the backend
- :meth:`_stat`             — ``IOStats`` round-trip (kind + size + mtime)
- :meth:`_ls`               — list children
- :meth:`_mkdir`            — create directory
- :meth:`_remove_file`      — unlink one file
- :meth:`_remove_dir`       — rmtree
- :meth:`_bread`            — positional read → :class:`BytesIO`
- :meth:`_bwrite`           — positional write ← :class:`BytesIO`

The pure-path API (parts, name, parent, suffix, joinpath, …)
delegates straight to :class:`URL` — :class:`Path` adds no parsing
of its own. Subclass dispatch is a small registry keyed by URL
scheme; :meth:`Path.from_` resolves a candidate via the registry
and falls back to :class:`LocalPath`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterator, List, Tuple

from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.data.enums import Mode
from yggdrasil.io.holder import Holder
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.url import URL

__all__ = ["Path"]


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------


class Path(Holder, os.PathLike, ABC):
    """Abstract URL-addressed byte holder with filesystem semantics.

    Two layers, no shared state between them:

    1. **Holder I/O** — :meth:`read_mv` / :meth:`write_mv` /
       :meth:`truncate` / :meth:`clear` / :attr:`size` route through
       :meth:`_bread` / :meth:`_bwrite`. Whole-file by default;
       backends with native positional I/O (LocalPath via
       ``os.pread``/``os.pwrite``) override the holder primitives
       directly.
    2. **Filesystem** — :meth:`stat`, :meth:`exists`, :meth:`is_file`,
       :meth:`is_dir`, :meth:`iterdir`, :meth:`mkdir`, :meth:`unlink`,
       :meth:`remove`. All thin wrappers over the abstract hooks.

    Pure-path manipulation (:attr:`parts`, :attr:`name`, :attr:`parent`,
    :attr:`suffix`, :meth:`joinpath`, :meth:`with_suffix`, …) delegates
    straight to :attr:`url` — Path adds no parsing.
    """

    scheme: ClassVar[str] = ""

    __slots__ = ()

    # ==================================================================
    # Construction / coercion
    # ==================================================================

    @classmethod
    def from_(cls, obj: Any, **kwargs: Any) -> "Path":
        """Coerce *obj* (str / URL / pathlib / Path) into a :class:`Path`.

        When called on the abstract :class:`Path`, dispatches via the
        :class:`Holder` scheme registry to the subclass registered for
        the URL's scheme. When called on a concrete subclass, returns
        an instance of that subclass.
        """
        if isinstance(obj, Path):
            if cls is Path or isinstance(obj, cls):
                return obj
            obj = obj.url

        url = URL.from_(obj)
        if cls is Path:
            # Holder.__new__ dispatches on url.scheme via _HOLDER_SCHEMES.
            return Holder(url=url, **kwargs)
        return cls(url=url, **kwargs)

    def _from_url(self, url: URL) -> "Path":
        """Build a sibling :class:`Path` of the same concrete type."""
        return type(self)(url=url)

    # ==================================================================
    # Backing-shape predicates
    # ==================================================================

    @property
    def is_memory(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        return not self.is_local_path

    # ==================================================================
    # Abstract hooks — backends implement these eight
    # ==================================================================

    @abstractmethod
    def full_path(self) -> str:
        """Backend-native string form of this path's URL."""

    @abstractmethod
    def _stat(self) -> IOStats:
        """One round-trip: ``kind`` + ``size`` + ``mtime``.

        Returns an :class:`IOStats` with ``kind=IOKind.MISSING`` when
        the path does not exist; never raises ``FileNotFoundError``.
        """

    @abstractmethod
    def _ls(self, recursive: bool = False) -> Iterator["Path"]:
        """Yield children. Empty when missing or not a directory."""

    @abstractmethod
    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        """Create directory at this path."""

    @abstractmethod
    def _remove_file(self, missing_ok: bool = True) -> None:
        """Unlink the file at this path."""

    @abstractmethod
    def _remove_dir(self, recursive: bool = True, missing_ok: bool = True) -> None:
        """Remove the directory at this path."""

    @abstractmethod
    def _bread(self, n: int, pos: int, mode: Mode) -> BytesIO:
        """Positional read → fresh :class:`BytesIO`. ``n < 0`` → to EOF.

        Caller owns the returned buffer (must close it). Whole-file
        backends materialize the full payload and slice.
        """

    @abstractmethod
    def _bwrite(self, data: BytesIO, pos: int, mode: Mode) -> int:
        """Splice *data* at *pos* on the backing. Returns bytes written.

        ``mode`` carries disposition (OVERWRITE / APPEND /
        ERROR_IF_EXISTS) for backends that need it on every call;
        positional backends honour caller intent regardless.
        """

    # ==================================================================
    # Holder primitives — built on _bread / _bwrite
    # ==================================================================

    @property
    def size(self) -> int:
        return int(self._stat().size)

    def stat(self) -> IOStats:
        s = self._stat()
        s.media_type = self.url.media_type
        return s

    def _read_mv(self, n: int, pos: int) -> memoryview:
        bio = self._bread(n, pos, Mode.READ_ONLY)
        try:
            return memoryview(bio.to_bytes())
        finally:
            bio.close()

    def _write_mv(self, data: memoryview, pos: int) -> int:
        scratch = BytesIO(bytes(data))
        scratch.open()
        try:
            return self._bwrite(scratch, pos, Mode.OVERWRITE)
        finally:
            scratch.close()

    def reserve(self, n: int) -> None:
        """No-op by default — files have no separate capacity layer."""
        del n

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        current = self.size
        if n == current:
            return n
        bio = self._bread(-1, 0, Mode.READ_ONLY)
        try:
            bio.truncate(n)
            bio.seek(0)
            self._bwrite(bio, 0, Mode.OVERWRITE)
            return n
        finally:
            bio.close()

    def clear(self) -> None:
        """:class:`Holder` primitive: drop the backing entirely (idempotent)."""
        self._remove_file(missing_ok=True)

    # ==================================================================
    # Filesystem surface — thin wrappers over the abstract hooks
    # ==================================================================

    def exists(self) -> bool:
        return self._stat().kind != IOKind.MISSING

    def is_file(self) -> bool:
        return self._stat().kind == IOKind.FILE

    def is_dir(self) -> bool:
        return self._stat().kind == IOKind.DIRECTORY

    @property
    def mtime(self) -> float:
        s = self._stat()
        return float(s.mtime or 0.0) if s.kind != IOKind.MISSING else 0.0

    def iterdir(self) -> Iterator["Path"]:
        yield from self._ls(recursive=False)

    def ls(self, *, recursive: bool = False) -> Iterator["Path"]:
        yield from self._ls(recursive=recursive)

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> "Path":
        self._mkdir(parents=parents, exist_ok=exist_ok)
        return self

    def unlink(self, missing_ok: bool = True) -> None:
        kind = self._stat().kind
        if kind == IOKind.MISSING:
            if not missing_ok:
                raise FileNotFoundError(f"{self.full_path()!r} does not exist")
            return
        if kind == IOKind.DIRECTORY:
            raise IsADirectoryError(
                f"Cannot unlink directory {self.full_path()!r}; use remove()."
            )
        self._remove_file(missing_ok=missing_ok)

    def remove(self, recursive: bool = True, missing_ok: bool = True) -> "Path":
        kind = self._stat().kind
        if kind == IOKind.FILE:
            self._remove_file(missing_ok=missing_ok)
        elif kind == IOKind.DIRECTORY:
            self._remove_dir(recursive=recursive, missing_ok=missing_ok)
        elif kind == IOKind.MISSING and not missing_ok:
            raise FileNotFoundError(f"{self.full_path()!r} does not exist")
        return self

    def touch(self) -> "Path":
        if not self.exists():
            self.write_bytes(b"")
        return self

    # ==================================================================
    # open(mode) — returns a BytesIO bound to self
    # ==================================================================

    def open(self, mode: "Mode | str | None" = None) -> "BytesIO | Path":
        """Two shapes:

        - ``path.open()``       → lifecycle acquire; returns ``self``.
        - ``path.open("rb")``   → :class:`BytesIO` bound to ``self``.
        """
        if mode is None:
            return Holder.open(self)
        return BytesIO(path=self, mode=Mode.from_(mode).os_mode)

    # ==================================================================
    # Pure-path API — all delegated to URL
    # ==================================================================

    @property
    def parts(self) -> List[str]:
        return self.url.parts

    @property
    def name(self) -> str:
        return self.url.name

    @property
    def stem(self) -> str:
        return self.url.stem

    @property
    def suffix(self) -> str:
        exts = self.url.extensions
        return "." + exts[-1] if exts else ""

    @property
    def suffixes(self) -> List[str]:
        return ["." + e for e in self.url.extensions]

    @property
    def parent(self) -> "Path":
        return self._from_url(self.url.parent)

    @property
    def parents(self) -> Tuple["Path", ...]:
        return tuple(self._from_url(u) for u in self.url.parents)

    @property
    def is_absolute(self) -> bool:
        return self.url.is_absolute

    def joinpath(self, *segments: Any) -> "Path":
        return self._from_url(self.url.joinpath(*segments))

    def with_name(self, name: str) -> "Path":
        if not name or "/" in name:
            raise ValueError(f"Invalid name {name!r}")
        return self.parent / name

    def with_suffix(self, suffix: str) -> "Path":
        if suffix and not suffix.startswith("."):
            raise ValueError(f"Invalid suffix {suffix!r}: must start with '.'")
        return self.with_name(self.stem + suffix)

    def with_stem(self, stem: str) -> "Path":
        return self.with_name(stem + self.suffix)

    def __truediv__(self, other: Any) -> "Path":
        return self.joinpath(other)

    def __rtruediv__(self, other: Any) -> "Path":
        return Path.from_(other) / self

    # ==================================================================
    # Dunder
    # ==================================================================

    def __fspath__(self) -> str:
        return self.url.__fspath__()

    def __hash__(self) -> int:
        return hash((type(self), self.url))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self.url == other.url
        if isinstance(other, str):
            return self.full_path() == other
        return NotImplemented

    def __str__(self) -> str:
        return self.full_path()
