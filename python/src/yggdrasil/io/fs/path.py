"""Abstract filesystem path — ``pathlib.Path``-like API over :class:`URL`.

Path is a backend-agnostic byte holder. Positional ops
(:meth:`pread` / :meth:`pwrite` / :meth:`truncate`) route directly
through the subclass's :meth:`_bread` / :meth:`_bwrite` whole-file
primitives — there is no internal buffering at this layer. Callers
that want to coalesce writes wrap the path in a
:class:`yggdrasil.io.buffer.bytes_io.BytesIO`, which carries its
own :class:`Memory`/:class:`Path` holder and only commits on flush.

The fd-driven fast path lives entirely in
:class:`yggdrasil.io.fs.local_path.LocalPath` — the only backend
that holds a kernel file descriptor. Other backends (S3, Databricks,
in-memory) implement :meth:`_bread` / :meth:`_bwrite` directly
against their SDK and pay the whole-file cost on every call.

Subclasses implement seven hooks: :meth:`full_path`, :meth:`_stat`,
:meth:`_ls`, :meth:`_mkdir`, :meth:`_remove_file`, :meth:`_remove_dir`,
:meth:`_bread`, :meth:`_bwrite`.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import pathlib
import re
import shutil
import time
from abc import ABC, abstractmethod
from typing import (
    IO,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.disposable import Disposable
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaType, Mode
from yggdrasil.io.holder import Holder
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL
from yggdrasil.lazy_imports import local_path_class, tabular_io_class, PATH_SCHEME_FACTORY

__all__ = ["Path", "register_path_class"]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subclass registry
# ---------------------------------------------------------------------------

_PATH_REGISTRY: List[type] = []


def register_path_class(cls: type) -> type:
    """Register *cls* as a candidate for :meth:`Path.__new__` dispatch."""
    if cls not in _PATH_REGISTRY:
        _PATH_REGISTRY.append(cls)
    return cls


def _select_path_class(obj: Any, default: type = ...) -> type:
    """Pick the best :class:`Path` subclass for *obj*."""
    if isinstance(obj, Path):
        target = type(obj)
        if target is not Path:
            return target

    for candidate in _PATH_REGISTRY:
        try:
            if candidate.handles(obj):
                return candidate
        except Exception:
            continue

    if hasattr(obj, "scheme"):
        scheme = obj.scheme
        factory = PATH_SCHEME_FACTORY.get(scheme)
        if factory is not None:
            return factory()

    default = local_path_class() if default is ... else default
    return default


# ---------------------------------------------------------------------------
# Staging sweep rate-limit — TTL'd dict keyed by parent full_path()
# ---------------------------------------------------------------------------

_STAGING_SWEEP_INTERVAL_S: float = 300.0       # 5 minutes default
_STAGING_SWEEP_MAX_KEYS: int = 256
_STAGING_SWEPT: ExpiringDict[str, bool] = ExpiringDict(
    default_ttl=_STAGING_SWEEP_INTERVAL_S,
    max_size=_STAGING_SWEEP_MAX_KEYS,
)

# Match a TTL-encoded staging filename: ``<prefix>-<start>-<end>-<seed>(.ext)*``
_STAGING_TMP_RE: re.Pattern = re.compile(r"-(\d+)-(\d+)-[0-9a-f]+(?:\.[^/]+)?$")


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------


class Path(TabularIO[CastOptions], Holder, os.PathLike, ABC):
    """Abstract filesystem path with :class:`pathlib.Path`-like behaviour.

    Stateless I/O surface. Positional ops (:meth:`pread`, :meth:`pwrite`,
    :meth:`truncate`) route straight through the subclass's
    :meth:`_bread` / :meth:`_bwrite` whole-file primitives — no
    internal buffering. :class:`LocalPath` adds an fd fast path on
    top via :meth:`acquire_io`; remote backends pay one round-trip
    per call (callers coalesce by wrapping in :class:`BytesIO`).

    Concrete backends override the seven abstract hooks
    (:meth:`full_path`, :meth:`_stat`, :meth:`_ls`, :meth:`_mkdir`,
    :meth:`_remove_file`, :meth:`_remove_dir`, :meth:`_bread`,
    :meth:`_bwrite`) plus :meth:`_open_fd` for local-fd backends.
    """

    scheme: ClassVar[str] = ""
    __slots__ = (
        "url",
        "temporary",
        "_mode",
    )

    _STAGING_SWEEP_INTERVAL: ClassVar[float] = _STAGING_SWEEP_INTERVAL_S

    # ==================================================================
    # Construction / dispatch
    # ==================================================================

    @classmethod
    def default_media_type(cls):
        """Path is format-agnostic — never auto-register against a mime type."""
        return None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        try:
            super().__init_subclass__(**kwargs)
        except TypeError:
            pass
        register_path_class(cls)

    def __new__(cls, obj: Any = None, *args: Any, **kwargs: Any) -> "Path":
        del args, kwargs
        if cls is Path:
            target = _select_path_class(obj)
            return target.__new__(target, obj)
        return object.__new__(cls)

    def __init__(
        self,
        obj: Any = None,
        *,
        url: URL | None = None,
        temporary: bool = False,
        mode: str = "rb+",
        auto_open: bool = True,
    ) -> None:
        TabularIO.__init__(self, media_type=None)

        if url is not None:
            resolved = URL.from_(url)
        elif obj is None:
            resolved = URL.empty()
        elif isinstance(obj, Path):
            resolved = obj.url
        else:
            resolved = URL.from_(obj)

        self.url = resolved
        self.temporary = bool(temporary)
        self._mode = mode

        if auto_open:
            Disposable.open(self)

    # ==================================================================
    # Disposable hooks — lifecycle is cheap; subclasses add per-open IO
    # ==================================================================

    def _acquire(self) -> None:
        # Stateless at the base. :class:`LocalPath` overrides to open
        # a long-lived fd; everything else has nothing to acquire.
        return

    def _release(self) -> None:
        try:
            self.unpersist()
        except Exception:
            pass
        try:
            self.close_io()
        except Exception:
            pass
        if not self.temporary:
            return
        try:
            self.unlink(missing_ok=True)
        except Exception:
            pass

    # ==================================================================
    # I/O acquire — subclass-specific (fd for local, no-op otherwise)
    # ==================================================================

    @property
    def io_open(self) -> bool:
        """True when a per-open IO backing (subclass-specific) is active.

        Default: ``False`` — the base path holds no per-open state.
        :class:`LocalPath` overrides to track its fd.
        """
        return False

    @property
    def mode(self) -> str:
        """The mode used for the next acquire (or already-open backing)."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if self.io_open and value != self._mode:
            raise RuntimeError(
                f"Cannot change mode while {self!r} is open. "
                "Call close_io() first."
            )
        self._mode = value

    @property
    def is_writing(self) -> bool:
        """True when the active mode includes any write semantics."""
        return any(c in self._mode for c in "wax+")

    def acquire_io(self, mode: Optional[str] = None) -> "Path":
        """Open the per-open IO backing (subclass-specific) explicitly.

        Idempotent for the same mode. If a different mode was already
        open, closes and reopens. Default base impl just records the
        new mode; :class:`LocalPath` overrides to manage its fd.
        Returns ``self`` so the call chains.
        """
        if mode is not None and mode != self._mode:
            if self.io_open:
                self.close_io()
            self._mode = mode
        return self

    def close_io(self) -> None:
        """Release per-open IO state. Default no-op.

        Subclasses with extra per-open resources (fd for
        :class:`LocalPath`) override.
        """

    @contextlib.contextmanager
    def opened(self, mode: str = "rb+") -> "Iterator[Path]":
        """Context manager: acquire I/O backing, release on exit."""
        prev_mode = self._mode
        was_open = self.io_open
        self.acquire_io(mode)
        try:
            yield self
        finally:
            if not was_open:
                self.close_io()
            if not was_open:
                self._mode = prev_mode

    def flush(self) -> None:
        """No-op at the base — every :meth:`pwrite` already committed.

        :class:`LocalPath`'s fd writes hit the kernel via :func:`os.pwrite`;
        remote :meth:`_bwrite` calls already issued the upload. There's
        nothing buffered between :class:`Path` and the durable backing.
        """

    # ==================================================================
    # TabularIO hooks — open the path, dispatch to its BytesIO
    # ==================================================================

    def _read_arrow_batches(self, options: CastOptions) -> Iterator["pa.RecordBatch"]:
        buf = self.open("rb")
        try:
            yield from buf.read_arrow_batches(options=options)
        finally:
            buf.close()

    def _write_arrow_batches(
        self,
        batches: Iterable["pa.RecordBatch"],
        options: CastOptions,
    ) -> None:
        buf = self.open("wb")
        try:
            buf.write_arrow_batches(batches, options=options)
        finally:
            buf.close()

    # ==================================================================
    # Temporary-flag builders
    # ==================================================================

    def as_temporary(self) -> "Path":
        self.temporary = True
        return self

    def as_persistent(self) -> "Path":
        self.temporary = False
        return self

    def with_tmp_name(
        self,
        prefix: str = "tmp-",
        suffix: str = "",
        ttl: int | None = 86400,
        append: bool = True,
        *,
        temporary: bool = True,
    ) -> "Path":
        """Mint a unique sibling/child path with a TTL-encoded name."""
        seed = os.urandom(8).hex()
        prefix = prefix or ""
        suffix = suffix or ""

        if ttl is None:
            name = f"{prefix}{seed}{suffix}"
        else:
            start = int(time.time())
            end = start + ttl
            name = f"{prefix}{start:012d}-{end:012d}-{seed}{suffix}"

        out = (self if append else self.parent) / name
        if temporary:
            out.as_temporary()
        return out

    # ==================================================================
    # Staging — generic, rate-limited, sweep-aware
    # ==================================================================

    def make_staging(
        self,
        path: Union[str, Iterable[str], None] = None,
        *,
        ttl: int = 3600,
        media_type: Union["MediaType", str, None] = None,
        sweep: bool = True,
        force_sweep: bool = False,
    ) -> "Path":
        """Mint a fresh temporary staging file under this directory."""
        parent = self if path is None else self._join_segments(path)

        if sweep:
            try:
                parent._sweep_expired_staging(force=force_sweep)
            except Exception:
                LOGGER.debug(
                    "Staging sweep failed for %s; continuing",
                    parent, exc_info=True,
                )

        ext = self._staging_extension(media_type)
        suffix = f".{ext}" if ext else ""
        return parent.with_tmp_name(
            prefix="tmp-",
            suffix=suffix,
            ttl=ttl,
            append=True,
            temporary=True,
        )

    def _join_segments(self, path: Union[str, Iterable[str]]) -> "Path":
        if isinstance(path, str):
            return self / path
        return self.joinpath(*path)

    @staticmethod
    def _staging_extension(media_type: Union["MediaType", str, None]) -> str:
        if media_type is None:
            return ""
        try:
            from yggdrasil.io.enums import MediaType, MediaTypes
            mt = MediaType.from_(media_type, default=MediaTypes.PARQUET)
            return mt.full_extension or ""
        except Exception:
            if isinstance(media_type, str):
                return media_type.lstrip(".")
            return ""

    def _sweep_expired_staging(self, *, force: bool = False) -> bool:
        key = self.full_path()
        if not force:
            if key in _STAGING_SWEPT:
                return False
        _STAGING_SWEPT[key] = True

        now_ts = int(time.time())
        try:
            for candidate in self.ls(recursive=True, allow_not_found=True):
                match = _STAGING_TMP_RE.search(candidate.name)
                if match is None:
                    continue
                try:
                    end_ts = int(match.group(2))
                except (TypeError, ValueError):
                    continue
                if end_ts >= now_ts:
                    continue
                try:
                    candidate.remove(recursive=False, allow_not_found=True)
                except Exception:
                    LOGGER.debug(
                        "Failed to remove expired staging file %s",
                        candidate, exc_info=True,
                    )
        except Exception:
            LOGGER.debug(
                "Failed to sweep expired staging files under %s",
                self, exc_info=True,
            )
        return True

    @classmethod
    def reset_staging_sweep_state(cls, parent_full_path: Optional[str] = None) -> None:
        if parent_full_path is None:
            _STAGING_SWEPT.clear()
        else:
            try:
                del _STAGING_SWEPT[parent_full_path]
            except KeyError:
                pass

    # ==================================================================
    # Classification — for the dispatch registry
    # ==================================================================

    @classmethod
    def handles(cls, obj: Any) -> bool:
        if not cls.scheme:
            return False
        if isinstance(obj, URL):
            return obj.scheme == cls.scheme
        if isinstance(obj, str):
            return obj.startswith(f"{cls.scheme}:/")
        try:
            return URL.from_(obj).scheme == cls.scheme
        except (ValueError, TypeError):
            return False

    @classmethod
    def is_pathish(cls, obj: Any) -> bool:
        if isinstance(obj, str):
            if not obj:
                return True
            if len(obj) > 256 * 1024:
                return False
            if any(c in obj for c in "\t\n\r\f\v{}[]*?"):
                return False
            return True
        if isinstance(obj, (Path, pathlib.PurePath, os.PathLike)):
            return True
        try:
            return URL.is_pathish(obj)
        except Exception:
            return False

    @property
    def is_memory(self) -> bool:
        """Paths are never raw memory holders — :class:`Memory` is the
        in-process bytearray shape, not anything reached via path
        addressing."""
        return False

    @property
    def is_remote_path(self) -> bool:
        """Default: every path that isn't a local-fs path is remote."""
        return not self.is_local_path

    # ==================================================================
    # Coercion entry points
    # ==================================================================

    @classmethod
    def from_(
        cls,
        obj: Any,
        default: Any = ...,
        *,
        temporary: bool = False,
        **kwargs
    ) -> "Path":
        if isinstance(obj, Path):
            same_type = type(obj) is cls or cls is Path
            if same_type:
                if temporary:
                    obj.temporary = True
                return obj
            return cls.from_url(obj.url, default=default, temporary=temporary, **kwargs)

        try:
            url = URL.from_(obj)
        except (ValueError, TypeError):
            if default is ...:
                raise
            return default

        return cls.from_url(url, default=default, temporary=temporary, **kwargs)

    @classmethod
    def from_url(
        cls,
        url: URL,
        default: Any = ...,
        *,
        temporary: bool = False,
        **kwargs
    ) -> "Path":
        try:
            resolved = URL.from_(url)
        except (ValueError, TypeError):
            if default is ...:
                raise
            return default

        target = _select_path_class(resolved) if cls is Path else cls
        return target(url=resolved, temporary=temporary, **kwargs)

    @classmethod
    def from_pathlib(
        cls,
        path: pathlib.PurePath,
        default: Any = ...,
        *,
        temporary: bool = False,
    ) -> "Path":
        try:
            url = URL.from_(path)
        except (ValueError, TypeError):
            if default is ...:
                raise
            return default
        return cls.from_url(url, default=default, temporary=temporary)

    # ==================================================================
    # Abstract hooks
    # ==================================================================

    @abstractmethod
    def full_path(self) -> str: ...

    @abstractmethod
    def _stat(self) -> IOStats: ...

    @abstractmethod
    def _ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]: ...

    @abstractmethod
    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None: ...

    @abstractmethod
    def _remove_file(self, allow_not_found: bool = True) -> None: ...

    @abstractmethod
    def _remove_dir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None: ...

    @abstractmethod
    def _bread(self, n: int, pos: int, mode: Mode) -> BytesIO:
        """Positional read primitive — return up to ``n`` bytes at ``pos``
        wrapped in a fresh :class:`BytesIO`.

        ``n < 0`` reads from ``pos`` to EOF. ``mode`` is the
        :class:`Mode` the caller is opening at; subclasses use it
        when their backend differentiates read shapes (e.g.
        :class:`Mode.READ_ONLY` may skip cache invalidation that an
        in-place edit needs).

        The returned :class:`BytesIO` is autonomous — the caller is
        responsible for closing it. Whole-file backends (S3,
        Databricks) implement this by downloading the full payload
        and slicing into a fresh buffer; backends with native
        positional IO (LocalPath via ``os.pread``) override directly.
        """

    @abstractmethod
    def _bwrite(self, data: BytesIO, pos: int, mode: Mode) -> int:
        """Positional write primitive — splice ``data`` at ``pos``.

        ``data`` is a :class:`BytesIO` carrying the bytes to splice;
        the implementation reads from it (typically from offset 0)
        and lays them down starting at ``pos`` on the backing.
        Returns the number of bytes actually written. Grows the
        backing if ``pos + data.size`` exceeds the current size.
        ``mode`` carries the disposition (overwrite / append / …)
        for backends that need it on every call; positional writes
        otherwise honour the caller's intent regardless.

        Whole-file backends implement via download → splice →
        upload (read-modify-write); native-positional backends
        (LocalPath via ``os.pwrite``) override directly.
        """

    # ==================================================================
    # I/O entry points
    # ==================================================================

    def open(
        self,
        mode: "Mode | str | None" = None,
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> "BytesIO | Path":
        """Open the path — pathlike when *mode* is given, lifecycle otherwise.

        Two shapes:

        - ``path.open()`` (no mode) — runs the :class:`Disposable`
          lifecycle ``open`` (sets ``_acquired``); returns ``self``.
          Equivalent to the bare ``Disposable.open(self)`` call sites
          relied on before, kept for ``with path:`` and other lifecycle
          contexts.
        - ``path.open(mode)`` — opens a binary file handle at *mode*
          and returns a :class:`BytesIO` bound to *self*. Mirrors
          :meth:`pathlib.Path.open` for byte-level I/O. ``mode``
          accepts a :class:`Mode`, an OS-style mode string
          (``"rb"`` / ``"wb"`` / ``"ab"`` / ``"rb+"`` / …), or any
          alias :meth:`Mode.from_` understands. The text-IO knobs
          (``encoding`` / ``errors`` / ``newline``) are accepted for
          signature parity with :class:`pathlib.Path` but ignored —
          this is a binary surface.
        """
        if mode is None:
            del encoding, errors, newline, auto_open, touch
            Disposable.open(self)
            return self

        del encoding, errors, newline  # binary-only at this layer

        if not self.opened:
            Disposable.open(self)

        parsed = Mode.from_(mode)
        os_mode = parsed.os_mode if isinstance(mode, Mode) or not isinstance(mode, str) else mode
        if touch and parsed is Mode.READ_ONLY and not self.exists():
            self.touch(exist_ok=True, parents=True)

        return BytesIO(path=self, mode=os_mode, auto_open=auto_open)

    # ==================================================================
    # URL-delegated pure-path API
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
    def extensions(self) -> List[str]:
        return self.url.extensions

    @property
    def media_type(self):
        return self.url.media_type

    @property
    def mime_type(self):
        return self.url.mime_type

    @property
    def codec(self):
        return self.url.codec

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

    def __truediv__(self, other: Any) -> "Path":
        return self.joinpath(other)

    def __rtruediv__(self, other: Any) -> "Path":
        return Path.from_(other) / self

    def with_name(self, name: str) -> "Path":
        if not self.parts:
            raise ValueError(
                f"Cannot set name on empty path {self!r}. "
                "Build a non-empty path first, e.g. Path('dir') / name."
            )
        if not name:
            raise ValueError("Name cannot be empty")
        if "/" in name:
            raise ValueError(
                f"Invalid name {name!r}: cannot contain '/'. "
                "Use joinpath() for nested paths."
            )
        return self.parent / name

    def with_suffix(self, suffix: str) -> "Path":
        if suffix and not suffix.startswith("."):
            raise ValueError(
                f"Invalid suffix {suffix!r}: must start with '.' "
                "(e.g. '.parquet') or be '' to strip the existing suffix."
            )
        return self.with_name(self.stem + suffix)

    def with_stem(self, stem: str) -> "Path":
        return self.with_name(stem + self.suffix)

    def match_pattern(self, pattern: str) -> bool:
        return self.url.match_pattern(pattern)

    def matches_patterns(self, patterns: Iterable[str] | None) -> bool:
        return self.url.matches_patterns(patterns)

    def is_relative_to(self, other: Any) -> bool:
        if isinstance(other, Path):
            if type(self) is not type(other):
                return False
            return self.url.is_relative_to(other.url)
        try:
            other_path = Path.from_(other)
        except (ValueError, TypeError):
            return False
        if type(self) is not type(other_path):
            return False
        return self.url.is_relative_to(other_path.url)

    def relative_to(self, other: Any) -> "Path":
        if isinstance(other, Path):
            other_path = other
        else:
            other_path = Path.from_(other)

        if not self.is_relative_to(other_path):
            raise ValueError(
                f"{self.full_path()!r} is not relative to "
                f"{other_path.full_path()!r}. Use is_relative_to() to test "
                "the relationship first."
            )
        return self._from_url(self.url.relative_to(other_path.url))

    # ==================================================================
    # Stat — uncached, every call hits the backend
    # ==================================================================

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks
        return self._stat().kind != IOKind.MISSING

    def is_file(self) -> bool:
        return self._stat().kind == IOKind.FILE

    def is_dir(self) -> bool:
        return self._stat().kind == IOKind.DIRECTORY

    def is_symlink(self) -> bool:
        return False

    def is_dir_sink(self) -> bool:
        if self.url.path.endswith("/"):
            return True
        return self.is_dir()

    @property
    def size(self) -> int:
        return int(self._stat().size)

    @property
    def mtime(self) -> float:
        s = self._stat()
        if s.kind == IOKind.MISSING:
            return 0.0
        return float(s.mtime or 0.0)

    def stat(self) -> IOStats:
        """One backend round-trip → ``IOStats`` (kind + size + mtime + mode + media_type)."""
        s = self._stat()
        s.media_type = self.media_type
        return s

    # ==================================================================
    # Listing / walking
    # ==================================================================

    def iterdir(self) -> Iterator["Path"]:
        yield from self._ls(recursive=False, allow_not_found=True)

    def ls(
        self,
        *,
        recursive: bool = False,
        allow_not_found: bool = True,
        include_patterns: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        exclude_private: bool = False,
    ) -> Iterator["Path"]:
        includes = _materialize(include_patterns)
        excludes = _materialize(exclude_patterns)

        if includes is None and excludes is None and not exclude_private:
            yield from self._ls(recursive=recursive, allow_not_found=allow_not_found)
            return

        def _dropped(child: "Path") -> bool:
            if exclude_private and child.name.startswith("."):
                return True
            if excludes and child.matches_patterns(excludes):
                return True
            return False

        if not recursive:
            for child in self._ls(recursive=False, allow_not_found=allow_not_found):
                if _dropped(child):
                    continue
                if includes is None or child.matches_patterns(includes):
                    yield child
            return

        stack: List[Path] = [self]
        while stack:
            current = stack.pop()
            for child in current._ls(
                recursive=False, allow_not_found=allow_not_found
            ):
                if _dropped(child):
                    continue
                if includes is None or child.matches_patterns(includes):
                    yield child
                if child.is_dir():
                    stack.append(child)

    def glob(self, pattern: str) -> Iterator["Path"]:
        for child in self._ls(recursive=True, allow_not_found=True):
            if child.match_pattern(pattern):
                yield child

    def rglob(self, pattern: str) -> Iterator["Path"]:
        yield from self.glob(pattern)

    def walk(self) -> Iterator[Tuple["Path", List["Path"], List["Path"]]]:
        dirs: List[Path] = []
        files: List[Path] = []
        for child in self.iterdir():
            (dirs if child.is_dir() else files).append(child)
        yield self, dirs, files
        for sub in dirs:
            yield from sub.walk()

    # ==================================================================
    # Filesystem mutators
    # ==================================================================

    def mkdir(
        self,
        mode: int = 0o777,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> "Path":
        del mode
        self._mkdir(parents=parents, exist_ok=exist_ok)
        return self

    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> "Path":
        self._remove_dir(
            recursive=recursive,
            allow_not_found=allow_not_found,
            with_root=with_root,
        )
        return self

    def rmfile(self, allow_not_found: bool = True) -> "Path":
        self._remove_file(allow_not_found=allow_not_found)
        return self

    def unlink(self, missing_ok: bool = True) -> None:
        kind = self._stat().kind
        if kind == IOKind.MISSING:
            if missing_ok:
                return
            raise FileNotFoundError(f"{self.full_path()!r} does not exist")
        if kind == IOKind.DIRECTORY:
            raise IsADirectoryError(
                f"Cannot unlink directory {self.full_path()!r}; "
                "use rmdir() or remove() for trees."
            )
        self._remove_file(allow_not_found=missing_ok)

    def remove(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
    ) -> "Path":
        kind = self._stat().kind
        if kind == IOKind.FILE:
            self._remove_file(allow_not_found=allow_not_found)
        elif kind == IOKind.DIRECTORY:
            self._remove_dir(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=True,
            )
        elif kind == IOKind.MISSING:
            if not allow_not_found:
                raise FileNotFoundError(f"{self!r} does not exist")
        return self

    def rename(self, target: Any) -> "Path":
        target_path = Path.from_(target)
        self.copy_to(target_path)
        self.remove(recursive=True)
        return target_path

    def touch(
        self,
        mode: int = 0o666,
        exist_ok: bool = True,
        parents: bool = True,
    ) -> None:
        del mode
        if not exist_ok:
            try:
                self.write_bytes(b"", mode="xb", parents=parents)
                return
            except FileExistsError:
                raise FileExistsError(
                    f"Path already exists: {self.full_path()!r}"
                )
        self.write_bytes(b"", parents=parents)

    def resolve(self, *, strict: bool = False) -> "Path":
        del strict
        return self

    def absolute(self) -> "Path":
        return self

    # ==================================================================
    # Bytes / text I/O
    # ==================================================================

    def pread(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes:
        """Positional read.

        Resolves negative *pos* against :attr:`size` (so ``-1`` reads
        the last byte, ``-n`` reads from N bytes before EOF). Routes
        through the :meth:`_bread` whole-file primitive and unwraps
        its :class:`BytesIO` into a fresh :class:`bytes`.
        """
        if pos < 0:
            pos = max(0, int(self.size) + pos)
        if n == 0:
            return b""

        try:
            bio = self._bread(
                n, pos, Mode.from_(self._mode, default=Mode.READ_ONLY),
            )
        except (OSError, ValueError):
            if default is ...:
                raise
            return default
        try:
            return bio.to_bytes()
        finally:
            bio.close()

    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        """Positional write.

        Resolves negative *pos* against :attr:`size` (``-1`` splices
        before the last byte; ``-n`` splices N bytes before EOF).
        Wraps ``data`` in a :class:`BytesIO` and routes through the
        :meth:`_bwrite` positional primitive — whole-file backends
        absorb the read-modify-write cost internally.
        """
        del parents
        mv = memoryview(data)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        n = len(mv)
        if n == 0:
            return 0
        if pos < 0:
            pos = max(0, int(self.size) + pos)

        scratch = BytesIO(bytes(mv))
        try:
            return self._bwrite(
                scratch, pos, Mode.from_(self._mode, default=Mode.AUTO),
            )
        finally:
            scratch.close()

    def truncate(self, n: int, *, parents: bool = True) -> int:
        del parents
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")

        stat = self._stat()
        if stat.kind == IOKind.MISSING:
            raise FileNotFoundError(
                f"Cannot truncate non-existent path {self.full_path()!r}. "
                "Call touch() first if you want create-or-resize semantics."
            )
        if stat.kind == IOKind.DIRECTORY:
            raise IsADirectoryError(
                f"Cannot truncate directory {self.full_path()!r}"
            )

        current = int(stat.size)
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
        """:class:`Holder` primitive: drop the backing file entirely.

        Unlinks via :meth:`unlink` with ``missing_ok=True`` so a
        missing target is a no-op.
        """
        self.unlink(missing_ok=True)

    # ==================================================================
    # Holder primitives — read_mv / write_mv / reserve
    # ==================================================================

    def read_mv(self, n: int, pos: int) -> memoryview:
        """:class:`Holder` primitive: memoryview slice at ``pos``."""
        if pos < 0:
            raise ValueError(f"read_mv pos must be >= 0, got {pos!r}")
        # Materialize via pread; fd-mode and transaction-buffer-mode
        # both copy the bytes out. The returned view backs onto the
        # produced bytes object so the caller owns it.
        data = self.pread(n, pos, default=b"")
        return memoryview(data)

    def write_mv(self, data: memoryview, pos: int) -> int:
        """:class:`Holder` primitive: splice memoryview at ``pos``."""
        if data.format != "B" or data.ndim != 1 or data.itemsize != 1:
            data = data.cast("B")
        if not data.c_contiguous:
            data = memoryview(bytes(data))
        return self.pwrite(data, pos)

    def reserve(self, n: int) -> None:
        """:class:`Holder` primitive: pre-grow capacity.

        Files have no separate "capacity" knob — :func:`posix_fallocate`
        isn't portable, and remote backends don't have the concept.
        Default no-op; backends with a meaningful pre-grow override.
        """
        del n

    def fileno(self) -> int:
        """Backends with a real fd override; the default has none."""
        raise OSError(
            f"{type(self).__name__} has no underlying file descriptor"
        )

    def read_bytes(self, *, raise_error: bool = True) -> bytes:
        """Whole-file read."""
        try:
            bio = self._bread(-1, 0, Mode.READ_ONLY)
        except (OSError, ValueError):
            if raise_error:
                raise
            return b""
        try:
            return bio.to_bytes()
        finally:
            bio.close()

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview],
        *,
        mode: "Mode | str" = "wb",
        parents: bool = True,
    ) -> int:
        """Whole-file write. Routes through :meth:`_bwrite` with the
        :class:`Mode` derived from *mode* (``wb`` → ``OVERWRITE``,
        ``ab`` → ``APPEND``, ``xb`` → ``ERROR_IF_EXISTS``).

        Subclasses with native atomic-append (``O_APPEND``) or
        atomic-create (``O_EXCL``) honour the disposition inside
        :meth:`_bwrite`; backends without those primitives fall back
        to whole-file RMW with a best-effort ``exists()`` guard for
        ``xb``.
        """
        del parents

        parsed = Mode.from_(mode, default=Mode.OVERWRITE)
        if parsed not in (Mode.OVERWRITE, Mode.APPEND, Mode.ERROR_IF_EXISTS):
            raise ValueError(
                f"write_bytes mode must resolve to OVERWRITE, APPEND, or "
                f"ERROR_IF_EXISTS; got {mode!r} → {parsed.name}"
            )

        mv = memoryview(data)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        if mv.format != "B" or mv.ndim != 1 or mv.itemsize != 1:
            mv = mv.cast("B")
        n = len(mv)

        bio = BytesIO()
        bio.open()
        try:
            if n > 0:
                bio.write(bytes(mv))
            bio.seek(0)
            return self._bwrite(bio, 0, parsed)
        finally:
            bio.close()

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
        raise_error: bool = True,
    ) -> str:
        return self.read_bytes(raise_error=raise_error).decode(
            encoding, errors=errors,
        )

    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: Optional[str] = None,
        parents: bool = True,
    ) -> int:
        del newline
        encoded = data.encode(encoding, errors=errors)
        self.write_bytes(encoded, parents=parents)
        return len(encoded)

    def write_bytes_io(
        self,
        buffer: BytesIO,
        *,
        batch_size: int = 1024 * 1024,
        parents: bool = True,
    ):
        buffer = BytesIO.from_(buffer)
        return self._write_bytes_io(buffer, batch_size=batch_size, parents=parents)

    def _write_bytes_io(
        self,
        buffer: BytesIO,
        *,
        batch_size: int = 1024 * 1024,
        parents: bool = True,
    ):
        del batch_size
        buffer = BytesIO.from_(buffer)
        if not buffer.opened:
            buffer.open()
            try:
                return self._write_bytes_io(buffer, parents=parents)
            finally:
                buffer.close()

        if (
            buffer.spilled
            and buffer.is_local
            and self.is_local
            and buffer.path is not None
        ):
            if parents:
                self.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(buffer.path.full_path(), self.full_path())
            except shutil.SameFileError:
                pass
            return buffer.size

        prev_pos = buffer.tell()
        buffer.seek(0)
        try:
            self._bwrite(buffer, 0, Mode.OVERWRITE)
        finally:
            try:
                buffer.seek(prev_pos)
            except Exception:
                pass
        return buffer.size

    # ==================================================================
    # Streaming copy / write_from_stream
    # ==================================================================

    def copy_to(
        self,
        dest: Any,
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        dest_path = Path.from_(dest)
        if dest_path == self:
            return self.size

        if parents:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_local and dest_path.is_local:
            src_full = self.full_path()
            dst_full = dest_path.full_path()
            try:
                shutil.copyfile(src_full, dst_full)
            except shutil.SameFileError:
                return self.size
            try:
                return int(dest_path._stat().size)
            except Exception:
                return self.size

        return self._copy_to_via_stream(dest_path, batch_size=batch_size)

    def _copy_to_via_stream(
        self,
        dest_path: "Path",
        *,
        batch_size: int,
    ) -> int:
        total = 0
        with self.open("rb") as src, dest_path.open("wb") as dst:
            if (
                isinstance(src, BytesIO)
                and src.spilled
                and src.is_local
            ):
                mv = src.memoryview()
                try:
                    n = dst.write(mv)
                    return int(n) if n is not None else len(mv)
                finally:
                    del mv

            while True:
                chunk = src.read(batch_size)
                if not chunk:
                    break
                dst.write(chunk)
                total += len(chunk)
        return total

    def write_stream(
        self,
        src: IO[bytes],
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        if isinstance(src, BytesIO):
            return self.write_bytes_io(src)

        if isinstance(src, io.BytesIO):
            start = src.tell()
            payload = src.getvalue()[start:]
            self.write_bytes(payload, parents=parents)
            return len(payload)

        total = 0
        with self.open("wb") as dst:
            while True:
                chunk = src.read(batch_size)
                if not chunk:
                    break
                dst.write(chunk)
                total += len(chunk)
        return total

    # ==================================================================
    # memoryview / mmap
    # ==================================================================

    def memoryview(
        self,
        *,
        offset: int = 0,
        size: Optional[int] = None,
        raise_error: bool = True,
    ) -> memoryview:
        if offset < 0:
            raise ValueError("memoryview offset must be >= 0")
        n = -1 if size is None else int(size)
        return memoryview(self.pread(n=n, pos=offset, default=raise_error))

    def open_mmap(self, mode: str = "r"):
        del mode
        return None

    # ==================================================================
    # Media type
    # ==================================================================

    def infer_media_type(self, *, default: "MediaType" = ...) -> "MediaType":
        return MediaType.from_path(self, default=default)

    def as_media(self, media_type: MediaType | None = None):
        return tabular_io_class().from_path(self, media_type=media_type)

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _from_url(self, url: URL) -> "Path":
        return type(self)(url=url)

    # ==================================================================
    # Dunder
    # ==================================================================

    def __fspath__(self) -> str:
        return self.url.__fspath__()

    def __hash__(self) -> int:
        return hash((type(self), self.url))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return type(self) is type(other) and self.url == other.url
        if isinstance(other, str):
            return self.full_path() == other
        return NotImplemented

    def __str__(self) -> str:
        return self.full_path()

    def __repr__(self) -> str:
        if self.temporary:
            return f"{type(self).__name__}({self.url!r}, temporary=True)"
        return f"{type(self).__name__}({self.url!r})"

    # ==================================================================
    # Context manager
    # ==================================================================

    def __enter__(self) -> "Path":
        if not self._acquired:
            Disposable.open(self)
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._depth > 0:
            self._depth -= 1
        if self._depth > 0:
            return
        self.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _materialize(patterns: Iterable[str] | None) -> Optional[Tuple[str, ...]]:
    if patterns is None:
        return None
    out = patterns if isinstance(patterns, tuple) else tuple(patterns)
    return out if out else None
