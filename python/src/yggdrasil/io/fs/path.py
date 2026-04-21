"""Abstract filesystem path — ``pathlib.Path``-like API, no inheritance.

Modeled on :class:`yggdrasil.databricks.fs.path.DatabricksPath` so local,
remote, and object-store paths share the same surface. Concrete subclasses
fill in a small set of hooks; everything else (pure-path manipulation,
``read_text`` / ``write_text``, ``touch``, ``glob`` / ``rglob``, ``copy_to``)
comes from this base class for free.

Hierarchy
---------
::

    Path (abstract, dataclass)
    ├── LocalPath     — pathlib-backed, scheme ``"file"``
    └── …             — other backends plug in separately

Side classes
------------
- :class:`PathKind`   — file / directory / symlink / other / missing enum
- :class:`StatResult` — dataclass mirroring the useful bits of ``os.stat_result``
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from .filesystem import FileSystem

__all__ = ["PathKind", "StatResult", "Path", "register_path_class"]


# ---------------------------------------------------------------------------
# Path class registry — used by Path.__new__ to pick a concrete subclass
# ---------------------------------------------------------------------------
#
# Registration is automatic via ``Path.__init_subclass__``. The order
# matters: the first class whose :meth:`_match` returns True wins.
# :class:`LocalPath` is explicitly the fallback and is skipped during
# iteration so more specific backends (Databricks, S3, …) take priority.

_PATH_REGISTRY: List[type] = []


def register_path_class(cls: type) -> type:
    """Register *cls* as a candidate for :meth:`Path.__new__` dispatch.

    Subclasses are auto-registered via ``__init_subclass__``; call this
    only for something unusual (e.g. a dynamically-created subclass).
    """
    if cls not in _PATH_REGISTRY:
        _PATH_REGISTRY.append(cls)
    return cls


def _select_path_class(obj: Any) -> type:
    """Pick the best :class:`Path` subclass for *obj*.

    Exact-type match wins (``Path(some_path)`` stays the same type).
    Then registered subclasses get a shot via their ``_match`` classmethod.
    :class:`LocalPath` is the fallback when nothing else claims the input.
    """
    # Local import to avoid a circular import at module load time.
    from .local import LocalPath

    if isinstance(obj, Path):
        return type(obj)

    for candidate in _PATH_REGISTRY:
        if candidate is LocalPath:
            continue
        try:
            if candidate._match(obj):
                return candidate
        except Exception:
            # A misbehaving _match must not break dispatch — we just try
            # the next candidate and fall back to LocalPath if needed.
            continue
    return LocalPath


_SEP = "/"


# ---------------------------------------------------------------------------
# PathKind + StatResult side classes
# ---------------------------------------------------------------------------


class PathKind(str, Enum):
    """What a path points at. ``MISSING`` is the explicit "nothing here" value
    so callers don't have to handle ``None`` for non-existent paths.
    """

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    OTHER = "other"
    MISSING = "missing"

    @property
    def exists(self) -> bool:
        return self is not PathKind.MISSING

    @property
    def is_file(self) -> bool:
        return self is PathKind.FILE

    @property
    def is_dir(self) -> bool:
        return self is PathKind.DIRECTORY


@dataclass(frozen=True, slots=True)
class StatResult:
    """Minimal ``os.stat_result`` stand-in usable across backends.

    Backends with richer metadata (ETag, content-type, owner…) should
    subclass and extend rather than cram extras into ``mode``.
    """

    size: int = 0
    mtime: float = 0.0
    kind: PathKind = PathKind.MISSING
    mode: int = 0

    # os.stat_result is subscript-compatible — keep that affordance.
    def __getitem__(self, idx: int) -> Any:
        return (self.mode, 0, 0, 0, 0, 0, self.size, 0, self.mtime, 0)[idx]

    # Drop-in aliases for code that already reads os.stat_result.
    @property
    def st_size(self) -> int:
        return self.size

    @property
    def st_mtime(self) -> float:
        return self.mtime

    @property
    def st_mode(self) -> int:
        return self.mode


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _split(raw: str) -> List[str]:
    """Split a path string into segments, normalizing backslashes."""
    if not raw:
        return []
    return [p for p in raw.replace("\\", _SEP).split(_SEP) if p]


def _flatten(parts: Any) -> List[str]:
    """Flatten a path-ish input into a list of string segments.

    Accepts :class:`Path`, ``pathlib.PurePath``, strings, bytes, or any
    iterable of those. Mirrors how :class:`pathlib.PurePath` treats its
    constructor arguments — permissive on input shape, strict on contents.
    """
    if parts is None:
        return []
    if isinstance(parts, Path):
        return list(parts.parts)
    if isinstance(parts, (str, bytes, os.PathLike)):
        raw = os.fspath(parts) if isinstance(parts, os.PathLike) else parts
        if isinstance(raw, bytes):
            raw = raw.decode()
        return _split(raw)
    if isinstance(parts, (list, tuple, set)):
        out: List[str] = []
        for item in parts:
            out.extend(_flatten(item))
        return out
    # Last-ditch string coercion.
    return _split(str(parts))


def _coerce_mtime(value: Any) -> Optional[float]:
    """Normalize an mtime-ish value to float seconds since epoch.

    Handles Python datetime, ISO-8601 strings, and millisecond epoch ints
    that backends sometimes hand back.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.timestamp()
    if isinstance(value, str):
        return dt.datetime.fromisoformat(value).timestamp()
    out = float(value)
    # Anything past y2286 is almost certainly milliseconds, not seconds.
    if out > 10_000_000_000:
        out = out / 1000.0
    return out


# ---------------------------------------------------------------------------
# Path — abstract, pathlib.Path-like, dataclass-shaped like DatabricksPath
# ---------------------------------------------------------------------------


@dataclass(eq=False, init=False)
class Path(ABC):
    """Abstract filesystem path with ``pathlib.Path``-like behavior.

    Shape is deliberately aligned with
    :class:`yggdrasil.databricks.fs.path.DatabricksPath`:

    - ``parts`` is a plain ``List[str]`` of segments, root-free.
    - ``anchor`` is the absolute prefix (``""``, ``"/"``, ``"C:\\"``…).
      Backends with a namespace rendering (``DBFS``, ``S3``) typically
      leave this empty and override :meth:`full_path` instead.
    - Cached metadata (``_is_file``, ``_is_dir``, ``_size``, ``_mtime``)
      is filled in lazily by :meth:`_refresh_metadata`. Subclasses should
      populate these in a single round-trip when possible.

    Dispatching constructor
    -----------------------
    ``Path("dbfs:/x")``, ``Path("/tmp/y")``, ``Path(pathlib.Path(...))``
    pick the right concrete subclass automatically. The first registered
    class whose ``_match`` claims the input wins; :class:`LocalPath` is
    the fallback.

    Abstract hooks (minimal set a backend must provide)
    ---------------------------------------------------
    - :meth:`full_path`          — absolute string rendering
    - :meth:`_refresh_metadata`  — fetch remote status into cache
    - :meth:`_ls_impl`           — list children (recursive flag)
    - :meth:`_mkdir_impl`        — create the directory
    - :meth:`_remove_file_impl`  — delete this file
    - :meth:`_remove_dir_impl`   — delete this directory
    - :meth:`open`               — return a file-like handle

    Everything else (``name``, ``stem``, ``parent``, ``parents``,
    ``joinpath``, ``read_text`` / ``write_text``, ``touch``, ``glob``,
    ``copy_to``, …) is derived here and stays consistent across backends.
    """

    #: URL-style scheme for this backend. Empty for local-style paths.
    scheme: ClassVar[str] = ""

    # ── Core fields ──────────────────────────────────────────────────
    parts: List[str] = field(default_factory=list)
    anchor: str = ""

    # ── Cached metadata (excluded from eq/hash/repr) ─────────────────
    _is_file: Optional[bool] = field(
        repr=False, hash=False, compare=False, default=None
    )
    _is_dir: Optional[bool] = field(repr=False, hash=False, compare=False, default=None)
    _size: Optional[int] = field(repr=False, hash=False, compare=False, default=None)
    _mtime: Optional[float] = field(repr=False, hash=False, compare=False, default=None)

    # Optional filesystem handle. Set by :meth:`FileSystem.path`.
    _fs: Optional["FileSystem"] = field(
        repr=False, hash=False, compare=False, default=None
    )

    # ================================================================ #
    # Registry + dispatching constructor                                #
    # ================================================================ #

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Every concrete Path subclass is a dispatch candidate.
        register_path_class(cls)

    def __new__(cls, obj: Any = None, *args: Any, **kwargs: Any) -> "Path":
        """Dispatch to the correct subclass when called on :class:`Path`.

        ``Path("/tmp/foo")``       → :class:`LocalPath`
        ``Path("dbfs:/x/y")``      → Databricks / DBFS path (if registered)
        ``Path(existing_path)``    → same type as ``existing_path``

        When called on a concrete subclass, returns a plain instance —
        no dispatch — so internal ``cls(parts=…, anchor=…)`` calls work
        as usual.
        """
        del args, kwargs  # consumed by __init__
        if cls is Path:
            target = _select_path_class(obj)
            # target.__new__ won't recurse: target is a concrete subclass
            # so the ``cls is Path`` branch above doesn't fire.
            return target.__new__(target, obj)
        return object.__new__(cls)

    def __init__(
        self,
        obj: Any = None,
        *,
        parts: Optional[List[str]] = None,
        anchor: str = "",
        _is_file: Optional[bool] = None,
        _is_dir: Optional[bool] = None,
        _size: Optional[int] = None,
        _mtime: Optional[float] = None,
        _fs: Optional["FileSystem"] = None,
    ) -> None:
        """Construct a :class:`Path`.

        Two usage shapes, detected by argument type:

        * ``Path(obj)`` — parse a string / :class:`os.PathLike` /
          :class:`pathlib.PurePath` / other :class:`Path`. The class
          typically runs ``type(self).parse(obj)`` to get fields.
        * ``Path(parts=[...], anchor="/")`` — build from explicit fields.
          This is what :func:`dataclasses.replace` (used by
          :meth:`_copy_with`) and the FS factory reach for.
        """
        if obj is not None and not isinstance(obj, list):
            # String / PathLike / Path — delegate to the class's parser.
            if isinstance(obj, Path):
                parsed = obj
            else:
                parsed = type(self).parse(obj)
            self.parts = list(parsed.parts)
            self.anchor = parsed.anchor
        else:
            # Explicit field form — also handles ``obj=[...]`` for symmetry
            # with how pathlib.PurePath accepts a list of segments.
            if obj is not None and parts is None:
                parts = obj  # type: ignore[assignment]
            self.parts = list(parts) if parts is not None else []
            self.anchor = anchor
        self._is_file = _is_file
        self._is_dir = _is_dir
        self._size = _size
        self._mtime = _mtime
        self._fs = _fs

    @classmethod
    def _match(cls, obj: Any) -> bool:
        """Return True if ``cls`` is the right Path type for *obj*.

        Default rule: match the URL-style scheme (``scheme://`` or
        ``scheme:/``) against the class's :attr:`scheme`. Backends with
        namespace prefixes (``/Volumes/``, ``/Workspace/``, …) should
        override to inspect the head of the input.
        """
        if not cls.scheme or not isinstance(obj, str):
            return False
        head = f"{cls.scheme}:"
        return obj.startswith(f"{head}//") or obj.startswith(f"{head}/")

    @classmethod
    def parse(cls, obj: Any) -> "Path":
        """Build an instance from a string, list, or another :class:`Path`.

        Called on the abstract :class:`Path`, dispatches through the
        registry (same rules as :meth:`__new__`). Subclasses override to
        handle namespace prefixes (``dbfs:/…``) or URL schemes.
        """
        if cls is Path:
            target = _select_path_class(obj)
            return target.parse(obj)
        if isinstance(obj, Path):
            return obj
        raw = os.fspath(obj) if isinstance(obj, os.PathLike) else str(obj or "")
        anchor = _SEP if raw.startswith(_SEP) or raw.startswith("\\") else ""
        # Construct via the dataclass field form so we don't recurse through
        # __init__'s string-parsing branch.
        inst: Path = object.__new__(cls)
        Path.__init__(inst, parts=_split(raw), anchor=anchor)
        return inst

    # ================================================================ #
    # Abstract hooks — implement per backend                            #
    # ================================================================ #

    @abstractmethod
    def full_path(self) -> str:
        """Absolute string rendering of this path."""

    @abstractmethod
    def _refresh_metadata(self) -> None:
        """Fetch backend status into ``_is_file`` / ``_is_dir`` / ``_size`` / ``_mtime``."""

    @abstractmethod
    def _ls_impl(
        self, recursive: bool = False, allow_not_found: bool = True
    ) -> Iterator["Path"]:
        """Yield children of this directory."""

    @abstractmethod
    def _mkdir_impl(self, parents: bool = True, exist_ok: bool = True) -> None:
        """Create the directory at this path."""

    @abstractmethod
    def _remove_file_impl(self, allow_not_found: bool = True) -> None:
        """Delete the file at this path."""

    @abstractmethod
    def _remove_dir_impl(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> None:
        """Delete the directory at this path."""

    @abstractmethod
    def open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> IO:
        """Open the file and return a file-like object (pathlib.Path.open)."""

    # ================================================================ #
    # pathlib.PurePosixPath — pure-path manipulation                    #
    # ================================================================ #

    @property
    def name(self) -> str:
        """Last component — ``pathlib.PurePath.name``."""
        return self.parts[-1] if self.parts else ""

    @property
    def suffix(self) -> str:
        """Last extension with the leading dot."""
        name = self.name
        idx = name.rfind(".")
        return name[idx:] if idx > 0 else ""

    @property
    def suffixes(self) -> List[str]:
        """All extensions, dotted."""
        name = self.name
        if not name or name.startswith("."):
            return []
        pieces = name.split(".")
        return [f".{p}" for p in pieces[1:]] if len(pieces) > 1 else []

    @property
    def stem(self) -> str:
        """Name minus the last suffix."""
        name = self.name
        idx = name.rfind(".")
        return name[:idx] if idx > 0 else name

    @property
    def extension(self) -> str:
        """Dot-less extension convenience (``'txt'``, not ``'.txt'``)."""
        return self.suffix.lstrip(".")

    @property
    def parent(self) -> "Path":
        """Parent path."""
        if not self.parts:
            return self
        return self._copy_with(
            parts=self.parts[:-1],
            _is_file=False,
            _is_dir=True,
            _size=0,
        )

    @property
    def parents(self) -> Tuple["Path", ...]:
        """All ancestors, closest first."""
        out: List[Path] = []
        cur: Path = self
        while cur.parts:
            cur = cur.parent
            out.append(cur)
        return tuple(out)

    def is_absolute(self) -> bool:
        """True when the path has a root anchor.

        Backends whose :meth:`full_path` always renders an absolute string
        (e.g. ``dbfs:/…``) should override this to return ``True``.
        """
        return bool(self.anchor)

    def as_posix(self) -> str:
        """POSIX-style rendering without scheme decoration."""
        body = _SEP.join(self.parts)
        return self.anchor + body if self.anchor else body

    def joinpath(self, *others: Any) -> "Path":
        """Join one or more path components."""
        cur: Path = self
        for other in others:
            cur = cur / other
        return cur

    def with_name(self, name: str) -> "Path":
        if not self.parts:
            raise ValueError(
                f"Cannot set name on empty path {self!r}. "
                "Build a non-empty path first, e.g. Path('dir') / name."
            )
        if not name or _SEP in name or "\\" in name:
            raise ValueError(
                f"Invalid name {name!r}. Names cannot be empty or contain "
                "path separators — use joinpath() for nested paths."
            )
        return self._copy_with(parts=self.parts[:-1] + [name])

    def with_suffix(self, suffix: str) -> "Path":
        if suffix and not suffix.startswith("."):
            raise ValueError(
                f"Invalid suffix {suffix!r}. Suffixes must start with '.' "
                "(e.g. '.parquet') or be '' to strip the existing suffix."
            )
        return self.with_name(self.stem + suffix)

    def with_stem(self, stem: str) -> "Path":
        return self.with_name(stem + self.suffix)

    def match(self, pattern: str) -> bool:
        """Glob match against the name, then the full rendering."""
        return fnmatch.fnmatch(self.name, pattern) or fnmatch.fnmatch(
            self.full_path(), pattern
        )

    def is_relative_to(self, other: Any) -> bool:
        other = self._coerce(other)
        if type(self) is not type(other):
            return False
        if self.anchor != other.anchor:
            return False
        return self.parts[: len(other.parts)] == other.parts

    def relative_to(self, other: Any) -> "Path":
        other = self._coerce(other)
        if not self.is_relative_to(other):
            raise ValueError(
                f"{self.full_path()!r} is not relative to {other.full_path()!r}. "
                "Use is_relative_to() to test the relationship first."
            )
        return self._copy_with(parts=self.parts[len(other.parts) :], anchor="")

    # ================================================================ #
    # pathlib.Path — concrete I/O                                       #
    # ================================================================ #

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks  # signature parity with pathlib.Path
        return bool(self.is_file() or self.is_dir())

    def is_file(self) -> Optional[bool]:
        if self._is_file is None:
            self._refresh_metadata()
        return self._is_file

    def is_dir(self) -> Optional[bool]:
        if self._is_dir is None:
            self._refresh_metadata()
        return self._is_dir

    def is_symlink(self) -> bool:
        return False

    def stat(self) -> StatResult:
        """Refresh cached metadata and return a :class:`StatResult`."""
        self._refresh_metadata()
        if self._is_file:
            kind = PathKind.FILE
        elif self._is_dir:
            kind = PathKind.DIRECTORY
        else:
            kind = PathKind.MISSING
        return StatResult(
            size=self._size or 0,
            mtime=self._mtime or 0.0,
            kind=kind,
            mode=0,
        )

    def iterdir(self) -> Iterator["Path"]:
        """One level of children."""
        yield from self._ls_impl(recursive=False, allow_not_found=True)

    def ls(
        self,
        recursive: bool = False,
        allow_not_found: bool = True,
    ) -> Iterator["Path"]:
        """Flexible listing — matches the DatabricksPath legacy helper."""
        yield from self._ls_impl(recursive=recursive, allow_not_found=allow_not_found)

    def glob(self, pattern: str) -> Iterator["Path"]:
        """Recursive glob filtered by *pattern* (matches DatabricksPath)."""
        for child in self._ls_impl(recursive=True, allow_not_found=True):
            if fnmatch.fnmatch(child.name, pattern):
                yield child

    def rglob(self, pattern: str) -> Iterator["Path"]:
        """Alias for :meth:`glob` — always recursive."""
        yield from self.glob(pattern)

    def walk(self) -> Iterator[Tuple["Path", List["Path"], List["Path"]]]:
        """``os.walk``-style traversal rooted at this path."""
        dirs: List[Path] = []
        files: List[Path] = []
        for child in self.iterdir():
            (dirs if child.is_dir() else files).append(child)
        yield self, dirs, files
        for sub in dirs:
            yield from sub.walk()

    def mkdir(
        self,
        mode: int = 0o777,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> "Path":
        del mode  # Most remote backends ignore POSIX mode bits.
        self._mkdir_impl(parents=parents, exist_ok=exist_ok)
        return self

    def rmdir(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
        with_root: bool = True,
    ) -> "Path":
        self._remove_dir_impl(
            recursive=recursive,
            allow_not_found=allow_not_found,
            with_root=with_root,
        )
        return self

    def unlink(self, missing_ok: bool = True) -> None:
        """``pathlib.Path.unlink`` — delegates to :meth:`remove`."""
        self.remove(recursive=True, allow_not_found=missing_ok)

    def rmfile(self, allow_not_found: bool = True) -> "Path":
        self._remove_file_impl(allow_not_found=allow_not_found)
        return self

    def remove(
        self,
        recursive: bool = True,
        allow_not_found: bool = True,
    ) -> "Path":
        """File or directory, whichever this path happens to be."""
        if self.is_file():
            self._remove_file_impl(allow_not_found=allow_not_found)
        elif self.is_dir():
            self._remove_dir_impl(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=True,
            )
        elif not allow_not_found:
            raise FileNotFoundError(f"{self!r} does not exist")
        return self

    def rename(self, target: Any) -> "Path":
        """``pathlib.Path.rename`` — copy + remove by default."""
        target_path = self._coerce(target)
        self.copy_to(target_path)
        self.remove(recursive=True)
        return target_path

    def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
        """``pathlib.Path.touch``.

        Default write-empty-bytes implementation. Backends with native
        touch semantics (e.g. S3 zero-byte PUT) should override.
        """
        del mode
        if self.exists():
            if not exist_ok:
                raise FileExistsError(f"Path already exists: {self.full_path()!r}")
            return
        self.write_bytes(b"")

    def resolve(self, *, strict: bool = False) -> "Path":
        del strict
        return self

    def absolute(self) -> "Path":
        return self

    # ── Bytes / text helpers ───────────────────────────────────────

    def read_bytes(self) -> bytes:
        with self.open("rb") as fh:
            return fh.read()

    def write_bytes(self, data: Union[bytes, bytearray, memoryview]) -> int:
        with self.open("wb") as fh:
            fh.write(bytes(data))
        return len(data)

    def read_text(
        self,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        return self.read_bytes().decode(encoding, errors=errors)

    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: Optional[str] = None,
    ) -> int:
        del newline  # Signature parity; encoding runs before write_bytes.
        encoded = data.encode(encoding, errors=errors)
        self.write_bytes(encoded)
        return len(encoded)

    # ── Copy ───────────────────────────────────────────────────────

    def copy_to(self, dest: Any, allow_not_found: bool = True) -> "Path":
        """Copy this file or directory to ``dest``.

        Default implementation streams bytes for files and recurses for
        directories. Backends with a cheap server-side copy should override.
        """
        dest_path = self._coerce(dest)
        if self.is_file():
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(self.read_bytes())
        elif self.is_dir():
            dest_path.mkdir(parents=True, exist_ok=True)
            skip = len(self.parts)
            for child in self.ls(recursive=True, allow_not_found=True):
                if child.is_file():
                    target = dest_path._copy_with(
                        parts=dest_path.parts + child.parts[skip:],
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(child.read_bytes())
        elif not allow_not_found:
            raise FileNotFoundError(f"{self!r} does not exist")
        return dest_path

    # ================================================================ #
    # Metadata                                                          #
    # ================================================================ #

    @property
    def content_length(self) -> int:
        if self._size is None:
            self._refresh_metadata()
        return self._size or 0

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        self._size = value

    @property
    def mtime(self) -> Optional[float]:
        if self._mtime is None:
            self._refresh_metadata()
        return self._mtime

    @mtime.setter
    def mtime(self, value: Any) -> None:
        self._mtime = _coerce_mtime(value)

    def refresh_status(self) -> "Path":
        self._refresh_metadata()
        return self

    def reset_metadata(
        self,
        is_file: Optional[bool] = None,
        is_dir: Optional[bool] = None,
        size: Optional[int] = None,
        mtime: Any = None,
    ) -> "Path":
        self._is_file = is_file
        self._is_dir = is_dir
        self._size = None if size is None else int(size)
        self._mtime = _coerce_mtime(mtime)
        return self

    # ================================================================ #
    # FileSystem binding                                                #
    # ================================================================ #

    @property
    def filesystem(self) -> Optional["FileSystem"]:
        return self._fs

    def bind(self, fs: "FileSystem") -> "Path":
        self._fs = fs
        return self

    # ================================================================ #
    # Clone / coerce                                                    #
    # ================================================================ #

    def _copy_with(self, **overrides: Any) -> "Path":
        """Internal clone. Forwards to :func:`dataclasses.replace`.

        Cached metadata is cleared unless the caller passes it explicitly,
        so a cloned path doesn't inherit stale status from the original.
        """
        defaults = dict(
            _is_file=None,
            _is_dir=None,
            _size=None,
            _mtime=None,
            _fs=self._fs,
        )
        defaults.update(overrides)
        return replace(self, **defaults)

    def _coerce(self, obj: Any) -> "Path":
        """Coerce *obj* to a path of this class."""
        if isinstance(obj, Path) and type(obj) is type(self):
            return obj
        parsed = type(self).parse(obj)
        if self._fs is not None and parsed._fs is None:
            parsed._fs = self._fs
        return parsed

    # ================================================================ #
    # Dunder                                                            #
    # ================================================================ #

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.full_path()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return (
                type(self) is type(other)
                and self.anchor == other.anchor
                and self.parts == other.parts
            )
        if isinstance(other, str):
            return self.full_path() == other
        return NotImplemented

    def __truediv__(self, other: Any) -> "Path":
        if other is None or other == "":
            return self
        # An absolute RHS replaces the LHS — matches pathlib semantics for
        # ``Path('/a') / '/b' == Path('/b')``.
        if isinstance(other, Path):
            if other.anchor:
                return self._copy_with(parts=list(other.parts), anchor=other.anchor)
            return self._copy_with(parts=self.parts + list(other.parts))
        raw = os.fspath(other) if isinstance(other, os.PathLike) else str(other)
        raw_norm = raw.replace("\\", _SEP)
        if raw_norm.startswith(_SEP):
            return self._copy_with(parts=_split(raw_norm), anchor=_SEP)
        return self._copy_with(parts=self.parts + _split(raw_norm))

    def __rtruediv__(self, other: Any) -> "Path":
        return self._coerce(other) / self

    def __str__(self) -> str:
        return self.full_path()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.full_path()!r})"

    def __fspath__(self) -> str:
        return self.full_path()
