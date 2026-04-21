"""Abstract filesystem path — ``pathlib.Path``-like API without inheritance.

This module defines a backend-agnostic :class:`Path` abstraction. It mirrors
the surface of :class:`pathlib.Path` (pure-path manipulation + concrete I/O)
without inheriting from it, so the same call sites can transparently work on
local files, Databricks paths, object stores, and anything else that plugs in.

Layout
------
- :class:`PathKind`   — enum for file / directory / other / missing
- :class:`StatResult` — minimal ``os.stat_result``-compatible dataclass
- :class:`PurePath`   — name/parts manipulation, no I/O
- :class:`Path`       — concrete I/O contract, abstract per backend

Design notes
------------
- Paths are POSIX-style (forward slash). Windows-drive parsing is intentionally
  not baked in; a local-filesystem subclass can override ``_parse`` if it needs
  Windows semantics.
- ``parts`` carries the full split, including a leading ``"/"`` token when the
  path is absolute. This matches ``pathlib.PurePosixPath.parts``.
- Concrete subclasses override only the small set of abstract hooks; all
  pure-path properties (``name``, ``stem``, ``suffix``…) and the bytes↔text
  helpers (``read_text``, ``write_text``) come from the base class and stay
  consistent across backends.
"""

from __future__ import annotations

import fnmatch
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from .filesystem import FileSystem

__all__ = [
    "PathKind",
    "StatResult",
    "PurePath",
    "Path",
]


_SEP = "/"
_ROOT = "/"


# ---------------------------------------------------------------------------
# PathKind + StatResult side classes
# ---------------------------------------------------------------------------


class PathKind(str, Enum):
    """What a path points at. Mirrors what ``os.stat`` would tell you.

    ``MISSING`` is the explicit "nothing here" value — callers should not get
    ``None`` when the path simply does not exist.
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

    Keeps the field set small and typed; backends that have more metadata
    (ETag, content-type, owner…) should subclass and add fields rather than
    cramming them into ``st_mode``.
    """

    size: int = 0
    mtime: float = 0.0
    kind: PathKind = PathKind.MISSING
    mode: int = 0

    # os.stat_result is subscript-compatible — keep that convenience.
    def __getitem__(self, idx: int) -> Any:
        return (self.mode, 0, 0, 0, 0, 0, self.size, 0, self.mtime, 0)[idx]

    # Aliases to match os.stat_result for drop-in use.
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
# Parse helpers
# ---------------------------------------------------------------------------


def _split(raw: str) -> Tuple[bool, List[str]]:
    """Split a posix-ish path string into ``(is_absolute, parts)``.

    Backslashes are normalized to forward slashes so Windows-style inputs
    don't trip callers. Empty segments are dropped.
    """
    if not raw:
        return False, []
    cleaned = raw.replace("\\", "/")
    is_abs = cleaned.startswith(_SEP)
    return is_abs, [p for p in cleaned.split(_SEP) if p]


def _flatten(parts: Iterable[Any]) -> Tuple[bool, List[str]]:
    """Flatten a heterogeneous input into ``(is_absolute, segments)``.

    Accepts strings, other PurePaths, ``pathlib.PurePath`` objects, and lists
    of any of the above. Mirrors how ``pathlib`` handles the constructor.
    """
    is_abs = False
    out: List[str] = []
    first = True
    for item in parts:
        if item is None:
            continue
        if isinstance(item, PurePath):
            if first and item._is_absolute:
                is_abs = True
            out.extend(item._parts)
        elif hasattr(item, "parts") and not isinstance(item, (str, bytes)):
            # pathlib.PurePath and anything else that exposes .parts.
            raw = os.fspath(item) if hasattr(item, "__fspath__") else str(item)
            abs_, segs = _split(raw)
            if first and abs_:
                is_abs = True
            out.extend(segs)
        else:
            raw = os.fspath(item) if hasattr(item, "__fspath__") else str(item)
            abs_, segs = _split(raw)
            if first and abs_:
                is_abs = True
            out.extend(segs)
        first = False
    return is_abs, out


# ---------------------------------------------------------------------------
# PurePath — pure manipulation, no I/O
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class PurePath:
    """Backend-agnostic pure path. All manipulation, zero I/O.

    Subclass when you want different parsing rules (e.g. Windows drives, URL
    schemes) but don't need I/O yet. Most concrete backends will subclass
    :class:`Path` instead.
    """

    # The scheme is advisory — it lets ``__str__`` / ``url()`` produce a
    # stable rendering for non-local backends (e.g. ``"dbfs"``, ``"s3"``).
    # Local filesystems should leave it as ``""``.
    scheme: ClassVar[str] = ""

    _parts: List[str] = field(default_factory=list)
    _is_absolute: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, *segments: Any) -> None:
        abs_, parts = _flatten(segments)
        self._parts = parts
        self._is_absolute = abs_

    # Dataclass-like copy helper used by ``with_*`` and ``joinpath``.
    def _replace(
        self, parts: Sequence[str], is_absolute: Optional[bool] = None
    ) -> "PurePath":
        clone = object.__new__(type(self))
        clone._parts = list(parts)
        clone._is_absolute = self._is_absolute if is_absolute is None else is_absolute
        return clone

    # ------------------------------------------------------------------
    # pathlib.PurePath read-only properties
    # ------------------------------------------------------------------

    @property
    def parts(self) -> Tuple[str, ...]:
        """Path components, including the root sentinel for absolute paths.

        Mirrors ``pathlib.PurePosixPath.parts``: absolute paths prepend
        ``"/"`` so callers can distinguish ``/a/b`` from ``a/b``.
        """
        if self._is_absolute:
            return (_ROOT, *self._parts)
        return tuple(self._parts)

    @property
    def anchor(self) -> str:
        """``"/"`` for absolute paths, ``""`` otherwise."""
        return _ROOT if self._is_absolute else ""

    @property
    def root(self) -> str:
        return self.anchor

    @property
    def drive(self) -> str:
        # POSIX default; Windows-style subclasses can override.
        return ""

    @property
    def name(self) -> str:
        return self._parts[-1] if self._parts else ""

    @property
    def suffix(self) -> str:
        name = self.name
        idx = name.rfind(".")
        return name[idx:] if idx > 0 else ""

    @property
    def suffixes(self) -> List[str]:
        name = self.name
        if not name or name.startswith("."):
            return []
        pieces = name.split(".")
        return [f".{p}" for p in pieces[1:]] if len(pieces) > 1 else []

    @property
    def stem(self) -> str:
        name = self.name
        idx = name.rfind(".")
        return name[:idx] if idx > 0 else name

    @property
    def parent(self) -> "PurePath":
        if not self._parts:
            return self
        return self._replace(self._parts[:-1])

    @property
    def parents(self) -> Tuple["PurePath", ...]:
        out: List[PurePath] = []
        cur = self
        while cur._parts:
            cur = cur.parent
            out.append(cur)
        return tuple(out)

    # ------------------------------------------------------------------
    # pathlib.PurePath manipulation
    # ------------------------------------------------------------------

    def is_absolute(self) -> bool:
        return self._is_absolute

    def as_posix(self) -> str:
        body = _SEP.join(self._parts)
        return (_ROOT + body) if self._is_absolute else body

    def joinpath(self, *others: Any) -> "PurePath":
        cur = self
        for other in others:
            cur = cur / other
        return cur

    def with_name(self, name: str) -> "PurePath":
        if not self._parts:
            raise ValueError(
                f"Cannot set name on empty path: {self!r}. "
                "Build a non-empty path first, e.g. Path('dir') / name."
            )
        if not name or _SEP in name or "\\" in name:
            raise ValueError(
                f"Invalid name {name!r}. Names cannot be empty or contain "
                "path separators — use joinpath() for nested paths."
            )
        return self._replace(self._parts[:-1] + [name])

    def with_suffix(self, suffix: str) -> "PurePath":
        if suffix and not suffix.startswith("."):
            raise ValueError(
                f"Invalid suffix {suffix!r}. Suffixes must start with '.' "
                "(e.g. '.parquet'), or be '' to strip the existing suffix."
            )
        return self.with_name(self.stem + suffix)

    def with_stem(self, stem: str) -> "PurePath":
        return self.with_name(stem + self.suffix)

    def match(self, pattern: str) -> bool:
        """Glob-match against the name or full string.

        Matches ``pathlib.PurePath.match`` loosely: tries the filename first,
        then the full rendering. Backends with scheme-aware rendering stay
        consistent because both forms go through ``as_posix``.
        """
        return fnmatch.fnmatch(self.name, pattern) or fnmatch.fnmatch(
            self.as_posix(), pattern
        )

    def is_relative_to(self, other: Any) -> bool:
        other = self._coerce(other)
        if self._is_absolute != other._is_absolute:
            return False
        return self._parts[: len(other._parts)] == other._parts

    def relative_to(self, other: Any) -> "PurePath":
        other = self._coerce(other)
        if not self.is_relative_to(other):
            raise ValueError(
                f"{self.as_posix()!r} is not relative to {other.as_posix()!r}. "
                "Use is_relative_to() to test the relationship first."
            )
        return self._replace(self._parts[len(other._parts) :], is_absolute=False)

    # ------------------------------------------------------------------
    # Coercion / dunder
    # ------------------------------------------------------------------

    def _coerce(self, obj: Any) -> "PurePath":
        """Coerce ``obj`` to a path of this class, preserving abs/rel."""
        if isinstance(obj, PurePath):
            return obj
        abs_, parts = _flatten((obj,))
        return self._replace(parts, is_absolute=abs_)

    def __truediv__(self, other: Any) -> "PurePath":
        if other is None or other == "":
            return self
        abs_, segs = _flatten((other,))
        # Right-hand absolute paths replace the left-hand side, matching
        # pathlib semantics for ``Path('/a') / '/b' == Path('/b')``.
        if abs_:
            return self._replace(segs, is_absolute=True)
        return self._replace(self._parts + segs)

    def __rtruediv__(self, other: Any) -> "PurePath":
        abs_, segs = _flatten((other,))
        return self._replace(segs + self._parts, is_absolute=abs_ or self._is_absolute)

    def __str__(self) -> str:
        rendered = self.as_posix() or "."
        if self.scheme:
            return (
                f"{self.scheme}://{rendered}"
                if not rendered.startswith(_SEP)
                else f"{self.scheme}:{rendered}"
            )
        return rendered

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.as_posix()!r})"

    def __fspath__(self) -> str:
        return self.as_posix()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._is_absolute, tuple(self._parts)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PurePath):
            return (
                type(self) is type(other)
                and self._is_absolute == other._is_absolute
                and self._parts == other._parts
            )
        if isinstance(other, str):
            return self.as_posix() == other
        return NotImplemented


# ---------------------------------------------------------------------------
# Path — concrete I/O contract
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Path(PurePath, ABC):
    """Abstract filesystem path with ``pathlib.Path``-like I/O.

    Subclasses must implement the small set of abstract hooks; everything
    else (``read_text``, ``write_text``, ``touch``, ``parents``, ``name``,
    ``match``…) comes from this class or :class:`PurePath`.

    Subclasses typically hold a reference to a :class:`FileSystem` (set via
    ``filesystem`` / ``_fs``) to route I/O. Backends without a separable FS
    layer (pure pathlib-style local paths) can just implement the hooks.
    """

    # Concrete backends fill this in; it's optional so tests and lightweight
    # implementations aren't forced to build a full FileSystem first.
    _fs: Optional["FileSystem"] = field(default=None, repr=False, compare=False)

    def __init__(
        self, *segments: Any, filesystem: Optional["FileSystem"] = None
    ) -> None:
        PurePath.__init__(self, *segments)
        self._fs = filesystem

    # ------------------------------------------------------------------
    # Filesystem binding
    # ------------------------------------------------------------------

    @property
    def filesystem(self) -> Optional["FileSystem"]:
        """The :class:`FileSystem` this path is bound to, if any."""
        return self._fs

    def _replace(
        self, parts: Sequence[str], is_absolute: Optional[bool] = None
    ) -> "Path":
        clone = object.__new__(type(self))
        clone._parts = list(parts)
        clone._is_absolute = self._is_absolute if is_absolute is None else is_absolute
        clone._fs = self._fs
        return clone

    # ------------------------------------------------------------------
    # Abstract I/O hooks — the minimum a backend must provide
    # ------------------------------------------------------------------

    @abstractmethod
    def stat(self) -> StatResult:
        """Return filesystem metadata as a :class:`StatResult`.

        Must return a ``StatResult`` with ``kind=MISSING`` for non-existent
        paths rather than raising — callers rely on that to implement
        ``exists()`` without a try/except.
        """

    @abstractmethod
    def iterdir(self) -> Iterator["Path"]:
        """Yield one level of children (no recursion).

        Raises ``FileNotFoundError`` only when the directory truly does not
        exist; empty directories yield nothing.
        """

    @abstractmethod
    def mkdir(self, *, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a directory at this path."""

    @abstractmethod
    def unlink(self, *, missing_ok: bool = False) -> None:
        """Remove this file. Directories should use :meth:`rmdir`."""

    @abstractmethod
    def rmdir(self, *, recursive: bool = False, missing_ok: bool = False) -> None:
        """Remove this directory. ``recursive=True`` drops its contents too."""

    @abstractmethod
    def open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> IO:
        """Open the file and return a file-like object."""

    @abstractmethod
    def rename(self, target: Any) -> "Path":
        """Rename to ``target`` and return the resulting :class:`Path`."""

    # ------------------------------------------------------------------
    # Concrete helpers — derived from the abstract hooks
    # ------------------------------------------------------------------

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks  # Left in the signature for pathlib.Path parity.
        return self.stat().kind.exists

    def is_file(self) -> bool:
        return self.stat().kind.is_file

    def is_dir(self) -> bool:
        return self.stat().kind.is_dir

    def is_symlink(self) -> bool:
        return self.stat().kind is PathKind.SYMLINK

    # ``resolve`` / ``absolute`` default to "I already am" — concrete
    # backends override if they need to handle ``.``, ``..``, or symlinks.
    def resolve(self, *, strict: bool = False) -> "Path":
        del strict
        return self

    def absolute(self) -> "Path":
        return self

    # ------------------------------------------------------------------
    # Read/write convenience — written in terms of ``open``
    # ------------------------------------------------------------------

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
        del newline  # Signature parity; encoding happens before write_bytes.
        encoded = data.encode(encoding, errors=errors)
        self.write_bytes(encoded)
        return len(encoded)

    def touch(self, *, exist_ok: bool = True) -> None:
        """Create an empty file, updating mtime when it already exists.

        Backends with native ``touch`` should override — the default here
        rewrites the file with empty bytes, which is correct but not cheap.
        """
        if self.exists():
            if not exist_ok:
                raise FileExistsError(f"Path already exists: {self.as_posix()!r}")
            return
        self.write_bytes(b"")

    # ------------------------------------------------------------------
    # Listing helpers
    # ------------------------------------------------------------------

    def glob(self, pattern: str) -> Iterator["Path"]:
        """Non-recursive glob of direct children by name."""
        for child in self.iterdir():
            if fnmatch.fnmatch(child.name, pattern):
                yield child

    def rglob(self, pattern: str) -> Iterator["Path"]:
        """Recursive glob. Walks into directories depth-first."""
        for child in self.iterdir():
            if fnmatch.fnmatch(child.name, pattern):
                yield child
            if child.is_dir():
                yield from child.rglob(pattern)

    def walk(self) -> Iterator[Tuple["Path", List["Path"], List["Path"]]]:
        """Yield ``(root, dirs, files)`` tuples, like :func:`os.walk`."""
        dirs: List[Path] = []
        files: List[Path] = []
        for child in self.iterdir():
            (dirs if child.is_dir() else files).append(child)
        yield self, dirs, files
        for sub in dirs:
            yield from sub.walk()
