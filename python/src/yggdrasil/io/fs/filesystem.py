"""Abstract filesystem backend for the :class:`Path` hierarchy.

A :class:`FileSystem` is the lightest possible "where do these bytes live"
contract: enough to open, list, stat, and mutate, without forcing every
backend to also ship a custom :class:`Path` subclass for every operation.

Typical pairings:

- ``LocalFileSystem``          ↔ ``pathlib.Path``-backed paths
- ``DatabricksFileSystem``     ↔ DBFS / Workspace / Volumes paths
- ``S3FileSystem`` etc.        ↔ bucket-based paths

Two usage patterns are supported:

1. **FS-first.** Call ``fs.path("some/path")`` to get a :class:`Path`
   already bound to this backend. Path methods route through the FS.
2. **Path-first.** A concrete :class:`Path` subclass implements the I/O
   hooks directly and doesn't need a FS at all. Useful for simple cases
   (e.g. a pathlib-backed local path).

Both styles coexist — ``Path`` methods fall back to the FS only when the
subclass doesn't override them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import IO, Any, ClassVar, Dict, Iterator, Optional, Type, Union

from .path import Path, StatResult

__all__ = [
    "FileSystem",
    "get_filesystem",
    "register_filesystem",
]


# ---------------------------------------------------------------------------
# Registry — schemes → FileSystem classes
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type["FileSystem"]] = {}


def register_filesystem(scheme: str, cls: Type["FileSystem"]) -> Type["FileSystem"]:
    """Register a :class:`FileSystem` subclass under ``scheme``.

    Called by concrete backends on import (``local``, ``dbfs``, ``s3``…).
    Schemes are lowercased; re-registering the same scheme replaces the
    previous entry — last write wins so tests can swap implementations.
    """
    if not scheme:
        raise ValueError("scheme must be a non-empty string")
    _REGISTRY[scheme.lower()] = cls
    return cls


def get_filesystem(scheme: str) -> Type["FileSystem"]:
    """Look up a registered :class:`FileSystem` class by scheme.

    Raises a :class:`KeyError` with the available schemes when the lookup
    misses so users can see what backends actually shipped in this build.
    """
    key = scheme.lower()
    if key not in _REGISTRY:
        available = sorted(_REGISTRY) or ["(none registered)"]
        raise KeyError(
            f"No filesystem registered for scheme {scheme!r}. "
            f"Available: {available}. "
            "Import the backend module (e.g. yggdrasil.io.fs.local) to register it."
        )
    return _REGISTRY[key]


# ---------------------------------------------------------------------------
# FileSystem — the abstract backend contract
# ---------------------------------------------------------------------------


class FileSystem(ABC):
    """Abstract backend-agnostic filesystem.

    Subclasses implement the small set of abstract hooks and declare the
    :class:`Path` class they produce via :attr:`path_class`. Everything else
    — text/bytes helpers, recursive walks, existence checks — is derived
    here so behavior stays consistent across backends.
    """

    #: Scheme used to build ``scheme://...`` renderings and for the registry.
    scheme: ClassVar[str] = ""

    #: Path class this FS hands out. Set on the subclass.
    path_class: ClassVar[Type[Path]]

    # ------------------------------------------------------------------
    # Path factory
    # ------------------------------------------------------------------

    def path(self, *segments: Any) -> Path:
        """Build a :class:`Path` bound to this filesystem."""
        p = self.path_class(*segments, filesystem=self)
        return p

    # Convenience alias that reads more naturally at call sites.
    def __call__(self, *segments: Any) -> Path:
        return self.path(*segments)

    # ------------------------------------------------------------------
    # Required I/O hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def stat(self, path: Union[str, Path]) -> StatResult:
        """Return metadata for ``path`` (``kind=MISSING`` when absent)."""

    @abstractmethod
    def open(
        self,
        path: Union[str, Path],
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> IO:
        """Open ``path`` and return a file-like object."""

    @abstractmethod
    def ls(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = False,
    ) -> Iterator[Path]:
        """Yield children of the directory at ``path``.

        ``recursive=True`` walks the full tree in depth-first order.
        """

    @abstractmethod
    def mkdir(
        self,
        path: Union[str, Path],
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        """Create the directory at ``path``."""

    @abstractmethod
    def rm(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = False,
        missing_ok: bool = False,
    ) -> None:
        """Remove the file or directory at ``path``.

        For directories, ``recursive=True`` is required — matches the
        ``rm``/``rm -rf`` split most users already know.
        """

    @abstractmethod
    def rename(self, src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """Rename ``src`` → ``dst`` and return the resulting path."""

    # ------------------------------------------------------------------
    # Derived helpers — all concrete, written in terms of the hooks above
    # ------------------------------------------------------------------

    def exists(self, path: Union[str, Path]) -> bool:
        return self.stat(path).kind.exists

    def is_file(self, path: Union[str, Path]) -> bool:
        return self.stat(path).kind.is_file

    def is_dir(self, path: Union[str, Path]) -> bool:
        return self.stat(path).kind.is_dir

    def read_bytes(self, path: Union[str, Path]) -> bytes:
        with self.open(path, "rb") as fh:
            return fh.read()

    def write_bytes(
        self,
        path: Union[str, Path],
        data: Union[bytes, bytearray, memoryview],
    ) -> int:
        with self.open(path, "wb") as fh:
            fh.write(bytes(data))
        return len(data)

    def read_text(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        return self.read_bytes(path).decode(encoding, errors=errors)

    def write_text(
        self,
        path: Union[str, Path],
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> int:
        encoded = data.encode(encoding, errors=errors)
        self.write_bytes(path, encoded)
        return len(encoded)

    def touch(
        self,
        path: Union[str, Path],
        *,
        exist_ok: bool = True,
    ) -> None:
        """Create ``path`` as an empty file, or refresh mtime when it exists."""
        if self.exists(path):
            if not exist_ok:
                raise FileExistsError(f"Path already exists: {path!r}")
            return
        self.write_bytes(path, b"")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        scheme = self.scheme or "?"
        return f"{type(self).__name__}(scheme={scheme!r})"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register concrete subclasses that declare a scheme so imports
        # stay the single source of truth. Abstract subclasses can leave
        # ``scheme`` as the empty default to opt out.
        if cls.scheme and not getattr(cls, "__abstractmethods__", None):
            register_filesystem(cls.scheme, cls)
