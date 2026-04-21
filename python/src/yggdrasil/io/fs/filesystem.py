"""Abstract filesystem backend for the :class:`Path` hierarchy.

A :class:`FileSystem` is the "where do these bytes live" handle — enough
to build paths, stat, open, list, make dirs, and remove. Most of the
surface is already on :class:`Path`; this class exists so callers who'd
rather work FS-first (``fs.open(path)``) than path-first (``path.open()``)
have an ergonomic entry point, and so backends can bind a default client
or config to every path they hand out.

Pattern
-------
Subclasses declare a :attr:`path_class` and :attr:`scheme`. Subclasses
with a non-empty ``scheme`` are auto-registered; look them up with
:func:`get_filesystem`.

The default method implementations route through the :class:`Path` API,
so most backends only need to implement :attr:`path_class` and override
what they want to customize.
"""

from __future__ import annotations

from abc import ABC
from typing import IO, Any, ClassVar, Dict, Iterator, Optional, Type, Union

from .path import Path, StatResult

__all__ = ["FileSystem", "get_filesystem", "register_filesystem"]


# ---------------------------------------------------------------------------
# Registry — scheme → FileSystem class
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type["FileSystem"]] = {}


def register_filesystem(scheme: str, cls: Type["FileSystem"]) -> Type["FileSystem"]:
    """Register *cls* under *scheme*. Last write wins."""
    if not scheme:
        raise ValueError("scheme must be a non-empty string")
    _REGISTRY[scheme.lower()] = cls
    return cls


def get_filesystem(scheme: str) -> Type["FileSystem"]:
    """Look up a :class:`FileSystem` class by scheme."""
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
# FileSystem
# ---------------------------------------------------------------------------


class FileSystem(ABC):
    """Abstract, backend-agnostic filesystem handle.

    Subclasses typically only need to set :attr:`scheme` and
    :attr:`path_class`. Everything else routes through :class:`Path`.
    """

    #: Scheme used by :func:`register_filesystem` and URL rendering.
    scheme: ClassVar[str] = ""

    #: Concrete :class:`Path` subclass this FS hands out.
    path_class: ClassVar[Type[Path]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Auto-register concrete subclasses with a declared scheme. Abstract
        # subclasses (those without a path_class yet) stay out of the registry
        # so half-built bases aren't handed to users.
        if cls.scheme and getattr(cls, "path_class", None) is not None:
            register_filesystem(cls.scheme, cls)

    def __new__(cls, *args: Any, **kwargs: Any) -> "FileSystem":
        # Guard direct instantiation of the base. ABC alone doesn't block this
        # because we have no @abstractmethod — but a FileSystem without a
        # path_class can't do anything useful, so we enforce it here.
        if cls is FileSystem or getattr(cls, "path_class", None) is None:
            raise TypeError(
                f"Cannot instantiate abstract FileSystem {cls.__name__}. "
                "Subclass and set `path_class` (and optionally `scheme`)."
            )
        return super().__new__(cls)

    # ------------------------------------------------------------------
    # Path factory
    # ------------------------------------------------------------------

    def path(self, obj: Any = "") -> Path:
        """Build a :class:`Path` bound to this filesystem."""
        return self.path_class.parse(obj).bind(self)

    def __call__(self, obj: Any = "") -> Path:
        return self.path(obj)

    # ------------------------------------------------------------------
    # Thin delegations — route to the Path API
    # ------------------------------------------------------------------

    def stat(self, path: Union[str, Path]) -> StatResult:
        return self._as_path(path).stat()

    def exists(self, path: Union[str, Path]) -> bool:
        return self._as_path(path).exists()

    def is_file(self, path: Union[str, Path]) -> bool:
        return bool(self._as_path(path).is_file())

    def is_dir(self, path: Union[str, Path]) -> bool:
        return bool(self._as_path(path).is_dir())

    def open(
        self,
        path: Union[str, Path],
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ) -> IO:
        return self._as_path(path).open(
            mode, encoding=encoding, errors=errors, newline=newline
        )

    def ls(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = False,
    ) -> Iterator[Path]:
        yield from self._as_path(path).ls(recursive=recursive, allow_not_found=True)

    def iterdir(self, path: Union[str, Path]) -> Iterator[Path]:
        yield from self._as_path(path).iterdir()

    def glob(self, path: Union[str, Path], pattern: str) -> Iterator[Path]:
        yield from self._as_path(path).glob(pattern)

    def mkdir(
        self,
        path: Union[str, Path],
        *,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> Path:
        return self._as_path(path).mkdir(parents=parents, exist_ok=exist_ok)

    def rm(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> None:
        self._as_path(path).remove(recursive=recursive, allow_not_found=missing_ok)

    def rmdir(
        self,
        path: Union[str, Path],
        *,
        recursive: bool = True,
        missing_ok: bool = True,
        with_root: bool = True,
    ) -> None:
        self._as_path(path).rmdir(
            recursive=recursive,
            allow_not_found=missing_ok,
            with_root=with_root,
        )

    def rename(self, src: Union[str, Path], dst: Union[str, Path]) -> Path:
        return self._as_path(src).rename(self._as_path(dst))

    def read_bytes(self, path: Union[str, Path]) -> bytes:
        return self._as_path(path).read_bytes()

    def write_bytes(
        self,
        path: Union[str, Path],
        data: Union[bytes, bytearray, memoryview],
    ) -> int:
        return self._as_path(path).write_bytes(data)

    def read_text(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        return self._as_path(path).read_text(encoding=encoding, errors=errors)

    def write_text(
        self,
        path: Union[str, Path],
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> int:
        return self._as_path(path).write_text(data, encoding=encoding, errors=errors)

    def touch(
        self,
        path: Union[str, Path],
        *,
        exist_ok: bool = True,
    ) -> None:
        self._as_path(path).touch(exist_ok=exist_ok)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_path(self, obj: Any) -> Path:
        """Coerce *obj* to a :class:`Path` bound to this filesystem."""
        if isinstance(obj, Path) and type(obj) is self.path_class:
            if obj._fs is None:
                obj._fs = self
            return obj
        return self.path(obj)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scheme={self.scheme or '?'!r})"
