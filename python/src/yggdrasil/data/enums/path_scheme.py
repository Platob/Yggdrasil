"""Centralized URL-scheme enum for filesystem / holder dispatch.

Across Yggdrasil, a "filesystem path" can live behind several
backends — the local OS (``file://``), in-memory buffers
(``memory://``), the two Databricks filesystem surfaces
(``dbfs://`` and ``volumes://``), S3, HTTP(S). Workspace objects
(notebooks / files under ``/Workspace``) are managed through a
different SDK API and are *not* a filesystem in the same sense, so
they live outside this enum even though :class:`WorkspacePath`
shares the :class:`Holder` base. Each backend's holder class
declares its scheme as a
plain string ``ClassVar[str]`` so :class:`yggdrasil.io.holder.Holder`
can dispatch ``Holder(url="...")`` to the right subclass via the
runtime ``_HOLDER_SCHEMES`` registry (populated on subclass import).

That registry is great when the caller has already imported the
backend module — but ``Holder(url="dbfs://...")`` from a fresh
process raises ``ValueError: Unknown scheme 'dbfs'`` because the
import side-effect that registers :class:`DBFSPath` hasn't fired
yet. :class:`PathScheme` centralizes the scheme tokens *and* the
lazy-import shape so a caller can hand off a URL without first
guessing which sub-package to import.

The enum exposes:

* canonical members for every shipped filesystem backend
  (``FILE``, ``MEMORY``, ``DBFS``, ``VOLUMES``, ``S3``, ``HTTP``,
  ``HTTPS``);
* :meth:`from_` — forgiving coercion (string / :class:`PathScheme` /
  ``None``) with alias support (``"file://"`` / ``"FILE"`` /
  ``"s3a"`` all land on the right member);
* :meth:`path_class` — lazy import of the concrete
  :class:`yggdrasil.io.holder.Holder` subclass, so the first call
  triggers the side-effect import that registers the scheme into
  :data:`_HOLDER_SCHEMES` (``yggdrasil.databricks.fs``,
  ``yggdrasil.aws.fs``, …) and returns the class.

Each backend's path class continues to declare ``scheme = "dbfs"``
etc. as a plain string for backward compat with the existing
runtime registry; :class:`PathScheme` is the cross-cutting view onto
that data, not a parallel registry.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.io.holder import Holder


__all__ = ["PathScheme"]


# ---------------------------------------------------------------------------
# Lazy-import targets — module + attribute for each scheme. Imported on
# first :meth:`PathScheme.path_class` call; cached on the class via
# :data:`_PATH_CLASS_CACHE` to avoid re-importing on every dispatch.
# ---------------------------------------------------------------------------

# Per-process cache of resolved path classes — populated on first
# :meth:`PathScheme.path_class` call. Lives at module scope (rather
# than on the enum class) because :class:`Enum` would otherwise
# interpret a class-level dict as an aliased member value.
_PATH_CLASS_CACHE: dict[str, type] = {}


_PATH_CLASS_TARGETS: dict[str, tuple[str, str]] = {
    "file":    ("yggdrasil.io.path.local_path", "LocalPath"),
    "memory":  ("yggdrasil.io.memory", "Memory"),
    "dbfs":    ("yggdrasil.databricks.fs.dbfs_path", "DBFSPath"),
    "volumes": ("yggdrasil.databricks.fs.volume_path", "VolumePath"),
    "s3":      ("yggdrasil.aws.fs.path", "S3Path"),
    "http":    ("yggdrasil.io.http_.path", "HTTPPath"),
    "https":   ("yggdrasil.io.http_.path", "HTTPPath"),
}


# ---------------------------------------------------------------------------
# Alias table — every spelling we accept on input, normalized to the
# canonical scheme token. Case-insensitive; trailing ``://`` and
# whitespace are stripped before lookup.
# ---------------------------------------------------------------------------

_PATHSCHEME_ALIASES: dict[str, str] = {
    "":        "file",   # bare path — local
    "file":    "file",
    "local":   "file",
    "memory":  "memory",
    "mem":     "memory",
    "dbfs":    "dbfs",
    "volumes": "volumes",
    "volume":  "volumes",
    "uc":      "volumes",
    "s3":      "s3",
    "s3a":     "s3",
    "s3n":     "s3",
    "http":    "http",
    "https":   "https",
}


class PathScheme(str, Enum):
    """Canonical URL-scheme token for a Yggdrasil filesystem backend.

    Subclasses :class:`str` so a member is interchangeable with its
    scheme token everywhere a string is expected (``url.scheme ==
    PathScheme.DBFS`` works, ``f"{PathScheme.S3}://bucket/key"`` reads
    naturally). The lazy-import resolver lives on :meth:`path_class`.
    """

    FILE    = "file"
    MEMORY  = "memory"
    DBFS    = "dbfs"
    VOLUMES = "volumes"
    S3      = "s3"
    HTTP    = "http"
    HTTPS   = "https"

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "PathScheme":
        """Coerce *value* to a :class:`PathScheme` member.

        Accepts:

        * :class:`PathScheme` (returned as-is);
        * a scheme string — case-insensitive, trailing ``://`` and
          whitespace tolerated, common aliases (``"s3a"`` → :attr:`S3`,
          ``"local"`` → :attr:`FILE`, ``"volume"`` → :attr:`VOLUMES`);
        * ``None`` — returns *default* if supplied, else raises.

        ``default`` swallows unknown / unparseable input. Without it,
        unknown tokens raise :class:`ValueError` and unsupported types
        raise :class:`TypeError`.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            if default is not ...:
                return default
            raise ValueError("PathScheme cannot be derived from None")

        if isinstance(value, str):
            token = value.strip().lower()
            if token.endswith("://"):
                token = token[:-3]
            canonical = _PATHSCHEME_ALIASES.get(token)
            if canonical is None:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Unknown PathScheme: {value!r}. "
                    f"Valid schemes: {sorted({m.value for m in cls})!r}"
                )
            return cls(canonical)

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive PathScheme from {type(value).__name__}: {value!r}"
        )

    @classmethod
    def is_valid(cls, value: Any) -> bool:
        """``True`` when :meth:`from_` would succeed for *value*."""
        try:
            cls.from_(value)
            return True
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Lazy resolver — scheme → concrete Holder subclass
    # ------------------------------------------------------------------

    def path_class(self) -> "type[Holder]":
        """Return the concrete :class:`Holder` subclass for this scheme.

        Lazy: the backend module is imported on first use, which fires
        the ``__init_subclass__`` side-effect that registers the class
        into :data:`yggdrasil.io.holder._HOLDER_SCHEMES`. Subsequent
        calls hit the per-class cache.

        Raises :class:`ImportError` when the backend's optional
        dependencies aren't installed (``databricks-sdk`` for the
        Databricks schemes, ``boto3`` for ``s3``, …) — the message
        names the missing extra so the caller can install it.
        """
        cached = _PATH_CLASS_CACHE.get(self.value)
        if cached is not None:
            return cached

        target = _PATH_CLASS_TARGETS.get(self.value)
        if target is None:
            raise ValueError(
                f"PathScheme {self!r} has no registered path class. "
                f"Add it to _PATH_CLASS_TARGETS in path_scheme.py."
            )
        module_name, attr = target
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot resolve PathScheme.{self.name} → {attr}: "
                f"backend module {module_name!r} failed to import "
                f"({exc}). Install the matching extra (e.g. "
                f"`pip install ygg[databricks]` for dbfs/volumes/workspace, "
                f"`pip install ygg[aws]` for s3) and retry."
            ) from exc

        klass = getattr(module, attr, None)
        if klass is None:
            raise ImportError(
                f"Module {module_name!r} does not expose {attr!r}; "
                f"PathScheme.{self.name} cannot be resolved."
            )
        _PATH_CLASS_CACHE[self.value] = klass
        return klass

    @classmethod
    def resolve(cls, value: Any, *, default: Any = ...) -> "type[Holder]":
        """Shortcut: ``cls.from_(value).path_class()``.

        Useful where the caller has a raw scheme string (or URL) and
        just wants the concrete holder class. ``default`` is forwarded
        to :meth:`from_` for forgiving lookup; it does not catch
        :class:`ImportError` from :meth:`path_class`.
        """
        return cls.from_(value, default=default).path_class()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"PathScheme.{self.name}"

    def __str__(self) -> str:
        return self.value
