"""Centralized URL-scheme enum for :class:`URLBased` dispatch.

Across Yggdrasil, anything addressable by a URL — a filesystem
path (``file://``, ``s3://``, ``dbfs://``, ``volumes://``), an
in-memory buffer (``mem://``), a workspace object
(``workspace://``), an HTTP endpoint (``http://`` / ``https://``)
— exposes itself as a :class:`yggdrasil.io.url.URLBased` subclass.
That base owns a single registry keyed by :class:`Scheme`, and
``URLBased.from_url(url)`` dispatches to the concrete subclass for
``url.scheme`` without the caller knowing which sub-package owns it.

This module centralizes the scheme tokens and the lazy-import
shape so a caller can hand off a URL without first guessing which
sub-package implements it: :meth:`Scheme.path_class` imports the
backend module on demand, the import side-effect wires the
subclass into the URLBased registry, and the resolved class is
returned.

The enum exposes:

* canonical members for every shipped backend (``FILE``, ``MEMORY``,
  ``DBFS``, ``VOLUMES``, ``WORKSPACE``, ``S3``, ``HTTP``, ``HTTPS``);
* :meth:`from_` — forgiving coercion (string / :class:`Scheme` /
  ``None``) with alias support (``"file://"`` / ``"FILE"`` /
  ``"s3a"`` all land on the right member);
* :meth:`path_class` — lazy import of the concrete subclass;
* :meth:`resolve` — the ``from_(...) → path_class()`` shortcut.

Each backend's class declares ``scheme = Scheme.DBFS`` (etc.) on
the class body; :meth:`URLBased.__init_subclass__` reads that
member and registers the class into the cross-cutting registry.
"""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from yggdrasil.io.url import URLBased


__all__ = ["Scheme"]


# ---------------------------------------------------------------------------
# Lazy-import targets — module + attribute for each scheme. Imported on
# first :meth:`Scheme.path_class` call; cached on a per-process dict
# (:data:`_PATH_CLASS_CACHE`) to avoid re-importing on every dispatch.
# ---------------------------------------------------------------------------

# Per-process cache of resolved path classes. Lives at module scope
# (rather than on the enum class) because :class:`Enum` would
# otherwise interpret a class-level dict as an aliased member value.
_PATH_CLASS_CACHE: dict[str, type] = {}


_PATH_CLASS_TARGETS: dict[str, tuple[str, str]] = {
    "file":           ("yggdrasil.io.path.local_path", "LocalPath"),
    "mem":            ("yggdrasil.io.memory", "Memory"),
    # ``dbfs://`` resolves to the abstract Databricks dispatcher —
    # :class:`DatabricksPath` inspects the URL and forwards to the
    # concrete subclass (DBFS surface, Volumes, Workspace).
    "dbfs":           ("yggdrasil.databricks.fs.path", "DatabricksPath"),
    "dbfs+dbfs":      ("yggdrasil.databricks.fs.dbfs_path", "DBFSPath"),
    "dbfs+volume":    ("yggdrasil.databricks.fs.volume_path", "VolumePath"),
    "dbfs+workspace": ("yggdrasil.databricks.fs.workspace_path", "WorkspacePath"),
    "s3":             ("yggdrasil.aws.fs.path", "S3Path"),
    "http":           ("yggdrasil.io.http_.path", "HTTPPath"),
    "https":          ("yggdrasil.io.http_.path", "HTTPPath"),
}


# ---------------------------------------------------------------------------
# Alias table — every spelling we accept on input, normalized to the
# canonical scheme token. Case-insensitive; trailing ``://`` and
# whitespace are stripped before lookup.
# ---------------------------------------------------------------------------

_SCHEME_ALIASES: dict[str, str] = {
    "":               "file",   # bare path — local
    "file":           "file",
    "local":          "file",
    "mem":            "mem",
    "memory":         "mem",
    "dbfs":           "dbfs",
    "dbfs+dbfs":      "dbfs+dbfs",
    "dbfs+volume":    "dbfs+volume",
    "dbfs+volumes":   "dbfs+volume",
    "volumes":        "dbfs+volume",
    "volume":         "dbfs+volume",
    "uc":             "dbfs+volume",
    "dbfs+workspace": "dbfs+workspace",
    "workspace":      "dbfs+workspace",
    "ws":             "dbfs+workspace",
    "s3":             "s3",
    "s3a":            "s3",
    "s3n":            "s3",
    "http":           "http",
    "https":          "https",
}


class Scheme(str, Enum):
    """Canonical URL-scheme token for a Yggdrasil :class:`URLBased`
    subclass.

    Subclasses :class:`str` so a member is interchangeable with its
    scheme token everywhere a string is expected (``url.scheme ==
    Scheme.DBFS`` works, ``f"{Scheme.S3}://bucket/key"`` reads
    naturally). The lazy-import resolver lives on :meth:`path_class`.
    """

    FILE   = "file"
    MEMORY = "mem"
    S3     = "s3"
    HTTP   = "http"
    HTTPS  = "https"

    #: Databricks family root — ``dbfs://`` URLs route to the
    #: abstract :class:`DatabricksPath` dispatcher, which inspects
    #: the URL and forwards to the right concrete subclass (DBFS
    #: surface, Volumes, Workspace).
    DBFS                 = "dbfs"

    #: Concrete scheme for the DBFS surface itself (``dbfs+dbfs://``).
    #: Distinct from :attr:`DBFS` so the family root can stay a
    #: dispatcher; the same `dbfs+<surface>` convention applies to
    #: Volumes and Workspace.
    DATABRICKS_DBFS      = "dbfs+dbfs"
    DATABRICKS_VOLUME    = "dbfs+volume"
    DATABRICKS_WORKSPACE = "dbfs+workspace"

    # ------------------------------------------------------------------
    # Coercion
    # ------------------------------------------------------------------

    @classmethod
    def from_(cls, value: Any, *, default: Any = ...) -> "Scheme":
        """Coerce *value* to a :class:`Scheme` member.

        Accepts:

        * :class:`Scheme` (returned as-is);
        * a scheme string — case-insensitive, trailing ``://`` and
          whitespace tolerated, common aliases (``"s3a"`` → :attr:`S3`,
          ``"local"`` → :attr:`FILE`, ``"volume"`` → :attr:`VOLUMES`,
          ``"memory"`` → :attr:`MEMORY`);
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
            raise ValueError("Scheme cannot be derived from None")

        if isinstance(value, str):
            token = value.strip().lower()
            if token.endswith("://"):
                token = token[:-3]
            canonical = _SCHEME_ALIASES.get(token)
            if canonical is None:
                if default is not ...:
                    return default
                raise ValueError(
                    f"Unknown Scheme: {value!r}. "
                    f"Valid schemes: {sorted({m.value for m in cls})!r}"
                )
            return cls(canonical)

        if default is not ...:
            return default
        raise TypeError(
            f"Cannot derive Scheme from {type(value).__name__}: {value!r}"
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
    # Lazy resolver — scheme → concrete URLBased subclass
    # ------------------------------------------------------------------

    def path_class(self) -> "type[URLBased]":
        """Return the concrete :class:`URLBased` subclass for this scheme.

        Lazy: the backend module is imported on first use, which fires
        the :meth:`URLBased.__init_subclass__` side-effect that
        registers the class. Subsequent calls hit the per-process cache.

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
                f"Scheme {self!r} has no registered path class. "
                f"Add it to _PATH_CLASS_TARGETS in scheme.py."
            )
        module_name, attr = target
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot resolve Scheme.{self.name} → {attr}: "
                f"backend module {module_name!r} failed to import "
                f"({exc}). Install the matching extra (e.g. "
                f"`pip install ygg[databricks]` for dbfs/volumes/workspace, "
                f"`pip install ygg[aws]` for s3) and retry."
            ) from exc

        klass = getattr(module, attr, None)
        if klass is None:
            raise ImportError(
                f"Module {module_name!r} does not expose {attr!r}; "
                f"Scheme.{self.name} cannot be resolved."
            )
        _PATH_CLASS_CACHE[self.value] = klass
        return klass

    @classmethod
    def resolve(cls, value: Any, *, default: Any = ...) -> "type[URLBased]":
        """Shortcut: ``cls.from_(value).path_class()``.

        Useful where the caller has a raw scheme string (or URL) and
        just wants the concrete URLBased class. ``default`` is forwarded
        to :meth:`from_` for forgiving lookup; it does not catch
        :class:`ImportError` from :meth:`path_class`.
        """
        return cls.from_(value, default=default).path_class()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Scheme.{self.name}"

    def __str__(self) -> str:
        return self.value
