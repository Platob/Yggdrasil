"""Abstract filesystem path — ``pathlib.Path``-like API over :class:`URL`.

Design
------

* :class:`Path` is *not* a dataclass and has no metadata cache.
  Pure-path manipulation (``parent``, ``joinpath``, ``name``, …)
  delegates to :class:`URL`. Filesystem-flavoured calls (``stat``,
  ``ls``, ``mkdir``, ``read_bytes``, ``copy_to``, …) live here on
  top of a small abstract surface.
* Subclasses implement seven hooks: :meth:`full_path`,
  :meth:`_stat`, :meth:`_ls`, :meth:`_mkdir`, :meth:`_remove_file`,
  :meth:`_remove_dir`, :meth:`_open`. Everything else derives.
* :class:`Path` participates in the :class:`Disposable` graph from
  :mod:`yggdrasil.disposable` so temp-file lifecycle and PathIO
  ownership compose with the rest of the codebase.

Lifecycle rules
---------------

Every Path is constructed in the **open** state — :meth:`Disposable.open`
runs from the constructor unless ``auto_open=False`` is passed —
so naive callers never have to think about it. Acquire is a true
no-op (Path doesn't materialize files just because someone built a
reference to one). Release honors :attr:`temporary` and unlinks the
backing file when set, dirty-bit-irrelevant.

The lifecycle ``open()`` is intentionally separate from the I/O
``open_io()``. ``Disposable.open()`` takes no arguments (it's the
lifecycle "acquire") and returns ``self``; ``open_io(mode, …)`` takes
a mode and returns a :class:`PathIO`. The earlier draft tried to
overload ``open`` to mean both, which produced an infinite recursion
inside the Path-level ``open(mode)`` shadow. The names are kept
distinct here on purpose.

Temporary paths
---------------

:attr:`temporary` is a settable flag. Default ``False``. When True,
:meth:`_release` unlinks the underlying file (``missing_ok=True``,
exceptions swallowed). :meth:`with_tmp_name` mints a unique sibling
or child path with the flag set; :meth:`as_temporary` /
:meth:`as_persistent` flip it on an existing instance.

When a :class:`PathIO` is opened against a temporary path, the
PathIO adds the path as an owned child of the IO via
:meth:`Disposable.add_owned`. The graph's claim refcount keeps
the path alive until every PathIO over it has closed AND the path
itself has closed — only then does the unlink fire.

Staging
-------

:meth:`make_staging` is the canonical entry point for "give me a
fresh temporary file under this directory, and clean up any
expired siblings while you're there." All backends share:

- the TTL-encoded filename format (``…-<start_ts>-<end_ts>.<ext>``)
  so external sweepers can age files lexically without coordinating
  with the in-process rate limiter;
- the per-parent rate-limited sweep (``ExpiringDict``-backed)
  so a high-throughput caller doesn't flood the backend's listing
  API on every staging call.

Subclasses that need backend-specific bring-up (UC hierarchy for
:class:`VolumePath`, prefix construction for an S3 bucket, …)
provide their own factory that builds the parent path, then call
:meth:`make_staging` on it.
"""

from __future__ import annotations

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
    Union, )

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.disposable import Disposable
from yggdrasil.io.buffer.base import TabularIO
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaType
from yggdrasil.io.path_stat import PathKind, PathStats
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
    """Pick the best :class:`Path` subclass for *obj*.

    Order: exact-type match, registered handles() match, LocalPath
    fallback. Lazy-loaded to break import cycles.
    """
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
# Staging sweep rate-limit
# ---------------------------------------------------------------------------
#
# Process-global TTL'd dict mapping ``parent.full_path()`` -> a sentinel
# value.  An entry's *presence* signals "swept recently"; its absence
# signals "due for a sweep."  ``ExpiringDict`` handles the expiry, the
# size bound, and the thread-safety; we just use it as a "have we done
# this in the last N seconds" oracle.

_STAGING_SWEEP_INTERVAL_S: float = 300.0       # 5 minutes default
_STAGING_SWEEP_MAX_KEYS: int = 256
_STAGING_SWEPT: ExpiringDict[str, bool] = ExpiringDict(
    default_ttl=_STAGING_SWEEP_INTERVAL_S,
    max_size=_STAGING_SWEEP_MAX_KEYS,
)


# Match a TTL-encoded staging filename's trailing ``-<start>-<end>(.ext)*``.
# Time-sortable layout: prefix-{start}-{end}-{seed}(.ext)*
# Group 1 is the start_ts (epoch seconds), group 2 is the end_ts.
# Anchored at start (after a dash-separated prefix) so the regex
# captures the leading timestamps without being fooled by digits
# that appear inside the random seed.
_STAGING_TMP_RE: re.Pattern = re.compile(r"-(\d+)-(\d+)-[0-9a-f]+(?:\.[^/]+)?$")


# ---------------------------------------------------------------------------
# Path — abstract, URL-delegating
# ---------------------------------------------------------------------------


class Path(TabularIO[CastOptions], os.PathLike, ABC):
    """Abstract filesystem path with :class:`pathlib.Path`-like behaviour.

    See the module docstring for the design and lifecycle rules.

    Abstract hooks
    --------------
    - :meth:`full_path`     — absolute string rendering.
    - :meth:`_stat`         — backend stat → :class:`PathStats`.
    - :meth:`_ls`           — list children.
    - :meth:`_mkdir`        — create directory.
    - :meth:`_remove_file`  — delete file.
    - :meth:`_remove_dir`   — delete directory.
    - :meth:`_open`         — open file, return :class:`PathIO`.
    - :meth:`pread`         — positional read.
    - :meth:`pwrite`        — positional write.

    For backends without a native random-access primitive, the
    :meth:`pread` / :meth:`pwrite` overrides can delegate to the
    concrete helpers :meth:`_pread_via_io` (open + read) and
    :meth:`_pwrite_via_rmw` (read-modify-write). Those helpers are
    correct but slow; subclasses with a real positional API
    (``os.pread``, HTTP range requests) should implement their
    own. Making these abstract — rather than providing the slow
    helpers as defaults — forces every backend to think about
    whether it has a fast path, instead of silently inheriting an
    O(file size) write.
    """

    scheme: ClassVar[str] = ""
    __slots__ = ("url", "temporary")

    # Override on a subclass to tighten/loosen the staging-sweep rate
    # limit for that backend.  The module-level default applies to any
    # backend that doesn't override.  Currently the rate limit is shared
    # across backends via the module-level ExpiringDict — overriding here
    # changes only the *check* threshold the override caller sees, not
    # the dict TTL.  In practice 5 min is fine for every known backend.
    _STAGING_SWEEP_INTERVAL: ClassVar[float] = _STAGING_SWEEP_INTERVAL_S

    # ==================================================================
    # Construction / dispatch
    # ==================================================================

    @classmethod
    def default_mime_type(cls):
        """Path is format-agnostic — never auto-register against a mime type.

        :class:`TabularIO.__init_subclass__` registers concrete subclasses
        against their default mime; returning ``None`` opts Path (and every
        scheme-specific subclass) out cleanly. The actual format for a
        given Path comes from its URL extension at I/O time.
        """
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
        auto_open: bool = True,
    ) -> None:
        # TabularIO.__init__ runs Disposable.__init__ and seeds the
        # cache / spill-path slots every TabularIO carries.
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

        if auto_open:
            Disposable.open(self)

    # ==================================================================
    # Disposable hooks
    # ==================================================================

    def _acquire(self) -> None:
        return

    def _release(self) -> None:
        # Drop any persisted Arrow / Spark cache TabularIO is holding.
        try:
            self.unpersist()
        except Exception:
            pass
        if not self.temporary:
            return
        try:
            self.unlink(missing_ok=True)
        except Exception:
            pass

    # ==================================================================
    # TabularIO hooks — open the local file, dispatch to its BytesIO
    # ==================================================================

    # ``cached`` / ``persist`` / ``unpersist`` come from
    # :class:`TabularIO` — shared ``_persisted_data`` slot driver.

    def _read_arrow_batches(self, options: CastOptions) -> Iterator["pa.RecordBatch"]:
        """Stream Arrow batches by opening the file locally and delegating.

        ``open_io("rb")`` returns a :class:`BytesIO` (or, when the URL
        carries a tabular extension, the registered format leaf via
        :meth:`BytesIO.__new__`'s registry dispatch). Either way the
        buffer's ``_read_arrow_batches`` knows how to decode the bytes;
        we just bridge open → read → close.
        """
        buf = self.open_io("rb")
        try:
            yield from buf.read_arrow_batches(options=options)
        finally:
            buf.close()

    def _write_arrow_batches(
        self,
        batches: Iterable["pa.RecordBatch"],
        options: CastOptions,
    ) -> None:
        buf = self.open_io("wb")
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
        """Mint a unique sibling/child path with a TTL-encoded name.

        Filename layout (time-sortable):
        ``{prefix}{start}-{end}-{token}{suffix}``.

        ``start`` and ``end`` come first so a plain lexical sort of
        a directory's tmp files yields chronological order — handy
        for tailing a stream of staged writes or for time-based
        sweeps that prefer the oldest files. ``token`` is a random
        16-char hex tiebreaker.

        :data:`yggdrasil.io.fs.path._STAGING_TMP_RE` matches the
        leading timestamps for the cleanup sweep.
        """
        seed = os.urandom(8).hex()
        prefix = prefix or ""
        suffix = suffix or ""

        if ttl is None:
            name = f"{prefix}{seed}{suffix}"
        else:
            # Zero-pad to 12 digits so lexical order matches numeric
            # order across the full epoch range we'll see in
            # practice (today is ~1.7e9, 12 digits covers up to
            # year 33658).
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
        """Mint a fresh temporary staging file under this directory.

        ``path`` (optional) joins onto ``self`` first — pass a string
        like ``"sub/dir"`` or a list of segments ``["sub", "dir"]`` to
        nest the staging area under a subpath.  ``None`` (default) uses
        ``self`` as the parent directly.

        Pre-cleans the parent directory of expired siblings with the
        same TTL-encoded name format, rate-limited per-parent so
        high-throughput callers don't flood the backend's listing API.
        Default rate limit: one sweep per parent per
        :data:`_STAGING_SWEEP_INTERVAL_S` seconds (process-global).

        Returns a ``temporary=True`` :class:`Path` whose name encodes
        the lifetime so external sweepers can age it lexically.

        Subclass hook
        -------------
        Subclasses needing backend-specific bring-up (e.g. UC hierarchy
        creation for :class:`VolumePath`) compute the parent path and
        call ``super().make_staging(...)``.  See
        :meth:`VolumePath.staging_path` for the canonical example.
        """
        parent = self if path is None else self._join_segments(path)

        if sweep:
            try:
                parent._sweep_expired_staging(force=force_sweep)
            except Exception:
                # Sweep is best-effort.  A failed listing on a
                # not-yet-existing parent is fine — the staging write
                # below will create it.
                LOGGER.debug(
                    "Staging sweep failed for %s; continuing",
                    parent, exc_info=True,
                )

        # Build the file: parent / "{tmp_prefix}{token}-{start}-{end}{ext}"
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
        """Resolve ``path`` against ``self``, accepting str or segment list."""
        if isinstance(path, str):
            return self / path
        # Iterable of segments — fall through joinpath.
        return self.joinpath(*path)

    @staticmethod
    def _staging_extension(media_type: Union["MediaType", str, None]) -> str:
        """Resolve the staging file extension from a media-type hint."""
        if media_type is None:
            return ""
        try:
            from yggdrasil.io.enums import MediaType, MediaTypes
            mt = MediaType.from_(media_type, default=MediaTypes.PARQUET)
            return mt.full_extension or ""
        except Exception:
            # If MediaType resolution fails (no enum match), treat the
            # input as a literal extension string.
            if isinstance(media_type, str):
                return media_type.lstrip(".")
            return ""

    def _sweep_expired_staging(self, *, force: bool = False) -> bool:
        """Best-effort sweep of expired staging files under this directory.

        Returns ``True`` if a sweep ran, ``False`` if the rate limiter
        skipped this call.

        Rate-limited per parent ``full_path()`` via the module-level
        :data:`_STAGING_SWEPT` :class:`ExpiringDict` — at most one
        sweep per :data:`_STAGING_SWEEP_INTERVAL_S` per parent per
        process.  The check-and-stamp is one ``ExpiringDict`` op so
        concurrent callers in the same parent converge to one sweep
        per interval.
        """
        key = self.full_path()

        if not force:
            # `__contains__` on ExpiringDict drops stale entries; if the
            # key is still alive we've already swept this interval.
            if key in _STAGING_SWEPT:
                return False

        # Stamp BEFORE the walk so concurrent callers see "swept" and
        # skip.  Even if the walk fails, the stamp gates retries to the
        # next interval (failures are best-effort).
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
        """Test/maintenance hook: drop the rate-limit stamp.

        ``None`` clears every tracked parent (next call to any parent
        will sweep); a specific path clears just that parent.
        """
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
    def is_local(self) -> bool:
        return False

    # ==================================================================
    # Concurrency — sidecar locks
    # ==================================================================

    def lock_path(self, *, read: bool = False, write: bool = True) -> str:
        """Return the canonical sidecar lock-file path for this path.

        The lock filename carries an access-intent suffix
        (``.r.lock`` / ``.w.lock`` / ``.rw.lock``), so external
        tooling can identify what kind of lock is held without
        opening the file. Read locks are typically *skippable* by
        cleanup or monitoring tools — multiple of them coexist by
        design.

        Override on a subclass that needs a different sidecar
        location (e.g. a backend that forbids hidden files in the
        target directory). The default builds ``<dir>/.<basename>.{suffix}.lock``
        from :meth:`full_path`.
        """
        from yggdrasil.io.buffer._concurrency import lock_path_for
        return lock_path_for(self.full_path(), read=read, write=write)

    def lock(
        self,
        *,
        read: bool = False,
        write: bool = True,
        wait: Any = None,
        stale_after_seconds: Any = None,
    ) -> "Any":
        """Build (but don't acquire) a cross-process lock for this path.

        Usage::

            with path.lock(write=True, wait=30):
                path.write_bytes(payload)

        Backend dispatch:

        - **Local paths** → :class:`FileLock` (``fcntl`` /
          ``msvcrt``). Kernel-enforced, OS-released on process
          death. Honours shared/exclusive (``LOCK_SH`` / ``LOCK_EX``).
        - **Remote paths** (S3, GCS, ABFS, Databricks volumes,
          memory FS, …) → :class:`AtomicLock`. The sidecar is
          created with ``xb`` (atomic exclusive); other writers
          poll until it disappears. ``shared`` is accepted for
          parity but ignored — the lock is always exclusive.

        ``wait`` is a :class:`WaitingConfig` argument: ``None``
        (wait forever), a number (seconds), a ``dict`` of overrides,
        or a ready-made :class:`WaitingConfig`. Backoff and retry
        come from the config's ``interval`` / ``backoff`` /
        ``max_interval`` knobs; on contention the lock loop calls
        :meth:`WaitingConfig.sleep` and raises :class:`TimeoutError`
        once the deadline is reached.

        ``stale_after_seconds`` (atomic-lock only) controls how long
        a lingering sidecar is tolerated before the next acquirer
        force-unlinks it. Zero / negative disables staleness
        recovery (correctness > liveness). Defaults to 15 minutes
        — long enough that an honest holder heartbeating once per
        minute is safe.

        Semantics for the suffix:

        - ``read=True, write=False`` → ``.r.lock``
        - ``read=False, write=True`` (default) → ``.w.lock``
        - ``read=True, write=True`` → ``.rw.lock``

        On platforms / backends without :mod:`fcntl` /
        :mod:`msvcrt` support and without ``xb`` mode, the returned
        lock degrades to a no-op — the caller's I/O still proceeds,
        just without cross-process coordination. Subclasses for
        backends with a native CAS primitive (S3 conditional PUT,
        GCS preconditions) can override this method entirely.
        """
        # Lazy import — _concurrency imports are leaves but the full
        # ``yggdrasil.io.buffer`` package pulls in NestedIO which in
        # turn imports ``Path``, so module-level imports here would
        # close the cycle.
        from yggdrasil.io.buffer._concurrency import AtomicLock, FileLock

        suffix_path_str = self.lock_path(read=read, write=write)
        if self.is_local:
            return FileLock(
                suffix_path_str,
                shared=(read and not write),
                wait=wait,
            )
        # Build a sibling :class:`Path` of the same backend pointing
        # at the sidecar location. Atomic-create needs the same
        # filesystem semantics as the target (exclusive-create,
        # unlink, stat) — co-locating the sidecar is the simplest
        # way to inherit them without the user wiring per-backend
        # knobs.
        try:
            sidecar = type(self).from_(suffix_path_str)
        except Exception:
            sidecar = Path.from_(suffix_path_str)
        return AtomicLock(
            sidecar,
            shared=(read and not write),
            wait=wait,
            stale_after_seconds=stale_after_seconds,
        )

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
    def _stat(self) -> PathStats: ...

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
    def _open(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> BytesIO:
        """Open the underlying file, return a :class:`BytesIO` (or subclass)."""

    # ==================================================================
    # I/O entry point — distinct from the lifecycle Disposable.open()
    # ==================================================================

    def open_io(
        self,
        mode: str = "rb",
        *,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        auto_open: bool = True,
        touch: bool = False,
    ) -> BytesIO:
        if not self.opened:
            Disposable.open(self)

        io = self._open(
            mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
            auto_open=auto_open,
            touch=touch,
        )

        return io

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

    def stat(self) -> PathStats:
        return self._stat()

    def exists(self, *, follow_symlinks: bool = True) -> bool:
        del follow_symlinks
        return self._stat().kind != PathKind.MISSING

    def is_file(self) -> bool:
        return self._stat().kind == PathKind.FILE

    def is_dir(self) -> bool:
        return self._stat().kind == PathKind.DIRECTORY

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
    def mtime(self) -> Optional[float]:
        s = self._stat()
        if s.kind == PathKind.MISSING:
            return None
        return s.mtime

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
        if kind == PathKind.MISSING:
            if missing_ok:
                return
            raise FileNotFoundError(f"{self.full_path()!r} does not exist")
        if kind == PathKind.DIRECTORY:
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
        if kind == PathKind.FILE:
            self._remove_file(allow_not_found=allow_not_found)
        elif kind == PathKind.DIRECTORY:
            self._remove_dir(
                recursive=recursive,
                allow_not_found=allow_not_found,
                with_root=True,
            )
        elif kind == PathKind.MISSING:
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
        # EAFP: try atomic exclusive-create when ``exist_ok=False``.
        # The backend tells us via FileExistsError whether the file
        # was already there — no separate ``exists()`` round-trip.
        if not exist_ok:
            try:
                self.write_bytes(b"", mode="xb", parents=parents)
                return
            except FileExistsError:
                raise FileExistsError(
                    f"Path already exists: {self.full_path()!r}"
                )
        # exist_ok=True: ``wb`` is fine — it overwrites an empty
        # file with empty bytes (no-op semantically) or creates one.
        self.write_bytes(b"", parents=parents)

    def resolve(self, *, strict: bool = False) -> "Path":
        del strict
        return self

    def absolute(self) -> "Path":
        return self

    # ==================================================================
    # Bytes / text I/O
    # ==================================================================

    @abstractmethod
    def pread(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes: ...

    @abstractmethod
    def pwrite(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int: ...

    def _pread_via_io(
        self,
        n: int,
        pos: int,
        *,
        default: Any = ...,
    ) -> bytes:
        if pos < 0:
            raise ValueError("pread position must be >= 0")
        if n == 0:
            return b""
        try:
            with self.open_io("rb") as src:
                if n < 0:
                    src.seek(pos)
                    return src.read()
                return src.pread(n=n, pos=pos)
        except (OSError, ValueError):
            if default is ...:
                raise
            return default

    def _pwrite_via_rmw(
        self,
        data: Union[bytes, bytearray, memoryview],
        pos: int,
        *,
        parents: bool = True,
    ) -> int:
        mv = memoryview(data)
        if not mv.c_contiguous:
            mv = memoryview(bytes(mv))
        n = len(mv)
        if n == 0:
            return 0
        if pos < 0:
            raise ValueError("pwrite position must be >= 0")

        # EAFP: read directly. ``read_bytes(raise_error=False)``
        # returns ``b""`` when the file is missing, saving the extra
        # ``exists()`` round-trip on remote backends.
        existing = self.read_bytes(raise_error=False)
        end = pos + n

        if end <= len(existing):
            buf = bytearray(existing)
            buf[pos:end] = mv
            out = bytes(buf)
        else:
            buf = bytearray(end)
            if existing:
                buf[: len(existing)] = existing
            buf[pos:end] = mv
            out = bytes(buf)

        self.write_bytes(out, parents=parents)
        return n

    def read_bytes(self, *, raise_error: bool = True) -> bytes:
        try:
            with self.open_io("rb") as fh:
                return fh.read()
        except (OSError, ValueError):
            if raise_error:
                raise
            return b""

    def write_bytes(
        self,
        data: Union[bytes, bytearray, memoryview],
        *,
        mode: str = "wb",
        parents: bool = True,
    ) -> int:
        def _write() -> int:
            with self.open_io(mode) as fh:
                return fh.write(bytes(data))

        try:
            return _write()
        except (OSError, ValueError):
            if not parents:
                raise
            self.parent.mkdir(parents=True, exist_ok=True)
            return _write()

    def truncate(self, n: int, *, parents: bool = True) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")

        stat = self._stat()
        if stat.kind == PathKind.MISSING:
            raise FileNotFoundError(
                f"Cannot truncate non-existent path {self.full_path()!r}. "
                "Call touch() first if you want create-or-resize semantics."
            )
        if stat.kind == PathKind.DIRECTORY:
            raise IsADirectoryError(
                f"Cannot truncate directory {self.full_path()!r}"
            )

        current = int(stat.size)
        if n == current:
            return n
        if n == 0:
            self.write_bytes(b"", parents=parents)
            return 0

        if n < current:
            prefix = self.pread(n=n, pos=0)
            if len(prefix) != n:
                raise OSError(
                    f"Short pread during truncate: requested {n} bytes, "
                    f"got {len(prefix)}"
                )
            self.write_bytes(prefix, parents=parents)
            return n

        existing = self.read_bytes()
        if len(existing) != current:
            raise OSError(
                f"Stat/read mismatch during truncate: stat={current}, "
                f"read={len(existing)}"
            )
        buf = bytearray(n)
        buf[:current] = existing
        self.write_bytes(bytes(buf), parents=parents)
        return n

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
        parents: bool = True
    ):
        buffer = BytesIO.from_(buffer)
        return self._write_bytes_io(buffer, batch_size=batch_size, parents=parents)

    def _write_bytes_io(
        self,
        buffer: BytesIO,
        *,
        batch_size: int = 1024 * 1024,
        parents: bool = True
    ):
        buffer = BytesIO.from_(buffer)
        if not buffer.opened:
            buffer.open()

            try:
                return self.write_bytes_io(
                    buffer, batch_size=batch_size, parents=parents
                )
            finally:
                buffer.close()

        # src local-spilled + self local → kernel copy.
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

        if buffer.spilled and buffer.is_local:
            mv = buffer.memoryview()
            try:
                self.write_bytes(mv, parents=parents)
                return len(mv)
            finally:
                del mv

        if not buffer.spilled:
            with self.open_io("wb", auto_open=False) as fh:
                fh.write_bytes_io(buffer, batch_size=batch_size)

        return 0


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
        """Copy self to *dest*. Path-to-Path; chooses the fastest available shape.

        Fast paths, in priority order:

        1. **Local → local** — :func:`shutil.copyfile`, which uses
           ``os.sendfile`` (Linux), ``fcopyfile`` (macOS), or
           ``CopyFileEx`` (Windows) under the hood. Kernel-side
           zero-copy on supported platforms; no userspace bytes touched.
        2. **Same-type non-local → same-type non-local** — left to
           subclasses to override (e.g. S3 server-side copy). Falls
           through to the stream loop here.
        3. **Streaming loop** — read the source via ``open_io("rb")``,
           write the dest via ``open_io("wb")``, ``batch_size`` chunks.
           Always correct, memory-bounded by ``batch_size``.

        For mid-size cross-backend copies (one local, one remote)
        where the buffered ``BytesIO`` already holds the whole payload
        in memory or in a single mmap-able file, the dest's
        ``write_bytes`` path is one round-trip vs. N for the loop.
        That's handled implicitly: ``write_stream`` from the source's
        ``open_io`` falls back to the loop, which is fine — backends
        that benefit from single-shot uploads (Databricks, S3) already
        aggregate the loop's chunks into one PUT in their ``BytesIO``
        subclass's ``flush``.
        """
        dest_path = Path.from_(dest)
        if dest_path == self:
            return self.size

        if parents:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Fast path 1: local → local via shutil.copyfile ---------------
        # Both ends are local files on the same kernel — let the OS do
        # the copy. shutil.copyfile picks sendfile/fcopyfile/CopyFileEx
        # based on platform and falls back to its own optimized loop
        # if those aren't available.
        if self.is_local and dest_path.is_local:
            src_full = self.full_path()
            dst_full = dest_path.full_path()
            # Confirm source is a regular file before handing to shutil.
            # shutil.copyfile would raise SameFileError on os.path.samefile;
            # we already short-circuited equal paths above, but the local
            # paths could differ textually while pointing to the same
            # inode (symlinks, bind mounts). Let shutil's own check handle
            # that — it raises SameFileError, which is the right behavior.
            try:
                shutil.copyfile(src_full, dst_full)
            except shutil.SameFileError:
                return self.size
            # shutil.copyfile doesn't return bytes copied; stat the
            # destination. Avoids a second open just to len() the source.
            try:
                return int(dest_path._stat().size)
            except Exception:
                return self.size

        # --- Fallback: streaming loop -------------------------------------
        return self._copy_to_via_stream(dest_path, batch_size=batch_size)

    def _copy_to_via_stream(
        self,
        dest_path: "Path",
        *,
        batch_size: int,
    ) -> int:
        """Fallback path-to-path copy via paired open_io streams."""
        total = 0
        with self.open_io("rb") as src, dest_path.open_io("wb") as dst:
            # If the source BytesIO is local-spilled, hand its mmap
            # memoryview to write() in one shot — saves N chunked
            # syscalls for medium files, and the dst's write() handles
            # buffer-protocol inputs without an extra copy in the
            # bytes-fast path.
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
                    # Release the mmap before src closes — some platforms
                    # are strict about closing a file with live maps.
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
        """Stream *src* into self.

        Fast paths when *src* is a yggdrasil :class:`BytesIO`:

        - **src local-spilled, self local** → :func:`shutil.copyfile`
          between the two backing files. Kernel-side on supported
          platforms; bypasses the open/loop/close cycle entirely.
        - **src local-spilled, self non-local** → hand the source's
          mmap :meth:`memoryview` to ``self.write_bytes`` in one
          shot. One upload round-trip instead of N chunked writes.
        - **src memory-mode** → ``self.write_bytes(src.to_bytes())``
          directly. The source's bytes are already contiguous; the
          stream loop would just chunk them back up.

        Stdlib :class:`io.BytesIO` is also single-shotted via
        :meth:`getvalue`. Anything else (HTTP response, pipe, named
        pipe, stdin) falls through to the streaming loop.
        """
        # Fast paths for yggdrasil BytesIO sources -------------------------
        if isinstance(src, BytesIO):
            return self.write_bytes_io(src)

        # Stdlib io.BytesIO — single-shot getvalue() also wins here.
        if isinstance(src, io.BytesIO):
            # tell() so we respect any pre-positioned cursor, the same
            # way the loop would.
            start = src.tell()
            payload = src.getvalue()[start:]
            self.write_bytes(payload, parents=parents)
            return len(payload)

        # --- Fallback: streaming loop -------------------------------------
        total = 0
        with self.open_io("wb") as dst:
            while True:
                chunk = src.read(batch_size)
                if not chunk:
                    break
                dst.write(chunk)
                total += len(chunk)
        return total

    # ==================================================================
    # memoryview / mmap — default fallbacks
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
    # Context manager — single-shot, file-like idiom
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
        if exc_type is not None:
            self._dirty = False
        self.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _materialize(patterns: Iterable[str] | None) -> Optional[Tuple[str, ...]]:
    if patterns is None:
        return None
    out = patterns if isinstance(patterns, tuple) else tuple(patterns)
    return out if out else None