"""Abstract filesystem path — :class:`Holder` over a :class:`URL`.

A :class:`Path` is a byte holder addressed by a URL. The holder
contract (:meth:`read_mv` / :meth:`write_mv` / :meth:`reserve` /
:meth:`truncate` / :meth:`clear` / :attr:`size`) routes through two
positional primitives — :meth:`_read_mv` / :meth:`_write_mv` — that
subclasses override. There is no buffering at this layer; callers
that want to coalesce wrap the path in
:class:`yggdrasil.io.holder.IO`.

Subclasses implement seven hooks:

- :meth:`full_path`         — string form of the URL on the backend
- :meth:`_stat`             — ``IOStats`` round-trip (kind + size + mtime)
- :meth:`_ls`               — list children
- :meth:`_mkdir`            — create directory
- :meth:`_remove_file`      — unlink one file
- :meth:`_remove_dir`       — rmtree
- :meth:`_read_mv`          — positional read → :class:`memoryview`
- :meth:`_write_mv`         — positional write ← :class:`memoryview`

The pure-path API (parts, name, parent, suffix, joinpath, …)
delegates straight to :class:`URL` — :class:`Path` adds no parsing
of its own. Subclass dispatch is a small registry keyed by URL
scheme; :meth:`Path.from_` resolves a candidate via the registry
and falls back to :class:`LocalPath`.
"""

from __future__ import annotations

import datetime as dt
import importlib
import os
import re
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, List, Tuple, Optional

if TYPE_CHECKING:
    from yggdrasil.io.holder import Holder

from yggdrasil.enums import Mode
from yggdrasil.dataclasses import WaitingConfigArg, WaitingConfig
from yggdrasil.io.base import IO
from yggdrasil.io.io_stats import IOKind, IOStats, TimeLike
from yggdrasil.url import URL

__all__ = ["Path"]


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

TS_PATTERN = re.compile(r"-(\d+)-(\d+)-")


class Path(IO, os.PathLike, ABC):
    """Abstract URL-addressed byte holder with filesystem semantics.

    Two layers, no shared state between them:

    1. **Holder I/O** — :meth:`read_mv` / :meth:`write_mv` /
       :meth:`truncate` / :meth:`clear` / :attr:`size` route through
       :meth:`_read_mv` / :meth:`_write_mv`. Backends with native
       positional I/O (LocalPath via ``os.pread``/``os.pwrite``)
       override these directly; whole-blob backends (RemotePath) splice
       via a download / re-upload.
    2. **Filesystem** — :meth:`stat`, :meth:`exists`, :meth:`is_file`,
       :meth:`is_dir`, :meth:`iterdir`, :meth:`mkdir`, :meth:`unlink`,
       :meth:`remove`. All thin wrappers over the abstract hooks.

    Pure-path manipulation (:attr:`parts`, :attr:`name`, :attr:`parent`,
    :attr:`suffix`, :meth:`joinpath`, :meth:`with_suffix`, …) delegates
    straight to :attr:`url` — Path adds no parsing.
    """

    scheme: ClassVar[str] = ""

    # The stat-cache machinery (``STAT_CACHE_TTL`` / ``_stat_cached`` /
    # ``_stat_cached_at`` slots, ``_stat_cached_fresh`` /
    # ``_persist_stat_cache`` / the ``_touch_stat`` propagation, and the
    # ``_TRANSIENT_STATE_ATTRS`` pickle filter) lives on the base
    # :class:`~yggdrasil.io.holder.Holder` so every holder shares one
    # IOStats snapshot mutated in :meth:`Holder._touch_stat`. Path only
    # reads off it through the filesystem predicates below and overrides
    # :attr:`STAT_CACHE_TTL` (e.g. :class:`RemotePath`) when a snapshot
    # must not outlive the backend's mutation window.

    # ==================================================================
    # Construction / coercion
    # ==================================================================

    @classmethod
    def from_(cls, obj: Any, **kwargs: Any) -> "Path":
        """Coerce *obj* (str / URL / pathlib / Path) into a :class:`Path`.

        When called on the abstract :class:`Path`, dispatches via the
        :class:`Holder` scheme registry to the subclass registered for
        the URL's scheme; defaults to :class:`LocalPath` for path-shaped
        URLs without an explicit scheme. When called on a concrete
        subclass, returns an instance of that subclass.
        """
        if isinstance(obj, Path):
            if cls is Path or isinstance(obj, cls):
                return obj
            obj = obj.url

        url = URL.from_(obj)
        if cls.__subclasses__():
            # Holder.__new__ dispatches on scheme; force LocalPath for
            # the no-scheme path case so callers don't accidentally
            # land on :class:`Memory`.
            scheme = url.scheme
            if scheme:
                from yggdrasil.url import URLBased

                try:
                    target = URLBased.for_scheme(scheme)
                except (ValueError, ImportError) as exc:
                    raise ValueError(
                        f"Unknown scheme {scheme!r} for Path.from_({obj!r})."
                    ) from exc
                return target(url=url, **kwargs)
            from yggdrasil.path.local_path import LocalPath

            return LocalPath(url=url, **kwargs)
        return cls(url=url, **kwargs)

    # ``_from_url`` lives on :class:`Holder` — Path is a top-level
    # storage (``_parent is None``), so the default ``type(self)(url=url)``
    # branch fires here and the previous Path-local override is gone.

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
    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        """Yield children. Empty when missing or not a directory.

        ``singleton_ttl`` is forwarded to child path construction so
        backends backed by :class:`Singleton` can opt the listed
        children out of the per-class ``_INSTANCES`` cache. The
        default ``False`` mirrors the contract on hot listing call
        sites: an ``iterdir`` of N entries pays at most one Python
        allocation per child, with nothing pinned in the bounded
        singleton cache. Callers that want listing children to share
        identity with later constructor calls pass ``...`` (class
        default TTL), ``None`` (process lifetime), or a number of
        seconds. Backends whose children are not :class:`Singleton`
        accept and ignore the kwarg.
        """

    @abstractmethod
    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        """Create directory at this path."""

    @abstractmethod
    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        """Unlink the file at this path."""

    @abstractmethod
    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
    ) -> None:
        """Remove the directory at this path."""

    @abstractmethod
    def _read_mv(self, n: int, pos: int) -> memoryview:
        """Positional read → :class:`memoryview`. ``n < 0`` → to EOF.

        Receives a non-negative, in-range ``(n, pos)`` from
        :meth:`read_mv` (bounds + negative-index normalization happen
        there). Whole-blob backends materialize the object and slice;
        positional backends (LocalPath) read the window directly. A
        missing object raises :class:`FileNotFoundError`.
        """

    @abstractmethod
    def _write_mv(self, data: memoryview, pos: int) -> int:
        """Splice *data* at *pos* on the backing. Returns bytes written.

        Positional backends write the window in place; whole-blob
        backends (RemotePath) read-modify-write the object. Size
        management and dirty-marking happen in :meth:`write_mv`.
        """

    # ==================================================================
    # Holder primitives — built on _read_mv / _write_mv
    # ==================================================================

    @property
    def size(self) -> int:
        # Fast path: a seeded :class:`IOStats` (from a listing entry
        # or a previous probe) skips the per-call backend round trip.
        # ``_stat_cached_fresh`` lives on the base :class:`Holder`.
        cached = self._stat_cached_fresh()
        if cached is not None:
            return int(cached.size)
        return int(self._stat().size)

    def _stat_cached_fresh(self) -> "IOStats | None":
        """Return the cached :class:`IOStats` when still inside its TTL.

        ``None`` when the slot is empty *or* the entry has expired
        past :attr:`stat_cache_ttl`. Subclasses that always want a
        fresh probe (e.g. test scaffolding) override
        :attr:`stat_cache_ttl` to ``0`` so this method always returns
        ``None``.
        """
        cached = self._stat_cached
        if cached is None:
            return None
        ttl = self.STAT_CACHE_TTL
        if ttl is None:
            return cached
        if ttl <= 0:
            return None
        if (time.monotonic() - self._stat_cached_at) <= ttl:
            return cached
        return None

    def reserve(self, n: int) -> None:
        """No-op by default — files have no separate capacity layer."""
        del n

    def truncate(self, n: int) -> int:
        """Resize to exactly *n* bytes via read-modify-write.

        The generic whole-file fallback: download, slice / zero-extend,
        re-write. Backends with a native resize (LocalPath
        ``os.ftruncate``, RemotePath's single upload) override this.
        """
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        current = self.size
        if n == current:
            return n
        existing = bytes(self._read_mv(current, 0)) if current > 0 else b""
        if n <= len(existing):
            payload = existing[:n]
        else:
            payload = existing + b"\x00" * (n - len(existing))
        self._write_mv(memoryview(payload), 0)
        # The backing is now ``n`` bytes — reflect it in the stat cache so a
        # seeded entry (from a listing) doesn't keep reporting the old size.
        self._touch_stat(size=n)
        return n

    def _clear(self) -> None:
        """:class:`Holder` primitive: drop the backing entirely (idempotent)."""
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))
        # The backing is gone — a cached FILE/size snapshot would now lie.
        # Drop it so the next probe re-checks the backend (and reports
        # MISSING) instead of trusting the pre-clear state.
        self._stat_cached = None
        self._stat_cached_at = 0.0

    # ==================================================================
    # Filesystem surface — thin wrappers over the abstract hooks
    # ==================================================================

    def exists(self) -> bool:
        cached = self._stat_cached_fresh()
        if cached is not None:
            return cached.kind != IOKind.MISSING
        return self._stat().kind != IOKind.MISSING

    def is_file(self) -> bool:
        cached = self._stat_cached_fresh()
        if cached is not None:
            return cached.kind == IOKind.FILE
        return self._stat().kind == IOKind.FILE

    def is_dir(self) -> bool:
        cached = self._stat_cached_fresh()
        if cached is not None:
            return cached.kind == IOKind.DIRECTORY
        return self._stat().kind == IOKind.DIRECTORY

    @property
    def mtime(self) -> float:
        cached = self._stat_cached_fresh()
        if cached is not None:
            return float(cached.mtime or 0.0) if cached.kind != IOKind.MISSING else 0.0
        s = self._stat()
        return float(s.mtime or 0.0) if s.kind != IOKind.MISSING else 0.0

    # ``_stat_cached_fresh`` / ``_persist_stat_cache`` and the
    # ``_touch_stat`` write-propagation live on the base
    # :class:`~yggdrasil.io.holder.Holder` — a Path is just a holder
    # whose cached :class:`IOStats` carries a filesystem ``kind``.

    def invalidate_singleton(self, remove_global: bool = True) -> None:
        """Drop the cached :class:`IOStats` *and* pop self from
        ``_INSTANCES``. Single canonical invalidator for paths.

        Mutating ops (writes, deletes) call this so the next read on
        the same logical path picks up fresh state. Subclasses with
        their own caches (schema, table info, column lists) override
        and chain ``super().invalidate_singleton(...)``.
        """
        self._stat_cached = None
        self._stat_cached_at = 0.0
        super().invalidate_singleton(remove_global=remove_global)

    def iterdir(
        self, *, limit: "int | None" = None, singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        yield from self.ls(recursive=False, limit=limit, singleton_ttl=singleton_ttl)

    def ls(
        self,
        *,
        recursive: bool = False,
        limit: "int | None" = None,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        """Yield children lazily. ``limit`` caps how many are produced — the
        underlying listing stays incremental, so a bounded ``ls`` over a huge
        prefix never materialises (or fetches) more than it needs."""
        children = self._ls(recursive=recursive, singleton_ttl=singleton_ttl)
        if limit is None:
            yield from children
            return
        if limit <= 0:
            return
        for i, child in enumerate(children):
            yield child
            if i + 1 >= limit:
                return

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> "Path":
        self._mkdir(parents=parents, exist_ok=exist_ok)
        # The path is now a directory. Refresh the stat cache to match —
        # a slot left at MISSING (from an ``exists()`` probe taken before
        # the create) would make a follow-up ``exists`` / ``is_dir``
        # report the directory as absent until the entry aged out. Seed
        # DIRECTORY directly so remote backends skip a re-probe round trip.
        self._persist_stat_cache(
            IOStats(
                kind=IOKind.DIRECTORY,
                size=0,
                mtime=time.time(),
                media_type=self.media_type,
            )
        )
        return self

    def _delete(
        self,
        predicate: "Any" = None,
        *,
        remove_path: bool = False,
        recursive: bool = True,
        files_only: bool = False,
        missing_ok: bool = True,
        wait: WaitingConfigArg = True,
        fresher_than: Optional[TimeLike] = None,
        older_than: Optional[TimeLike] = None,
        **kwargs: Any,
    ) -> int:
        """The single deletion primitive for a path — backends override
        *this* (and nothing else) to customise removal.

        Two modes, both centralised here:

        - **path removal** — ``remove_path=True`` (set by :meth:`remove` /
          :meth:`unlink` / :meth:`rm`) *or* a ``None`` *predicate* removes the
          backing object itself: the file, or the whole subtree when
          ``recursive``. ``files_only`` refuses a directory (``unlink``
          semantics). Honours the ``fresher_than`` / ``older_than`` window.
        - **row delete** — a non-``None`` *predicate* deletes matching rows,
          delegated to the byte-leaf rewrite via :class:`IO`.

        ``remove`` and ``unlink`` are thin wrappers over the path-removal
        mode so there is exactly one deletion code path through the IO layer.
        """
        # Path removal is the default for a no-predicate delete. Only a
        # *non-None* predicate is a row-level delete — and only that path
        # reads the leaf back as Arrow batches. A bare ``path.delete()``
        # (predicate=None) must NEVER fall through to the byte-leaf rewrite:
        # on a directory / a non-tabular file / a missing path that rewrite
        # raises ``NotImplementedError: IO has no tabular decoder`` instead
        # of just removing the object. So no predicate ⇒ remove the file or
        # folder, full stop.
        if not remove_path and predicate is not None:
            return super()._delete(
                predicate, missing_ok=missing_ok, wait=wait, **kwargs,
            )

        wait = WaitingConfig.from_(wait)
        # A no-predicate ``.delete()`` is idempotent path removal — an
        # already-absent path has nothing to delete, so it's a no-op rather
        # than an error. ``remove`` / ``unlink`` arrive with ``remove_path``
        # set and still honour an explicit ``missing_ok=False`` (raise on a
        # ghost path); the implicit Tabular ``.delete()`` contract does not.
        if not remove_path:
            missing_ok = True
        # A delete must look at the object's *current* state, not a snapshot a
        # prior read left in the stat cache: an entry cached MISSING (so the
        # delete silently no-ops) or cached the wrong kind (file vs directory)
        # is exactly how a deletion appears not to work. Drop the cached stat
        # so the probe below hits the backend.
        self._stat_cached = None
        self._stat_cached_at = 0.0
        stat = self._stat()
        kind = stat.kind

        if kind == IOKind.MISSING:
            if not missing_ok:
                raise FileNotFoundError(f"{self.full_path()!r} does not exist")
            return 0

        if files_only and kind == IOKind.DIRECTORY:
            raise IsADirectoryError(
                f"{self.full_path()!r} is a directory — use ``remove()`` "
                "for the recursive-delete path; ``unlink`` is files-only."
            )

        if fresher_than or older_than:
            fresher_than = IOStats.normalize_timestamp(fresher_than, default=0.0)
            older_than = IOStats.normalize_timestamp(
                fresher_than, default=dt.datetime.now(tz=dt.timezone.utc).timestamp()
            )

            if kind == IOKind.FILE:
                if stat.is_between_timestamp(start=fresher_than, end=older_than):
                    self._remove_file(missing_ok=missing_ok, wait=wait)
            elif kind == IOKind.DIRECTORY:
                for child in self.ls(recursive=False):
                    child._delete(
                        remove_path=True,
                        recursive=recursive,
                        missing_ok=missing_ok,
                        wait=wait,
                        fresher_than=fresher_than,
                        older_than=older_than,
                    )
                    if child.is_empty():
                        child._delete(remove_path=True, missing_ok=missing_ok, wait=False)
        else:
            # Be aggressive about the file-vs-directory verdict: a single
            # stat probe (especially the off-cluster Files-API heuristic) can
            # misclassify a leaf, so if the kind-specific remove rejects the
            # object as the wrong type, retry as the other type rather than
            # leaving it behind.
            if kind == IOKind.FILE:
                try:
                    self._remove_file(missing_ok=missing_ok, wait=wait)
                except IsADirectoryError:
                    self._remove_dir(recursive=recursive, missing_ok=missing_ok, wait=wait)
            elif kind == IOKind.DIRECTORY:
                try:
                    self._remove_dir(recursive=recursive, missing_ok=missing_ok, wait=wait)
                except NotADirectoryError:
                    self._remove_file(missing_ok=missing_ok, wait=wait)

        # The object is gone: drop the cached FILE/DIRECTORY snapshot the probe
        # above left behind so a follow-up ``exists`` / ``is_file`` re-probes
        # the backend instead of reporting the deleted path as still present.
        self.invalidate_singleton()
        return 0

    def unlink(self, missing_ok: bool = True, wait: WaitingConfigArg = True) -> None:
        """Remove the leaf — pathlib-compatible: refuses directories.

        Mirrors :meth:`pathlib.Path.unlink`: succeeds for files, raises
        :class:`IsADirectoryError` for directories so callers don't
        accidentally recursive-delete via ``unlink``. Use :meth:`remove`
        for the directory case. Thin wrapper over :meth:`_delete`'s
        path-removal mode.
        """
        self._delete(
            remove_path=True, files_only=True, recursive=False,
            missing_ok=missing_ok, wait=wait,
        )

    def remove(
        self,
        recursive: bool = True,
        missing_ok: bool = True,
        wait: WaitingConfigArg = True,
        fresher_than: Optional[TimeLike] = None,
        older_than: Optional[TimeLike] = None,
    ) -> "Path":
        """Remove this path — the file, or the whole subtree when *recursive*.

        Thin wrapper over :meth:`_delete`'s path-removal mode (the single
        deletion primitive). ``fresher_than`` / ``older_than`` scope the
        removal to children inside that mtime window.
        """
        self._delete(
            remove_path=True,
            recursive=recursive,
            missing_ok=missing_ok,
            wait=wait,
            fresher_than=fresher_than,
            older_than=older_than,
        )
        return self

    rm = remove

    def wait_until_gone(self, wait: WaitingConfigArg = True) -> "Path":
        """Block until :meth:`exists` reports ``False`` or *wait* expires.

        Polls the backend with a fresh probe each iteration — the
        stat cache is invalidated between checks so a TTL'd hit
        can't mask a deletion that landed after the cache was
        filled. Useful when a fire-and-forget unlink (e.g.
        ``WarehouseStatementBatch.clear_temporary_resources``) means
        the caller can't observe completion through the original
        operation's return value.

        Raises :class:`TimeoutError` when *wait*'s deadline elapses
        and the path is still present.
        """
        cfg = WaitingConfig.from_(wait)
        start = time.monotonic()
        iteration = 0
        while True:
            self._stat_cached = None
            self._stat_cached_at = 0.0
            if not self.exists():
                return self
            if cfg.timeout > 0 and (time.monotonic() - start) >= cfg.timeout:
                raise TimeoutError(
                    f"{self.full_path()!r} still exists after "
                    f"{cfg.timeout:.1f}s — wait_until_gone deadline expired"
                )
            cfg.sleep(iteration)
            iteration += 1

    def is_empty(self):
        stat = self._stat()
        kind = stat.kind

        if kind == IOKind.MISSING:
            return True
        elif kind == IOKind.FILE:
            return stat.size == 0
        elif kind == IOKind.DIRECTORY:
            return not any(self.ls(recursive=True))
        else:
            raise ValueError(f"is_empty: unknown kind {kind!r}")

    def touch(self) -> "Path":
        """Create the path as an empty file if it doesn't exist.

        ``write_bytes(b"")`` short-circuits in the holder fast path
        (zero bytes, no flush), which would leave a missing file behind
        — open + close around the empty write so the holder actually
        materialises the entry on the backing store.
        """
        if not self.exists():
            with self:
                self.write_bytes(b"")
        return self

    # ==================================================================
    # Byte transfer — optimized hooks for Path-to-Path and Path-to-Holder
    # ==================================================================

    def _transfer_to(self, target: "Holder | IO") -> None:
        """Path-side override of :meth:`Holder._transfer_to`.

        Two filesystem-aware fast paths skip materialising the full
        payload into Python bytes:

        1. **Local→local** — both ends back a local file: hand off to
           :func:`shutil.copyfile`, which uses ``sendfile`` /
           ``copy_file_range`` / ``fclonefileat`` under the hood.
        2. **Local→remote holder** — self is a local file, target is
           a non-IO :class:`Holder` (any remote :class:`Path`, a
           :class:`Memory`, …): stream via
           :meth:`Holder.write_local_path` so a multi-GB file
           uploads in :data:`_COPY_CHUNK`-sized chunks instead of
           one giant in-memory ``read_bytes()``.

        Everything else falls back to the generic
        :meth:`Holder._transfer_to` (bytes copy).
        """
        from yggdrasil.io.base import IO  # local to break the import cycle

        if isinstance(target, IO):
            return super()._transfer_to(target)
        if isinstance(target, Path) and self.is_local_path and target.is_local_path:
            import shutil

            shutil.copyfile(os.fspath(self), os.fspath(target))
            return
        if self.is_local_path:
            target.write_local_path(os.fspath(self))
            return
        return super()._transfer_to(target)

    # ==================================================================
    # Module upload / import — share local Python packages over the wire
    # ==================================================================

    def upload_module(
        self,
        module: Any,
        *,
        name: str | None = None,
        overwrite: bool = True,
    ) -> "Path":
        """Zip a local module / package and write it under this path.

        *module* is anything :func:`resolve_module_root` accepts —
        an importable module name (``"yggdrasil.io"``), a
        :class:`os.PathLike` pointing at a package directory or an
        existing ``.zip`` / ``.whl`` archive, or a callable
        carrying a ``__module__`` attribute. The module is packed
        into a deflated zip whose top-level entry is the package
        directory itself, so the archive can be added to
        ``sys.path`` directly (or fed to
        :meth:`SparkSession.addArtifacts` with ``pyfile=True``).

        Destination shape on *self*:

        - *self* names a file with a ``.zip`` / ``.whl`` suffix —
          archive bytes land at that exact path.
        - *self* is anything else — archive lands at
          ``self / <name or "<module>.zip">``.

        Returns the concrete :class:`Path` that now holds the
        archive. ``overwrite=False`` raises
        :class:`FileExistsError` when the destination already
        exists.
        """
        from yggdrasil.path._module_pack import (
            build_module_archive,
            resolve_module_root,
        )

        local_root = resolve_module_root(module)
        suffix = self.suffix.lower()
        archive_default = (
            name
            if name is not None
            else (
                local_root.name
                if local_root.is_file() and suffix in (".zip", ".whl")
                else f"{local_root.name}.zip"
            )
        )

        target: "Path" = self if suffix in (".zip", ".whl") else self / archive_default

        if not overwrite and target.exists():
            raise FileExistsError(
                f"upload_module: destination {target.full_path()!r} "
                f"already exists. Pass overwrite=True to replace it."
            )

        archive_path = build_module_archive(local_root, dest=None)
        try:
            target.write_bytes(archive_path.read_bytes())
        finally:
            # ``build_module_archive`` writes into the staging dir
            # when dest is None; remove the local copy after the
            # remote write succeeds so we don't leak the file.
            if local_root != archive_path:
                try:
                    archive_path.unlink()
                except OSError:
                    pass
        return target

    def import_module(
        self,
        module_name: str | None = None,
        *,
        install: bool = True,
        cache_dir: "Any" = None,
    ) -> Any:
        """Download a module archive at this path and import it.

        Inverse of :meth:`upload_module`: fetch the archive bytes
        at *self*, drop them on local disk, prepend the archive (or
        its extracted parent) to :data:`sys.path`, and return the
        live module via :func:`importlib.import_module`.

        *module_name* defaults to the archive's stem (filename
        minus suffix). *cache_dir* picks where the archive lands
        locally (default: a fresh
        :meth:`LocalPath.staging_path`-style directory).

        ``install=True`` (the default) preserves the archive on
        disk so subsequent imports in the same process hit the
        cache. ``install=False`` makes the cache-dir lifetime the
        caller's problem.
        """

        suffix = self.suffix.lower()
        if suffix not in (".zip", ".whl"):
            raise ValueError(
                f"import_module: {self.full_path()!r} does not look "
                f"like a Python archive (expected .zip or .whl, got "
                f"{suffix or '<none>'!r})."
            )

        stem = self.stem
        resolved_name = module_name or stem

        if cache_dir is None:
            target_dir = tempfile.mkdtemp(prefix="ygg-module-")
        else:
            target_dir = os.fspath(cache_dir)
            os.makedirs(target_dir, exist_ok=True)

        local_archive = os.path.join(target_dir, self.name)
        with open(local_archive, "wb") as fh:
            fh.write(bytes(self.read_mv(-1, 0)))

        if local_archive not in sys.path:
            sys.path.insert(0, local_archive)

        importlib.invalidate_caches()
        try:
            return importlib.import_module(resolved_name)
        except ModuleNotFoundError:
            if not install:
                raise
            # ``.whl`` archives don't sit on ``sys.path``
            # directly — installing them is the supported path.
            from yggdrasil.environ import PyEnv

            PyEnv.current().install(local_archive, raise_error=True)
            importlib.invalidate_caches()
            return importlib.import_module(resolved_name)

    def as_media(self, media_type: "Any" = None) -> "Any":
        """Wrap this path in the format leaf for its media type.

        .. deprecated::
            Use :meth:`open` with a ``media_type`` instead —
            ``path.open(media_type=...)`` already dispatches to the
            right format leaf and gives a properly acquired cursor with
            lifecycle handling. ``as_media`` returns an un-acquired leaf
            and is kept only for callers that haven't migrated.

        Resolution: explicit ``media_type`` first, else the holder's
        :class:`MediaType` (path extension, magic-byte sniff, or
        content-type header). The resolved class is looked up in the
        :class:`Holder` format registry and instantiated bound to this
        path.

        Raises :class:`KeyError` when the path's media type isn't
        registered as a tabular format.
        """
        import warnings

        warnings.warn(
            "Path.as_media() is deprecated; use open(media_type=...) which "
            "dispatches to the format leaf and manages the cursor lifecycle.",
            DeprecationWarning,
            stacklevel=2,
        )
        return IO.for_holder(self, media_type=media_type)

    # ==================================================================
    # open(mode) — returns an IO bound to self
    # ==================================================================

    def open(
        self,
        mode: "Mode | str | None" = None,
        **kwargs: Any,
    ) -> "IO":
        """Acquire the path and return an :class:`IO` cursor bound to it.

        ``mode`` accepts a :class:`Mode` member, an alias string, or
        a stdlib ``open()`` mode string. ``None`` falls through to
        :meth:`Holder.open` which uses ``"rb+"``. Other keyword
        arguments (``owns_holder``, ``media_type``, ``auto_open``,
        …) ride through to :meth:`Holder.open`.
        """
        if mode is None:
            return IO.open(self, **kwargs)
        return IO.open(self, mode=Mode.from_(mode).os_mode, **kwargs)

    # ==================================================================
    # Pure-path API — all delegated to URL
    # ==================================================================

    @property
    def parts(self) -> List[str]:
        return self.url.parts or []

    @property
    def name(self) -> str:
        return self.url.name or ""

    @property
    def stem(self) -> str:
        return self.url.stem or ""

    def start_end_timestamp(self) -> Tuple[float, float]:
        m = TS_PATTERN.search(self.name)
        if m is None:
            return None
        return float(m.group(1)), float(m.group(2))

    @property
    def suffix(self) -> str:
        exts = self.url.extensions
        return "." + exts[-1] if exts else ""

    @property
    def suffixes(self) -> List[str]:
        return ["." + e for e in self.url.extensions]

    # ``parent`` / ``parents`` / ``joinpath`` / ``__truediv__`` all
    # live on :class:`Holder` — Path inherits them unchanged. The
    # URL-parent branch in :attr:`Holder.parent` matches Path's
    # filesystem-navigation semantics; :class:`Memory` overrides
    # :meth:`_url_parent` to opt out (its URLs are synthetic).

    @property
    def is_absolute(self) -> bool:
        return self.url.is_absolute

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

    def __rtruediv__(self, other: Any) -> "Path":
        return Path.from_(other) / self

    # ==================================================================
    # Fast whole-file write
    # ==================================================================

    # ==================================================================
    # Dunder
    # ==================================================================

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.url!r})"

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


# ---------------------------------------------------------------------------
# Cast registry — Any → Path routes through :meth:`Path.from_`
# ---------------------------------------------------------------------------

from yggdrasil.data.cast.registry import register_converter  # noqa: E402


@register_converter(Any, Path)
def any_to_path(value: Any, opts: Any = None) -> Path:
    """Coerce *value* to a :class:`Path` via :meth:`Path.from_`.

    Accepts ``str`` / :class:`pathlib.PurePath` / :class:`URL` /
    existing :class:`Path` instances (identity on same-class).
    Dispatches by URL scheme to the matching :class:`Path` subclass
    (``LocalPath`` for the no-scheme case).
    """
    return Path.from_(value)
