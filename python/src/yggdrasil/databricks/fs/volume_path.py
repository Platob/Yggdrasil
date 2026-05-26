""":class:`VolumePath` — Databricks Unity Catalog Volume via Files API.

Volumes carry a Unity Catalog hierarchy (catalog → schema → volume →
path) and are the SQL engine's preferred staging surface. Reads /
writes go through ``workspace.files.*``: ``download``, ``upload``,
``list_directory_contents``, ``create_directory``, ``delete``.

The :class:`Holder` byte primitives map onto these:

- :meth:`_read_mv` — ``files.download`` returns a streaming body;
  we slice into the requested range. (Files API doesn't expose
  range reads.)
- :meth:`_write_mv` — read-modify-rewrite via ``files.upload``.
- :meth:`truncate` — ``files.upload`` of the head N bytes.
- :meth:`_clear` — ``files.delete``.

The catalog-management surface (grants, volume metadata, staging
factories) lives in dedicated modules; this class covers the
filesystem contract.

Native storage fast path
------------------------

For S3-backed volumes the SDK's Files API is just a translation
layer over the underlying object store. :meth:`storage_location`,
:meth:`temporary_credentials`, :meth:`aws`, and :meth:`s3_path`
expose the volume's UC-vended S3 storage directly so callers can
bypass ``workspace.files`` entirely — the resulting :class:`S3Path`
carries a botocore :class:`RefreshableCredentials`-backed session
that re-invokes :meth:`temporary_credentials` on every near-expiry
refresh cycle. One fewer hop per read / write, no Unity Catalog
quota burn for the bulk transfer.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional

from databricks.sdk.errors import PermissionDenied

from yggdrasil.concurrent import Job
from yggdrasil.data.cast import any_to_datetime, parse_http_date
from yggdrasil.enums import Mode, ModeLike, Scheme
from yggdrasil.enums.media_type import MediaType
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.url import URL
from ..path import DatabricksPath

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.schema.schema import UCSchema
    from yggdrasil.databricks.volume.volume import Volume

from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials

# ``VolumeCredentialsRefresher`` is kept as an alias for the public name
# used in older releases / external callers and existing test fixtures.
VolumeCredentialsRefresher = AWSDatabricksVolumeCredentials

__all__ = ["VolumePath", "VolumeCredentialsRefresher"]


logger = logging.getLogger(__name__)
_VOLUME_DOTTED_NAME_RE = re.compile(
    r"Volume\s+'(?P<catalog>[\w-]+)\.(?P<schema>[\w-]+)\.(?P<volume>[\w-]+)'"
)


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/<cat>/<sch>/<vol>/...`` via the Files API.

    Per-volume metadata (``VolumeInfo``, storage location, temporary
    credentials, AWS client) lives on the :class:`Volume` resource
    accessible via :attr:`volume`. Every :class:`VolumePath` pointing at
    the same UC volume collapses to the same :class:`Volume` singleton,
    so the SDK round trip and the auto-refreshing :class:`AWSClient`
    are shared process-wide.
    """

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_VOLUME
    NAMESPACE_PREFIX: ClassVar[str] = "/Volumes/"

    # ``_SERVICE_CLASS`` is bound below the class body to avoid the
    # ``volume.volumes`` → ``volume.volume`` → ``fs.volume_path``
    # import cycle.

    def __init__(
        self,
        data: Any = None,
        *,
        url: "URL | None" = None,
        volume: "Volume | None" = None,
        service: Any = None,
        **kwargs: Any,
    ) -> None:
        # Idempotent under ``Singleton`` caching — see ``DatabricksPath.__init__``.
        if getattr(self, "_initialized", False):
            return

        self._volume: Optional["Volume"] = volume

        # A bound :class:`Volume` carries the service — prefer the
        # Volume's service so the resource stays navigable
        # (``volume_path.volume`` short-circuits to the cached
        # instance without re-resolving).
        if volume is not None and service is None:
            service = volume.service

        super().__init__(
            data=data,
            service=service,
            url=url,
            **kwargs,
        )

    @property
    def explore_url(self) -> URL:
        return self.volume.explore_url.add_param(
            key="volumePath",
            value=self.full_path(),
        )

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = (self.url.path or "").lstrip("/")
        return "/Volumes/" + p if p else "/Volumes"

    @property
    def api_path(self) -> str:
        return self.full_path()

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        files = self.client.workspace_client().files
        api_path = self.api_path
        # Heuristic: a leaf with a ``.`` is almost always a file
        # (``foo.parquet`` / ``part-….json``); a bare leaf is almost
        # always a directory (``/Volumes/cat/sch/vol``,
        # ``/Volumes/cat/sch/vol/tmp``). Probe that side first so the
        # common case is one Files-API round trip instead of two.
        # The 0.6.21 ``get_volume_status`` used the same heuristic;
        # the rewrite dropped it and silently doubled stat latency
        # for every directory probe.
        file_first = "." in (self.url.path or "").rsplit("/", 1)[-1]
        if file_first:
            first_probe, first_kind = files.get_metadata, IOKind.FILE
            second_probe, second_kind = files.get_directory_metadata, IOKind.DIRECTORY
        else:
            first_probe, first_kind = files.get_directory_metadata, IOKind.DIRECTORY
            second_probe, second_kind = files.get_metadata, IOKind.FILE
        try:
            info = self._call(first_probe, api_path)
        except Exception:
            info = None
        if info is not None:
            if first_kind is IOKind.FILE:
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(getattr(info, "content_length", 0) or 0),
                    mtime=_mtime(info),
                    media_type=_media_type_from_response(info),
                )
            return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=_mtime(info))
        try:
            info = self._call(second_probe, api_path)
        except Exception:
            info = None
        if info is not None:
            if second_kind is IOKind.FILE:
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(getattr(info, "content_length", 0) or 0),
                    mtime=_mtime(info),
                    media_type=_media_type_from_response(info),
                )
            return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=_mtime(info))
        # Implicit-directory fallback. ``files.upload`` to a brand-new
        # ``/Volumes/<...>/parent/file.bin`` silently materialises the
        # file without creating an explicit ``parent`` entry, so
        # ``get_directory_metadata(parent)`` returns NotFound even
        # though listing the path enumerates ``file.bin``. Without
        # this probe, ``remove(parent, recursive=True, missing_ok=False)``
        # raises ``FileNotFoundError`` against a parent that the caller
        # just wrote into. One extra round trip pays for the negative
        # case only — both metadata probes already missed.
        try:
            entries = self._call(files.list_directory_contents, api_path)
            first = next(iter(entries), None)
        except Exception:
            first = None
        if first is not None:
            return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=0.0)
        return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)

    @property
    def size(self) -> int:
        return int(self._stat().size)

    # ==================================================================
    # Unity Catalog volume metadata — storage location + temp creds
    # ==================================================================

    def _split_volume(self) -> Optional[tuple[str, str, str]]:
        """``/cat/sch/vol/...`` → ``("cat", "sch", "vol")`` or ``None``.

        Returns ``None`` when the URL path has fewer than three
        segments (i.e. it doesn't address a volume at all — typically
        a stat probe at ``/Volumes`` itself or a malformed path).
        """
        parts = (self.url.path or "/").lstrip("/").split("/")
        parts = [p for p in parts if p]
        if len(parts) < 3:
            return None
        return parts[0], parts[1], parts[2]

    @property
    def catalog_name(self) -> Optional[str]:
        """The Unity Catalog catalog this volume lives under, or ``None``."""
        triple = self._split_volume()
        return triple[0] if triple else None

    @property
    def schema_name(self) -> Optional[str]:
        """The Unity Catalog schema this volume lives under, or ``None``."""
        triple = self._split_volume()
        return triple[1] if triple else None

    @property
    def volume_name(self) -> Optional[str]:
        """The Unity Catalog volume name, or ``None`` when the URL path
        doesn't address a volume."""
        triple = self._split_volume()
        return triple[2] if triple else None

    @property
    def volume(self) -> "Volume":
        """Return the :class:`Volume` resource backing this path.

        Lazily resolved on first access and cached on the instance.
        Because :class:`Volume` instances are singletons per
        ``(host, catalog, schema, name)``, every :class:`VolumePath`
        on the same UC volume shares the same live metadata cache,
        the same :class:`VolumeInfo` snapshot, and the same
        credentials refresher.

        Raises :class:`ValueError` when the URL path doesn't address
        a volume (no ``/cat/sch/vol`` prefix).
        """
        if self._volume is not None:
            return self._volume

        triple = self._split_volume()
        if triple is None:
            raise ValueError(
                f"{type(self).__name__}.volume requires a path under "
                f"/Volumes/<cat>/<sch>/<vol>/... — got {self.full_path()!r}."
            )
        catalog, schema, volume_name = triple
        from yggdrasil.databricks.volume.volume import Volume
        from yggdrasil.databricks.volume.volumes import Volumes

        # Bind through a fresh ``Volumes`` over this path's
        # :attr:`client` so the Volume sees the same workspace
        # context — using ``self.client.volumes`` would resolve via
        # whatever attribute the client exposes (a real
        # :class:`Volumes`, or a test-side mock), which breaks
        # workspace-client identity in mocked test setups.
        self._volume = Volume(
            service=Volumes(client=self.client),
            catalog_name=catalog,
            schema_name=schema,
            volume_name=volume_name,
        )
        return self._volume

    @property
    def catalog(self) -> "UCCatalog":
        """Return a :class:`Catalog` instance for this volume's parent catalog.

        Delegates to :attr:`volume`.catalog so the underlying
        :class:`Catalog` instance is reused across every path on this
        UC volume.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix).
        """
        return self.volume.catalog

    @property
    def schema(self) -> "UCSchema":
        """Return a :class:`Schema` instance for this volume's parent schema.

        Delegates to :attr:`volume`.schema so the underlying
        :class:`Schema` instance is reused across every path on this
        UC volume.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix).
        """
        return self.volume.schema

    def volume_info(self, refresh: bool = False) -> Any:
        """Return the SDK's :class:`VolumeInfo` for this volume.

        Delegates to :meth:`Volume.read_info`. The result is shared
        across every :class:`VolumePath` on this UC volume (via the
        :class:`Volume` singleton) and refreshed when the cached
        snapshot is past the 5-minute TTL.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume.
        """
        return self.volume.read_info(refresh=refresh)

    def storage_location(self, refresh: bool = False) -> str:
        """Volume's backing storage URL string. Delegates to
        :meth:`Volume.storage_location`."""
        return self.volume.storage_location(refresh=refresh)

    def storage_path(
        self,
        *,
        mode: ModeLike = Mode.AUTO,
        region: Optional[str] = None,
        refresh: bool = False,
    ) -> Any:
        """Return the volume's root storage :class:`Path`. Delegates to
        :meth:`Volume.storage_path` — see there for the semantics."""
        return self.volume.storage_path(mode=mode, region=region, refresh=refresh)

    def temporary_credentials(
        self,
        *,
        mode: ModeLike = Mode.AUTO,
    ) -> Any:
        """Vend temporary cloud credentials for this volume. Delegates to
        :meth:`Volume.temporary_credentials`."""
        return self.volume.temporary_credentials(mode=mode)

    def credentials_refresher(self) -> "AWSDatabricksVolumeCredentials":
        """Return the process-wide singleton credentials provider for
        this volume. Delegates to :meth:`Volume.credentials_refresher`."""
        return self.volume.credentials_refresher()

    def aws(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from :meth:`temporary_credentials`. Delegates to :meth:`Volume.aws`."""
        return self.volume.aws(mode=mode, region=region)

    def arrow_filesystem(
        self,
        *,
        operation: Any = None,
        region: Optional[str] = None,
    ) -> Any:
        """Build a :class:`pyarrow.fs.S3FileSystem` for this volume.

        Delegates to :meth:`Volume.arrow_filesystem`. ``operation`` is
        passed through as the credential mode.
        """
        return self.volume.arrow_filesystem(mode=operation, region=region)

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["VolumePath"]:
        files = self.client.workspace_client().files
        try:
            entries = self._call(files.list_directory_contents, self.api_path)
        except PermissionDenied as e:
            logger.warning(
                "Permission denied listing volume directory %r: %r",
                self,
                e,
            )
            return
        except Exception:
            return
        if logger.isEnabledFor(logging.DEBUG):
            entries = list(entries)
            logger.debug(
                "Listing volume directory %r -> %d entries (recursive=%s)",
                self,
                len(entries),
                recursive,
            )
        for info in entries:
            child_path = getattr(info, "path", None)
            if not child_path:
                continue
            # The Files API returns canonical ``/Volumes/<cat>/<sch>/<vol>/...``
            # POSIX paths; route through the constructor so the same POSIX
            # coercion that built ``self`` (``/Volumes/...`` →
            # ``dbfs+volume:///...``) builds the child. Earlier code did
            # ``child_path.lstrip('/Volumes')`` which strips the *character
            # set* ``/Volumes`` and then yielded ``dbfs+volume://<cat>/...``,
            # which URL-parses ``<cat>`` as a host and drops it.
            # ``singleton_ttl`` defaults to ``False`` so the bounded
            # ``DatabricksPath._INSTANCES`` cache doesn't fill with
            # thousands of short-lived listing children. Callers that
            # explicitly want cached children (``singleton_ttl=None``
            # / class default) pass it through ``iterdir`` / ``ls``.
            child = type(self)(
                child_path,
                service=self.service,
                singleton_ttl=singleton_ttl,
            )
            # The listing entry already carries ``is_directory`` /
            # ``file_size`` / ``last_modified`` — seed the child's stat
            # cache so the caller's ``is_file()`` / ``size`` /
            # ``exists()`` per child collapses to a local hit. Without
            # this, iterating an N-entry directory floods the Files
            # API with N extra ``get_metadata`` round trips. (0.6.21
            # already did this; the rewrite dropped it.)
            is_directory = bool(getattr(info, "is_directory", False))
            child._persist_stat_cache(
                IOStats(
                    kind=IOKind.DIRECTORY if is_directory else IOKind.FILE,
                    size=(
                        0
                        if is_directory
                        else int(
                            getattr(info, "file_size", 0) or 0,
                        )
                    ),
                    mtime=_mtime(info),
                )
            )
            yield child
            if recursive and is_directory:
                yield from child._ls(recursive=True, singleton_ttl=singleton_ttl)

    # ``_call_ensuring_parents`` is inherited from :class:`DatabricksPath`
    # — the volume-specific recovery lives on :meth:`_ensure_parents`
    # below, which the base class invokes on NotFound.

    def _ensure_parents(self, exc: "BaseException | None" = None) -> bool:
        """Recovery hook for :meth:`_call_ensuring_parents`.

        Cheap-path first: if *self* lives below the volume root,
        ``files.create_directory`` on the parent fixes the common
        case (only a sub-directory was missing). If that also
        NotFounds — or if *exc* already named the volume as
        missing — fall back to :meth:`_ensure_volume` and retry
        the parent ``mkdir``. Blind creates swallow ``AlreadyExists``
        so the idempotent path costs at most three SDK calls.
        """
        triple = self._split_volume()
        if triple is None:
            return False

        parent = self.parent
        pparts = [p for p in (parent.url.path or "/").lstrip("/").split("/") if p]
        has_subdir = len(pparts) > 3  # parent strictly below ``/cat/sch/vol``
        volume_missing = exc is not None and _looks_like_volume_not_found(exc)

        if has_subdir and not volume_missing:
            try:
                self._call(
                    self.client.workspace_client().files.create_directory,
                    parent.api_path,
                )
                return True
            except Exception as inner:
                if _looks_like_already_exists(inner):
                    return True
                if not _looks_like_not_found(inner):
                    raise
                # Parent missing because volume itself is missing —
                # fall through to volume creation.

        self._ensure_volume()

        if has_subdir:
            try:
                self._call(
                    self.client.workspace_client().files.create_directory,
                    parent.api_path,
                )
            except Exception as inner:
                if not _looks_like_already_exists(inner):
                    raise
        return True

    def _ensure_volume(self) -> bool:
        """Top-down create of the missing pieces of catalog / schema / volume.

        Routes the volume create through :meth:`Volume.create` so the
        managed-volume-type default (``VolumeType.MANAGED`` enum, not
        a bare ``"MANAGED"`` string the SDK rejects) lives in one
        place. ``AlreadyExists`` is swallowed by ``missing_ok=True``;
        if the volume create NotFounds because schema (or catalog) is
        missing, falls through to :func:`_ensure_parents_for` to
        materialise the parents before a single retry.
        """
        if self._split_volume() is None:
            return False
        volume = self.volume

        try:
            volume.create(missing_ok=True)
            return True
        except Exception as exc:
            if _looks_like_already_exists(exc):
                return True
            if not _looks_like_not_found(exc):
                raise

        from yggdrasil.databricks.volume.volumes import _ensure_parents_for

        _ensure_parents_for(
            self.client.workspace_client(),
            catalog_name=volume.catalog_name,
            schema_name=volume.schema_name,
        )
        try:
            volume.create(missing_ok=True)
        except Exception as exc:
            if not _looks_like_already_exists(exc):
                raise
        return True

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        logger.debug("Creating volume directory %r", self)
        try:
            self._call_ensuring_parents(
                self.client.workspace_client().files.create_directory,
                self.api_path,
            )
            logger.info(
                "Created volume directory %r (parents=%s)",
                self,
                parents,
            )
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        del wait
        logger.info("Deleting volume file %r", self)
        try:
            self._call(self.client.workspace_client().files.delete, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self.invalidate_singleton()

    def _remove_dir(
        self,
        recursive: bool,
        missing_ok: bool,
        wait: WaitingConfig,
        pool: "int | ThreadPoolExecutor | None" = None,
    ) -> None:
        logger.info(
            "Deleting volume directory %r (recursive=%s)",
            self,
            recursive,
        )
        # ``files.delete_directory`` is non-recursive — its docstring is
        # explicit: "To delete a non-empty directory, first delete all
        # of its contents." Hitting it on a non-empty directory returns
        # ``BadRequest: The directory is not empty.`` So when the
        # caller asks for ``recursive=True`` we list + delete contents
        # ourselves, then drop the now-empty directory.
        #
        # File deletions for a given directory are fanned out to a
        # ``ThreadPoolExecutor`` (default 4 workers); the executor is
        # forwarded through recursive ``_remove_dir`` calls so the
        # whole subtree shares one pool. Subdirectory recursion stays
        # synchronous on the caller thread — submitting recursive
        # calls back onto the same pool would deadlock once every
        # worker is blocked waiting on its own children.
        if recursive:
            owns_pool = not isinstance(pool, ThreadPoolExecutor)
            if owns_pool:
                executor = ThreadPoolExecutor(
                    max_workers=pool if isinstance(pool, int) else 4,
                    thread_name_prefix="volume-rmdir",
                )
            else:
                executor = pool
            try:
                file_futures = []
                for child in self._ls(recursive=False):
                    cached = child._stat_cached
                    is_dir = cached is not None and cached.kind is IOKind.DIRECTORY
                    if is_dir:
                        child._remove_dir(
                            recursive=True,
                            missing_ok=missing_ok,
                            wait=wait,
                            pool=executor,
                        )
                    else:
                        file_futures.append(
                            executor.submit(
                                child._remove_file,
                                missing_ok=missing_ok,
                                wait=wait,
                            )
                        )
                for fut in file_futures:
                    fut.result()
            finally:
                if owns_pool:
                    executor.shutdown(wait=True)

        logger.info(
            "Deleted volume directory %r (recursive=%s)",
            self,
            recursive,
        )

        if wait:
            try:
                self._call(
                    self.client.workspace_client().files.delete_directory, self.api_path
                )
            except Exception:
                if not missing_ok:
                    raise
        else:
            Job.make(
                self.client.workspace_client().files.delete_directory,
                self.api_path,
            )
        self.invalidate_singleton()

    # ==================================================================
    # Holder I/O
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        try:
            response = self._call(
                self.client.workspace_client().files.download, self.api_path
            )
        except Exception as exc:
            if _looks_like_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            raise

        body = getattr(response, "contents", None) or response
        try:
            data = body.read()
        except AttributeError:
            data = bytes(body)
        logger.debug(
            "Downloaded volume file %r -> %d bytes (slice pos=%d n=%s)",
            self,
            len(data),
            pos,
            "EOF" if n < 0 else n,
        )

        media_type = _media_type_from_response(response)
        try:
            mtime = (
                parse_http_date(response.last_modified)
                if response.last_modified
                else None
            )
        except Exception:
            mtime = None
        mtime = mtime.timestamp() if mtime else time.time()
        if not self._stat_cached:
            self._persist_stat_cache(
                stats=IOStats(
                    size=len(data),
                    kind=IOKind.FILE,
                    mtime=mtime,
                    media_type=media_type,
                )
            )
        else:
            self._stat_cached.size = len(data)
            self._stat_cached.mtime = mtime
            if media_type is not None and self._stat_cached.media_type is None:
                self._stat_cached.media_type = media_type
            # Re-stamp the TTL — this download IS the freshest size we
            # could observe; the entry should outlive the original
            # probe's window from this point on.
            self._persist_stat_cache(self._stat_cached)

        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

    def _write_stream(
        self,
        src: Any,
        *,
        offset: int,
        size: int = -1,
        **kwargs: Any,
    ) -> int:
        """Override the base chunked stream — Volumes wants one PUT.

        The Files API does whole-object PUTs only, so a chunked
        :meth:`Holder._write_stream` would issue one RMW per
        chunk. Hand the live :class:`IO[bytes]` to :meth:`_upload`
        which does seek-on-retry around a single ``files.upload``
        call. ``size>=0`` (capped read) or non-zero ``offset``
        fall back to the chunked base path because the API can't
        splice at a range. ``batch_size`` only matters for that
        fallback — the atomic upload doesn't chunk.
        """
        if offset != 0 or size >= 0:
            return super()._write_stream(src, offset=offset, size=size, **kwargs)
        return self._upload(src)

    def _upload(self, content: Any) -> int:
        """Upload *content* through ``files.upload`` with retry semantics.

        Accepts either a bytes-like payload or a seekable binary
        stream. ``FilesExt.upload`` requires a ``BinaryIO`` — it
        probes ``contents.seekable()`` — so bytes-like payloads are
        wrapped in a fresh :class:`io.BytesIO` per attempt, and
        seekable streams are rewound to origin on every retry so
        transient-error / parent-recovery re-tries PUT the full
        body, not an empty tail.

        Returns the byte count when known (bytes-like input) or
        ``-1`` when the input is a stream of unknown length.
        """
        size = len(content) if hasattr(content, "__len__") else -1
        logger.debug(
            "Uploading volume file %r (%s bytes)",
            self,
            size if size >= 0 else "?",
        )
        upload = self.client.workspace_client().files.upload
        api_path = self.api_path

        if hasattr(content, "seek"):
            stream = content
            try:
                pos = content.tell()
                if size == -1:
                    content.seek(0, io.SEEK_END)
                    size = content.tell()
                    content.seek(pos, io.SEEK_SET)
            except Exception:
                pos = 0

            def _do_upload() -> None:
                stream.seek(pos)
                upload(file_path=api_path, contents=stream, overwrite=True)

        else:
            # ``FilesExt.upload`` calls ``contents.seekable()`` — wrap
            # raw bytes in a fresh ``IO`` each attempt so retries
            # always PUT the full body from offset zero.
            payload = bytes(content)

            def _do_upload() -> None:
                upload(
                    file_path=api_path,
                    contents=io.BytesIO(payload),
                    overwrite=True,
                )

        self._call_ensuring_parents(_do_upload)
        if size >= 0:
            self._persist_stat_cache(
                IOStats(
                    size=size,
                    kind=IOKind.FILE,
                    mtime=time.time(),
                    media_type=self.media_type,
                )
            )
            logger.info("Uploaded volume file %r (size=%d)", self, size)
        else:
            logger.info("Uploaded volume file %r (size=stream)", self)
        return size

    def _clear(self) -> None:
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _media_type_from_response(info) -> "MediaType | None":
    """Resolve a :class:`MediaType` from a Files API response.

    The SDK surfaces the object's MIME type as ``content_type``
    (download responses) or via the metadata payload. Returns
    ``None`` when the field is absent or doesn't map to a known
    media type — the caller falls back to URL-extension inference.
    """
    if info is None:
        return None
    mime = getattr(info, "content_type", None) or getattr(info, "mime_type", None)
    if not mime:
        return None
    return MediaType.from_(mime, default=None)


def _mtime(info) -> float:
    val = getattr(info, "last_modified", None) or getattr(
        info, "modification_time", None
    )

    if val is None:
        return 0.0

    try:
        return float(any_to_datetime(val, tz=dt.timezone.utc).timestamp())
    except Exception:
        try:
            return float(val) / 1000.0
        except Exception:
            return 0.0


def _looks_like_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("NotFound", "ResourceDoesNotExist", "FileNotFoundError"):
        return True
    if isinstance(exc, FileNotFoundError):
        return True
    return "does not exist" in str(exc).lower()


# ``\bvolume\b`` matches the bare word; ``/Volumes/`` in a directory-missing
# path lowercases to ``/volumes/`` and ``volumes`` (with the trailing ``s``)
# does *not* satisfy the second word boundary — so this stays clear of the
# path-prefix false positive.
_VOLUME_TOKEN_RE = re.compile(r"\bvolume\b", re.IGNORECASE)


def _looks_like_volume_not_found(exc: BaseException) -> bool:
    """True when *exc* names the Unity Catalog volume itself as missing.

    Distinct from a missing sub-directory inside an existing volume:
    Databricks' Files API surfaces the former as a NotFound carrying
    the word ``Volume`` (e.g. ``Volume 'cat.sch.vol' does not exist``),
    while a missing sub-path mentions ``Path``/``directory`` instead.
    Used by :meth:`VolumePath._ensure_parents` to skip the cheap
    ``files.create_directory`` probe and create the volume directly.
    """
    if not _looks_like_not_found(exc):
        return False
    return _VOLUME_TOKEN_RE.search(str(exc)) is not None


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError"):
        return True
    return "already exists" in str(exc).lower()


# Late-bound: ``VolumePath._SERVICE_CLASS`` is ``Volumes`` once the
# volume package finishes importing — avoids the
# ``fs.volume_path → volume.volumes → volume.volume → fs.volume_path``
# cycle by deferring the attribute set to module-load tail.
from ..volume.volumes import Volumes as _Volumes  # noqa: E402

VolumePath._SERVICE_CLASS = _Volumes
