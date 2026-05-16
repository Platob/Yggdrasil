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
import io as _stdio
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional, Tuple

from databricks.sdk.errors import PermissionDenied

from yggdrasil.concurrent import Job
from yggdrasil.data.cast import any_to_datetime, parse_http_date
from yggdrasil.data.enums import Mode, ModeLike, Scheme
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.dataclasses import WaitingConfig
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL
from ..path import DatabricksPath
from ..client import DatabricksClient

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.catalog.catalog import Catalog
    from yggdrasil.databricks.schema.schema import Schema
    from yggdrasil.databricks.volume.volume import Volume

from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials

# ``VolumeCredentialsRefresher`` is kept as an alias for the public name
# used in older releases / external callers and existing test fixtures.
VolumeCredentialsRefresher = AWSDatabricksVolumeCredentials

__all__ = ["VolumePath", "VolumeCredentialsRefresher"]


logger = logging.getLogger(__name__)


# Filename produced by ``staging_path``:
#     tmp-{start_epoch_s}-{end_epoch_s}-{seed}.parquet
# ``end_epoch_s`` is the TTL the external sweepers honour, so the in-process
# sweeper keys off the same field.
_STAGING_LEAF_RE = re.compile(
    r"^tmp-(?P<start>\d+)-(?P<end>\d+)-[0-9a-f]+\.parquet$",
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
    namespace_prefix: ClassVar[str] = "/Volumes/"

    # ``_SERVICE_CLASS`` is bound below the class body to avoid the
    # ``volume.volumes`` → ``volume.volume`` → ``fs.volume_path``
    # import cycle.

    # Process-wide "already swept" set, keyed by ``(catalog, schema, resource)``
    # so concurrent ``staging_path`` calls collapse to one sweep per staging
    # directory. Insert under the lock *before* launching the sweeper thread
    # so duplicate triggers don't double-launch.
    _STAGING_SWEPT: ClassVar["set[Tuple[str, str, str]]"] = set()
    _STAGING_SWEPT_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        data: Any = None,
        *,
        url: "URL | None" = None,
        volume: "Volume | None" = None,
        service: Any = None,
        client: "DatabricksClient | None" = None,
        **kwargs: Any,
    ) -> None:
        # Idempotent under ``Singleton`` caching — see ``DatabricksPath.__init__``.
        if getattr(self, "_initialized", False):
            return

        self._volume: Optional["Volume"] = volume

        # A bound :class:`Volume` carries both the service and the
        # client — prefer the Volume's service so the resource stays
        # navigable (``volume_path.volume`` short-circuits to the
        # cached instance without re-resolving).
        if volume is not None and service is None and client is None:
            service = volume.service

        super().__init__(
            data=data, service=service, client=client, url=url, **kwargs,
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
        # Construct the Volume singleton directly rather than routing
        # through ``client.volumes.volume(...)`` — VolumePath callers
        # frequently pass a mocked or partially-configured client, and
        # the SDK round trips that matter (``volumes.read`` /
        # ``volumes.create``) hit ``client.workspace_client()`` either
        # way. The singleton dance in :class:`Volume.__new__` still
        # collapses to the same instance across every path on this UC
        # volume.
        from yggdrasil.databricks.volume.volume import Volume
        from yggdrasil.databricks.volume.volumes import Volumes
        self._volume = Volume(
            service=Volumes(client=self.client),
            catalog_name=catalog,
            schema_name=schema,
            volume_name=volume_name,
        )
        return self._volume

    @property
    def catalog(self) -> "Catalog":
        """Return a :class:`Catalog` instance for this volume's parent catalog.

        Delegates to :attr:`volume`.catalog so the underlying
        :class:`Catalog` instance is reused across every path on this
        UC volume.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix).
        """
        return self.volume.catalog

    @property
    def schema(self) -> "Schema":
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
    # SQL staging factory
    # ==================================================================

    @classmethod
    def staging_path(
        cls,
        *,
        catalog_name: str,
        schema_name: str,
        resource_name: Optional[str] = None,
        temporary: bool = True,
        client: Any = None,
        max_lifetime: Optional[float] = None,
        tabular: Any = None,
    ) -> "VolumePath":
        """Mint a fresh Parquet staging file under
        ``/Volumes/<cat>/<sch>/tmp/.sql/<cat>/<sch>/<resource>/part-...``.

        The leaf filename is unique per call (epoch-ms + 8 bytes of
        randomness). Pass ``temporary=False`` to keep the file past
        process exit; otherwise it is unlinked when the holder is
        released.

        Pass ``client`` — a :class:`DatabricksClient` aggregator — to
        bind the freshly-minted path explicitly. When omitted, the
        path lazy-resolves through :meth:`DatabricksClient.current`
        on first use.

        ``max_lifetime`` is accepted for backwards compatibility —
        external sweepers honour it via the ``part-{epoch_ms}-...``
        filename convention.

        ``tabular`` — optional :class:`Tabular` (or anything
        :meth:`Tabular.write_table` accepts: ``pa.Table`` / pandas /
        polars / pyspark frames, list of dicts, ...).  When supplied,
        the data is written to the freshly-minted path as Parquet
        before returning, so a single call yields a populated staging
        file ready to reference from SQL.  Cleanup matches the
        ``temporary`` flag: a write failure unlinks the path when
        ``temporary=True``.

        Side effect: on first call per ``(cat, sch, resource)`` in
        this process, a background sweep of the staging directory
        deletes files whose embedded TTL (``end_epoch_s`` segment of
        the leaf name) has already passed. The sweep is fire-and-
        forget; staging never blocks on it and failures are logged
        only.
        """
        cat = _staging_clean_part(catalog_name)
        sch = _staging_clean_part(schema_name)
        tbl = _staging_clean_part(resource_name or "default")

        start_epoch_s = int(time.time())
        end_epoch_s = start_epoch_s + int(max_lifetime or 3600)
        seed = os.urandom(4).hex()
        leaf = f"tmp-{start_epoch_s}-{end_epoch_s}-{seed}.parquet"
        path = f"/{cat}/{sch}/tmp_{tbl}/.sql/{leaf}"

        staged = cls(
            url=URL(scheme=cls.scheme, path=path),
            client=client,
            temporary=temporary,
        )

        # Opportunistic, one-shot-per-process staging sweep. Runs in
        # the background — the new staging path is returned immediately.
        cls._maybe_sweep_staging(
            catalog=cat,
            schema=sch,
            resource=tbl,
            client=client,
        )

        if tabular is None:
            return staged

        try:
            staged.as_media(media_type=MediaTypes.PARQUET).write_table(
                tabular,
                mode=Mode.OVERWRITE
            )
        except Exception:
            if staged.temporary:
                staged.clear()
            raise
        return staged

    # ==================================================================
    # Staging sweep — one-time per (cat, sch, resource) per process
    # ==================================================================

    @classmethod
    def _maybe_sweep_staging(
        cls,
        *,
        catalog: str,
        schema: str,
        resource: str,
        client: Any,
    ) -> None:
        """Launch a one-shot background sweep of the staging directory.

        Idempotent per process: the ``(catalog, schema, resource)`` key is
        inserted into :attr:`_STAGING_SWEPT` under the lock *before* the
        thread is launched, so concurrent ``staging_path`` calls on the
        same directory don't double-launch. The thread is a daemon — it
        never blocks process exit and never raises into the caller.
        """
        key = (catalog, schema, resource)

        if key in cls._STAGING_SWEPT:
            return

        with cls._STAGING_SWEPT_LOCK:
            if key in cls._STAGING_SWEPT:
                return
            cls._STAGING_SWEPT.add(key)

        thread = threading.Thread(
            target=cls._sweep_staging,
            kwargs={
                "catalog": catalog,
                "schema": schema,
                "resource": resource,
                "client": client,
            },
            name=f"volume-staging-sweep-{catalog}.{schema}.{resource}",
            daemon=True,
        )
        thread.start()

    @classmethod
    def _sweep_staging(
        cls,
        *,
        catalog: str,
        schema: str,
        resource: str,
        client: Any,
    ) -> None:
        """List the staging directory and delete files whose TTL has passed.

        Best-effort: every failure mode (missing directory, listing
        permission denied, individual delete failures) is swallowed
        and logged. The sweep MUST NOT raise — it is invoked from a
        daemon thread and any uncaught exception would just disappear
        into the void anyway, but explicit handling keeps the log
        trail readable.

        Filenames are matched against :data:`_STAGING_LEAF_RE`; only
        files whose embedded ``end_epoch_s`` is strictly in the past
        are deleted. Anything else (foreign files, in-flight stagers
        from other processes whose TTL hasn't elapsed, oddly-named
        leftovers) is left alone.
        """
        staging_dir = cls(
            url=URL(
                scheme=cls.scheme,
                path=f"/{catalog}/{schema}/tmp_{resource}/.sql",
            ),
            client=client,
        )
        now = int(time.time())
        deleted = 0
        scanned = 0
        try:
            for child in staging_dir._ls(recursive=False):
                scanned += 1
                leaf = (child.url.path or "").rsplit("/", 1)[-1]
                match = _STAGING_LEAF_RE.match(leaf)
                if not match:
                    continue
                try:
                    end_epoch_s = int(match.group("end"))
                except (TypeError, ValueError):
                    continue
                if end_epoch_s >= now:
                    continue
                try:
                    child._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))
                    deleted += 1
                except Exception:
                    logger.debug(
                        "Staging sweep failed to delete %r",
                        child,
                        exc_info=True,
                    )
        except Exception:
            # Directory missing, permission denied, transient SDK
            # error — all benign. Log at debug so production logs
            # stay quiet but the trail exists for diagnosis.
            logger.debug(
                "Staging sweep aborted for /%s/%s/tmp_%s/.sql",
                catalog, schema, resource,
                exc_info=True,
            )
            return
        if deleted or scanned:
            logger.debug(
                "Staging sweep /%s/%s/tmp_%s/.sql scanned=%d deleted=%d",
                catalog, schema, resource, scanned, deleted,
            )

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
                self, e,
            )
            return
        except Exception:
            return
        if logger.isEnabledFor(logging.DEBUG):
            entries = list(entries)
            logger.debug(
                "Listing volume directory %r -> %d entries (recursive=%s)",
                self, len(entries), recursive,
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
            child._seed_stat_cache(IOStats(
                kind=IOKind.DIRECTORY if is_directory else IOKind.FILE,
                size=0 if is_directory else int(
                    getattr(info, "file_size", 0) or 0,
                ),
                mtime=_mtime(info),
            ))
            yield child
            if recursive and is_directory:
                yield from child._ls(recursive=True, singleton_ttl=singleton_ttl)

    # ==================================================================
    # Parent / volume auto-creation
    # ==================================================================

    def _ensure_parents(self) -> bool:
        """Recovery hook for ``_call_ensuring_parents`` after NotFound.

        Cheap-path first: if *self* lives strictly below the volume
        root, try a single ``files.create_directory`` on the parent
        — that's the common case where only a sub-directory was
        missing. Only if that call also fails NotFound (which
        indicates the volume itself doesn't exist) do we fall back
        to :meth:`_ensure_volume` and a parent ``mkdir`` retry. No
        upfront ``catalogs.get`` / ``schemas.get`` / ``volumes.read``
        probes — blind creates swallow ``AlreadyExists`` so the
        idempotent path costs at most three SDK calls.
        """
        triple = self._split_volume()
        if triple is None:
            return False

        parent = self.parent
        pparts = [p for p in (parent.url.path or "/").lstrip("/").split("/") if p]
        has_subdir = len(pparts) > 3  # parent strictly below ``/cat/sch/vol``

        if has_subdir:
            try:
                self._call(
                    self.client.workspace_client().files.create_directory, parent.api_path,
                )
                return True
            except Exception as exc:
                if _looks_like_already_exists(exc):
                    return True
                if not _looks_like_not_found(exc):
                    raise
                # Parent missing because volume itself is missing —
                # fall through to volume creation.

        self._ensure_volume()

        if has_subdir:
            try:
                self._call(
                    self.client.workspace_client().files.create_directory, parent.api_path,
                )
            except Exception as exc:
                if not _looks_like_already_exists(exc):
                    raise
        return True

    def _ensure_volume(self) -> bool:
        """Bottom-up create of the missing pieces of catalog / schema / volume.

        Delegates to :meth:`Volume._ensure_volume` — the actual
        recovery logic lives on the singleton resource so both
        ``VolumePath`` writes and direct ``Volume.read_info`` calls
        share one implementation.

        Returns ``False`` if the URL path doesn't address a volume
        (recovery isn't applicable); otherwise returns whatever the
        underlying :meth:`Volume._ensure_volume` returns.
        """
        if self._split_volume() is None:
            return False
        return self.volume._ensure_volume()

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        logger.debug("Creating volume directory %r", self)
        try:
            self._call_ensuring_parents(
                self.client.workspace_client().files.create_directory, self.api_path,
            )
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))
        logger.info(
            "Created volume directory %r (parents=%s)",
            self, parents,
        )

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
            self, recursive,
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
                            recursive=True, missing_ok=missing_ok,
                            wait=wait, pool=executor,
                        )
                    else:
                        file_futures.append(executor.submit(
                            child._remove_file, missing_ok=missing_ok, wait=wait,
                        ))
                for fut in file_futures:
                    fut.result()
            finally:
                if owns_pool:
                    executor.shutdown(wait=True)

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "files.delete_directory %s (recursive=%s)",
                self.api_path, recursive,
            )

        if wait:
            try:
                self._call(self.client.workspace_client().files.delete_directory, self.api_path)
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
            response = self._call(self.client.workspace_client().files.download, self.api_path)
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
            self, len(data), pos, "EOF" if n < 0 else n,
        )

        media_type = _media_type_from_response(response)
        try:
            mtime = parse_http_date(response.last_modified) if response.last_modified else None
        except Exception:
            mtime = None
        mtime = mtime.timestamp() if mtime else time.time()
        if not self._stat_cached:
            self._seed_stat_cache(stats=IOStats(
                size=len(data),
                kind=IOKind.FILE,
                mtime=mtime,
                media_type=media_type,
            ))
        else:
            self._stat_cached.size = len(data)
            self._stat_cached.mtime = mtime
            if media_type is not None and self._stat_cached.media_type is None:
                self._stat_cached.media_type = media_type
            # Re-stamp the TTL — this download IS the freshest size we
            # could observe; the entry should outlive the original
            # probe's window from this point on.
            self._seed_stat_cache(self._stat_cached)

        if pos:
            data = data[pos:]
        if n > 0:
            data = data[:n]
        return memoryview(data)

    def _write_mv(self, data: memoryview, pos: int) -> int:
        n = len(data)
        if n == 0:
            return 0
        if pos == 0:
            payload = bytes(data)
        else:
            # Positional write: pull the existing blob in a single
            # ``files.download`` round trip (no preceding
            # ``get_metadata`` probe). Volumes downloads the whole
            # object regardless of the requested window, so asking
            # for "to EOF" is free.
            try:
                existing = bytes(self._read_mv(-1, 0))
            except FileNotFoundError:
                existing = b""
            if pos > len(existing):
                existing = existing + b"\x00" * (pos - len(existing))
            payload = existing[:pos] + bytes(data) + existing[pos + n:]
        self._upload(payload)
        return n

    def _upload(self, payload: bytes) -> None:
        size = len(payload)

        logger.debug(
            "Uploading volume file %r (%d bytes)", self, size,
        )
        self._call_ensuring_parents(
            self.client.workspace_client().files.upload,
            file_path=self.api_path,
            contents=_stdio.BytesIO(payload),
            overwrite=True,
        )
        logger.info("Wrote %r (%d bytes)", self, size)
        self._seed_stat_cache(IOStats(
            size=len(payload),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=self.media_type,
        ))
        return None

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n == 0:
            return 0
        # One ``files.download`` round trip — skip the ``get_metadata``
        # probe and read the whole object. A missing target collapses
        # to "no existing bytes" and we upload a fresh zero-padded
        # head of size ``n``.
        try:
            existing = bytes(self._read_mv(-1, 0))
        except FileNotFoundError:
            existing = b""
        if n <= len(existing):
            head = existing[:n]
        else:
            head = existing + b"\x00" * (n - len(existing))
        self._upload(head)
        return n

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
    mime = (
        getattr(info, "content_type", None)
        or getattr(info, "mime_type", None)
    )
    if not mime:
        return None
    return MediaType.from_(mime, default=None)


def _mtime(info) -> float:
    val = getattr(info, "last_modified", None) or getattr(info, "modification_time", None)

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


def _looks_like_already_exists(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("AlreadyExists", "ResourceAlreadyExists", "FileExistsError"):
        return True
    return "already exists" in str(exc).lower()


def _staging_clean_part(value: str) -> str:
    """Strip backticks/whitespace and forbid ``/`` in path segments."""
    return str(value).strip().strip("`").replace("/", "_")

# Late-bound: ``VolumePath._SERVICE_CLASS`` is ``Volumes`` once the
# volume package finishes importing — avoids the
# ``fs.volume_path → volume.volumes → volume.volume → fs.volume_path``
# cycle by deferring the attribute set to module-load tail.
from ..volume.volumes import Volumes as _Volumes  # noqa: E402
VolumePath._SERVICE_CLASS = _Volumes
