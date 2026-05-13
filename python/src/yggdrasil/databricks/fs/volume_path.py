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

import io as _stdio
import logging
import os
import threading
import time
import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional, Tuple

from databricks.sdk.service.catalog import VolumeOperation

from yggdrasil.data.cast import any_to_datetime
from yggdrasil.data.enums import Mode, ModeLike, Scheme
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.path import Path
from yggdrasil.io.url import URL
from .path import DatabricksPath

if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.sql.catalog import Catalog
    from yggdrasil.databricks.sql.schema import Schema

from yggdrasil.databricks.aws import AWSDatabricksVolumeCredentials

# ``VolumeCredentialsRefresher`` is kept as an alias for the public name
# used in older releases / external callers and existing test fixtures.
VolumeCredentialsRefresher = AWSDatabricksVolumeCredentials

__all__ = ["VolumePath", "VolumeCredentialsRefresher"]


logger = logging.getLogger(__name__)


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/<cat>/<sch>/<vol>/...`` via the Files API."""

    __slots__ = ("_volume_info", "_storage_path", "_catalog", "_schema")

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_VOLUME
    namespace_prefix: ClassVar[str] = "/Volumes/"

    _VOLUME_INFO_CACHE: ClassVar["dict[Tuple[str, str, str], Any]"] = {}
    _VOLUME_INFO_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        data: Any = None,
        *,
        url: "URL | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data=data, url=url, **kwargs)
        self._volume_info: Any = None
        self._storage_path: Any = None
        self._catalog: Any = None
        self._schema: Any = None

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
    def catalog(self) -> "Catalog":
        """Return a :class:`Catalog` instance for this volume's parent catalog.

        Lazily resolved on first access and cached on the instance —
        every subsequent access returns the same :class:`Catalog`. The
        instance is bound to :attr:`client` via
        ``client.catalogs.catalog(...)`` so it inherits the same auth
        / retry / caching behavior as the rest of the SQL surface.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix), since there's no catalog
        to bind to.
        """
        if self._catalog is not None:
            return self._catalog
        catalog_name = self.catalog_name
        if not catalog_name:
            raise ValueError(
                f"{type(self).__name__}.catalog requires a path under "
                f"/Volumes/<cat>/<sch>/<vol>/... — got {self.full_path()!r}."
            )
        self._catalog = self.client.catalogs.catalog(catalog_name)
        return self._catalog

    @property
    def schema(self) -> "Schema":
        """Return a :class:`Schema` instance for this volume's parent schema.

        Lazily resolved on first access and cached on the instance.
        Bound to :attr:`client` via
        ``client.schemas.schema(catalog_name=..., schema_name=...)`` so
        it shares the workspace-scoped schema cache.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix).
        """
        if self._schema is not None:
            return self._schema
        triple = self._split_volume()
        if triple is None:
            raise ValueError(
                f"{type(self).__name__}.schema requires a path under "
                f"/Volumes/<cat>/<sch>/<vol>/... — got {self.full_path()!r}."
            )
        catalog_name, schema_name, _ = triple
        self._schema = self.client.schemas.schema(
            catalog_name=catalog_name,
            schema_name=schema_name,
        )
        return self._schema

    def volume_info(self, refresh: bool = False) -> Any:
        """Return the SDK's :class:`VolumeInfo` for this volume.

        Lookups go through the process-wide
        :attr:`_VOLUME_INFO_CACHE`, keyed by
        ``(catalog, schema, volume)`` — every :class:`VolumePath`
        instance on the same volume in this process reads from the
        same cached info, so only the very first construction pays
        the ``volumes.read`` SDK round trip. The per-instance slot
        snapshots the cached entry once populated so subsequent
        ``self._volume_info`` reads stay a single attribute hop.

        Pass ``refresh=True`` to force a fresh SDK call and overwrite
        both caches (e.g. after an external schema migration).

        If the underlying ``volumes.read`` raises :class:`NotFound`
        (or a ``BadRequest``-shaped "does not exist" — UC isn't fully
        consistent here), the missing pieces of catalog / schema /
        volume are created on demand via :meth:`_ensure_volume` and
        the read is retried exactly once. Subsequent failures
        propagate. This is the only place the credentials path
        auto-creates a volume, so ``temporary_credentials`` /
        ``aws`` / ``s3_path`` / ``arrow_filesystem`` all inherit the
        "first-touch creates" behavior for free.

        Raises :class:`ValueError` when the URL path doesn't address a
        volume (no ``/cat/sch/vol`` prefix).
        """
        if self._volume_info is not None and not refresh:
            return self._volume_info
        triple = self._split_volume()
        if triple is None:
            raise ValueError(
                f"{type(self).__name__}.volume_info requires a path under "
                f"/Volumes/<cat>/<sch>/<vol>/... — got {self.full_path()!r}."
            )
        catalog, schema, volume = triple

        # Class-level cache — prefer it whenever available so two
        # ``VolumePath`` instances on the same volume share a single
        # SDK round trip. ``refresh=True`` skips the cache entirely
        # and overwrites the entry after the fresh read lands.
        if not refresh:
            cached = self._VOLUME_INFO_CACHE.get(triple)
            if cached is not None:
                self._adopt_volume_info(cached)
                return cached

        full_name = f"{catalog}.{schema}.{volume}"
        try:
            info = self._call(self.client.workspace_client().volumes.read, full_name)
        except Exception as exc:
            if not _looks_like_not_found(exc):
                raise
            # Volume (and possibly its catalog / schema) doesn't
            # exist yet — let ``_ensure_volume`` create the missing
            # pieces, then re-read. ``_ensure_volume`` swallows
            # ``AlreadyExists`` on each rung so concurrent creators
            # don't fight, and only raises if both the create AND
            # the parent-walking failed.
            if not self._ensure_volume():
                raise
            info = self._call(self.client.workspace_client().volumes.read, full_name)

        # Publish to the process-wide cache under the lock so
        # concurrent constructions converge on one entry. ``setdefault``
        # would be racy with the eager refresh case — we want the
        # latest read to win when ``refresh=True``.
        with self._VOLUME_INFO_CACHE_LOCK:
            self._VOLUME_INFO_CACHE[triple] = info
        self._adopt_volume_info(info, refresh=refresh)
        return info

    def _adopt_volume_info(self, info: Any, *, refresh: bool = False) -> None:
        """Mirror ``info`` onto the per-instance slot.

        Kept separate from :meth:`volume_info` so cache-hit and
        cache-miss paths share one snapshot routine. Also drops the
        per-instance ``_storage_path`` cache when ``refresh=True``,
        so the next :meth:`storage_path` call rebuilds against the
        fresh ``VolumeInfo``.
        """
        self._volume_info = info
        if refresh:
            self._storage_path = None

    def storage_location(self, refresh: bool = False) -> str:
        """Volume's backing storage URL string (e.g. ``s3://bucket/...``).

        Pure read from :meth:`volume_info` — no AWS auth resolution,
        no Path construction. Use this when you only need the URL
        rendering for logging / config plumbing; reach for
        :meth:`storage_path` when you'll do actual I/O against the
        location (it carries the auto-refreshing client).

        Raises :class:`ValueError` when the SDK doesn't return a
        storage location (managed volumes always do; external volumes
        may return ``None`` while the metastore is still finalizing
        registration).
        """
        info = self.volume_info(refresh=refresh)
        raw = getattr(info, "storage_location", None)
        if not raw:
            raise ValueError(
                f"{type(self).__name__}: volume has no storage_location yet. "
                f"Volume info: {info!r}."
            )
        return str(raw)

    def storage_path(
        self,
        *,
        mode: ModeLike = Mode.AUTO,
        region: Optional[str] = None,
        refresh: bool = False,
    ) -> Any:
        """Return the volume's root storage :class:`Path`.

        Resolves the volume's ``storage_location`` URL (via
        :meth:`volume_info`) and dispatches to the right
        :class:`URLBased` :class:`Path` subclass — :class:`S3Path`
        for ``s3://``, :class:`AzureBlobPath` for ``abfss://`` (when
        registered), etc. The returned :class:`Path` is cached on
        the instance and carries the auto-refreshing
        :class:`AWSClient` session minted via
        :meth:`credentials_refresher` so reads / writes through it
        survive STS token rotation without caller-side rebinding.

        ``operation`` and ``region`` are forwarded to :meth:`aws` to
        bind the right credential scope and S3 region. ``refresh``
        forces a re-resolution (drops the cached :class:`Path` and
        re-reads :class:`VolumeInfo`).

        Raises :class:`ValueError` when the SDK doesn't return a
        storage location yet.
        """
        if self._storage_path is not None and not refresh:
            return self._storage_path

        raw = self.storage_location(refresh=refresh)
        scheme = URL.from_str(raw).scheme or ""

        if scheme.startswith("s3"):
            storage_path = self.aws(mode=mode, region=region).s3.path(raw)
        else:
            # Non-S3 backends (Azure abfss://, GCS gs://) just go
            # through the generic Path registry — we don't yet have a
            # UC-credential bridge for those clouds.
            storage_path = Path.from_url(raw)

        self._storage_path = storage_path
        return storage_path

    def temporary_credentials(
        self,
        *,
        mode: ModeLike = Mode.AUTO,
    ) -> Any:
        """Vend temporary cloud credentials for this volume.

        Wraps ``temporary_volume_credentials.generate_temporary_volume_credentials``
        — Unity Catalog issues short-lived AWS / Azure / GCP creds
        scoped to the volume's storage root.

        ``operation`` accepts a :class:`VolumeOperation` enum, a
        :class:`Mode` / mode-like string (``"read"`` / ``"overwrite"`` /
        ``"append"`` / …), or ``None``. ``None`` defaults to
        ``READ_VOLUME``; read-only modes map to ``READ_VOLUME``,
        everything else to ``WRITE_VOLUME``.
        """
        info = self.volume_info()
        volume_id = getattr(info, "volume_id", None)
        if not volume_id:
            raise ValueError(
                f"{type(self).__name__}: volume_info has no volume_id; "
                f"cannot vend temporary credentials. Info: {info!r}."
            )

        mode = Mode.from_(mode, default=Mode.READ_ONLY)
        if mode is Mode.READ_ONLY:
            operation = VolumeOperation.READ_VOLUME
        else:
            operation = VolumeOperation.WRITE_VOLUME

        return self._call(
            self.client.workspace_client().temporary_volume_credentials.generate_temporary_volume_credentials,
            volume_id=volume_id,
            operation=operation,
        )

    def credentials_refresher(self) -> "AWSDatabricksVolumeCredentials":
        """Return the process-wide singleton credentials provider for
        this volume.

        Keyed by ``volume_id`` — every :class:`VolumePath` instance
        pointing at the same UC volume collapses to one provider. The
        provider handles both read and write modes internally and
        caches its :class:`AWSClient` per ``(mode, region)``, so the
        boto session, :class:`RefreshableCredentials`, and STS vending
        are shared across every reader / writer on the volume in this
        process.

        The bound :class:`DatabricksClient` is updated on each call —
        the latest live handle wins so subsequent refresh cycles use
        the freshest auth context.
        """
        info = self.volume_info()
        volume_id = getattr(info, "volume_id", None)
        if not volume_id:
            raise ValueError(
                f"{type(self).__name__}: volume_info has no volume_id; "
                f"cannot mint a credentials provider. Info: {info!r}."
            )
        return AWSDatabricksVolumeCredentials(
            volume_id=volume_id,
            client=self.client,
        )

    def aws(
        self,
        *,
        mode: ModeLike = None,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from :meth:`temporary_credentials`.

        Routes through :meth:`credentials_refresher` — every
        ``VolumePath`` on the same ``(volume_id, operation)`` shares
        one refresher and, via that refresher's per-region cache, one
        :class:`AWSClient`. Every signing request that runs after the
        token's near-expiry window re-invokes the refresher and
        rotates the underlying creds in place. No caller-side refresh
        dance, and the STS vend is paid once per refresh cycle no
        matter how many callers are reading the volume concurrently.

        ``region`` is optional — when omitted, botocore resolves it
        from env / config / instance metadata. Pass it explicitly when
        the volume sits in a non-default region.
        """
        return self.credentials_refresher().aws_client(mode=mode, region=region)

    def arrow_filesystem(
        self,
        *,
        operation: Any = None,
        region: Optional[str] = None,
    ) -> Any:
        """Build a :class:`pyarrow.fs.S3FileSystem` for this volume.

        Routes through the cached :meth:`storage_path` so the
        filesystem inherits the same :class:`AWSClient` (and its
        auto-refreshing boto session) as direct :class:`S3Path` I/O.
        Credentials are sourced via the centralized
        :meth:`S3Service.arrow_filesystem` so any future tweak to
        the pyarrow filesystem construction (custom endpoint, retry
        strategy, …) lives in one place.

        Returns a fresh pyarrow filesystem snapshot bound to the
        currently-vended STS token. For long-running operations,
        call this once per refresh window (botocore will rotate the
        underlying creds in the meantime, so the snapshot stays
        valid until the token expires).
        """
        # Ensure the volume's storage path is resolved (and the
        # underlying AWSClient session is live) before snapshotting
        # credentials.
        self.storage_path(mode=operation, region=region)
        aws_client = self.aws(mode=operation, region=region)
        return aws_client.s3.arrow_filesystem(region=region)

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
        """
        del max_lifetime  # filename carries the timestamp; unused here

        cat = _staging_clean_part(catalog_name)
        sch = _staging_clean_part(schema_name)
        tbl = _staging_clean_part(resource_name or "default")

        epoch_ms = int(time.time() * 1000)
        seed = os.urandom(8).hex()
        leaf = f"part-{epoch_ms}-{seed}.parquet"
        path = f"/{cat}/{sch}/tmp/.sql/{cat}/{sch}/{tbl}/{leaf}"

        staged = cls(
            url=URL(scheme=cls.scheme, path=path),
            client=client,
            temporary=temporary,
        )
        if tabular is None:
            return staged

        # Local imports — keep optional engine deps off the import path
        # for a plain ``staging_path()`` call that doesn't write.
        from yggdrasil.data.enums import MediaTypes

        try:
            staged.as_media(media_type=MediaTypes.PARQUET).write_table(tabular)
        except Exception:
            if temporary:
                try:
                    staged.unlink(missing_ok=True)
                except Exception:
                    pass
            raise
        return staged

    # ==================================================================
    # Listing
    # ==================================================================

    def _ls(self, recursive: bool = False) -> Iterator["VolumePath"]:
        files = self.client.workspace_client().files
        try:
            entries = list(self._call(files.list_directory_contents, self.api_path))
        except Exception:
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "files.list_directory_contents %s -> %d entries (recursive=%s)",
                self.api_path, len(entries), recursive,
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
            child = type(self)(
                child_path,
                client=self._client,
            )
            yield child
            if recursive and getattr(info, "is_directory", False):
                yield from child._ls(recursive=True)

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

        Production callers usually have catalog and schema already in
        place and only need the managed volume created (and frequently
        lack permission to create catalogs at all). So the order is:
        try volume first, walk up only when a NotFound proves the
        next ancestor is also missing.

        Each ``create`` swallows ``AlreadyExists`` so re-runs are free.
        Returns ``True`` if at least one ``create`` landed.
        """
        triple = self._split_volume()
        if triple is None:
            return False
        catalog, schema, volume = triple
        ws = self.client.workspace_client()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "VolumePath._ensure_volume %s.%s.%s",
                catalog, schema, volume,
            )

        def _create_volume() -> Any:
            return ws.volumes.create(
                catalog_name=catalog,
                schema_name=schema,
                name=volume,
                volume_type=_managed_volume_type(),
            )

        # 1) Try volume only — common case where catalog + schema exist.
        try:
            _create_volume()
            return True
        except Exception as exc:
            if _looks_like_already_exists(exc):
                return False
            if not _looks_like_not_found(exc):
                raise
            # Fall through: a parent (schema or catalog) is missing.

        # 2) Schema may be missing — create it, then retry volume.
        try:
            ws.schemas.create(name=schema, catalog_name=catalog)
        except Exception as exc:
            if _looks_like_already_exists(exc):
                pass
            elif _looks_like_not_found(exc):
                # 3) Catalog also missing — create catalog, then schema.
                _safe_create(lambda: ws.catalogs.create(name=catalog))
                _safe_create(
                    lambda: ws.schemas.create(
                        name=schema, catalog_name=catalog,
                    ),
                )
            else:
                raise

        _safe_create(_create_volume)
        return True

    # ==================================================================
    # Mutators
    # ==================================================================

    def _mkdir(self, parents: bool = True, exist_ok: bool = True) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("files.create_directory %s", self.api_path)
        try:
            self._call_ensuring_parents(
                self.client.workspace_client().files.create_directory, self.api_path,
            )
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._seed_stat_cache(IOStats(kind=IOKind.DIRECTORY))

    def _remove_file(self, missing_ok: bool = True) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("files.delete %s", self.api_path)
        try:
            self._call(self.client.workspace_client().files.delete, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True,
    ) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "files.delete_directory %s (recursive=%s)",
                self.api_path, recursive,
            )
        try:
            self._call(self.client.workspace_client().files.delete_directory, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "files.download %s -> %d bytes (slice pos=%d n=%s)",
                self.api_path, len(data), pos, "EOF" if n < 0 else n,
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "files.upload %s -> %d bytes", self.api_path, len(payload),
            )
        self._call_ensuring_parents(
            self.client.workspace_client().files.upload,
            file_path=self.api_path,
            contents=_stdio.BytesIO(payload),
            overwrite=True,
        )
        self._seed_stat_cache(IOStats(
            size=len(payload),
            kind=IOKind.FILE,
            mtime=time.time(),
            media_type=self.media_type,
        ))

    def truncate(self, n: int) -> int:
        if n < 0:
            raise ValueError(f"truncate size must be >= 0, got {n!r}")
        if n == 0:
            self._upload(b"")
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
        self._remove_file(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sub_volume_tail(path: str) -> str:
    """Strip the ``/cat/sch/vol`` prefix off a volume-relative URL path.

    ``/cat/sch/vol/sub/dir/x.parquet`` → ``"/sub/dir/x.parquet"``.
    ``/cat/sch/vol``                    → ``""``.

    Used to map a :class:`VolumePath` URL onto the volume's S3
    storage root — the storage location already carries the catalog
    / schema / volume coordinates, so the sub-volume tail is the
    only part we need to append.
    """
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3:
        return ""
    return "/" + "/".join(parts[3:]) if len(parts) > 3 else ""


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
    val = getattr(info, "last_modified", None) or getattr(info, "modification_time", None) or getattr(info, "last_modified_time", None)

    if val is None:
        return 0.0

    try:
        return float(any_to_datetime(val, tz=dt.datetime.utc).timestamp())
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


def _safe_create(create: Any) -> bool:
    """Run *create()*; treat ``AlreadyExists`` as success (idempotent)."""
    try:
        create()
    except Exception as exc:
        if _looks_like_already_exists(exc):
            return False
        raise
    return True


def _managed_volume_type() -> Any:
    """Resolve the SDK's ``VolumeType.MANAGED`` enum, falling back to a string.

    The Databricks SDK accepts the enum or the literal ``"MANAGED"``;
    the string fallback keeps the helper usable in test environments
    that mock the workspace client without the SDK installed.
    """
    try:
        from databricks.sdk.service.catalog import VolumeType
        return VolumeType.MANAGED
    except Exception:
        return "MANAGED"


def _staging_clean_part(value: str) -> str:
    """Strip backticks/whitespace and forbid ``/`` in path segments."""
    return str(value).strip().strip("`").replace("/", "_")
