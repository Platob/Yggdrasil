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
import os
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional, Tuple

from yggdrasil.data.enums import Mode, Scheme
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.io.url import URL

from .path import DatabricksPath


if TYPE_CHECKING:
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.aws.config import AwsCredentials
    from yggdrasil.aws.fs.path import S3Path


__all__ = ["VolumePath", "VolumeCredentialsRefresher"]


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/<cat>/<sch>/<vol>/...`` via the Files API."""

    __slots__ = ("_volume_info", "_storage_path")

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_VOLUME
    namespace_prefix: ClassVar[str] = "/Volumes/"

    #: Process-wide :class:`VolumeInfo` cache keyed by
    #: ``(catalog, schema, volume)``. Unity Catalog volume metadata
    #: (storage_location, volume_id, volume_type) is stable for the
    #: volume's lifetime, so the very first :meth:`volume_info` call
    #: against a given triple pays the SDK round trip and every
    #: subsequent ``VolumePath`` on the same volume reads from this
    #: cache. Lock-guarded so concurrent constructions don't race the
    #: SDK call.
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
        # Per-instance pointer into the shared
        # :attr:`_VOLUME_INFO_CACHE`. ``None`` until
        # :meth:`volume_info` populates it; once set it aliases the
        # class-level cached entry, so the instance's view stays
        # consistent with every other :class:`VolumePath` on the
        # same volume.
        self._volume_info: Any = None
        # Cached :class:`Path` (typically :class:`S3Path`) at the
        # volume's root storage location. Resolved lazily by
        # :meth:`storage_path` — once built it carries the
        # auto-refreshing :class:`AWSClient` session, so reusing the
        # cached instance keeps the boto pool and the credential
        # vending session shared across calls.
        self._storage_path: Any = None

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
        files = self.workspace.files
        try:
            info = self._call(files.get_metadata, self.api_path)
        except Exception:
            info = None
        if info is not None:
            return IOStats(
                kind=IOKind.FILE,
                size=int(getattr(info, "content_length", 0) or 0),
                mtime=_mtime(info),
            )
        try:
            dir_info = self._call(files.get_directory_metadata, self.api_path)
        except Exception:
            dir_info = None
        if dir_info is not None:
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
            info = self._call(self.workspace.volumes.read, full_name)
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
            info = self._call(self.workspace.volumes.read, full_name)

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
        operation: Any = None,
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

        info = self.volume_info(refresh=refresh)
        raw = getattr(info, "storage_location", None)
        if not raw:
            raise ValueError(
                f"{type(self).__name__}: volume has no storage_location yet. "
                f"Volume info: {info!r}."
            )

        url = URL.from_(str(raw))
        scheme = (url.scheme or "").lower()
        if scheme in ("s3", "s3a", "s3n"):
            # Route through the shared :class:`AWSClient` so the
            # boto session + refreshable credentials follow the same
            # singleton that :meth:`aws` returns. We bind the
            # ``client=`` directly (S3Path stores it on ``_client``)
            # so every sibling built off this root reuses one boto
            # session.
            from yggdrasil.aws.fs.path import S3Path
            aws_client = self.aws(operation=operation, region=region)
            storage_path = S3Path(url=url, client=aws_client.s3.boto_client)
        else:
            # Azure / GCS / etc. — fall back to ``URLBased`` dispatch.
            # The path won't carry refreshable Databricks creds, but
            # it'll still address the right object store; callers can
            # plug in their own credential refresh strategy.
            from yggdrasil.io.url import URLBased
            try:
                target = URLBased.for_scheme(scheme)
            except (ValueError, ImportError) as exc:
                raise ValueError(
                    f"{type(self).__name__}: storage_location scheme "
                    f"{scheme!r} has no registered Path class. URL: {raw!r}."
                ) from exc
            storage_path = target.from_url(url)

        self._storage_path = storage_path
        return storage_path

    def temporary_credentials(
        self,
        *,
        operation: Any = None,
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
        op = _resolve_volume_operation(operation)
        info = self.volume_info()
        volume_id = getattr(info, "volume_id", None)
        if not volume_id:
            raise ValueError(
                f"{type(self).__name__}: volume_info has no volume_id; "
                f"cannot vend temporary credentials. Info: {info!r}."
            )
        return self._call(
            self.workspace.temporary_volume_credentials
            .generate_temporary_volume_credentials,
            volume_id=volume_id,
            operation=op,
        )

    def credentials_refresher(
        self,
        *,
        operation: Any = None,
    ) -> "VolumeCredentialsRefresher":
        """Return the process-wide singleton refresher for this volume.

        Keyed by ``(volume_id, operation)`` — every :class:`VolumePath`
        instance pointing at the same UC volume and asking for the
        same op collapses to one refresher. That refresher caches its
        :class:`AWSClient` per region, so the boto session,
        :class:`RefreshableCredentials`, and STS vending are shared
        across every reader / writer on the volume in this process.

        The bound client is updated on each call — the latest live
        handle wins so subsequent refresh cycles use the freshest auth
        context (useful when callers rotate Databricks clients between
        sessions). The SDK workspace client is reached through
        ``client.workspace_client()`` at refresh time.
        """
        op = _resolve_volume_operation(operation)
        info = self.volume_info()
        volume_id = getattr(info, "volume_id", None)
        if not volume_id:
            raise ValueError(
                f"{type(self).__name__}: volume_info has no volume_id; "
                f"cannot mint a credentials refresher. Info: {info!r}."
            )
        return VolumeCredentialsRefresher(
            volume_id=volume_id,
            operation=op,
            client=self.client,
        )

    def aws(
        self,
        *,
        operation: Any = None,
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
        return self.credentials_refresher(operation=operation).aws_client(region=region)

    def s3_path(
        self,
        *,
        operation: Any = None,
        region: Optional[str] = None,
    ) -> "S3Path":
        """Return an :class:`S3Path` over the volume's S3 storage.

        Joins this path's sub-volume tail (``/sub/dir/file.parquet``)
        onto the cached :meth:`storage_path` so reads and writes
        bypass the SDK's Files API and go directly against S3 —
        cheaper on Unity Catalog quota and faster on the wire. The
        returned :class:`S3Path` shares the boto client that
        :meth:`storage_path` minted via the singleton
        :class:`AWSClient`, so credentials auto-refresh and the boto
        session is reused across every reader / writer on the
        volume.

        Only S3-backed volumes are supported by this fast path; an
        Azure / GCP volume raises :class:`ValueError` from
        :meth:`storage_path` when it can't dispatch to an
        :class:`S3Path`.
        """
        root = self.storage_path(operation=operation, region=region)
        from yggdrasil.aws.fs.path import S3Path
        if not isinstance(root, S3Path):
            raise ValueError(
                f"{type(self).__name__}.s3_path requires an S3-backed "
                f"volume; got {type(root).__name__} at {root.full_path()!r}."
            )
        tail = _sub_volume_tail(self.url.path or "/")
        if not tail:
            target_url = root.url
        else:
            target_url = root.url.joinpath(tail.lstrip("/"))
        # Reuse the boto client the storage root is bound to so every
        # S3Path on this volume shares the same auto-refreshing session.
        return S3Path(
            url=target_url,
            client=root._client,
            temporary=self.temporary,
        )

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
        self.storage_path(operation=operation, region=region)
        aws_client = self.aws(operation=operation, region=region)
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
        files = self.workspace.files
        try:
            entries = self._call(files.list_directory_contents, self.api_path)
        except Exception:
            return
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
                    self.workspace.files.create_directory, parent.api_path,
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
                    self.workspace.files.create_directory, parent.api_path,
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
        ws = self.workspace

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
        try:
            self._call_ensuring_parents(
                self.workspace.files.create_directory, self.api_path,
            )
        except Exception as exc:
            if not exist_ok and _looks_like_already_exists(exc):
                raise
        self._invalidate_stat_cache()

    def _remove_file(self, missing_ok: bool = True) -> None:
        try:
            self._call(self.workspace.files.delete, self.api_path)
        except Exception:
            if not missing_ok:
                raise
        self._invalidate_stat_cache()

    def _remove_dir(
        self, recursive: bool = True, missing_ok: bool = True,
    ) -> None:
        try:
            self._call(self.workspace.files.delete_directory, self.api_path)
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
            response = self._call(self.workspace.files.download, self.api_path)
        except Exception as exc:
            if _looks_like_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            raise
        body = getattr(response, "contents", None) or response
        try:
            data = body.read()
        except AttributeError:
            data = bytes(body)
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
        self._call_ensuring_parents(
            self.workspace.files.upload,
            file_path=self.api_path,
            contents=_stdio.BytesIO(payload),
            overwrite=True,
        )
        self._invalidate_stat_cache()

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
# VolumeCredentialsRefresher — process-wide singleton, keyed by (volume_id, op)
# ---------------------------------------------------------------------------


class VolumeCredentialsRefresher:
    """Process-wide singleton refresher for a Unity Catalog volume.

    Keyed by ``(volume_id, operation)``: every caller asking for a
    refreshable credential vending session against the same UC volume
    and operation collapses to *one* refresher instance. Through that
    refresher's per-region :class:`AWSClient` cache, the boto session,
    :class:`RefreshableCredentials`, connection pool, and STS vending
    cycle are all shared too — one ``generate_temporary_volume_credentials``
    round trip per refresh window no matter how many readers / writers
    are touching the volume concurrently.

    Identity rules (mirrors :class:`AWSClient`):

    - Same ``(volume_id, operation)`` → same instance in the process
      (cached on the class-level :attr:`_INSTANCES` dict, guarded by
      :attr:`_INSTANCES_LOCK`).
    - ``__init__`` is idempotent — Python invokes it on every
      constructor call, but the singleton guard re-uses live state.
    - Pickling routes through :meth:`__getnewargs__` /
      :meth:`__setstate__` so a refresher unpickled in the same
      process collapses back to the live singleton; the bound client
      is transient and not transported.

    The bound :class:`DatabricksClient` is mutable — every constructor
    call updates :attr:`_client` with the freshest one, so STS
    refreshes that happen after a workspace rotation pick up the new
    auth context. Pass ``None`` to leave the existing binding in place.
    """

    # Class-level singleton registry. Two refreshers with the same
    # (volume_id, operation) collapse to one instance; the live AWS
    # client cache hangs off that instance so multiple paths share
    # both the credential vend and the boto session.
    _INSTANCES: ClassVar[
        "dict[Tuple[type, str, Any], VolumeCredentialsRefresher]"
    ] = {}
    _INSTANCES_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # Slots covering both the identity fields (compared) and the
    # transient runtime state (DatabricksClient + per-region AWS clients).
    __slots__ = (
        "volume_id",
        "operation",
        "_client",
        "_client_cache",
        "_client_cache_lock",
        "_initialized",
    )

    def __new__(
        cls,
        volume_id: str,
        operation: Any,
        client: Any = None,
    ) -> "VolumeCredentialsRefresher":
        key = (cls, str(volume_id), operation)
        with cls._INSTANCES_LOCK:
            existing = cls._INSTANCES.get(key)
            if existing is not None:
                # Re-bind the latest client so a follow-up refresh
                # cycle uses the freshest auth context — a stale ref
                # would silently 401 once the underlying creds expired.
                if client is not None:
                    existing._client = client
                return existing
            instance = super().__new__(cls)
            cls._INSTANCES[key] = instance
            return instance

    def __init__(
        self,
        volume_id: str,
        operation: Any,
        client: Any = None,
    ) -> None:
        # Idempotent init — Python always calls __init__ after __new__
        # returns the cached instance. Skip the second pass so the
        # live ``_client_cache`` survives.
        if getattr(self, "_initialized", False):
            if client is not None:
                self._client = client
            return
        self.volume_id: str = str(volume_id)
        self.operation: Any = operation
        self._client: Any = client
        # Per-region AWSClient cache. One client per region per
        # refresher — the cache key is the requested region (which can
        # legitimately be ``None``, letting botocore resolve via env).
        self._client_cache: "dict[Optional[str], AWSClient]" = {}
        self._client_cache_lock: threading.Lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Pickling — survive cross-process transport via the singleton cache
    # ------------------------------------------------------------------

    def __getnewargs__(self) -> "Tuple[Any, ...]":
        return (self.volume_id, self.operation)

    def __getstate__(self) -> dict[str, Any]:
        # ``_client`` and the AWSClient cache are transient — the
        # receiver re-binds the client via the next VolumePath
        # construction, and the AWSClient cache is rebuilt lazily.
        return {"volume_id": self.volume_id, "operation": self.operation}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # ``__new__`` may have returned the live singleton — leave its
        # client ref and AWSClient cache untouched.
        if getattr(self, "_initialized", False):
            return
        self.volume_id = state["volume_id"]
        self.operation = state["operation"]
        self._client = None
        self._client_cache = {}
        self._client_cache_lock = threading.Lock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Client / workspace accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """Bound :class:`DatabricksClient` (or :class:`_WorkspaceOnlyClient` shim).

        Lazily resolves to :meth:`DatabricksClient.current` when no
        client has been bound. The SDK workspace client is reached
        through :attr:`workspace` (a thin shortcut).
        """
        if self._client is None:
            from yggdrasil.lazy_imports import databricks_client_class
            self._client = databricks_client_class().current()
        return self._client

    @property
    def workspace(self) -> Any:
        """Shortcut for ``self.client.workspace_client()``."""
        return self.client.workspace_client()

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((type(self), self.volume_id, self.operation))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VolumeCredentialsRefresher):
            return NotImplemented
        return (
            self.volume_id == other.volume_id
            and self.operation == other.operation
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(volume_id={self.volume_id!r}, "
            f"operation={self.operation!r})"
        )

    # ------------------------------------------------------------------
    # Refresh — invoked by botocore's RefreshableCredentials hook
    # ------------------------------------------------------------------

    def with_client(self, client: Any) -> "VolumeCredentialsRefresher":
        """Replace the bound :class:`DatabricksClient`. Returns *self*."""
        self._client = client
        return self

    def __call__(self) -> "AwsCredentials":
        from yggdrasil.aws.config import AwsCredentials

        resp = (
            self.workspace.temporary_volume_credentials
            .generate_temporary_volume_credentials(
                volume_id=self.volume_id,
                operation=self.operation,
            )
        )
        aws = getattr(resp, "aws_temp_credentials", None)
        if aws is None:
            raise RuntimeError(
                f"{type(self).__name__}: temporary credentials for "
                f"volume_id={self.volume_id!r} carry no "
                f"``aws_temp_credentials`` — the volume is likely "
                f"backed by Azure or GCP, not S3. Inspect the raw "
                f"response via VolumePath.temporary_credentials() "
                f"to read the right credential shape."
            )
        return AwsCredentials(
            access_key_id=aws.access_key_id,
            secret_access_key=aws.secret_access_key,
            session_token=aws.session_token,
            expiration=_iso_or_str(getattr(resp, "expiration_time", None)),
        )

    # ------------------------------------------------------------------
    # AWSClient binding — one client per (refresher, region)
    # ------------------------------------------------------------------

    def aws_client(self, *, region: Optional[str] = None) -> "AWSClient":
        """Return the cached :class:`AWSClient` for this refresher / region.

        First call seeds a botocore :class:`RefreshableCredentials`
        backed session by invoking ``self()`` once; subsequent calls
        with the same *region* return the same live client (and
        therefore share the connection pool, boto-client cache, and
        in-flight refresh state). The refresher is wired in as the
        config's ``refresher`` field — botocore re-invokes ``self()``
        on every near-expiry cycle.

        Different *region* values mint different clients (one per
        region key, ``None`` included), since boto region is a
        per-client concern.
        """
        with self._client_cache_lock:
            existing = self._client_cache.get(region)
            if existing is not None:
                return existing
            from yggdrasil.lazy_imports import aws_config_class
            client = (
                aws_config_class()
                .from_refresher(self, region=region)
                .to_client()
            )
            self._client_cache[region] = client
            return client


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


def _iso_or_str(value: Any) -> Optional[str]:
    """Coerce an expiration timestamp into the ISO-8601 string botocore
    wants for ``RefreshableCredentials``' ``expiry_time``.

    The SDK returns ``expiration_time`` as a ``datetime`` (or
    ms-since-epoch ``int`` on some shapes); botocore accepts either an
    ISO string or a datetime, but normalizing to ISO keeps the
    refresher's return shape consistent with
    :class:`AwsCredentials.expiration`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        # SDK responses occasionally carry ms-since-epoch — convert
        # to a UTC isoformat so botocore can parse it.
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            float(value) / 1000.0, tz=_dt.timezone.utc,
        ).isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _resolve_volume_operation(operation: Any) -> Any:
    """Map a caller-supplied operation hint to a :class:`VolumeOperation`.

    Accepts the SDK enum (passes through), a :class:`Mode` /
    mode-like string (``"read"`` / ``"overwrite"`` / …, normalized
    via :meth:`Mode.from_`), or ``None`` (defaults to
    ``READ_VOLUME``). Anything :meth:`Mode.from_` recognizes as a
    read-only mode (``AUTO`` / ``READ_ONLY``) collapses to
    ``READ_VOLUME``; everything else is a write and gets
    ``WRITE_VOLUME``.
    """
    # ``VolumeOperation`` only landed in databricks-sdk 0.34+. Older SDKs
    # accept the bare literal "READ_VOLUME" / "WRITE_VOLUME" on the wire
    # all the same — mirror the ``_managed_volume_type`` fallback.
    try:
        from databricks.sdk.service.catalog import VolumeOperation as _VO
    except ImportError:
        _VO = None

    read_volume = _VO.READ_VOLUME if _VO is not None else "READ_VOLUME"
    write_volume = _VO.WRITE_VOLUME if _VO is not None else "WRITE_VOLUME"

    if _VO is not None and isinstance(operation, _VO):
        return operation
    if operation is None:
        return read_volume

    mode = Mode.from_(operation, default=Mode.AUTO)
    if mode in (Mode.AUTO, Mode.READ_ONLY):
        return read_volume
    return write_volume


def _mtime(info) -> float:
    val = getattr(info, "last_modified", None) or getattr(info, "modification_time", None)
    if val is None:
        return 0.0
    try:
        return float(val.timestamp())
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
