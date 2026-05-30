""":class:`VolumePath` — Databricks Unity Catalog Volume via Files API.

Volumes carry a Unity Catalog hierarchy (catalog → schema → volume →
path) and are the SQL engine's preferred staging surface. Off-cluster,
reads / writes go through the Databricks **Files REST API** —
``/api/2.0/fs/files{path}`` for files and
``/api/2.0/fs/directories{path}`` for directories — issued over
yggdrasil's own :class:`~yggdrasil.http_.HTTPSession` (see
:meth:`DatabricksClient.files_session`) rather than the SDK's
``workspace.files`` client. The session brings a keep-alive connection
pool, status-aware 429 / 5xx retry, and — through the
:class:`~yggdrasil.http_.stream.HTTPStream` response body —
resume-on-disconnect for the SSL ``UNEXPECTED_EOF`` / connection-reset
failures the SDK's Files client handles poorly. Auth still flows through
the SDK Config, so every credential type keeps working.

The :class:`Holder` byte primitives map onto these:

- :meth:`_read_mv` — ``GET /files`` streams the body through
  :class:`HTTPStream`; we slice into the requested range.
- :meth:`_write_mv` — read-modify-rewrite via ``PUT /files``.
- :meth:`truncate` — ``PUT /files`` of the head N bytes.
- :meth:`_clear` — ``DELETE /files``.

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

Cluster-mount fast path
-----------------------

Inside a Databricks runtime, ``/Volumes/...`` is exposed as a FUSE
mount. Reads, stats, listings, mkdirs, removes, and uploads all run
through the kernel mount via stdlib syscalls instead of paying a
Files-API round trip per operation. The probe lives in
:func:`_local_mount_available` and is gated on
``DatabricksClient.is_in_databricks_environment()`` AND
``os.path.isdir("/Volumes")``. Off-cluster the existing Files-API
path runs unchanged.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import os
import re
import stat as _stat
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional
from urllib.parse import quote

from databricks.sdk.errors import PermissionDenied

from yggdrasil.concurrent import Job
from yggdrasil.data.cast import any_to_datetime, parse_http_date
from yggdrasil.enums import Mode, ModeLike, Scheme
from yggdrasil.enums.media_type import MediaType
from yggdrasil.dataclasses import ExpiringDict, WaitingConfig
from yggdrasil.io.io_stats import IOStats, IOKind
from yggdrasil.path.remote_path import _STAT_CACHE_TTL
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


# ---------------------------------------------------------------------------
# Local /Volumes FUSE mount fast path
# ---------------------------------------------------------------------------
#
# Inside a Databricks runtime (cluster / DBR notebook), the workspace
# mounts every Unity Catalog volume the cluster has access to under
# ``/Volumes/<cat>/<sch>/<vol>/...`` as a native filesystem. Reads and
# writes through the kernel mount cost a syscall — no HTTPS round trip,
# no whole-object download, no credentials-refresh dance. Locally
# (off-cluster) the mount doesn't exist and we fall back to the Files
# REST API.
#
# The probe runs lazily so test environments and off-cluster paths pay
# nothing; once probed the result is cached for the process lifetime.

_LOCAL_MOUNT_PROBED: bool = False
_LOCAL_MOUNT_AVAILABLE: bool = False


def _local_mount_available() -> bool:
    """``True`` when ``/Volumes/...`` is reachable via the kernel mount.

    Conjunction of "this process runs inside a Databricks runtime"
    (``DATABRICKS_RUNTIME_VERSION`` env var set) and "``/Volumes``
    actually exists on disk" (catches the rare runtime that disables
    the mount or test environments that fake the env var). The
    result is logged once at INFO so an operator can see at a glance
    whether VolumePath is short-circuiting through the kernel or
    paying Files-API round trips.
    """
    global _LOCAL_MOUNT_PROBED, _LOCAL_MOUNT_AVAILABLE
    if _LOCAL_MOUNT_PROBED:
        return _LOCAL_MOUNT_AVAILABLE
    try:
        from yggdrasil.databricks.client import DatabricksClient
        in_runtime = DatabricksClient.is_in_databricks_environment()
    except Exception:
        in_runtime = False
    has_mount = bool(in_runtime) and os.path.isdir("/Volumes")
    _LOCAL_MOUNT_AVAILABLE = has_mount
    _LOCAL_MOUNT_PROBED = True
    if has_mount:
        logger.info(
            "VolumePath: /Volumes kernel mount detected — short-circuiting "
            "stat/read/ls/upload/mkdir/remove off the Files API.",
        )
    else:
        logger.debug(
            "VolumePath: /Volumes kernel mount unavailable "
            "(in_runtime=%s, /Volumes exists=%s) — routing through Files API.",
            in_runtime, os.path.isdir("/Volumes"),
        )
    return _LOCAL_MOUNT_AVAILABLE


def _reset_local_mount_probe() -> None:
    """Test hook — drop the cached probe result."""
    global _LOCAL_MOUNT_PROBED, _LOCAL_MOUNT_AVAILABLE
    _LOCAL_MOUNT_PROBED = False
    _LOCAL_MOUNT_AVAILABLE = False


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

    # Multipart upload tuning. At / above ``MULTIPART_MIN_SIZE`` an upload
    # runs as concurrent presigned parts (see :meth:`_try_multipart_upload`)
    # instead of one whole-object PUT — faster on large objects and past
    # the Files-API 5 GiB single-PUT ceiling. ``PART_SIZE`` is the target
    # part grain, raised as needed so the part count stays under
    # ``MAX_PARTS``. Tunable per instance / subclass.
    MULTIPART_MIN_SIZE: ClassVar[int] = 100 * 1024 * 1024
    MULTIPART_PART_SIZE: ClassVar[int] = 16 * 1024 * 1024
    MULTIPART_MAX_PARTS: ClassVar[int] = 1000
    MULTIPART_PARALLELISM: ClassVar[int] = 8

    # ``_read_mv`` range-reads via the Files API, so Parquet projection
    # can pull the footer + projected chunks only (see
    # :meth:`RemotePath.arrow_random_access_file`).
    SUPPORTS_RANGED_RANDOM_ACCESS: ClassVar[bool] = True

    # Arrow/Parquet writes spill the encode to a temp file and stream it to the
    # Files API in bounded chunks (see :meth:`_upload_stream`), so a multi-GB
    # write never materialises whole in memory.
    SUPPORTS_STREAMING_UPLOAD: ClassVar[bool] = True

    # ``_SERVICE_CLASS`` is bound below the class body to avoid the
    # ``volume.volumes`` → ``volume.volume`` → ``fs.volume_path``
    # import cycle.

    # Per-class singleton cache — partitioned away from DBFSPath /
    # WorkspacePath. No companion lock —
    # :class:`ExpiringDict.get_or_set` is GIL-atomic.
    _INSTANCES: ClassVar[ExpiringDict] = ExpiringDict(
        default_ttl=_STAT_CACHE_TTL,
        max_size=10_000,
    )

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
    # Files REST API transport
    # ==================================================================
    #
    # Off-cluster, every volume operation is a Databricks Files REST call
    # issued through yggdrasil's own :class:`HTTPSession` instead of the
    # SDK's ``workspace.files`` client — keep-alive pooling, tiered
    # 429/5xx retry, and resume-on-disconnect streaming, with auth still
    # vended by the SDK Config. ``kind`` selects the endpoint family:
    # ``"files"`` (``/api/2.0/fs/files{path}``) or ``"directories"``
    # (``/api/2.0/fs/directories{path}``).

    def _fs_request(
        self,
        method: str,
        kind: str,
        api_path: str,
        *,
        params: "dict[str, str] | None" = None,
        body: Any = None,
        json_body: Any = None,
        range_header: "str | None" = None,
        preload_content: bool = True,
    ) -> Any:
        """Issue one Files-API request; return the raw :class:`HTTPResponse`.

        Status translation is the caller's job — see
        :meth:`_raise_for_files_status`. The session retries transient
        wire failures (429 / 5xx / SSL EOF) internally before this
        returns; an open *range_header* (``bytes=<off>-``) resumes
        mid-stream from the last received byte. *json_body* JSON-encodes
        a mapping and stamps ``Content-Type: application/json`` (used by
        the multipart coordination calls).
        """
        url = self.client.base_url.with_path(
            f"/api/2.0/fs/{kind}{quote(api_path, safe='/')}"
        )
        if params:
            for key, value in params.items():
                url = url.add_param(key=key, value=value)
        headers = {"Authorization": self.client.files_authorization()}
        if range_header is not None:
            headers["Range"] = range_header
        if json_body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(json_body).encode()
        return self.client.files_session().fetch(
            method,
            url,
            headers=headers,
            body=body,
            preload_content=preload_content,
        )

    def _raise_for_files_status(self, resp: Any, api_path: str) -> None:
        """Translate a non-2xx Files-API response into the right exception.

        Maps the wire status onto the exception *shapes* the recovery /
        classification helpers key on: 404 → :class:`FileNotFoundError`
        (carrying the server message so :func:`_looks_like_volume_not_found`
        still fires), 401/403 → SDK :class:`PermissionDenied`, 409 →
        :class:`FileExistsError`, everything else → :class:`OSError`.
        2xx/3xx return cleanly.
        """
        if resp.ok:
            return
        status = resp.status
        message = None
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                message = payload.get("message")
        except Exception:
            message = None
        if not message:
            try:
                message = resp.text
            except Exception:
                message = ""
        detail = (message or "").strip() or f"Files API {status} for {api_path}"
        if status == 404:
            raise FileNotFoundError(detail)
        if status in (401, 403):
            raise PermissionDenied(detail)
        if status == 409:
            raise FileExistsError(detail)
        raise OSError(f"Files API {status} for {api_path}: {detail}")

    def _list_directory(self, api_path: str) -> Iterator[Any]:
        """Yield directory entries, following ``next_page_token`` pagination.

        Each entry is a :class:`SimpleNamespace` carrying the Files-API
        fields (``path`` / ``name`` / ``is_directory`` / ``file_size`` /
        ``last_modified``) so the attribute-based consumers
        (:meth:`_ls`, :func:`_mtime`) read it unchanged. The SDK's
        ``list_directory_contents`` auto-paginated; this restores that.
        """
        page_token: "str | None" = None
        while True:
            params = {"page_token": page_token} if page_token else None
            resp = self._fs_request("GET", "directories", api_path, params=params)
            self._raise_for_files_status(resp, api_path)
            payload = resp.json() or {}
            for entry in payload.get("contents") or []:
                yield SimpleNamespace(**entry)
            page_token = payload.get("next_page_token")
            if not page_token:
                return

    def _create_directory(self, api_path: str) -> None:
        """``PUT /directories`` — idempotent create (parents auto-materialised)."""
        resp = self._fs_request("PUT", "directories", api_path)
        self._raise_for_files_status(resp, api_path)

    def _delete_path(self, kind: str, api_path: str) -> None:
        """``DELETE`` a file / directory — raises FileNotFoundError on 404."""
        resp = self._fs_request("DELETE", kind, api_path)
        self._raise_for_files_status(resp, api_path)

    # ==================================================================
    # Multipart upload (parallel) — mirrors the SDK's FilesExt protocol
    # ==================================================================
    #
    # ``initiate-upload`` → ``create-upload-part-urls`` → parallel PUTs to
    # the presigned cloud-storage URLs (each returns an ETag) →
    # ``complete-upload``; ``create-abort-upload-url`` + DELETE on failure.
    # The coordination calls hit the workspace with our auth; the part
    # PUTs go straight to cloud storage and carry no Databricks auth (the
    # presigned URL self-authorizes). Lets large uploads run concurrent
    # parts and lifts the 5 GiB single-PUT ceiling.

    def _fs_coordination_post(self, endpoint: str, body: dict) -> dict:
        """``POST /api/2.0/fs/<endpoint>`` with auth + JSON body → parsed JSON."""
        url = self.client.base_url.with_path(f"/api/2.0/fs/{endpoint}")
        resp = self.client.files_session().fetch(
            "POST",
            url,
            headers={
                "Authorization": self.client.files_authorization(),
                "Content-Type": "application/json",
            },
            body=json.dumps(body).encode(),
            preload_content=True,
        )
        self._raise_for_files_status(resp, endpoint)
        return resp.json() or {}

    def _try_multipart_upload(self, api_path: str, payload: bytes) -> bool:
        """Upload *payload* in concurrent parts. Returns ``True`` on success.

        Returns ``False`` (so the caller falls back to a single PUT) when
        the workspace doesn't support multipart — the ``initiate-upload``
        probe fails or yields no session token. A failure *after* a
        successful initiate aborts the session and re-raises (the bytes
        never half-land).
        """
        resp = self._fs_request(
            "POST", "files", api_path,
            params={"action": "initiate-upload", "overwrite": "true"},
        )
        if not resp.ok:
            logger.info(
                "Multipart unsupported for %r (initiate %s) — single PUT.",
                self, resp.status,
            )
            return False
        token = ((resp.json() or {}).get("multipart_upload") or {}).get("session_token")
        if not token:
            return False

        size = len(payload)
        part_size = max(self.MULTIPART_PART_SIZE, -(-size // self.MULTIPART_MAX_PARTS))
        n_parts = -(-size // part_size)
        expire = (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            part_urls = self._fs_coordination_post(
                "create-upload-part-urls",
                {
                    "path": api_path,
                    "session_token": token,
                    "start_part_number": 1,
                    "count": n_parts,
                    "expire_time": expire,
                },
            ).get("upload_part_urls") or []
            if len(part_urls) < n_parts:
                raise OSError(
                    f"Multipart: server returned {len(part_urls)} part URLs "
                    f"for {n_parts} parts ({api_path})."
                )

            session = self.client.files_session()
            # Zero-copy part slicing — a ``memoryview`` slice doesn't copy
            # the (potentially multi-GiB) payload; each part materialises
            # only when its PUT body is encoded, so at most ``parallelism``
            # parts are resident at once instead of all N.
            payload_view = memoryview(payload)

            def _put_part(info: dict) -> "tuple[int, str]":
                part_number = int(info["part_number"])
                start = (part_number - 1) * part_size
                chunk = payload_view[start:start + part_size]
                headers = {"Content-Type": "application/octet-stream"}
                for h in info.get("headers") or []:
                    headers[h["name"]] = h["value"]
                # No Authorization — the presigned URL self-authorizes.
                r = session.fetch("PUT", info["url"], headers=headers, body=chunk)
                try:
                    if r.status not in (200, 201):
                        raise OSError(
                            f"Multipart part {part_number} failed: HTTP "
                            f"{r.status} ({api_path})."
                        )
                    etag = _header(r.headers, "ETag") or ""
                finally:
                    # Return the socket to the pool — parts reuse it.
                    r.drain_conn()
                    r.release_conn()
                return part_number, etag

            etags: "dict[int, str]" = {}
            workers = min(self.MULTIPART_PARALLELISM, n_parts)
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="volume-multipart",
            ) as pool:
                for part_number, etag in pool.map(_put_part, part_urls[:n_parts]):
                    etags[part_number] = etag

            complete = self._fs_request(
                "POST", "files", api_path,
                params={
                    "action": "complete-upload",
                    "upload_type": "multipart",
                    "session_token": token,
                },
                json_body={
                    "parts": [
                        {"part_number": pn, "etag": etags[pn]}
                        for pn in sorted(etags)
                    ]
                },
            )
            self._raise_for_files_status(complete, api_path)
        except Exception:
            self._multipart_abort(api_path, token, expire)
            raise
        logger.info(
            "Uploaded volume file %r via multipart (%d bytes, %d parts).",
            self, size, n_parts,
        )
        return True

    def _multipart_abort(self, api_path: str, token: str, expire: str) -> None:
        """Best-effort abort so a failed multipart leaves no orphan session."""
        try:
            info = self._fs_coordination_post(
                "create-abort-upload-url",
                {"path": api_path, "session_token": token, "expire_time": expire},
            ).get("abort_upload_url") or {}
            url = info.get("url")
            if not url:
                return
            headers = {}
            for h in info.get("headers") or []:
                headers[h["name"]] = h["value"]
            self.client.files_session().fetch("DELETE", url, headers=headers, body=b"")
        except Exception as exc:  # noqa: BLE001 — abort is best-effort
            logger.warning("Multipart abort failed for %r: %r", self, exc)

    # ==================================================================
    # Stat
    # ==================================================================

    def _stat_uncached(self) -> IOStats:
        api_path = self.api_path
        # Fast path on a Databricks cluster: the kernel mount knows
        # whether the path is a file / directory / missing in one
        # syscall — no HTTPS round trip needed.
        if _local_mount_available():
            try:
                st = os.stat(api_path)
            except FileNotFoundError:
                logger.debug("stat via kernel mount: %r -> MISSING", api_path)
                return IOStats(kind=IOKind.MISSING, size=0, mtime=0.0)
            except OSError as exc:
                logger.debug(
                    "stat via kernel mount: %r -> OSError %r, "
                    "falling back to Files API", api_path, exc,
                )
            else:
                if _stat.S_ISDIR(st.st_mode):
                    logger.debug(
                        "stat via kernel mount: %r -> DIRECTORY", api_path,
                    )
                    return IOStats(
                        kind=IOKind.DIRECTORY,
                        size=0,
                        mtime=st.st_mtime,
                    )
                logger.debug(
                    "stat via kernel mount: %r -> FILE size=%d",
                    api_path, st.st_size,
                )
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(st.st_size),
                    mtime=st.st_mtime,
                )
        # Off-cluster: probe via the Files REST API. Heuristic: a leaf
        # with a ``.`` is almost always a file (``foo.parquet`` /
        # ``part-….json``); a bare leaf is almost always a directory
        # (``/Volumes/cat/sch/vol``, ``/Volumes/cat/sch/vol/tmp``). Probe
        # that side first so the common case is one ``HEAD`` round trip
        # instead of two.
        file_first = "." in (self.url.path or "").rsplit("/", 1)[-1]
        if file_first:
            probes = (("files", IOKind.FILE), ("directories", IOKind.DIRECTORY))
        else:
            probes = (("directories", IOKind.DIRECTORY), ("files", IOKind.FILE))
        for kind, io_kind in probes:
            # ``HEAD`` probe — any failure (NotFound, permission,
            # transient after the session exhausted its retries) collapses
            # to "treat as absent" and falls through to the next probe.
            try:
                resp = self._fs_request("HEAD", kind, api_path)
            except Exception:
                continue
            if not resp.ok:
                continue
            lm = _header(resp.headers, "Last-Modified")
            parsed = parse_http_date(lm) if lm else None
            mtime = parsed.timestamp() if parsed else 0.0
            if io_kind is IOKind.FILE:
                ct = _header(resp.headers, "Content-Type")
                return IOStats(
                    kind=IOKind.FILE,
                    size=int(_header(resp.headers, "Content-Length") or 0),
                    mtime=mtime,
                    media_type=MediaType.from_(ct, default=None) if ct else None,
                )
            return IOStats(kind=IOKind.DIRECTORY, size=0, mtime=mtime)
        # Implicit-directory fallback. A ``PUT /files`` to a brand-new
        # ``/Volumes/<...>/parent/file.bin`` silently materialises the
        # file without creating an explicit ``parent`` entry, so a
        # ``HEAD /directories/parent`` returns 404 even though listing
        # the path enumerates ``file.bin``. Without this probe,
        # ``remove(parent, recursive=True, missing_ok=False)`` raises
        # ``FileNotFoundError`` against a parent the caller just wrote
        # into. One extra round trip pays for the negative case only —
        # both metadata probes already missed.
        try:
            first = next(iter(self._list_directory(api_path)), None)
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
        # Cluster fast path — scandir on the kernel mount returns
        # entries with stat info already populated, so we skip both
        # the listing round trip and the per-child get_metadata that
        # the Files API path otherwise pays for.
        if _local_mount_available():
            scan_root = self.api_path
            logical_root = self.full_path().rstrip("/")
            try:
                scan = os.scandir(scan_root)
            except FileNotFoundError:
                logger.debug(
                    "ls via kernel mount: %r -> not found", scan_root,
                )
                return
            except (NotADirectoryError, PermissionError) as exc:
                logger.warning(
                    "Cannot scan volume directory %r: %r", self, exc,
                )
                return
            yielded = 0
            with scan as it:
                for entry in it:
                    # Build the child against the logical
                    # ``/Volumes/...`` URL so the dispatcher /
                    # api_path / catalog navigation all stay
                    # consistent with the off-cluster path.
                    child_logical = f"{logical_root}/{entry.name}"
                    child = type(self)(
                        child_logical,
                        service=self.service,
                        singleton_ttl=singleton_ttl,
                    )
                    try:
                        st = entry.stat(follow_symlinks=False)
                        is_directory = _stat.S_ISDIR(st.st_mode)
                        child._persist_stat_cache(
                            IOStats(
                                kind=(
                                    IOKind.DIRECTORY
                                    if is_directory
                                    else IOKind.FILE
                                ),
                                size=0 if is_directory else int(st.st_size),
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        is_directory = entry.is_dir(follow_symlinks=False)
                    yielded += 1
                    yield child
                    if recursive and is_directory:
                        yield from child._ls(
                            recursive=True, singleton_ttl=singleton_ttl,
                        )
            logger.debug(
                "ls via kernel mount: %r -> %d entries (recursive=%s)",
                scan_root, yielded, recursive,
            )
            return
        # Off-cluster: list via the paginated Files REST API. Only the
        # listing fetch (``next(entries)``) is guarded — child
        # construction / recursion errors propagate so a bug there isn't
        # silently swallowed into an empty directory.
        entries = self._list_directory(self.api_path)
        while True:
            try:
                info = next(entries)
            except StopIteration:
                return
            except PermissionDenied as e:
                logger.warning(
                    "Permission denied listing volume directory %r: %r",
                    self,
                    e,
                )
                return
            except FileNotFoundError:
                return
            except Exception:
                return
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
                self._create_directory(parent.api_path)
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
                self._create_directory(parent.api_path)
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
        logger.debug(
            "Creating volume directory %r (parents=%s, exist_ok=%s)",
            self, parents, exist_ok,
        )
        # Cluster fast path — mkdir on the kernel mount. The mount
        # auto-materializes intermediate UC volume directories the
        # same way the Files API does, so ``parents=True`` maps
        # cleanly onto ``os.makedirs``.
        if _local_mount_available():
            api_path = self.api_path
            try:
                if parents:
                    os.makedirs(api_path, exist_ok=exist_ok)
                else:
                    os.mkdir(api_path)
            except FileExistsError:
                if not exist_ok:
                    raise
            self._persist_stat_cache(IOStats(kind=IOKind.DIRECTORY))
            logger.debug("mkdir via kernel mount: %r", api_path)
            return
        try:
            self._call_ensuring_parents(self._create_directory, self.api_path)
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
        logger.debug(
            "Deleting volume file %r (missing_ok=%s)", self, missing_ok,
        )
        if _local_mount_available():
            try:
                os.remove(self.api_path)
                logger.debug("rm via kernel mount: %r", self.api_path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            except IsADirectoryError:
                raise
            self.invalidate_singleton()
            return
        try:
            self._delete_path("files", self.api_path)
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
        logger.debug(
            "Deleting volume directory %r (recursive=%s, missing_ok=%s)",
            self, recursive, missing_ok,
        )
        # Cluster fast path — ``shutil.rmtree`` for recursive,
        # ``os.rmdir`` for the simple case. No thread pool needed
        # because the kernel walks the tree at filesystem speed,
        # and the Files-API per-leaf round trip the off-cluster
        # path needs to fan out doesn't exist here.
        if _local_mount_available():
            import shutil
            api_path = self.api_path
            try:
                if recursive:
                    shutil.rmtree(api_path)
                else:
                    os.rmdir(api_path)
            except FileNotFoundError:
                if not missing_ok:
                    raise
            self.invalidate_singleton()
            logger.debug(
                "rmdir via kernel mount: %r (recursive=%s)",
                api_path, recursive,
            )
            return
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
                self._delete_path("directories", self.api_path)
            except Exception:
                if not missing_ok:
                    raise
        else:
            Job.make(self._delete_path, "directories", self.api_path)
        self.invalidate_singleton()

    # ==================================================================
    # Holder I/O
    # ==================================================================

    def _read_mv(self, n: int, pos: int) -> memoryview:
        if n == 0:
            return memoryview(b"")
        # Cluster fast path — read straight off the kernel mount,
        # honouring offset + length without downloading the rest.
        if _local_mount_available():
            api_path = self.api_path
            try:
                with open(api_path, "rb") as fh:
                    if pos:
                        fh.seek(pos)
                    data = fh.read() if n < 0 else fh.read(n)
            except FileNotFoundError as exc:
                logger.debug(
                    "read via kernel mount: %r -> NOT FOUND", api_path,
                )
                raise FileNotFoundError(self.full_path()) from exc
            except OSError as exc:
                logger.debug(
                    "read via kernel mount: %r -> OSError %r, "
                    "falling back to Files API", api_path, exc,
                )
            else:
                logger.debug(
                    "read via kernel mount: %r -> %d bytes (pos=%d, n=%s)",
                    api_path, len(data), pos, "EOF" if n < 0 else n,
                )
                if not self._stat_cached:
                    try:
                        st = os.stat(api_path)
                        self._persist_stat_cache(
                            stats=IOStats(
                                size=int(st.st_size),
                                kind=IOKind.FILE,
                                mtime=st.st_mtime,
                            )
                        )
                    except OSError:
                        pass
                return memoryview(data)
        api_path = self.api_path
        # Critical: a bounded / offset read (``n > 0`` or ``pos > 0``)
        # asks for a *slice*, so send an HTTP ``Range`` header and let
        # the server return just those bytes (206) instead of pulling the
        # whole object and slicing locally — random-seek reads (Parquet
        # footers, column chunks, page-cache fills) would otherwise
        # transfer the entire file per call. The whole-file case
        # (``pos == 0`` and ``n < 0``) sends no Range so the body streams
        # resumably from byte zero.
        if n > 0:
            range_header = f"bytes={pos}-{pos + n - 1}"
        elif pos > 0:
            range_header = f"bytes={pos}-"
        else:
            range_header = None
        try:
            resp = self._fs_request(
                "GET", "files", api_path,
                range_header=range_header, preload_content=False,
            )
        except Exception as exc:
            if _looks_like_not_found(exc):
                raise FileNotFoundError(self.full_path()) from exc
            raise
        if resp.status == 404:
            raise FileNotFoundError(self.full_path())
        self._raise_for_files_status(resp, api_path)

        data = resp.data
        # The server honoured the Range (206) → ``data`` is already the
        # requested slice; the total object size comes from
        # ``Content-Range: bytes <s>-<e>/<total>``. A 200 means the
        # server ignored the Range (or none was sent) → ``data`` is the
        # whole object and ``len(data)`` IS the total, so slice locally.
        total_size = len(data)
        if range_header is not None and resp.status == 206:
            content_range = _header(resp.headers, "Content-Range") or ""
            tail = content_range.rsplit("/", 1)[-1].strip() if "/" in content_range else ""
            if tail.isdigit():
                total_size = int(tail)
        else:
            if pos:
                data = data[pos:]
            if n > 0:
                data = data[:n]
        logger.debug(
            "Read volume file %r -> %d bytes (pos=%d n=%s status=%d total=%d)",
            self, len(data), pos, "EOF" if n < 0 else n, resp.status, total_size,
        )

        ct = _header(resp.headers, "Content-Type")
        media_type = MediaType.from_(ct, default=None) if ct else None
        lm = _header(resp.headers, "Last-Modified")
        parsed = parse_http_date(lm) if lm else None
        mtime = parsed.timestamp() if parsed else time.time()
        if not self._stat_cached:
            self._persist_stat_cache(
                stats=IOStats(
                    size=total_size,
                    kind=IOKind.FILE,
                    mtime=mtime,
                    media_type=media_type,
                )
            )
        else:
            self._stat_cached.size = total_size
            self._stat_cached.mtime = mtime
            if media_type is not None and self._stat_cached.media_type is None:
                self._stat_cached.media_type = media_type
            # Re-stamp the TTL — this read observed the freshest total
            # size; the entry should outlive the original probe window.
            self._persist_stat_cache(self._stat_cached)

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
        which does a single ``PUT /files``.
        ``size>=0`` (capped read) or non-zero ``offset``
        fall back to the chunked base path because the API can't
        splice at a range. ``batch_size`` only matters for that
        fallback — the atomic upload doesn't chunk.
        """
        if offset != 0 or size >= 0:
            return super()._write_stream(src, offset=offset, size=size, **kwargs)
        return self._upload(src)

    def _upload(self, content: Any) -> int:
        """Upload *content* via ``PUT /api/2.0/fs/files`` (overwrite).

        Accepts either a bytes-like payload or a binary stream. The
        whole object is read into bytes up front so the
        :class:`HTTPSession`'s transient-error retry and the
        parent-recovery re-try both replay the identical body from
        offset zero — the bytes are inherently seekable, so there's no
        stream rewind dance.

        Returns the uploaded byte count.
        """
        size = len(content) if hasattr(content, "__len__") else -1
        logger.debug(
            "Uploading volume file %r (%s bytes)",
            self,
            size if size >= 0 else "?",
        )
        api_path = self.api_path
        # Cluster fast path — write straight to the kernel mount.
        if _local_mount_available():
            parent = os.path.dirname(api_path)
            if parent and not os.path.isdir(parent):
                logger.debug(
                    "upload via kernel mount: auto-creating parent %r",
                    parent,
                )
                os.makedirs(parent, exist_ok=True)
            if hasattr(content, "seek"):
                try:
                    pos = content.tell()
                    if size == -1:
                        content.seek(0, io.SEEK_END)
                        size = content.tell()
                    content.seek(pos, io.SEEK_SET)
                except Exception:
                    pos = 0
                bytes_written = 0
                with open(api_path, "wb") as fh:
                    while True:
                        chunk = content.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        bytes_written += len(chunk)
                if size == -1:
                    size = bytes_written
            else:
                payload = bytes(content)
                size = len(payload)
                with open(api_path, "wb") as fh:
                    fh.write(payload)
            logger.debug(
                "upload via kernel mount: %r -> %d bytes", api_path, size,
            )
            self._persist_stat_cache(
                IOStats(kind=IOKind.FILE, size=int(max(size, 0)),
                        mtime=time.time())
            )
            return int(max(size, -1))
        # Off-cluster: PUT the whole object through the Files REST API.
        # Read the source fully into bytes so transient / parent-recovery
        # retries replay the same body verbatim.
        if hasattr(content, "read"):
            if hasattr(content, "seek"):
                try:
                    content.seek(0)
                except Exception:
                    pass
            payload = content.read()
            if isinstance(payload, str):
                payload = payload.encode()
            else:
                payload = bytes(payload)
        else:
            payload = bytes(content)
        size = len(payload)

        # Large objects: run concurrent presigned parts (and lift the
        # 5 GiB single-PUT cap). Falls back to a single PUT when the
        # workspace doesn't support multipart.
        if size >= self.MULTIPART_MIN_SIZE and self._try_multipart_upload(api_path, payload):
            pass
        else:
            def _do_upload() -> None:
                resp = self._fs_request(
                    "PUT",
                    "files",
                    api_path,
                    params={"overwrite": "true"},
                    body=payload,
                )
                self._raise_for_files_status(resp, api_path)

            self._call_ensuring_parents(_do_upload)
            logger.info("Uploaded volume file %r (size=%d)", self, size)
        self._persist_stat_cache(
            IOStats(
                size=size,
                kind=IOKind.FILE,
                mtime=time.time(),
                media_type=self.media_type,
            )
        )
        self._cache_after_upload(payload, size)
        return size

    def _upload_stream(self, source: "Any") -> int:
        """Stream *source* (a seekable, sized Holder) to the Files API as the
        whole object, in bounded chunks — the memory-bounded counterpart to
        :meth:`_upload` driven by the Arrow/Parquet write path.

        The single PUT hands the session the Holder itself: its ``iter_mv``
        streams the body off disk window-by-window, and re-reads from byte 0
        on a transient retry (positional reads, no consumed cursor). Skips the
        read-after-write page cache that :meth:`_upload` populates — caching
        would re-materialise the whole body, defeating the bound; later reads
        re-fetch via Range instead.
        """
        size = int(source.size)
        api_path = self.api_path
        logger.debug("Streaming volume file %r (%d bytes)", self, size)
        # Cluster fast path — copy straight to the kernel mount, bounded.
        if _local_mount_available():
            parent = os.path.dirname(api_path)
            if parent and not os.path.isdir(parent):
                os.makedirs(parent, exist_ok=True)
            with open(api_path, "wb") as fh:
                for chunk in source.iter_mv():
                    fh.write(chunk)
            self._persist_stat_cache(
                IOStats(kind=IOKind.FILE, size=size, mtime=time.time())
            )
            self._note_streamed_upload(size)
            return size
        # Past the single-PUT ceiling, presigned multipart needs the bytes;
        # materialise only in that (already very large) case.
        if size >= self.MULTIPART_MIN_SIZE and self._try_multipart_upload(
            api_path, source.read_bytes()
        ):
            pass
        else:
            def _do_upload() -> None:
                resp = self._fs_request(
                    "PUT", "files", api_path,
                    params={"overwrite": "true"}, body=source,
                )
                self._raise_for_files_status(resp, api_path)

            self._call_ensuring_parents(_do_upload)
            logger.info("Streamed volume file %r (size=%d)", self, size)
        self._persist_stat_cache(
            IOStats(
                size=size,
                kind=IOKind.FILE,
                mtime=time.time(),
                media_type=self.media_type,
            )
        )
        self._note_streamed_upload(size)
        return size

    def _clear(self) -> None:
        self._remove_file(missing_ok=True, wait=WaitingConfig.from_(True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _header(headers, name: str) -> "str | None":
    """Case-insensitive single-header lookup.

    :class:`HTTPHeaders.get` is case-sensitive, but a server can return
    any casing — scan once for a case-folded match after the cheap
    direct hit misses.
    """
    if headers is None:
        return None
    value = headers.get(name)
    if value is not None:
        return value
    target = name.lower()
    for key in headers.keys():
        if str(key).lower() == target:
            return headers.get(key)
    return None


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
