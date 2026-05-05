""":class:`VolumePath` — ``/Volumes/catalog/schema/volume/...`` via Files API.

Volumes carry a Unity Catalog hierarchy (catalog → schema → volume → path)
plus access control (grants), and the SQL engine uses Volumes as a
SQL-staging landing zone.  This class folds in:

- **UC hierarchy bring-up** — :meth:`_ensure_uc_hierarchy` creates
  catalog/schema/volume gaps when ``mkdir`` lands on a missing
  parent.  Probe-before-create at each level so the already-exists
  path costs one ``read`` instead of an unconditional ``create``.
- **Grants** — :meth:`grants`, :meth:`grant`, :meth:`revoke`,
  :meth:`set_grants` for principal-level UC privilege management.
- **SQL staging factory** — :meth:`staging_path` resolves the
  Databricks-specific ``tmp/.sql/<cat>/<sch>/<tbl>/`` parent and
  delegates to :meth:`Path.make_staging` for the cross-backend
  rate-limited sweep + TTL-encoded filename.
- **Volume metadata** — :meth:`volume_info`, :meth:`storage_location`,
  :meth:`temporary_credentials` for cloud-credential vending.
"""

from __future__ import annotations

import logging
import time
from typing import ClassVar, Optional, Tuple, IO

from databricks.sdk.errors import InternalError
from databricks.sdk.errors.platform import (
    BadRequest,
    NotFound,
    PermissionDenied,
    ResourceDoesNotExist,
)
from databricks.sdk.service.catalog import PathOperation, VolumeInfo, VolumeType

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums import MediaType, MediaTypes
from yggdrasil.io.path_stat import PathKind, PathStats
from yggdrasil.io.url import URL
from ._errors import (
    ALREADY_EXISTS_ERRORS,
    SDK_ERRORS,
    coerce_mtime,
    retry_sdk_call,
)
from .path import DatabricksPath
from .path_kind import DatabricksPathKind
from .volumes import get_volume_metadata, get_volume_status
from ..client import DatabricksClient

__all__ = ["VolumePath"]


LOGGER = logging.getLogger(__name__)


# ===========================================================================
# VolumePath
# ===========================================================================


class VolumePath(DatabricksPath):
    """Path under ``/Volumes/cat/sch/vol/...`` via the Files API."""

    scheme: ClassVar[str] = "dbfs+volumes"
    _NAMESPACE_PREFIX: ClassVar[str] = "/Volumes/"

    # Multipart-upload knobs.  Defaults match the Files API SDK defaults.
    _UPLOAD_PART_SIZE: ClassVar[Optional[int]] = None
    _UPLOAD_USE_PARALLEL: ClassVar[bool] = True
    _UPLOAD_PARALLELISM: ClassVar[Optional[int]] = None
    _UPLOAD_MAX_RETRIES: ClassVar[int] = 4

    # Page size for ``files.list_directory_contents`` (None = SDK default).
    _PAGE_SIZE: ClassVar[Optional[int]] = None

    # Per-instance volume metadata cache.  Lazily fetched on first
    # :meth:`volume_info` access.
    __slots__ = ("_volume_info",)

    def __init__(self, *args, **kwargs) -> None:
        self._volume_info: Optional[VolumeInfo] = None
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.url!r})"

    def __str__(self) -> str:
        return self.__fspath__()

    def __fspath__(self) -> str:
        return self.full_path()

    @property
    def kind(self) -> DatabricksPathKind:
        return DatabricksPathKind.VOLUME

    @property
    def explore_url(self) -> URL:
        c, s, v, _ = self.sql_volume_or_table_parts()
        return (
            self.client.base_url
            .joinpath(f"/explore/data/volumes/{c}/{s}/{v}")
            .add_query_item("volumePath", self.full_path())
        )

    # ==================================================================
    # Path rendering
    # ==================================================================

    def full_path(self) -> str:
        p = self.url.path.lstrip("/")
        return "/Volumes/" + p if p else "/Volumes"

    # ==================================================================
    # UC decomposition
    # ==================================================================

    def sql_volume_or_table_parts(self) -> Tuple[
        Optional[str], Optional[str], Optional[str], list,
    ]:
        """Split the URL path into ``(catalog, schema, volume, rest)``."""
        parts = self.url.parts
        return (
            parts[0] if len(parts) > 0 else None,
            parts[1] if len(parts) > 1 else None,
            parts[2] if len(parts) > 2 else None,
            list(parts[3:]),
        )

    def volume_info(self) -> Optional[VolumeInfo]:
        """Return UC :class:`VolumeInfo` for this path's volume."""
        if self._volume_info is None:
            cat, sch, vol, _ = self.sql_volume_or_table_parts()
            if cat and sch and vol:
                self._volume_info = get_volume_metadata(
                    sdk=self._sdk(), full_name=f"{cat}.{sch}.{vol}",
                )
        return self._volume_info

    def storage_location(self) -> str:
        """Cloud storage URL for this path's bytes (s3://, abfss://, ...)."""
        info = self.volume_info()
        if info is None:
            raise NotFound(f"Volume {self!r} not found.")
        _, _, _, rel = self.sql_volume_or_table_parts()
        base = info.storage_location.rstrip("/")
        return f"{base}/{'/'.join(rel)}" if rel else base

    def temporary_credentials(self, operation: Optional[PathOperation] = None):
        """Vend short-lived cloud credentials for this path's storage URL."""
        return (
            self._sdk().temporary_path_credentials
            .generate_temporary_path_credentials(
                url=self.storage_location(),
                operation=operation or PathOperation.PATH_READ,
            )
        )

    # ==================================================================
    # SQL-staging
    # ==================================================================
    #
    # The cross-backend rate-limited sweep + TTL-encoded filename
    # generation lives on :meth:`Path.make_staging`.  This class adds
    # the Databricks-specific bits:
    #
    # 1. Resolve the canonical staging parent under the catalog/schema:
    #    ``/Volumes/<cat>/<sch>/tmp/.sql/<cat>/<sch>/<tbl>/``
    # 2. Sanitize the catalog/schema/table name segments.
    # 3. Default the resource name to ``"default"`` when unspecified.
    #
    # Once the parent is built, the abstract ``make_staging`` does the
    # rest — sweep, mint, return a ``temporary=True`` path.

    @staticmethod
    def _staging_clean_part(value: str) -> str:
        """Strip backticks/whitespace and forbid ``/`` in path segments."""
        text = str(value).strip().strip("`")
        return text.replace("/", "_")

    @staticmethod
    def _staging_lifetime_seconds(max_lifetime: Optional[float]) -> int:
        """Clamp ``max_lifetime`` into ``[1, 3600]``, defaulting to 3600."""
        if max_lifetime is None:
            return 3600
        try:
            value = int(max_lifetime)
        except (TypeError, ValueError):
            return 3600
        if value <= 0:
            return 1
        return min(value, 3600)

    @classmethod
    def staging_path(
        cls,
        catalog_name: str,
        schema_name: str,
        resource_name: str | None = None,
        *,
        max_lifetime: Optional[float] = None,
        temporary: bool = True,
        client: DatabricksClient | None = None,
        media_type: MediaType | str = MediaTypes.PARQUET,
        sweep: bool = True,
    ) -> "VolumePath":
        """Mint a Databricks-style staging file under ``tmp/.sql/<cat>/<sch>/<tbl>/``.

        Convenience wrapper around :meth:`Path.make_staging` that fills
        in the UC-aware parent location and the TTL clamp.  All the
        cross-backend behavior — sweep rate-limit, TTL-encoded
        filename, ``temporary=True`` lifecycle — comes from the
        abstract base.

        Pass ``sweep=False`` to skip the parent-cleanup walk on this
        call (useful when the caller knows the bucket is clean or
        when running in a hot loop where every Files API call counts).
        ``temporary`` defaults to True; pass False to keep the file
        past process exit (paired with manual cleanup or external
        TTL sweepers).
        """
        cat = cls._staging_clean_part(catalog_name)
        sch = cls._staging_clean_part(schema_name)
        tbl = cls._staging_clean_part(resource_name or "default")
        ttl = cls._staging_lifetime_seconds(max_lifetime)
        client = client or DatabricksClient.current()

        # Build the staging parent — the directory the abstract base
        # will sweep + mint a child under.  Constructing through the
        # class constructor (not from_/from_url) lets us bind the client
        # in one shot.
        parent_path = (
            f"/{cat}/{sch}/tmp/.sql/{cat}/{sch}/{tbl}"
        )
        parent = cls(
            url=URL(scheme=cls.scheme, path=parent_path),
            client=client,
            temporary=False,  # the parent dir lingers; only the file goes
        )

        result = parent.make_staging(
            ttl=ttl,
            media_type=media_type,
            sweep=sweep,
        )

        # The abstract ``make_staging`` returns a path bound to the same
        # backend (via ``with_tmp_name`` -> ``_from_url`` -> ``type(self)``),
        # so it's already a ``VolumePath``.  But ``_from_url`` doesn't
        # propagate the ``client`` attribute that ``DatabricksPath``
        # carries, so set it explicitly.  And honour the caller's
        # ``temporary`` flag (default True; False for opt-out).
        if not temporary:
            result.as_persistent()
        result = result.with_client(client)
        return result

    # ==================================================================
    # SDK hooks
    # ==================================================================

    def _remote_download(self, allow_not_found: bool = False) -> BytesIO:
        """Pull the full object via ``sdk.files.download``.

        Drains the SDK response into a project :class:`BytesIO`
        (spill-capable) so multi-GB volume objects don't blow out
        RAM. Transient transport flakes are retried by the base
        ``read_bytes`` wrapper (the whole ``_remote_download`` is
        replayed on failure).
        """
        from ._errors import NOT_FOUND_ERRORS, SDK_ERRORS
        sdk = self._sdk()
        out = BytesIO()
        try:
            resp = sdk.files.download(self.full_path())
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise FileNotFoundError(self.full_path())
        except SDK_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise

        try:
            stream = resp.contents
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        except NOT_FOUND_ERRORS:
            if allow_not_found:
                out.seek(0)
                return out
            raise FileNotFoundError(self.full_path())

        out.seek(0)
        return out

    def write_stream(
        self,
        src: IO[bytes],
        *,
        batch_size: int = 4 * 1024 * 1024,
        parents: bool = True,
    ) -> int:
        """Push the full payload via ``sdk.files.upload``.

        Retries on both Files API errors (``SDK_ERRORS`` —
        ``NotFound`` / ``BadRequest`` / ``InternalError``, with a
        one-shot parent-mkdir on the first failure) and transport
        flakes (``TRANSIENT_ERRORS`` — ``requests.ReadTimeout``,
        ``ConnectionError``, 5xx/429/503/504). The loop runs up to
        five attempts with a 1/2/4/8/16s backoff and seeks the
        payload back to its original position before each retry so
        the next attempt replays the same bytes.
        """
        from ._errors import TRANSIENT_ERRORS

        sdk = self._sdk()
        full_path = self.full_path()

        bio = BytesIO.from_(src)
        try:
            start_pos = bio.tell()
        except Exception:
            start_pos = 0

        LOGGER.debug("Uploading %r bytes to %s", bio, self)

        checked_parent = False
        last_exc: Exception | None = None
        retryable = SDK_ERRORS + TRANSIENT_ERRORS
        max_tries = 5

        for itry in range(max_tries):
            if itry > 0:
                try:
                    bio.seek(start_pos)
                except Exception:
                    LOGGER.debug(
                        "Could not seek payload back to %s before retry %d "
                        "(buffer may not be seekable); continuing.",
                        start_pos, itry,
                    )
            try:
                sdk.files.upload(
                    full_path,
                    bio,
                    overwrite=True,
                    part_size=self._UPLOAD_PART_SIZE,
                    use_parallel=self._UPLOAD_USE_PARALLEL,
                    parallelism=self._UPLOAD_PARALLELISM,
                )
                break
            except retryable as exc:
                last_exc = exc
                if (
                    isinstance(exc, SDK_ERRORS)
                    and parents
                    and not checked_parent
                ):
                    self.parent.mkdir(parents=True, exist_ok=True)
                    checked_parent = True
                elif itry == max_tries - 1:
                    if isinstance(exc, SDK_ERRORS):
                        raise FileNotFoundError(full_path) from exc
                    raise
                LOGGER.info(
                    "Volume upload to %s failed (attempt %d/%d): %r — "
                    "retrying",
                    full_path, itry + 1, max_tries, exc,
                )
                time.sleep(min(2 ** itry, 16))
        else:
            raise FileNotFoundError(full_path) from last_exc

        LOGGER.info("Wrote %r bytes to %s", bio, self)

    def _remote_upload(self, payload: BytesIO) -> None:
        return self.write_stream(src=payload, parents=True)

    # ------------------------------------------------------------------
    # Filesystem metadata
    # ------------------------------------------------------------------

    def _stat(self) -> PathStats:
        is_file, is_dir, size, mtime = get_volume_status(
            sdk=self._sdk(),
            full_path=self.full_path(),
            check_file_first="." in self.name,
            raise_error=False,
        )
        if is_file is None and is_dir is None:
            return PathStats(kind=PathKind.MISSING, size=0, mtime=None)
        return PathStats(
            kind=PathKind.FILE if is_file else PathKind.DIRECTORY,
            size=int(size or 0),
            mtime=coerce_mtime(mtime),
        )

    def _ls(self, recursive=False, allow_not_found=True):
        cat, sch, vol, _ = self.sql_volume_or_table_parts()
        if not vol:
            yield from self._ls_uc_hierarchy(
                self._sdk(), cat, sch, recursive, allow_not_found,
            )
            return

        try:
            for info in self._sdk().files.list_directory_contents(
                self.full_path(), page_size=self._PAGE_SIZE,
            ):
                api_path = info.path
                url_path = (
                    api_path[len("/Volumes"):]
                    if api_path.startswith("/Volumes")
                    else api_path
                )
                child = VolumePath(
                    url=URL(scheme=self.scheme, host=self.url.host, path=url_path),
                    client=self._client,
                )
                if recursive and info.is_directory:
                    yield from child._ls(
                        recursive=True, allow_not_found=allow_not_found,
                    )
                else:
                    yield child
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _ls_uc_hierarchy(self, sdk, catalog, schema, recursive, allow_not_found):
        """List catalogs / schemas / volumes when not yet inside a volume."""
        try:
            if not catalog:
                items = [[i.name] for i in sdk.catalogs.list()]
            elif not schema:
                items = [
                    [catalog, i.name]
                    for i in sdk.schemas.list(catalog_name=catalog)
                ]
            else:
                items = [
                    [i.catalog_name, i.schema_name, i.name]
                    for i in sdk.volumes.list(
                        catalog_name=catalog, schema_name=schema,
                    )
                ]
            for pts in items:
                child = VolumePath(
                    url=URL(
                        scheme="volumes",
                        host=self.url.host,
                        path="/" + "/".join(pts),
                    ),
                    client=self._client,
                )
                if recursive:
                    yield from child._ls(
                        recursive=True, allow_not_found=allow_not_found,
                    )
                else:
                    yield child
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _mkdir(self, parents=True, exist_ok=True):
        """Create the directory; bring up missing UC parents on demand."""
        if exist_ok:
            try:
                if self.is_dir():
                    return
            except SDK_ERRORS:
                pass

        sdk = self._sdk()
        path = self.full_path()
        last_exc: Optional[Exception] = None
        sleep_time: float = 1.0

        for _ in range(3):
            try:
                sdk.files.create_directory(path)
                return
            except (BadRequest, NotFound, ResourceDoesNotExist) as exc:
                if not parents:
                    raise
                if "not exist" in str(exc):
                    self._ensure_uc_hierarchy(sdk=sdk, exist_ok=exist_ok)
                last_exc = exc
            except InternalError as exc:
                if "already exists" in str(exc):
                    return
                last_exc = exc
            except ALREADY_EXISTS_ERRORS:
                if not exist_ok:
                    raise
                return

            sleep_time = min(sleep_time * 2, 30)
            time.sleep(sleep_time)

        if last_exc is not None:
            raise last_exc
        raise InternalError(f"Failed to create directory {path!r}")

    # ==================================================================
    # UC hierarchy bring-up
    # ==================================================================

    def _ensure_uc_hierarchy(self, sdk=None, exist_ok: bool = True) -> None:
        cat, sch, vol, _ = self.sql_volume_or_table_parts()
        if not (cat and sch and vol):
            raise ValueError(
                f"VolumePath {self!r} is not deep enough to identify "
                "a (catalog, schema, volume) — cannot bring up "
                "the UC hierarchy."
            )
        sdk = sdk or self._sdk()
        tags = None
        self._ensure_catalog(sdk, cat, tags, exist_ok=exist_ok)
        self._ensure_schema(sdk, cat, sch, tags, exist_ok=exist_ok)
        self._ensure_volume(sdk, cat, sch, vol, tags, exist_ok=exist_ok)

    @staticmethod
    def _ensure_catalog(sdk, cat: str, tags, *, exist_ok: bool) -> None:
        try:
            sdk.catalogs.get(name=cat)
            return
        except (NotFound, ResourceDoesNotExist):
            pass
        except PermissionDenied:
            return

        try:
            sdk.catalogs.create(name=cat)
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        except (PermissionDenied, InternalError) as exc:
            if not exist_ok:
                raise
            LOGGER.warning(
                "Catalog %s appears missing and could not be created (%s). "
                "Continuing — the file operation that triggered this will "
                "surface the real error if access is broken.",
                cat, exc,
            )

    @staticmethod
    def _ensure_schema(sdk, cat: str, sch: str, tags, *, exist_ok: bool) -> None:
        full = f"{cat}.{sch}"
        try:
            sdk.schemas.get(full_name=full)
            return
        except (NotFound, ResourceDoesNotExist):
            pass
        except PermissionDenied:
            return

        try:
            sdk.schemas.create(name=sch, catalog_name=cat)
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        except (PermissionDenied, InternalError) as exc:
            if not exist_ok:
                raise
            LOGGER.warning(
                "Schema %s appears missing and could not be created (%s).",
                full, exc,
            )

    @staticmethod
    def _ensure_volume(
        sdk, cat: str, sch: str, vol: str, tags, *, exist_ok: bool,
    ) -> None:
        full_name = f"{cat}.{sch}.{vol}"
        try:
            sdk.volumes.read(name=full_name)
            return
        except (NotFound, ResourceDoesNotExist):
            pass
        except PermissionDenied:
            return

        try:
            sdk.volumes.create(
                catalog_name=cat,
                schema_name=sch,
                name=vol,
                volume_type=VolumeType.MANAGED,
            )
        except ALREADY_EXISTS_ERRORS:
            if not exist_ok:
                raise
        except (PermissionDenied, InternalError) as exc:
            if not exist_ok:
                raise
            LOGGER.warning(
                "Volume %s appears missing and could not be created (%s).",
                full_name, exc,
            )

    def _remove_file(self, allow_not_found=True):
        try:
            self._sdk().files.delete(self.full_path())
        except SDK_ERRORS:
            if not allow_not_found:
                raise

    def _remove_dir(self, recursive=True, allow_not_found=True, with_root=True):
        cat, sch, vol, rel = self.sql_volume_or_table_parts()
        path = self.full_path()
        try:
            if rel:
                try:
                    self._sdk().files.delete_directory(path)
                except SDK_ERRORS as exc:
                    if recursive and "directory is not empty" in str(exc):
                        for child in self.ls():
                            if child.is_file():
                                child._remove_file(True)
                            else:
                                child._remove_dir(True, True, True)
                        if with_root:
                            self._sdk().files.delete_directory(path)
                    elif not allow_not_found:
                        raise
            elif vol:
                try:
                    self._sdk().volumes.delete(f"{cat}.{sch}.{vol}")
                except SDK_ERRORS:
                    if not allow_not_found:
                        raise
            elif sch:
                try:
                    self._sdk().schemas.delete(f"{cat}.{sch}", force=True)
                except SDK_ERRORS:
                    if not allow_not_found:
                        raise
        finally:
            pass
