from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.databricks.fs.path import DatabricksPath, VolumePath
from yggdrasil.environ import shutdown as yg_shutdown
from yggdrasil.io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.parquet_io import ParquetOptions
from yggdrasil.io.enums.media_type import MediaTypes

if TYPE_CHECKING:
    from yggdrasil.data.cast import CastOptions

__all__ = ["StagingPath"]


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strong-reference keepalive for staging paths with a live shutdown hook.
#
# The shutdown registry holds bound methods via weakref.WeakMethod. For a
# StagingPath that the caller still owns, the registration survives until
# cleanup() is called. For a StagingPath that the caller dropped on the
# floor (e.g. after a crash in the caller), we still want the cleanup hook
# to fire at process exit so the temporary parquet file doesn't leak on the
# volume. This set closes that gap by holding a strong reference for as
# long as the hook is registered.
# ---------------------------------------------------------------------------

_LIVE_STAGING: "set[StagingPath]" = set()
_LIVE_STAGING_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Bounded expired-sweep deduper.
#
# We only want to sweep expired files once per (catalog, schema, table) per
# process. In a long-running worker touching many tables, an unbounded set
# would leak memory. This is a simple FIFO-bounded OrderedDict used as a
# set; 4096 entries is plenty for any sensible workload.
# ---------------------------------------------------------------------------

_SWEEP_CAP = 4096


class _BoundedSeen:
    """FIFO-bounded 'have we seen this key?' tracker, thread-safe."""

    __slots__ = ("_lock", "_seen", "_cap")

    def __init__(self, cap: int) -> None:
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._cap = cap

    def add_if_new(self, key: str) -> bool:
        """Return True if ``key`` was newly added, False if already present."""
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return False
            self._seen[key] = None
            while len(self._seen) > self._cap:
                self._seen.popitem(last=False)
            return True


# ---------------------------------------------------------------------------
# StagingPath
# ---------------------------------------------------------------------------

@dataclass(slots=True, eq=False)
class StagingPath:
    """Temporary parquet staging path used by ``SQLEngine.arrow_insert_into``.

    Path format::

        /Volumes/{catalog}/{schema}/tmp/.sql/{catalog}/{schema}/{table}/
        tmp-{start_ts}-{end_ts}-{token}.parquet

    ``end_ts - start_ts`` is clamped to at most 3600 seconds.

    ``owned``
        ``True`` when the :class:`StagingPath` was created by
        :meth:`for_table`, so the engine is responsible for removing the file
        afterwards. ``False`` (the default for :meth:`from_volume`) means the
        path was supplied by the caller; :meth:`cleanup` becomes a best-effort
        no-op so user-owned volume files are never deleted behind the caller's
        back.

    Equality is identity-based (``eq=False``). The previous ``frozen=True``
    was a lie — several fields were mutated through ``object.__setattr__``,
    so freezing only served to confuse readers.
    """

    path: VolumePath
    catalog_name: str
    schema_name: str
    table_name: str
    start_ts: int
    end_ts: int
    token: str
    owned: bool = False

    last_read_frame: Any = field(default=None, repr=False, compare=False)

    # True iff this instance is currently registered with yg_shutdown.
    _shutdown_registered: bool = field(
        default=False, init=False, repr=False, compare=False,
    )

    _EXPIRED_SWEEP_SEEN: ClassVar[_BoundedSeen] = _BoundedSeen(cap=_SWEEP_CAP)
    _TMP_FILE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^tmp-(\d+)-(\d+)-[0-9a-fA-F]+\.parquet$"
    )

    # ------------------------------------------------------------------
    # Hashing (explicit since eq=False gives us identity hashing already,
    # but we want a stable content-based hash for any set/dict that wants
    # one)
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        # Stable over the object's lifetime: path + token uniquely identify
        # a staging file, and neither is mutated after construction.
        return hash((str(self.path), self.token))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_part(value: str) -> str:
        text = str(value).strip().strip("`")
        return text.replace("/", "_")

    @staticmethod
    def _lifetime_seconds(max_lifetime: Optional[float]) -> int:
        if max_lifetime is None:
            return 3600
        try:
            value = int(max_lifetime)
        except (TypeError, ValueError):
            value = 3600
        if value <= 0:
            return 1
        return min(value, 3600)

    # ------------------------------------------------------------------
    # Expired-file sweep (synchronous, one-shot per table per process)
    # ------------------------------------------------------------------

    @classmethod
    def _cleanup_expired_once(
        cls,
        *,
        client,
        catalog_name: str,
        schema_name: str,
        table_name: str,
    ) -> None:
        root_str = (
            f"/Volumes/{catalog_name}/{schema_name}/tmp/.sql/"
            f"{catalog_name}/{schema_name}/{table_name}"
        )

        if not cls._EXPIRED_SWEEP_SEEN.add_if_new(root_str):
            return

        root = DatabricksPath.parse(root_str, client=client)
        now_ts = int(time.time())

        try:
            for candidate in root.ls(recursive=True, allow_not_found=True):
                match = cls._TMP_FILE_RE.match(candidate.name)
                if match is None:
                    continue

                try:
                    end_ts = int(match.group(2))
                except Exception:
                    continue

                if end_ts >= now_ts:
                    continue

                try:
                    candidate.remove(recursive=True, allow_not_found=True)
                except Exception:
                    LOGGER.debug(
                        "Failed to remove expired staging file %s",
                        candidate,
                        exc_info=True,
                    )
        except Exception:
            LOGGER.debug(
                "Failed to sweep expired SQL staging files under %s",
                root,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Data I/O
    # ------------------------------------------------------------------

    def write_table(
        self,
        data: Any,
        *,
        cast_options: Optional["CastOptions"] = None,
        read_columns: Optional[list[str]] = None,
    ) -> "StagingPath":
        """Serialize ``data`` to Parquet and upload it to this staging path.

        Ensures the parent directory exists, converts ``data`` through
        :class:`MediaIO` using Parquet, and writes the resulting bytes to
        ``self.path``.

        ``last_read_frame`` is cleared on every write so stale data from a
        previous write can't leak downstream. The ``read_columns`` argument
        is accepted for API compatibility but projection is left to callers
        that actually need it.

        Returns ``self`` so calls can be chained.
        """
        del read_columns  # accepted for API compatibility; projection is caller-side

        self.path.parent.mkdir(parents=True, exist_ok=True)
        options = ParquetOptions(cast=cast_options) if cast_options else ParquetOptions()

        # Clear any frame from a previous write so stale data can't leak.
        self.last_read_frame = None

        with BytesIO() as buffer:
            mio = MediaIO.make(buffer=buffer, media=MediaTypes.PARQUET)
            mio.write_table(data, options=options)
            mio.holder.seek(0)
            # Convert memoryview -> bytes to avoid any write_bytes
            # implementations that don't accept a buffer protocol object.
            payload = mio.holder.memoryview()
            if not isinstance(payload, (bytes, bytearray)):
                payload = bytes(payload)
            self.path.write_bytes(payload)

        return self

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def for_table(
        cls,
        *,
        client,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        max_lifetime: float | None = 3600,
        start_ts: int | None = None,
        token: str | None = None,
        owned: bool = True,
    ) -> "StagingPath":
        """Build an engine-owned staging path under the ``tmp/.sql`` tree."""
        catalog = cls._clean_part(catalog_name)
        schema = cls._clean_part(schema_name)
        table = cls._clean_part(table_name)

        cls._cleanup_expired_once(
            client=client,
            catalog_name=catalog,
            schema_name=schema,
            table_name=table,
        )

        start = int(time.time() if start_ts is None else start_ts)
        lifetime = cls._lifetime_seconds(max_lifetime)
        end = start + lifetime
        rnd = token or os.urandom(4).hex()

        base = f"/Volumes/{catalog}/{schema}/tmp/.sql/{catalog}/{schema}/{table}"
        name = f"tmp-{start}-{end}-{rnd}.parquet"
        path = DatabricksPath.parse(f"{base}/{name}", client=client)
        if not isinstance(path, VolumePath):
            raise TypeError(
                f"StagingPath.for_table expected a VolumePath, got {type(path).__name__}"
            )

        return cls(
            path=path,
            catalog_name=catalog,
            schema_name=schema,
            table_name=table,
            start_ts=start,
            end_ts=end,
            token=rnd,
            owned=owned,
        )

    @classmethod
    def from_volume(
        cls,
        path: "VolumePath | str",
        *,
        client=None,
        owned: bool = False,
        start_ts: int | None = None,
        max_lifetime: float | None = None,
        token: str | None = None,
    ) -> "StagingPath":
        """Wrap an existing :class:`VolumePath` as a non-owned staging path.

        The engine treats the path as read-only (``owned=False``), so
        :meth:`cleanup` will leave the underlying file alone.
        """
        if isinstance(path, str):
            if client is None:
                raise ValueError(
                    "StagingPath.from_volume(str_path) requires ``client`` so "
                    "the path can be resolved against a workspace."
                )
            parsed = DatabricksPath.parse(path, client=client)
        else:
            parsed = path

        if not isinstance(parsed, VolumePath):
            raise TypeError(
                f"StagingPath.from_volume expected a VolumePath, "
                f"got {type(parsed).__name__}"
            )

        try:
            catalog, schema, volume_name, _rel = parsed.sql_volume_or_table_parts()
        except Exception:
            catalog, schema, volume_name = "", "", ""

        start = int(time.time() if start_ts is None else start_ts)
        end = start + cls._lifetime_seconds(max_lifetime)
        rnd = token or os.urandom(4).hex()

        return cls(
            path=parsed,
            catalog_name=catalog or "",
            schema_name=schema or "",
            table_name=volume_name or "",
            start_ts=start,
            end_ts=end,
            token=rnd,
            owned=owned,
        )

    # ------------------------------------------------------------------
    # Shutdown-hook integration
    # ------------------------------------------------------------------

    def register_shutdown_cleanup(self) -> None:
        """Register a best-effort process-exit cleanup callback.

        No-op for non-owned staging paths: those belong to the caller, and
        cleanup would silently delete someone else's volume file.
        """
        if self._shutdown_registered or not self.owned:
            return

        try:
            yg_shutdown.register(self._unsafe_cleanup)
        except Exception:
            LOGGER.debug(
                "Failed to register staging shutdown cleanup for %s",
                self.path,
                exc_info=True,
            )
            return

        with _LIVE_STAGING_LOCK:
            _LIVE_STAGING.add(self)
        self._shutdown_registered = True

    def unregister_shutdown_cleanup(self) -> None:
        """Remove the shutdown hook. Idempotent."""
        if not self._shutdown_registered:
            with _LIVE_STAGING_LOCK:
                _LIVE_STAGING.discard(self)
            return

        self._shutdown_registered = False
        try:
            yg_shutdown.unregister(self._unsafe_cleanup)
        except Exception:
            LOGGER.debug(
                "Failed to unregister staging shutdown cleanup for %s",
                self.path,
                exc_info=True,
            )
        finally:
            with _LIVE_STAGING_LOCK:
                _LIVE_STAGING.discard(self)

    def _unsafe_cleanup(self) -> None:
        """Shutdown-safe cleanup used as the atexit / signal callback.

        Swallows BaseException (shutdown hooks must never propagate) and
        defensively guards the logger because logging itself may already be
        torn down during late-stage shutdown.
        """
        try:
            self.cleanup(allow_not_found=True, unregister=False)
        except BaseException:  # noqa: BLE001 — shutdown hook must not raise
            try:
                LOGGER.debug(
                    "Shutdown cleanup of staging path %s failed",
                    self.path,
                    exc_info=True,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, *, allow_not_found: bool = True, unregister: bool = True) -> None:
        """Best-effort staging-file cleanup.

        Only engine-owned paths (``owned=True``) are deleted; user-supplied
        paths created via :meth:`from_volume` are left untouched so the
        caller retains control of their own data. Shutdown-hook
        unregistration runs regardless.
        """
        if unregister:
            self.unregister_shutdown_cleanup()

        if not self.owned:
            return

        try:
            self.path.remove(recursive=True, allow_not_found=allow_not_found)
        except Exception:
            LOGGER.debug("Failed to remove staging path %s", self.path, exc_info=True)

    # ------------------------------------------------------------------
    # Context-manager convenience (makes the owned/cleanup contract obvious)
    # ------------------------------------------------------------------

    def __enter__(self) -> "StagingPath":
        if self.owned:
            self.register_shutdown_cleanup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup(allow_not_found=True, unregister=True)