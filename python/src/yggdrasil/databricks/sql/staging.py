from __future__ import annotations

import logging
import os
import re
import threading
import time
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


@dataclass(frozen=True, slots=True)
class StagingPath:
    """Temporary parquet staging path used by ``SQLEngine.arrow_insert_into``.

    Path format:
        /Volumes/{catalog}/{schema}/tmp/.sql/{catalog}/{schema}/{table}/
        tmp-{start_ts}-{end_ts}-{token}.parquet

    ``end_ts - start_ts`` is always clamped to at most 3600 seconds.

    ``owned``
        ``True`` when the :class:`StagingPath` was created by
        :meth:`for_table` and so the engine is responsible for removing the
        file afterwards.  ``False`` (the default and the value used by
        :meth:`from_volume`) means the path was supplied by the caller;
        :meth:`cleanup` becomes a best-effort no-op so user-owned volume
        files are never deleted behind the caller's back.
    """

    path: VolumePath
    catalog_name: str
    schema_name: str
    table_name: str
    start_ts: int
    end_ts: int
    token: str
    owned: bool = False

    _shutdown_hook: Any = field(default=None, init=False, repr=False, compare=False)
    last_read_frame: Any = field(default=None, init=False, repr=False, compare=False)

    _EXPIRED_SWEEP_DONE: ClassVar[set[str]] = set()
    _EXPIRED_SWEEP_LOCK: ClassVar[threading.Lock] = threading.Lock()
    _TMP_FILE_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^tmp-(\d+)-(\d+)-[0-9a-fA-F]+\.parquet$"
    )

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

        with cls._EXPIRED_SWEEP_LOCK:
            if root_str in cls._EXPIRED_SWEEP_DONE:
                return
            cls._EXPIRED_SWEEP_DONE.add(root_str)

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

    def write_table(
        self,
        data: Any,
        *,
        cast_options: Optional["CastOptions"] = None,
        read_columns: Optional[list[str]] = None,
    ) -> "StagingPath":
        """Serialize ``data`` to Parquet and upload it to this staging path.

        Ensures the parent directory exists, converts ``data`` through
        :class:`yggdrasil.io.buffer.media_io.MediaIO` using Parquet, and
        writes the resulting bytes to ``self.path``.

        When ``read_columns`` is set, the in-memory Parquet buffer is
        projected back as a Polars frame (reading only the requested
        columns) and stashed on ``self.last_read_frame`` so callers —
        e.g. the MERGE-narrowing path in
        :meth:`~yggdrasil.databricks.sql.engine.SQLEngine.arrow_insert_into`
        — can compute distinct values without an extra round trip to
        the staged file. The previous value is cleared on every write
        so a stale frame is never observable.

        Args:
            data:
                Any tabular input supported by ``MediaIO.write_table``.
            cast_options:
                Optional :class:`CastOptions` forwarded to the Parquet writer.
            read_columns:
                Optional subset of columns to project back off the
                serialized Parquet buffer. When supplied the resulting
                :class:`polars.DataFrame` is stored on
                ``self.last_read_frame``.

        Returns:
            ``self`` so calls can be chained.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        options = ParquetOptions(cast=cast_options) if cast_options else ParquetOptions()

        # Clear any frame from a previous write so stale data can't leak.
        object.__setattr__(self, "last_read_frame", None)

        with BytesIO() as buffer:
            mio = MediaIO.make(buffer=buffer, media=MediaTypes.PARQUET)
            mio.write_table(data, options=options)
            mio.holder.seek(0)
            self.path.write_bytes(mio.holder.memoryview())

        return self

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
        owned: bool = True
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

        Useful when the caller already has a volume path they want the engine
        to insert from: the engine treats the path as read-only (``owned=False``),
        so :meth:`cleanup` will leave the underlying file alone.

        ``catalog_name`` / ``schema_name`` / ``table_name`` are parsed from
        the volume path's ``sql_volume_or_table_parts`` decomposition; when
        the path does not correspond to a volume root (edge case), they fall
        back to empty strings.
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
            catalog_name=catalog,
            schema_name=schema,
            table_name=volume_name,
            start_ts=start,
            end_ts=end,
            token=rnd,
            owned=owned,
        )

    def register_shutdown_cleanup(self) -> None:
        """Register a best-effort process-exit cleanup callback."""
        if self._shutdown_hook is not None:
            return

        def _cleanup() -> None:
            self.cleanup(allow_not_found=True, unregister=False)

        try:
            hook = yg_shutdown.register(_cleanup)
            object.__setattr__(self, "_shutdown_hook", hook)
        except Exception:
            LOGGER.debug(
                "Failed to register staging shutdown cleanup for %s",
                self.path,
                exc_info=True,
            )

    def unregister_shutdown_cleanup(self) -> None:
        hook = self._shutdown_hook
        object.__setattr__(self, "_shutdown_hook", None)
        if hook is None:
            return

        try:
            yg_shutdown.unregister(hook)
        except Exception:
            LOGGER.debug(
                "Failed to unregister staging shutdown cleanup for %s",
                self.path,
                exc_info=True,
            )

    def cleanup(self, *, allow_not_found: bool = True, unregister: bool = True) -> None:
        """Best-effort staging file cleanup.

        Only engine-owned paths (``owned=True``) are deleted; user-supplied
        paths created via :meth:`from_volume` are left untouched so the
        caller retains control of their own data.  Shutdown hook
        unregistration runs regardless, because a ``from_volume`` path that
        was opted into shutdown cleanup still needs its hook cleared.
        """
        if unregister:
            self.unregister_shutdown_cleanup()

        if not self.owned:
            return

        try:
            self.path.remove(recursive=True, allow_not_found=allow_not_found)
        except Exception:
            LOGGER.debug("Failed to remove staging path %s", self.path, exc_info=True)