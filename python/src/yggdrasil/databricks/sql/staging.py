from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from yggdrasil.environ import shutdown as yg_shutdown

from ..fs.path import DatabricksPath

__all__ = ["StagingPath"]


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StagingPath:
    """Temporary parquet staging path used by ``SQLEngine.arrow_insert_into``.

    Path format:
        /Volumes/{catalog}/{schema}/tmp/.sql/{catalog}/{schema}/{table}/
        tmp-{start_ts}-{end_ts}-{token}.parquet

    ``end_ts - start_ts`` is always clamped to at most 3600 seconds.
    """

    path: DatabricksPath
    catalog_name: str
    schema_name: str
    table_name: str
    start_ts: int
    end_ts: int
    token: str

    _shutdown_hook: Any = field(default=None, init=False, repr=False, compare=False)

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

    @classmethod
    def for_table(
        cls,
        *,
        client,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        max_lifetime: Optional[float] = 3600,
        start_ts: Optional[int] = None,
        token: Optional[str] = None,
    ) -> "StagingPath":
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

        return cls(
            path=path,
            catalog_name=catalog,
            schema_name=schema,
            table_name=table,
            start_ts=start,
            end_ts=end,
            token=rnd,
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
        """Best-effort staging file cleanup."""
        if unregister:
            self.unregister_shutdown_cleanup()

        try:
            self.path.remove(recursive=True, allow_not_found=allow_not_found)
        except Exception:
            LOGGER.debug("Failed to remove staging path %s", self.path, exc_info=True)