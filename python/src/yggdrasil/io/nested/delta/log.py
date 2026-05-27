"""Delta transaction-log parser.

Caching: directory listing and ``_last_checkpoint`` are cached per
instance. Commit JSON content is cached in a module-level
``ExpiringDict`` (60 s TTL, 1024 max, skip > 1 MiB) shared across
instances on the same table root.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable, Iterator, List, Mapping, Optional, Tuple

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME, LOG_DIR_NAME, SIDECARS_DIR_NAME,
    format_commit_name, version_from_log_name,
)
from yggdrasil.io.nested.delta.protocol import DeltaAction, parse_action

if TYPE_CHECKING:
    from yggdrasil.path import Path

__all__ = ["DeltaLog", "LogSegment"]

_VERSION_FMT = "{:020d}"
_CONTENT_CACHE_MAX_BYTES = 1 * 1024 * 1024

_content_cache: ExpiringDict[str, bytes] = ExpiringDict(
    default_ttl=60.0, max_size=1024,
)


@dataclasses.dataclass(slots=True)
class LogSegment:
    version: int
    checkpoint_version: int
    checkpoint_files: Tuple["Path", ...]
    commit_files: Tuple["Path", ...]
    checkpoint_kind: str = "none"

    def is_initial(self) -> bool:
        return self.checkpoint_version < 0 and not self.commit_files


class DeltaLog:
    __slots__ = ("table_root", "log_path", "_last_checkpoint", "_listing")

    def __init__(self, table_root: "Path") -> None:
        self.table_root = table_root
        self.log_path = table_root / LOG_DIR_NAME
        self._last_checkpoint: "Optional[Mapping[str, object]]" = None
        self._listing: "Optional[Tuple[str, ...]]" = None

    def invalidate(self) -> None:
        self._last_checkpoint = None
        self._listing = None

    def extend_listing(self, *names: str) -> None:
        if self._listing is None:
            return
        current = set(self._listing)
        added = [n for n in names if n not in current]
        if added:
            self._listing = tuple(sorted(current | set(added)))

    def _list_log_dir(self) -> "Tuple[str, ...]":
        if self._listing is not None:
            return self._listing
        try:
            entries = list(self.log_path.iterdir())
        except FileNotFoundError:
            self._listing = ()
            return self._listing
        names = sorted(e.name for e in entries if e.name and not e.name.startswith("."))
        self._listing = tuple(names)
        return self._listing

    def read_last_checkpoint(self) -> "Optional[Mapping[str, object]]":
        if self._last_checkpoint is not None:
            return self._last_checkpoint or None
        try:
            payload = ygg_json.loads(_read_small_file(self.log_path / LAST_CHECKPOINT_NAME))
        except (FileNotFoundError, Exception):
            self._last_checkpoint = {}
            return None
        self._last_checkpoint = payload or {}
        return self._last_checkpoint or None

    def _is_commit_json(self, name: str) -> bool:
        return name.endswith(".json") and not name.startswith("_") and ".checkpoint." not in name

    def latest_version(self) -> int:
        max_v = -1
        for name in self._list_log_dir():
            if self._is_commit_json(name):
                v = version_from_log_name(name)
                if v is not None and v > max_v:
                    max_v = v
        return max_v

    def segment(self, version: "Optional[int]" = None) -> LogSegment:
        listing = self._list_log_dir()
        last_ck = self.read_last_checkpoint()
        ck_hint = int(last_ck["version"]) if last_ck and "version" in last_ck else -1

        all_commits = sorted(
            (v, name) for name in listing
            if self._is_commit_json(name)
            for v in [version_from_log_name(name)] if v is not None
        )
        target = all_commits[-1][0] if (version is None and all_commits) else (int(version) if version is not None else -1)

        # Resolve checkpoint
        ck_version, ck_files, ck_kind = -1, (), "none"

        # Try hint from _last_checkpoint first
        if ck_hint >= 0 and ck_hint <= target and last_ck:
            files = self._checkpoint_files(listing, ck_hint, last_ck)
            if files:
                ck_version, ck_files = ck_hint, files
                ck_kind = "v2" if last_ck.get("v2Checkpoint") else "v1"

        # Fall back to listing scan
        if ck_version < 0:
            ck_versions: dict[int, list[str]] = {}
            for name in listing:
                if ".checkpoint" not in name:
                    continue
                v = version_from_log_name(name)
                if v is not None and v <= target:
                    ck_versions.setdefault(v, []).append(name)
            if ck_versions:
                best = max(ck_versions)
                files = self._checkpoint_files(listing, best, None)
                if files:
                    ck_version, ck_files = best, files
                    prefix = f"{_VERSION_FMT.format(best)}.checkpoint"
                    ck_kind = "v2" if any(n.startswith(prefix) and n.endswith(".json") for n in ck_versions[best]) else "v1"

        commits = tuple(self.log_path / name for v, name in all_commits if ck_version < v <= target)
        return LogSegment(version=target, checkpoint_version=ck_version,
                          checkpoint_files=ck_files, commit_files=commits,
                          checkpoint_kind=ck_kind)

    def _checkpoint_files(self, listing: "Tuple[str, ...]", version: int,
                          last_ck: "Optional[Mapping[str, object]]") -> "Tuple[Path, ...]":
        prefix = f"{_VERSION_FMT.format(int(version))}.checkpoint"
        v1_single, v1_parts, v2_manifests = [], [], []
        for name in listing:
            if not name.startswith(prefix):
                continue
            if name == f"{prefix}.parquet":
                v1_single.append(name)
            elif name.endswith(".parquet") and ".checkpoint." in name:
                v1_parts.append(name)
            elif name.endswith(".json"):
                v2_manifests.append(name)

        if v2_manifests:
            # Read manifest to get sidecar paths
            try:
                raw = _read_small_file(self.log_path / v2_manifests[0]).decode("utf-8")
            except Exception:
                return ()
            sidecars = []
            for line in raw.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    sc = ygg_json.loads(line).get("sidecar")
                    if sc and sc.get("path"): sidecars.append(sc["path"])
                except Exception: continue
            if sidecars:
                sd = self.log_path / SIDECARS_DIR_NAME
                return tuple(sd / name for name in sidecars)
            return ()

        if v1_single:
            return (self.log_path / v1_single[0],)
        if v1_parts:
            return tuple(self.log_path / n for n in sorted(v1_parts))

        # Fallback: sidecars from _last_checkpoint
        if last_ck and last_ck.get("v2Checkpoint"):
            v2 = last_ck["v2Checkpoint"]
            sidecars = v2.get("sidecarFiles") or v2.get("sidecars") or []  # type: ignore[union-attr]
            if sidecars:
                return tuple(self.table_root / LOG_DIR_NAME / SIDECARS_DIR_NAME / s["path"] for s in sidecars)
        return ()

    # ==================================================================
    # Replay
    # ==================================================================

    def replay(self, segment: LogSegment) -> "Iterator[DeltaAction]":
        for raw in self._replay_raw(segment):
            action = parse_action(raw)
            if action is not None:
                yield action

    def replay_raw(self, segment: LogSegment) -> "Iterator[Mapping[str, object]]":
        yield from self._replay_raw(segment)

    def _replay_raw(self, segment: LogSegment) -> "Iterator[Mapping[str, object]]":
        if segment.checkpoint_files:
            import pyarrow.parquet as pq
            for path in segment.checkpoint_files:
                local = _local_str(path)
                try:
                    if local is not None:
                        table = pq.read_table(local)
                    else:
                        import io as _io
                        table = pq.read_table(_io.BytesIO(_read_file_uncached(path)))
                except FileNotFoundError:
                    continue
                cols = table.column_names
                if not cols or table.num_rows == 0:
                    continue
                materialised = [table.column(c).to_pylist() for c in cols]
                for row_idx in range(table.num_rows):
                    for col_idx, col in enumerate(cols):
                        val = materialised[col_idx][row_idx]
                        if val is not None:
                            yield {col: val}
                            break

        for commit in segment.commit_files:
            try:
                blob = _read_small_file(commit).decode("utf-8")
            except FileNotFoundError:
                continue
            for line in blob.splitlines():
                line = line.strip()
                if not line: continue
                try: yield ygg_json.loads(line)
                except Exception: continue


# ---------------------------------------------------------------------------
# File read helpers
# ---------------------------------------------------------------------------

def _cache_key(path: "Path") -> str:
    fn = getattr(path, "full_path", None)
    return fn() if callable(fn) else str(path)

def _read_small_file(path: "Path") -> bytes:
    key = _cache_key(path)
    cached = _content_cache.get(key)
    if cached is not None:
        return cached
    raw = _read_file_uncached(path)
    if len(raw) <= _CONTENT_CACHE_MAX_BYTES:
        _content_cache[key] = raw
    return raw

def _read_file_uncached(path: "Path") -> bytes:
    with path.open("rb") as bio:
        return bio.read()

def _local_str(path: "Path") -> "Optional[str]":
    if not getattr(path, "is_local_path", False):
        return None
    fn = getattr(path, "full_path", None)
    return fn() if callable(fn) else None
