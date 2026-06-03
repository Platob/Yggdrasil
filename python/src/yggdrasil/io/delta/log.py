"""Delta transaction-log parser with content caching."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable, Iterator, List, Mapping, Optional, Tuple

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.delta._names import (
    LAST_CHECKPOINT_NAME, LOG_DIR_NAME, SIDECARS_DIR_NAME,
    format_commit_name, version_from_log_name,
)
from yggdrasil.io.delta.protocol import DeltaAction, parse_action

if TYPE_CHECKING:
    from yggdrasil.path import Path

__all__ = ["DeltaLog", "LogSegment"]

_VERSION_FMT = "{:020d}"
_CONTENT_CACHE_MAX_BYTES = 1 * 1024 * 1024
_content_cache: ExpiringDict[str, bytes] = ExpiringDict(default_ttl=60.0, max_size=1024)


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
        if self._listing is None: return
        current = set(self._listing)
        added = [n for n in names if n not in current]
        if added:
            self._listing = tuple(sorted(current | set(added)))

    def _list_log_dir(self) -> "Tuple[str, ...]":
        if self._listing is not None: return self._listing
        try: entries = list(self.log_path.iterdir())
        except FileNotFoundError:
            self._listing = (); return self._listing
        self._listing = tuple(sorted(e.name for e in entries if e.name and not e.name.startswith(".")))
        return self._listing

    def read_last_checkpoint(self) -> "Optional[Mapping[str, object]]":
        if self._last_checkpoint is not None:
            return self._last_checkpoint or None
        try:
            payload = ygg_json.loads(_read_cached(self.log_path / LAST_CHECKPOINT_NAME))
        except Exception:
            self._last_checkpoint = {}; return None
        self._last_checkpoint = payload or {}
        return self._last_checkpoint or None

    def latest_version(self) -> int:
        max_v = -1
        for name in self._list_log_dir():
            if name.endswith(".json") and not name.startswith("_") and ".checkpoint." not in name:
                v = version_from_log_name(name)
                if v is not None and v > max_v: max_v = v
        return max_v

    def commits_after(self, base_version: int) -> "Tuple[Path, ...]":
        """Commit-JSON paths with version > *base_version*, ascending.

        Drives an *incremental* snapshot advance: apply just these commits on
        top of a cached snapshot, without re-reading the checkpoint or the
        prior commits."""
        pairs = []
        for name in self._list_log_dir():
            if not (name.endswith(".json") and not name.startswith("_") and ".checkpoint." not in name):
                continue
            v = version_from_log_name(name)
            if v is not None and v > base_version:
                pairs.append((v, name))
        pairs.sort(key=lambda t: t[0])
        return tuple(self.log_path / name for _v, name in pairs)


    def segment(self, version: "Optional[int]" = None) -> LogSegment:
        listing = self._list_log_dir()
        last_ck = self.read_last_checkpoint()
        ck_hint = int(last_ck["version"]) if last_ck and "version" in last_ck else -1

        all_commits = sorted(
            (v, name) for name in listing
            if name.endswith(".json") and not name.startswith("_") and ".checkpoint." not in name
            for v in [version_from_log_name(name)] if v is not None
        )
        target = (all_commits[-1][0] if all_commits else -1) if version is None else int(version)

        # Resolve best checkpoint <= target
        ck_version, ck_files, ck_kind = -1, (), "none"

        if ck_hint >= 0 and ck_hint <= target and last_ck:
            files = self._checkpoint_files(listing, ck_hint, last_ck)
            if files:
                ck_version, ck_files = ck_hint, files
                ck_kind = "v2" if last_ck.get("v2Checkpoint") else "v1"

        if ck_version < 0:
            ck_versions: dict[int, list[str]] = {}
            for name in listing:
                if ".checkpoint" not in name: continue
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
                          checkpoint_files=ck_files, commit_files=commits, checkpoint_kind=ck_kind)

    def _checkpoint_files(self, listing: "Tuple[str, ...]", version: int,
                          last_ck: "Optional[Mapping[str, object]]") -> "Tuple[Path, ...]":
        prefix = f"{_VERSION_FMT.format(int(version))}.checkpoint"
        v1_single, v1_parts, v2_manifests = [], [], []
        for name in listing:
            if not name.startswith(prefix): continue
            if name == f"{prefix}.parquet": v1_single.append(name)
            elif name.endswith(".parquet") and ".checkpoint." in name: v1_parts.append(name)
            elif name.endswith(".json"): v2_manifests.append(name)

        if v2_manifests:
            try: raw = _read_cached(self.log_path / v2_manifests[0]).decode("utf-8")
            except Exception: return ()
            sidecars = []
            for line in raw.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    sc = ygg_json.loads(line).get("sidecar")
                    if sc and sc.get("path"): sidecars.append(sc["path"])
                except Exception: continue
            return tuple(self.log_path / SIDECARS_DIR_NAME / n for n in sidecars) if sidecars else ()

        if v1_single: return (self.log_path / v1_single[0],)
        if v1_parts: return tuple(self.log_path / n for n in sorted(v1_parts))

        if last_ck and last_ck.get("v2Checkpoint"):
            v2 = last_ck["v2Checkpoint"]
            sidecars = v2.get("sidecarFiles") or v2.get("sidecars") or []  # type: ignore[union-attr]
            if sidecars:
                return tuple(self.table_root / LOG_DIR_NAME / SIDECARS_DIR_NAME / s["path"] for s in sidecars)
        return ()

    def replay(self, segment: LogSegment) -> "Iterator[DeltaAction]":
        for raw in self.replay_raw(segment):
            action = parse_action(raw)
            if action is not None:
                yield action

    def replay_raw(self, segment: LogSegment) -> "Iterator[Mapping[str, object]]":
        if segment.checkpoint_files:
            import pyarrow.parquet as pq
            for path in segment.checkpoint_files:
                try:
                    local = path.full_path() if getattr(path, "is_local_path", False) else None
                    if local:
                        table = pq.read_table(local)
                    else:
                        import io as _io
                        with path.open("rb") as bio:
                            table = pq.read_table(_io.BytesIO(bio.read()))
                except FileNotFoundError: continue
                cols = table.column_names
                if not cols or table.num_rows == 0: continue
                mat = [table.column(c).to_pylist() for c in cols]
                for ri in range(table.num_rows):
                    for ci, col in enumerate(cols):
                        if mat[ci][ri] is not None:
                            yield {col: mat[ci][ri]}; break

        for commit in segment.commit_files:
            try: blob = _read_cached(commit).decode("utf-8")
            except FileNotFoundError: continue
            for line in blob.splitlines():
                line = line.strip()
                if not line: continue
                try: yield ygg_json.loads(line)
                except Exception: continue


def _read_cached(path: "Path") -> bytes:
    fn = getattr(path, "full_path", None)
    key = fn() if callable(fn) else str(path)
    cached = _content_cache.get(key)
    if cached is not None: return cached
    with path.open("rb") as bio:
        raw = bio.read()
    if len(raw) <= _CONTENT_CACHE_MAX_BYTES:
        _content_cache[key] = raw
    return raw
