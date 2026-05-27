"""Delta transaction-log parser.

The Delta log is a sequence of files under ``<table>/_delta_log``:

- ``00000000000000000000.json`` — version 0 commit.
- ``00000000000000000010.checkpoint.parquet`` — V1 checkpoint.
- ``00000000000000000010.checkpoint.<uuid>.json`` + sidecars — V2.
- ``_last_checkpoint`` — JSON pointer at the most recent checkpoint.

Reading a snapshot collapses to:

1. Read ``_last_checkpoint`` to find the most recent checkpoint version.
2. Load checkpoint actions (V1 or V2).
3. Apply JSON commits with version > checkpoint version.

Caching strategy
----------------

Remote backends (S3, DBFS, ABFS) charge per-request, so every saved
round trip matters. The log caches:

- **Directory listing** — one ``iterdir()`` per instance lifetime
  (extended in-place when we know the exact name of a new commit).
- **``_last_checkpoint``** — one read per instance lifetime.
- **Commit JSON content** — ``ExpiringDict`` keyed by path string,
  60 s TTL, capped at 1024 entries.  Commit files are small (typically
  200-2000 bytes) and immutable once written, so caching them is
  safe and high-value on repeated reads or checkpoint replays.
- **V2 manifest content** — cached alongside commit JSON (same dict).

Content larger than ``_CONTENT_CACHE_MAX_BYTES`` (1 MiB) is never
cached to keep memory bounded — checkpoint parquets for large tables
can be tens of megabytes.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable, Iterator, List, Mapping, Optional, Tuple

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME,
    LOG_DIR_NAME,
    SIDECARS_DIR_NAME,
    format_commit_name,
    version_from_log_name,
)
from yggdrasil.io.nested.delta.protocol import DeltaAction, parse_action

if TYPE_CHECKING:
    from yggdrasil.path import Path


__all__ = [
    "DeltaLog",
    "LogSegment",
]


_VERSION_FMT = "{:020d}"

_CONTENT_CACHE_TTL = 60.0
_CONTENT_CACHE_MAX_SIZE = 1024
_CONTENT_CACHE_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB

# Module-level content cache shared across DeltaLog instances on the
# same table root. Keyed by the path's full_path() string so entries
# are unique across tables. Commit JSON files are immutable once
# written, so a 60 s TTL is conservative — the only risk is a stale
# entry from a partially-written commit that another process later
# overwrites, which the Delta spec forbids.
_content_cache: ExpiringDict[str, bytes] = ExpiringDict(
    default_ttl=_CONTENT_CACHE_TTL,
    max_size=_CONTENT_CACHE_MAX_SIZE,
)


# ---------------------------------------------------------------------------
# Log segment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class LogSegment:
    """The minimal set of files that reconstructs a snapshot at *version*."""

    version: int
    checkpoint_version: int
    checkpoint_files: Tuple["Path", ...]
    commit_files: Tuple["Path", ...]
    checkpoint_kind: str = "none"

    def is_initial(self) -> bool:
        return self.checkpoint_version < 0 and not self.commit_files


# ---------------------------------------------------------------------------
# DeltaLog
# ---------------------------------------------------------------------------


class DeltaLog:
    """High-level reader for a Delta ``_delta_log`` directory.

    Caches the directory listing + ``_last_checkpoint`` per instance.
    Call :meth:`invalidate` between writes, or :meth:`extend_listing`
    when you know the exact name of a newly committed file (cheaper
    than a full re-list).
    """

    __slots__ = (
        "table_root",
        "log_path",
        "_last_checkpoint",
        "_listing",
    )

    def __init__(self, table_root: "Path") -> None:
        self.table_root = table_root
        self.log_path = table_root / LOG_DIR_NAME
        self._last_checkpoint: "Optional[Mapping[str, object]]" = None
        self._listing: "Optional[Tuple[str, ...]]" = None

    # ==================================================================
    # Listing / cache
    # ==================================================================

    def invalidate(self) -> None:
        self._last_checkpoint = None
        self._listing = None

    def extend_listing(self, *names: str) -> None:
        """Cheaply extend the cached listing with known new file names.

        After a successful commit, the writer knows the exact name of
        the new JSON file. Appending it to the cached listing avoids a
        full ``iterdir()`` round trip on the next snapshot resolution.
        """
        if self._listing is None:
            return
        current = set(self._listing)
        added = [n for n in names if n not in current]
        if added:
            self._listing = tuple(sorted(current | set(added)))

    def _list_log_dir(self) -> "Tuple[str, ...]":
        if self._listing is not None:
            return self._listing
        names: List[str] = []
        try:
            entries = list(self.log_path.iterdir())
        except FileNotFoundError:
            self._listing = ()
            return self._listing
        for entry in entries:
            n = entry.name
            if n and not n.startswith("."):
                names.append(n)
        names.sort()
        self._listing = tuple(names)
        return self._listing

    # ==================================================================
    # _last_checkpoint
    # ==================================================================

    def read_last_checkpoint(self) -> "Optional[Mapping[str, object]]":
        if self._last_checkpoint is not None:
            return self._last_checkpoint or None

        ptr = self.log_path / LAST_CHECKPOINT_NAME
        try:
            raw = _read_small_file(ptr)
            payload = ygg_json.loads(raw)
        except FileNotFoundError:
            self._last_checkpoint = {}
            return None
        except Exception:
            self._last_checkpoint = {}
            return None
        self._last_checkpoint = payload or {}
        return self._last_checkpoint or None

    # ==================================================================
    # Resolution: version -> LogSegment
    # ==================================================================

    def latest_version(self) -> int:
        max_v = -1
        for name in self._list_log_dir():
            if (
                name.endswith(".json")
                and not name.startswith("_")
                and ".checkpoint." not in name
            ):
                v = version_from_log_name(name)
                if v is not None and v > max_v:
                    max_v = v
        return max_v

    def segment(self, version: "Optional[int]" = None) -> LogSegment:
        listing = self._list_log_dir()

        last_ck = self.read_last_checkpoint()
        ck_version_hint = (
            int(last_ck["version"]) if last_ck and "version" in last_ck else -1
        )

        all_commits: list[tuple[int, str]] = []
        for name in listing:
            if (
                name.endswith(".json")
                and not name.startswith("_")
                and ".checkpoint." not in name
            ):
                v = version_from_log_name(name)
                if v is not None:
                    all_commits.append((v, name))
        all_commits.sort()

        if version is None:
            target = all_commits[-1][0] if all_commits else -1
        else:
            target = int(version)

        ck_version, ck_files, ck_kind = self._resolve_checkpoint(
            listing,
            target,
            ck_version_hint,
            last_ck,
        )

        commits = tuple(
            self.log_path / name for v, name in all_commits if ck_version < v <= target
        )

        return LogSegment(
            version=target,
            checkpoint_version=ck_version,
            checkpoint_files=ck_files,
            commit_files=commits,
            checkpoint_kind=ck_kind,
        )

    def _resolve_checkpoint(
        self,
        listing: "Tuple[str, ...]",
        target: int,
        hint: int,
        last_ck: "Optional[Mapping[str, object]]",
    ) -> "tuple[int, Tuple[Path, ...], str]":
        if hint >= 0 and hint <= target and last_ck:
            files = self._files_for_checkpoint(listing, hint, last_ck)
            if files:
                kind = "v2" if last_ck.get("v2Checkpoint") else "v1"
                return hint, files, kind

        ck_versions: dict[int, list[str]] = {}
        for name in listing:
            if ".checkpoint" not in name:
                continue
            v = version_from_log_name(name)
            if v is None or v > target:
                continue
            ck_versions.setdefault(v, []).append(name)

        if not ck_versions:
            return -1, (), "none"

        best = max(ck_versions)
        files = self._files_for_checkpoint(listing, best, None)
        if not files:
            return -1, (), "none"
        prefix = f"{_VERSION_FMT.format(best)}.checkpoint"
        kind = "v1"
        for n in ck_versions[best]:
            if n.startswith(prefix) and n.endswith(".json"):
                kind = "v2"
                break
        return best, files, kind

    def _files_for_checkpoint(
        self,
        listing: "Tuple[str, ...]",
        version: int,
        last_ck: "Optional[Mapping[str, object]]",
    ) -> "Tuple[Path, ...]":
        prefix = f"{_VERSION_FMT.format(int(version))}.checkpoint"

        v1_singletons: list[str] = []
        v1_parts: list[str] = []
        v2_manifests: list[str] = []
        for name in listing:
            if not name.startswith(prefix):
                continue
            if name == f"{prefix}.parquet":
                v1_singletons.append(name)
            elif name.endswith(".parquet") and ".checkpoint." in name:
                v1_parts.append(name)
            elif name.endswith(".json"):
                v2_manifests.append(name)

        if v2_manifests:
            return self._read_v2_sidecars(version, v2_manifests[0])

        if v1_singletons:
            return (self.log_path / v1_singletons[0],)

        if v1_parts:
            v1_parts.sort()
            return tuple(self.log_path / n for n in v1_parts)

        if last_ck and last_ck.get("v2Checkpoint"):
            v2_ck = last_ck["v2Checkpoint"]
            sidecars = (
                v2_ck.get("sidecarFiles")  # type: ignore[union-attr]
                or v2_ck.get("sidecars")  # type: ignore[union-attr]
                or []
            )
            if sidecars:
                return tuple(
                    self.table_root / LOG_DIR_NAME / SIDECARS_DIR_NAME / s["path"]
                    for s in sidecars
                )

        return ()

    def _read_v2_sidecars(
        self,
        version: int,
        manifest_name: str,
    ) -> "Tuple[Path, ...]":
        manifest_path = self.log_path / manifest_name
        try:
            raw = _read_small_file(manifest_path)
        except Exception:
            return ()

        sidecars: list[str] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = ygg_json.loads(line)
            except Exception:
                continue
            sc = obj.get("sidecar")
            if sc and sc.get("path"):
                sidecars.append(sc["path"])

        if not sidecars:
            return ()
        sidecar_dir = self.log_path / SIDECARS_DIR_NAME
        return tuple(sidecar_dir / name for name in sidecars)

    # ==================================================================
    # Replay
    # ==================================================================

    def replay(self, segment: LogSegment) -> "Iterator[DeltaAction]":
        if segment.checkpoint_files:
            yield from self._iter_checkpoint(segment.checkpoint_files)
        for commit in segment.commit_files:
            yield from self._iter_commit(commit)

    def replay_raw(self, segment: LogSegment) -> "Iterator[Mapping[str, object]]":
        if segment.checkpoint_files:
            yield from self._iter_checkpoint_raw(segment.checkpoint_files)
        for commit in segment.commit_files:
            yield from self._iter_commit_raw(commit)

    # ------------------------------------------------------------------
    # Iterators per file kind
    # ------------------------------------------------------------------

    def _iter_commit(self, path: "Path") -> "Iterator[DeltaAction]":
        for raw in self._iter_commit_raw(path):
            action = parse_action(raw)
            if action is not None:
                yield action

    def _iter_commit_raw(self, path: "Path") -> "Iterator[Mapping[str, object]]":
        try:
            blob = _read_small_file(path).decode("utf-8")
        except FileNotFoundError:
            return
        for line in blob.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield ygg_json.loads(line)
            except Exception:
                continue

    def _iter_checkpoint(
        self,
        files: "Iterable[Path]",
    ) -> "Iterator[DeltaAction]":
        for raw in self._iter_checkpoint_raw(files):
            action = parse_action(raw)
            if action is not None:
                yield action

    def _iter_checkpoint_raw(
        self,
        files: "Iterable[Path]",
    ) -> "Iterator[Mapping[str, object]]":
        import pyarrow.parquet as pq

        for path in files:
            local = _local_str(path)
            try:
                if local is not None:
                    table = pq.read_table(local)
                else:
                    blob = _read_file_uncached(path)
                    import io as _io
                    table = pq.read_table(_io.BytesIO(blob))
            except FileNotFoundError:
                continue

            cols = table.column_names
            if not cols or table.num_rows == 0:
                continue
            materialised = [table.column(c).to_pylist() for c in cols]
            for row_idx in range(table.num_rows):
                for col_idx, col in enumerate(cols):
                    val = materialised[col_idx][row_idx]
                    if val is None:
                        continue
                    yield {col: val}
                    break


# ---------------------------------------------------------------------------
# File read helpers with content caching
# ---------------------------------------------------------------------------


def _cache_key(path: "Path") -> str:
    """Stable string key for the content cache."""
    fn = getattr(path, "full_path", None)
    if callable(fn):
        return fn()
    return str(path)


def _read_small_file(path: "Path") -> bytes:
    """Read a small file through the content cache.

    Files larger than ``_CONTENT_CACHE_MAX_BYTES`` are read but not
    cached. Commit JSON and ``_last_checkpoint`` files are typically
    200-2000 bytes and benefit strongly from caching on remote stores.
    """
    key = _cache_key(path)
    cached = _content_cache.get(key)
    if cached is not None:
        return cached

    raw = _read_file_uncached(path)

    if len(raw) <= _CONTENT_CACHE_MAX_BYTES:
        _content_cache[key] = raw
    return raw


def _read_file_uncached(path: "Path") -> bytes:
    """Read a file without caching."""
    with path.open("rb") as bio:
        return bio.read()


def _local_str(path: "Path") -> "Optional[str]":
    if not getattr(path, "is_local_path", False):
        return None
    fn = getattr(path, "full_path", None)
    return fn() if callable(fn) else None
