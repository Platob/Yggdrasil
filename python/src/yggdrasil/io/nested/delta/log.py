"""Delta transaction-log parser.

The Delta log is a sequence of files under ``<table>/_delta_log``:

- ``00000000000000000000.json`` — version 0 commit.
- ``00000000000000000001.json`` — version 1 commit.
- … one JSON commit per version.
- ``00000000000000000010.checkpoint.parquet`` — V1 checkpoint covering
  versions 0..10 (one parquet file).
- ``00000000000000000010.checkpoint.<uuid>.json`` plus sidecar parquet
  files under ``_delta_log/_sidecars/`` — V2 checkpoints (UUID-named,
  optionally split into per-action-class sidecars).
- ``_last_checkpoint`` — JSON pointer at the most recent checkpoint:
  ``{"version": 10, "size": ..., "v2Checkpoint": {...}}``.

Reading a snapshot collapses to:

1. Read ``_last_checkpoint`` to find the most recent checkpoint version
   (or treat it as -1 if absent).
2. Load the checkpoint actions (V1 = one parquet, V2 = manifest +
   sidecars).
3. Apply every JSON commit with version > checkpoint version.

This module owns step 1 and step 2; :class:`Snapshot` owns step 3 +
reduction.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Iterable, Iterator, List, Mapping, Optional, Tuple

from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME,
    LOG_DIR_NAME,
    SIDECARS_DIR_NAME,
    version_from_log_name,
)
from yggdrasil.io.nested.delta.protocol import DeltaAction, parse_action

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = [
    "DeltaLog",
    "LogSegment",
]


# 20-zero-padded version, mirrored here so the listing-scanner can
# build a prefix filter without re-importing from ``_names``.
_VERSION_FMT = "{:020d}"


# ---------------------------------------------------------------------------
# Log segment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class LogSegment:
    """The minimal set of files that reconstructs a snapshot at *version*.

    Built once per :meth:`DeltaLog.segment` call; consumed by
    :class:`Snapshot.replay`. Keeping the resolution and the replay
    apart lets callers cache the segment (cheap to hash) without
    pinning the materialized actions.
    """

    version: int
    #: Checkpoint version (``-1`` when no checkpoint exists yet — the
    #: replay starts from commit 0). Files in :attr:`checkpoint_files`
    #: cover ``[0, checkpoint_version]``.
    checkpoint_version: int
    #: Parquet files containing the checkpoint actions. For V1 this is
    #: a single ``.checkpoint.parquet``; for V2 it's the sidecars
    #: referenced from the manifest. Files are read in order.
    checkpoint_files: Tuple["Path", ...]
    #: JSON commit files in version-ascending order, all strictly
    #: greater than :attr:`checkpoint_version` and less than or equal
    #: to :attr:`version`.
    commit_files: Tuple["Path", ...]
    #: ``"v1"`` (single parquet), ``"v2"`` (manifest + sidecars), or
    #: ``"none"`` when no checkpoint exists yet.
    checkpoint_kind: str = "none"

    def is_initial(self) -> bool:
        """True when this is the first read of an empty / brand-new table."""
        return self.checkpoint_version < 0 and not self.commit_files


# ---------------------------------------------------------------------------
# DeltaLog
# ---------------------------------------------------------------------------


class DeltaLog:
    """High-level reader for a Delta ``_delta_log`` directory.

    All metadata I/O routes through the bound :class:`Path`, so the
    same parser works for local, S3, DBFS, etc. The class caches the
    listing of ``_delta_log`` per instance so a snapshot read +
    schema-only follow-up don't re-list the directory; call
    :meth:`invalidate` between writes when the caller knows the log
    moved underneath them.
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
        """Drop cached log listing + last-checkpoint pointer."""
        self._last_checkpoint = None
        self._listing = None

    def _list_log_dir(self) -> "Tuple[str, ...]":
        """Return every entry in ``_delta_log`` (cached).

        One round trip per :class:`DeltaLog` instance covers every
        version-resolution + segment-build call. Remote backends with
        per-listing-call cost (S3 ``LIST``, Databricks REST) collapse
        to a single round trip per snapshot read.

        Goes straight to ``iterdir()`` — no ``exists()`` probe. On
        remote stores (S3, ABFS, DBFS) a HEAD-then-LIST pair doubles
        the round-trip count for the common case (log directory is
        present); ``FileNotFoundError`` from the missing case is just
        as cheap to handle.
        """
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
        """Decode ``_last_checkpoint`` once.

        Returns ``None`` when the pointer is missing or unreadable —
        the caller falls through to the "no checkpoint, replay from
        version 0" path. We never crash on a malformed pointer; the
        Delta spec explicitly allows readers to ignore it and recover
        by listing.

        No ``exists()`` probe: ``open("rb")`` either returns the
        payload or raises ``FileNotFoundError`` in one round trip.
        On remote stores (S3 / DBFS / ABFS) the HEAD-then-GET pair
        the probe imposes doubles the latency on the common case
        (pointer is present); local FS bottoms out in a single
        ``stat`` either way.
        """
        if self._last_checkpoint is not None:
            # ``{}`` distinguishes "we tried and there's nothing" from
            # "we haven't tried yet" (None).
            return self._last_checkpoint or None

        ptr = self.log_path / LAST_CHECKPOINT_NAME
        try:
            with ptr.open("rb") as bio:
                payload = ygg_json.loads(bio.read())
        except FileNotFoundError:
            self._last_checkpoint = {}
            return None
        except Exception:
            self._last_checkpoint = {}
            return None
        self._last_checkpoint = payload or {}
        return self._last_checkpoint or None

    # ==================================================================
    # Resolution: version → LogSegment
    # ==================================================================

    def latest_version(self) -> int:
        """Highest commit version currently visible in the log.

        ``-1`` for an empty / non-existent log (i.e., the table hasn't
        been created yet — the first commit will mint version 0).
        """
        max_v = -1
        for name in self._list_log_dir():
            if name.endswith(".json") and not name.startswith("_"):
                v = version_from_log_name(name)
                if v is not None and v > max_v:
                    max_v = v
        return max_v

    def segment(self, version: "Optional[int]" = None) -> LogSegment:
        """Build a :class:`LogSegment` for *version* (or HEAD when ``None``).

        Resolution does at most two metadata round trips: one for
        ``_last_checkpoint``, one for the log listing. Both are
        memoized — repeat calls within the lifetime of this
        :class:`DeltaLog` are free.
        """
        listing = self._list_log_dir()

        # 1. Find the highest checkpoint version ≤ requested version.
        last_ck = self.read_last_checkpoint()
        ck_version_hint = (
            int(last_ck["version"]) if last_ck and "version" in last_ck else -1
        )

        # 2. Resolve the requested version.
        all_commits: list[tuple[int, str]] = []
        for name in listing:
            if name.endswith(".json") and not name.startswith("_"):
                v = version_from_log_name(name)
                if v is not None:
                    all_commits.append((v, name))
        all_commits.sort()

        if version is None:
            target = all_commits[-1][0] if all_commits else -1
        else:
            target = int(version)

        # 3. Pick the best checkpoint at or below target, biased
        # toward the ``_last_checkpoint`` hint (one read instead of
        # scanning the whole listing).
        ck_version, ck_files, ck_kind = self._resolve_checkpoint(
            listing,
            target,
            ck_version_hint,
            last_ck,
        )

        # 4. Commits strictly above ck_version, up to and including target.
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
        """Return ``(version, files, kind)`` for the best checkpoint ≤ target.

        Tries the ``_last_checkpoint`` hint first (the spec's intended
        path — one fast lookup), then falls back to scanning the
        listing for the highest checkpoint ≤ target.
        """
        if hint >= 0 and hint <= target and last_ck:
            files = self._files_for_checkpoint(listing, hint, last_ck)
            if files:
                kind = "v2" if last_ck.get("v2Checkpoint") else "v1"
                return hint, files, kind

        # Listing scan — pick the largest checkpoint version ≤ target.
        # We don't need the manifest here: V1 (single parquet) and V2
        # (manifest + sidecars) are distinguished by the suffix shape.
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
        # V2 manifests end with ``.checkpoint.<uuid>.json``; classic
        # V1 checkpoints are a single ``.checkpoint.parquet``.
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
        """Resolve the parquet files that constitute the checkpoint at *version*.

        V1 — one ``.checkpoint.parquet``.
        V2 — manifest JSON points at sidecars under ``_delta_log/_sidecars``.
        """
        prefix = f"{_VERSION_FMT.format(int(version))}.checkpoint"

        # Multi-part V1 (rare): ``00000000000000000010.checkpoint.0000000001.0000000003.parquet``
        # — multiple parquet parts sharing the same version. The pointer
        # records ``{"size": N, "parts": K}``. We don't enforce the count
        # but glob for them.
        v1_singletons: list[str] = []
        v1_parts: list[str] = []
        v2_manifests: list[str] = []
        for name in listing:
            if not name.startswith(prefix):
                continue
            if name == f"{prefix}.parquet":
                v1_singletons.append(name)
            elif name.endswith(".parquet") and ".checkpoint." in name:
                # V2 sidecar names: ``...checkpoint.<uuid>.parquet`` —
                # but those land under ``_sidecars/`` in modern V2 and
                # in the log dir for legacy multi-part V1.
                # Distinguish by uuid-shape: V2 manifest pairs with
                # exactly one ``.json`` of the same uuid.
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

        # Last-checkpoint pointer can record sidecars directly (some
        # writers do that to avoid a manifest fetch). Use it as a
        # fallback when listing didn't produce anything.
        if last_ck and last_ck.get("v2Checkpoint"):
            sidecars = last_ck["v2Checkpoint"].get("sidecars") or []  # type: ignore[union-attr]
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
        """Decode a V2 manifest and return its sidecar paths."""
        manifest_path = self.log_path / manifest_name
        try:
            with manifest_path.open("rb") as bio:
                raw = bio.read().decode("utf-8")
        except Exception:
            return ()

        # Manifest is one-action-per-line JSON, like a commit. We pull
        # ``sidecar`` actions to build the parquet list.
        sidecars: list[str] = []
        for line in raw.splitlines():
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
    # Replay — yield raw actions from a segment
    # ==================================================================

    def replay(self, segment: LogSegment) -> "Iterator[DeltaAction]":
        """Yield typed actions for a segment in the right order.

        Order: checkpoint actions first, then commits in ascending
        version. Within each commit JSON, lines are kept in file
        order — :class:`Snapshot` collapses ``add`` / ``remove`` /
        ``metaData`` / ``protocol`` reductions over the stream.
        """
        if segment.checkpoint_files:
            yield from self._iter_checkpoint(segment.checkpoint_files)
        for commit in segment.commit_files:
            yield from self._iter_commit(commit)

    def replay_raw(self, segment: LogSegment) -> "Iterator[Mapping[str, object]]":
        """Yield every raw action dict — including ones we don't model.

        Useful for CDC consumers and audit tooling that need the
        full payload (``cdc``, ``checkpointMetadata``, …).
        """
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
            with path.open("rb") as bio:
                blob = bio.read().decode("utf-8")
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
        """Decode every parquet in *files* and emit one raw-action dict per row.

        Checkpoint parquets store one row per action; the columns are
        the action keys (``add``, ``remove``, ``metaData``, …) and the
        non-null column on a given row is the action that row carries.
        Pyarrow makes it cheap — read the table, walk rows, take the
        first non-null column as the action body.
        """
        # Lazy import — we want this module importable without
        # pyarrow.parquet on the path (the import lives in
        # snapshot/io anyway).
        import pyarrow.parquet as pq

        for path in files:
            local = _local_str(path)
            try:
                if local is not None:
                    table = pq.read_table(local)
                else:
                    with path.open("rb") as bio:
                        blob = bio.read()
                    import io as _io  # local — avoid the top-level shadow

                    table = pq.read_table(_io.BytesIO(blob))
            except FileNotFoundError:
                # Checkpoint parquet referenced by the listing but
                # gone underneath us (vacuum race) — skip rather than
                # crash. Saves one HEAD per checkpoint vs the
                # ``exists()`` probe.
                continue

            cols = table.column_names
            for row_idx in range(table.num_rows):
                for col in cols:
                    val = table.column(col)[row_idx].as_py()
                    if val is None:
                        continue
                    yield {col: val}
                    break  # one action per row


def _local_str(path: "Path") -> "Optional[str]":
    """Path's backend-native string when local; ``None`` for remote."""
    if not getattr(path, "is_local_path", False):
        return None
    fn = getattr(path, "full_path", None)
    return fn() if callable(fn) else None
