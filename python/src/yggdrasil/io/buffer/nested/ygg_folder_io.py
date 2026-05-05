"""Yggdrasil-managed folder format with a ``.ygg/`` sidecar.

:class:`YGGFolderIO` is a :class:`FolderIO` that adds an out-of-band
metadata surface — checkpoints, key/value attributes, schema
pinning, and stats — under a hidden ``.ygg/`` subdirectory. The
data layout is identical to :class:`FolderIO` (one tabular file per
child, optional Hive-style partitions); the sidecar is purely
additive, so a YGG folder is also a valid plain folder when read by
any other tool.

The sidecar:

- ``<root>/.ygg/checkpoints.jsonl`` — append-only log of checkpoint
  records emitted via :meth:`checkpoint`.
- ``<root>/.ygg/metadata/<key>.json`` — per-key value store
  exposed via :meth:`write_metadata` / :meth:`read_metadata`.
- ``<root>/.ygg/stats.arrow`` — Arrow IPC encoded
  :class:`yggdrasil.io.stats.Stats` for the entire folder, written
  by :meth:`write_stats` and read by :meth:`read_stats`.

The ``.ygg/`` folder is hidden from data enumeration via the
default :meth:`NestedIO._is_ignored_path` rule (dot-prefix). Reads
through :meth:`read_arrow_table` therefore never include sidecar
content.

Auto-detection
--------------

:func:`is_ygg_folder` probes for ``<path>/.ygg/`` and returns
``True`` when the sidecar is present. The plain :class:`FolderIO`
factory consults this probe on construction: a directory that
already carries a sidecar is upgraded to a :class:`YGGFolderIO`
automatically, so callers never have to choose between the two
classes by name.
"""

from __future__ import annotations

import itertools
import os
import time
import urllib.parse
from typing import Any, ClassVar, Iterable, Iterator, Mapping, TYPE_CHECKING

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.io.enums import MimeType, MimeTypes
from yggdrasil.io.fs import Path
from yggdrasil.io.stats import Stats, STATS_FILENAME
from .base import _run_in_threads
from .folder_io import FolderIO


if TYPE_CHECKING:
    pass


__all__ = ["YGGFolderIO", "is_ygg_folder"]


_METADATA_KEY_VALID = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
)


def _validate_metadata_key(key: str) -> str:
    """Reject path-separator / traversal / empty keys for ``.ygg/metadata``.

    The sidecar is metadata, not a free-form filesystem; allowing
    arbitrary names invites traversal (``../``) and collisions with
    the JSON-extension convention (``foo.json/`` as a directory).
    Keep the alphabet narrow.
    """
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"Metadata key must be a non-empty string; got {key!r}."
        )
    bad = [c for c in key if c not in _METADATA_KEY_VALID]
    if bad:
        raise ValueError(
            f"Metadata key {key!r} contains invalid characters "
            f"{sorted(set(bad))!r}. Allowed: alphanumerics, '_', '-', '.'."
        )
    if key in (".", ".."):
        raise ValueError(f"Metadata key {key!r} is reserved.")
    return key


def _parse_checkpoint_lines(blob: bytes) -> list[dict[str, Any]]:
    """Decode a JSONL checkpoint blob, skipping torn / non-dict lines.

    Shared between :meth:`YGGFolderIO.list_checkpoints` and the
    fallback path of :meth:`YGGFolderIO._read_last_checkpoint_record`.
    Splits on ``b"\\n"`` instead of ``splitlines`` so an embedded
    ``\\r`` in a (mistakenly) Windows-encoded payload doesn't get
    treated as a record boundary.
    """
    out: list[dict[str, Any]] = []
    for raw in blob.split(b"\n"):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json_module.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def is_ygg_folder(path: "Path | str | os.PathLike") -> bool:
    """Probe ``<path>/.ygg/`` to classify a folder.

    Returns ``True`` when the directory exists and has a ``.ygg``
    subdirectory; ``False`` for plain folders or missing paths. The
    probe is a single backend round-trip (``Path.exists``), kept
    cheap because :meth:`FolderIO.__new__` calls it on every folder
    construction.
    """
    if not isinstance(path, Path):
        try:
            path = Path.from_(path)
        except Exception:
            return False
    try:
        return (path / YGGFolderIO.YGG_DIR_NAME).exists()
    except Exception:
        return False


class YGGFolderIO(FolderIO):
    """:class:`FolderIO` with a ``.ygg/`` metadata sidecar.

    Same data layout as :class:`FolderIO`. Adds:

    - **Checkpoints** — :meth:`checkpoint` /
      :meth:`list_checkpoints` / :meth:`latest_checkpoint`.
      Append-only JSONL log under ``.ygg/checkpoints.jsonl``,
      serialised via a transient ``.w.lock`` so concurrent writers
      don't tear lines.
    - **Metadata** — :meth:`write_metadata` /
      :meth:`read_metadata` / :meth:`list_metadata_keys`.
      JSON-encoded values under ``.ygg/metadata/<key>.json``,
      writes are stage+rename atomic.
    - **Stats** — :meth:`collect_stats` (compute from current
      data), :meth:`write_stats` (persist as Arrow IPC under
      ``.ygg/stats.arrow``), :meth:`read_stats` (load).

    Construction parity with :class:`FolderIO` — pass ``path=`` and
    optionally ``schema=`` / ``partition_columns=`` /
    ``concurrent=``. Reads via :meth:`read_arrow_batches` work
    exactly the same; the sidecar is invisible to data enumeration.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        return MimeTypes.YGG_FOLDER

    def _default_child_media_type(self) -> Any:
        """Arrow IPC is the canonical YGG-folder leaf format.

        Plain :class:`FolderIO` defaults to Parquet — fine when
        cross-tool readability matters more than write speed, but a
        ``.ygg/`` cache is intentionally an internal layout: the
        sidecar advertises "this folder is owned by yggdrasil." We
        trade Parquet's compression + column pruning for Arrow IPC's
        zero-copy reads, much faster small-file writes (no encoder
        setup, no row-group machinery), and a stable on-disk
        representation that maps 1:1 to the in-memory
        :class:`pa.RecordBatch`. The compactor still bin-packs small
        files into ``OPTIMIZE_TARGET_BYTES`` chunks, so storage stays
        comparable; the win is on the hot append path.

        Callers who want Parquet anyway can pass ``child_media_type``
        on :class:`FolderOptions` — same override hook the base class
        documents.
        """
        return MimeTypes.ARROW_IPC

    #: Hidden subfolder used for metadata. Starts with ``.`` so the
    #: default :meth:`_is_ignored_path` rule already hides it from
    #: data enumeration.
    YGG_DIR_NAME: ClassVar[str] = ".ygg"

    #: File under :attr:`YGG_DIR_NAME` that holds the append-only
    #: checkpoint log. JSON Lines: one record per
    #: :meth:`checkpoint` call.
    CHECKPOINT_LOG_NAME: ClassVar[str] = "checkpoints.jsonl"

    #: Subfolder under :attr:`YGG_DIR_NAME` that holds key/value
    #: metadata as ``<key>.json`` files.
    METADATA_DIR_NAME: ClassVar[str] = "metadata"

    #: Filename under :attr:`YGG_DIR_NAME` that stores the
    #: Arrow-IPC encoded :class:`Stats` sidecar.
    STATS_FILENAME: ClassVar[str] = STATS_FILENAME

    # ==================================================================
    # Sidecar paths
    # ==================================================================

    @property
    def ygg_path(self) -> Path:
        """Hidden ``.ygg/`` subfolder used for out-of-band metadata."""
        return self.path / self.YGG_DIR_NAME

    @property
    def _ygg_checkpoint_log(self) -> Path:
        return self.ygg_path / self.CHECKPOINT_LOG_NAME

    @property
    def _ygg_metadata_dir(self) -> Path:
        return self.ygg_path / self.METADATA_DIR_NAME

    @property
    def _ygg_stats_path(self) -> Path:
        return self.ygg_path / self.STATS_FILENAME

    # ==================================================================
    # Checkpoints
    # ==================================================================

    #: Tail window (bytes) used by :meth:`latest_checkpoint` to skip
    #: re-parsing the entire log when only the most recent record is
    #: needed. 8 KiB comfortably fits dozens of typical records (a
    #: ``checkpoint`` JSON line is ~200 B with a handful of files
    #: listed); on the rare case the last record overflows this we
    #: fall back to a full read.
    _CHECKPOINT_TAIL_WINDOW: ClassVar[int] = 8192

    #: Upper bound on payload size for the lockless append fast-path
    #: in :meth:`checkpoint`. POSIX guarantees ``write()`` atomicity
    #: only up to ``PIPE_BUF`` (typically 4 KiB on Linux, 512 B on
    #: macOS); we play it safe at 512 B so an interleaved concurrent
    #: append from another process never tears a JSON line.
    _APPEND_ATOMIC_LIMIT: ClassVar[int] = 512

    def checkpoint(
        self,
        message: "str | None" = None,
        *,
        owner: "str | None" = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append a checkpoint record to ``.ygg/checkpoints.jsonl``.

        A checkpoint is the streaming-write equivalent of a commit:
        it records that the writer considers everything written up
        to *this point* a coherent unit, with optional caller-supplied
        ``message`` and arbitrary ``**extra`` fields. Concurrent
        ``checkpoint()`` callers serialise on a transient ``.w.lock``
        against the log file so JSON lines never tear; the lock is
        held only for the append, not for the (much more expensive)
        record assembly.

        ``owner`` is a compute-identifier URL (see
        :func:`yggdrasil.io.buffer._concurrency.compute_identifier_url`
        for the schemes). When omitted, the local process's URL is
        used. Pass an explicit ``owner`` when committing on behalf of
        another process — for example, a Spark driver recording that
        a distributed write across workers belongs to its own job.
        """
        from yggdrasil.io.buffer._concurrency import compute_identifier_url

        if owner is None:
            owner = compute_identifier_url()

        # Build the record outside the write lock — id discovery
        # (tail-read) and file listing both touch the filesystem and
        # would be wasted contention if held inside ``.w.lock``.
        record: dict[str, Any] = {
            "id": self._next_checkpoint_id(),
            "ts": time.time(),
            # ``pid`` retained for back-compat with readers that look
            # it up by name; ``owner`` is the canonical attribution
            # going forward (carries hostname + Databricks job/run/
            # task tags or whatever the caller supplied).
            "pid": os.getpid(),
            "owner": owner,
            "files": self._current_child_filenames(),
            "message": message,
        }
        record["num_files"] = len(record["files"])
        record.update(extra)

        line = (
            json_module.dumps(record, to_bytes=False).encode("utf-8") + b"\n"
        )

        self.ygg_path.mkdir(parents=True, exist_ok=True)
        log = self._ygg_checkpoint_log

        # Local files: rely on POSIX / Windows ``O_APPEND`` atomicity
        # for writes smaller than ``PIPE_BUF`` (typically 4 KiB on
        # Linux, 512 B on macOS). A checkpoint line is on the order
        # of 200–800 B even with a hundred files listed, so the
        # append is atomic and the ``.w.lock`` sidecar is unneeded —
        # cuts the lock-file clutter and the per-call sidecar
        # round-trip on the hot path. Remote backends (where atomic
        # append is not guaranteed) still serialise on the lock.
        if log.is_local and len(line) <= self._APPEND_ATOMIC_LIMIT:
            try:
                log.write_bytes(line, mode="ab")
                return record
            except OSError:
                pass

        with log.lock(write=True, wait=30):
            log.write_bytes(line, mode="ab")
        return record

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Read every checkpoint record from ``.ygg/checkpoints.jsonl``.

        Records are returned in append order (oldest first). Lines
        that fail to parse are silently skipped — the log is meant
        to survive partial / torn writes from a crashed peer. An
        empty or missing log returns ``[]``.
        """
        log = self._ygg_checkpoint_log
        try:
            blob = log.read_bytes(raise_error=False)
        except OSError:
            return []
        if not blob:
            return []
        return _parse_checkpoint_lines(blob)

    def latest_checkpoint(self) -> "dict[str, Any] | None":
        """The most recent checkpoint record, or ``None`` if none.

        Tail-reads the log via :meth:`Path.pread` so a million-record
        history doesn't pay a full scan + JSON parse to surface the
        last commit. Falls back to a full read in the (rare) case
        where the last record straddles the tail window.
        """
        return self._read_last_checkpoint_record()

    def _read_last_checkpoint_record(self) -> "dict[str, Any] | None":
        log = self._ygg_checkpoint_log
        try:
            size = log.size
        except (OSError, FileNotFoundError):
            return None
        if not size:
            return None

        window = self._CHECKPOINT_TAIL_WINDOW
        pos = max(0, size - window)
        try:
            blob = log.pread(n=size - pos, pos=pos, default=b"")
        except (OSError, ValueError):
            blob = b""

        if blob and pos > 0:
            # Drop the leading partial record; its newline boundary
            # is the first ``\n`` in the window.
            nl = blob.find(b"\n")
            if nl < 0:
                # Last record longer than the window — fall back to a
                # full read (vanishingly rare in practice).
                blob = b""
            else:
                blob = blob[nl + 1:]

        if not blob:
            try:
                blob = log.read_bytes(raise_error=False)
            except OSError:
                return None
            if not blob:
                return None

        for raw in reversed(blob.split(b"\n")):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json_module.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _next_checkpoint_id(self) -> int:
        prev = self._read_last_checkpoint_record()
        if prev is None:
            return 1
        try:
            return int(prev.get("id", 0)) + 1
        except (TypeError, ValueError):
            return 1

    def _current_child_filenames(self) -> list[str]:
        """Names of the *direct* tabular children at this moment.

        Used by :meth:`checkpoint` to record what was visible when
        the checkpoint was taken, so a reader replaying the log can
        reconstruct the folder's state at each commit point. Hidden
        / sidecar entries are filtered via :meth:`_is_ignored_path`.
        Single ``iterdir`` round-trip — no per-entry stat calls.
        """
        try:
            entries = self.path.iterdir()
        except (OSError, FileNotFoundError):
            return []
        is_ignored = self._is_ignored_path
        names = [e.name for e in entries if not is_ignored(e)]
        names.sort()
        return names

    # ==================================================================
    # Metadata
    # ==================================================================

    def write_metadata(self, key: str, value: Any) -> None:
        """Persist a JSON-serialisable value at ``.ygg/metadata/<key>.json``.

        ``key`` must be a non-empty ASCII slug (no path separators);
        this is sidecar metadata, not a free-form filesystem.
        Writes are atomic via stage+rename so a concurrent reader
        either sees the previous value or the new one — never a
        partial write.
        """
        slug = _validate_metadata_key(key)
        self._ygg_metadata_dir.mkdir(parents=True, exist_ok=True)
        target = self._ygg_metadata_dir / f"{slug}.json"
        payload = json_module.dumps(value)
        try:
            staging = self._ygg_metadata_dir.make_staging(
                media_type=MimeTypes.JSON,
            )
            staging.write_bytes(payload)
            staging.rename(target)
        except Exception:
            target.write_bytes(payload)

    def read_metadata(self, key: str, default: Any = None) -> Any:
        """Load the value at ``.ygg/metadata/<key>.json`` or return *default*."""
        slug = _validate_metadata_key(key)
        target = self._ygg_metadata_dir / f"{slug}.json"
        try:
            blob = target.read_bytes(raise_error=False)
        except OSError:
            return default
        if not blob:
            return default
        try:
            return json_module.loads(blob)
        except Exception:
            return default

    def list_metadata_keys(self) -> list[str]:
        """All keys present under ``.ygg/metadata/`` — sorted, slug form."""
        directory = self._ygg_metadata_dir
        try:
            entries = directory.iterdir()
        except (OSError, FileNotFoundError):
            return []
        keys: list[str] = []
        for entry in entries:
            name = entry.name
            if not name.endswith(".json"):
                continue
            keys.append(name[:-5])
        keys.sort()
        return keys

    # ==================================================================
    # Stats — Arrow IPC sidecar at ``.ygg/stats.arrow``
    # ==================================================================

    def collect_stats(
        self,
        *,
        distinct: bool = False,
        per_file: bool = False,
        options: Any = None,
    ) -> Stats:
        """Compute :class:`Stats` for the folder's data.

        :param distinct: include ``distinct_count`` per column
            (extra full scan; off by default).
        :param per_file: when ``True`` and the folder has multiple
            tabular children, compute stats per child file and merge
            them with an aggregate row, so consumers can do
            partition-pruning by file. ``False`` (default) computes
            a single aggregate over the whole folder.
        :param options: optional :class:`FolderOptions` passed
            through to the underlying read.
        """
        if not per_file:
            return Stats.compute(
                self.read_arrow_table(options=options),
                distinct=distinct,
            )

        # Per-child stats are independent — compute concurrently when
        # the caller passed a thread budget on ``options``. Each child
        # opens its own IO so there's no shared state to fight over.
        children = list(self.iter_children(options=options))
        max_workers = getattr(options, "max_workers", 0) or 0

        def _compute_child(child: Any) -> "Stats | None":
            try:
                return Stats.compute(
                    child.read_arrow_table(options=options),
                    name=child.path.name,
                    distinct=distinct,
                )
            except Exception:
                return None

        per_child: list[Stats] = [
            s for s in _run_in_threads(
                children, _compute_child, max_workers=max_workers,
            )
            if s is not None
        ]

        if not per_child:
            # Empty folder — return a zero-row Stats with the
            # declared schema if we have one, else an empty schema.
            schema = self.collect_schema(options=options)
            arrow_schema = schema.to_arrow_schema() if schema else pa.schema([])
            return Stats.compute(
                pa.Table.from_pylist([], schema=arrow_schema),
                distinct=distinct,
            )
        return Stats.merge(per_child, with_aggregate=True)

    def write_stats(
        self,
        stats: "Stats | None" = None,
        *,
        distinct: bool = False,
        per_file: bool = False,
    ) -> Stats:
        """Persist a :class:`Stats` sidecar at ``.ygg/stats.arrow``.

        When ``stats`` is ``None`` (default), computes one via
        :meth:`collect_stats` first. Returns whatever was actually
        written so callers don't have to make a second call to
        round-trip it.

        Atomic: the write goes through a staging file under the
        ``.ygg/`` directory and ``rename`` to the final name, so a
        concurrent reader either sees the previous value or the new
        one — never a partial Arrow IPC.
        """
        if stats is None:
            stats = self.collect_stats(distinct=distinct, per_file=per_file)
        self.ygg_path.mkdir(parents=True, exist_ok=True)
        payload = stats.to_ipc()
        target = self._ygg_stats_path
        try:
            staging = self.ygg_path.make_staging(
                media_type=MimeTypes.OCTET_STREAM,
            )
            staging.write_bytes(payload)
            staging.rename(target)
        except Exception:
            target.write_bytes(payload)
        return stats

    def read_stats(self) -> "Stats | None":
        """Load the persisted :class:`Stats` sidecar, or ``None``.

        Missing / corrupt / version-mismatched files all collapse
        to ``None`` — callers that want a fresh recompute should
        call :meth:`collect_stats` directly.
        """
        target = self._ygg_stats_path
        try:
            blob = target.read_bytes(raise_error=False)
        except OSError:
            return None
        if not blob:
            return None
        try:
            return Stats.from_ipc(blob)
        except Exception:
            return None

    # ==================================================================
    # Optimize — small-file compaction
    # ==================================================================

    #: Per-bucket target size for :meth:`optimize`. Mirrors the Delta /
    #: Iceberg default; large enough that the rewrite cost amortises
    #: across many rows, small enough that a single bucket fits
    #: comfortably in memory and in one parquet row group.
    OPTIMIZE_TARGET_BYTES: ClassVar[int] = 128 * 1024 * 1024

    #: Minimum small-file count per leaf folder before
    #: :meth:`optimize` will compact. Folders at or below this skip
    #: the rewrite — the win on a 3-file folder isn't worth a fsync.
    OPTIMIZE_MIN_FILES: ClassVar[int] = 5

    #: Default thread budget for :meth:`optimize`'s per-leaf fan-out.
    #: Compaction is I/O-bound (read sources, write merged, unlink
    #: sources) so a small pool dramatically shortens wall time on a
    #: tree of dozens of leaves without saturating the FS.
    OPTIMIZE_MAX_WORKERS: ClassVar[int] = 8

    def optimize(
        self,
        *,
        target_bytes: "int | None" = None,
        min_files: "int | None" = None,
        wait: Any = None,
        partitions: "Mapping[str, Iterable[Any]] | None" = None,
        max_workers: "int | None" = None,
    ) -> "YGGFolderIO":
        """Compact small leaf files into ``~target_bytes`` chunks.

        Walks every leaf folder under :attr:`path` (a leaf folder
        is any folder that contains data files, not subfolders).
        Folders carrying more than ``min_files`` files smaller
        than ``target_bytes`` are repacked: small files are
        bin-packed into buckets summing to ``~target_bytes``,
        each bucket is written to a single new file, and the
        consumed sources are removed. Files already at or above
        the target survive untouched, so a healthy folder pays
        nothing.

        ``partitions`` scopes the optimize to a specific set of
        partition tuples — a mapping from partition column name to
        the values to touch (e.g. ``{"partition_key": [123, 456]}``).
        Only the leaf folders matching the cartesian product of those
        values are visited; the rest of the tree is left alone.

        ``max_workers`` controls the per-leaf fan-out. Defaults to
        :attr:`OPTIMIZE_MAX_WORKERS`; pass ``1`` for sequential
        compaction.

        Concurrency model — *lock-free*. The compactor relies on
        write-then-unlink ordering for correctness: each merged
        file is staged + atomically renamed before any source is
        removed, so a concurrent reader sees one of three
        consistent shapes — ``{old sources}``, ``{old sources +
        merged}``, or ``{merged}``. The middle window briefly
        produces duplicate rows; the response cache's ``match_by``
        de-duplication on read absorbs them. Trading the
        per-source ``.rw.lock`` sidecars for this small duplicate
        window is a deliberate choice: lock files were the single
        biggest source of clutter and FS-stat traffic in the
        original design.

        ``wait`` is accepted for API parity but unused in the
        lock-free path. Returns ``self`` so call sites can chain
        ``cache.optimize().read_arrow_table()``.
        """
        del wait

        target = int(target_bytes if target_bytes is not None else self.OPTIMIZE_TARGET_BYTES)
        threshold = int(min_files if min_files is not None else self.OPTIMIZE_MIN_FILES)
        workers = int(
            max_workers if max_workers is not None else self.OPTIMIZE_MAX_WORKERS
        )

        if target <= 0 or threshold <= 0:
            return self
        if not self.path.exists():
            return self

        if partitions:
            leaves = list(self._partition_leaves(partitions))
        else:
            leaves = list(self._walk_optimize_leaves(self.path))

        if not leaves:
            return self

        def _compact_one(leaf_folder: Path) -> None:
            self._compact_leaf_folder(
                leaf_folder,
                target_bytes=target,
                min_files=threshold,
            )

        _run_in_threads(leaves, _compact_one, max_workers=workers)
        return self

    def _partition_leaves(
        self,
        partitions: "Mapping[str, Iterable[Any]]",
    ) -> "Iterator[Path]":
        """Yield the leaf folders matching ``partitions``.

        ``partitions`` maps partition column names to the values
        whose leaves should be visited; the cartesian product of
        the per-column value sets is materialised and translated
        into ``key=value/key=value`` paths under :attr:`path` —
        the exact layout
        :func:`yggdrasil.io.buffer.nested.folder_io._partition_path_segment`
        writes. Missing folders are silently skipped (a partition
        with no rows simply has no leaf yet), and unknown column
        names raise so a typo doesn't quietly become a no-op.
        """
        partition_cols = self._resolve_partition_columns()
        if not partition_cols:
            return iter(())

        names = [c.name for c in partition_cols]
        unknown = [k for k in partitions.keys() if k not in names]
        if unknown:
            raise ValueError(
                f"Unknown partition column(s) {unknown!r} for "
                f"{type(self).__name__} at {self.path!r}. "
                f"Known partition columns: {names!r}."
            )

        # Per-column ordered, de-duplicated value lists. Missing
        # columns fall through to "every value present on disk for
        # this column," letting callers prune on a subset of the
        # partition tuple (e.g. a date-partitioned table where only
        # ``partition_key`` is known).
        per_column: list[list[Any]] = []
        for name in names:
            if name in partitions:
                seen: set[Any] = set()
                values: list[Any] = []
                for v in partitions[name]:
                    if v is None:
                        # Null partition values can't round-trip
                        # through a Hive path segment — skip.
                        continue
                    if v in seen:
                        continue
                    seen.add(v)
                    values.append(v)
                if not values:
                    return iter(())
                per_column.append(values)
            else:
                per_column.append(self._existing_partition_values(name))

        return self._iter_partition_leaves(names, per_column)

    def _existing_partition_values(self, column: str) -> "list[Any]":
        """Enumerate the on-disk values for a single partition column.

        Walks the layer of the tree that carries ``column=<value>/``
        directories and returns the raw (un-decoded) values. Used
        only when a caller pins some partition columns and lets
        others fan out across whatever's already there.
        """
        prefix = f"{column}="
        out: list[Any] = []
        # The partition layer is rooted at ``self.path`` plus any
        # higher-level partition columns. We don't bother modelling
        # higher levels here — for the local cache the partition
        # layer is one deep (``partition_key=…``), and a caller
        # passing a deeper partition spec will have pinned that
        # column explicitly.
        try:
            entries = list(self.path.iterdir())
        except (OSError, FileNotFoundError):
            return out
        for entry in entries:
            try:
                if not entry.is_dir():
                    continue
            except Exception:
                continue
            name = entry.name
            if not name.startswith(prefix):
                continue
            raw = name[len(prefix):]
            try:
                out.append(urllib.parse.unquote(raw))
            except Exception:
                continue
        return out

    def _iter_partition_leaves(
        self,
        names: "list[str]",
        per_column: "list[list[Any]]",
    ) -> "Iterator[Path]":
        for combo in itertools.product(*per_column):
            segments = [
                f"{name}={urllib.parse.quote(str(value), safe='')}"
                for name, value in zip(names, combo)
            ]
            leaf = self.path.joinpath(*segments)
            try:
                if leaf.exists():
                    yield leaf
            except Exception:
                continue

    def _walk_optimize_leaves(self, root: Path) -> "Iterator[Path]":
        """Yield each folder containing data-file children under ``root``.

        Subfolders propagate the walk; ignored entries (dotfiles,
        in-flight staging files) don't count as data and never
        trigger a "leaf" classification on their own.
        """
        try:
            entries = list(root.iterdir())
        except (OSError, FileNotFoundError):
            return

        files: list[Path] = []
        subfolders: list[Path] = []
        for entry in entries:
            if self._is_ignored_path(entry):
                continue
            try:
                if entry.is_dir():
                    subfolders.append(entry)
                else:
                    files.append(entry)
            except Exception:
                continue

        if files:
            yield root

        for sub in subfolders:
            yield from self._walk_optimize_leaves(sub)

    def _compact_leaf_folder(
        self,
        leaf: Path,
        *,
        target_bytes: int,
        min_files: int,
    ) -> None:
        """Bin-pack the small files in ``leaf`` into ``~target_bytes`` chunks.

        Files at or above ``target_bytes`` survive untouched. Small
        files below the threshold are sorted (oldest first) and
        first-fit-decreasing-style packed into buckets; buckets
        carrying a single file are dropped (no compaction needed).
        Each surviving bucket is written to one staging file, then
        atomically renamed; the consumed sources are removed last
        so a reader sees either the old layout, the merged-plus-old
        intermediate (duplicate rows, deduped on read), or the new
        layout — never a half-deleted folder.

        ``min_files`` is checked against the *small file* count, not
        the total — an otherwise-fine folder with one stray small
        leaf doesn't get rewritten just because it crossed the
        threshold by accident.
        """
        try:
            entries = [e for e in leaf.iterdir() if not self._is_ignored_path(e)]
        except (OSError, FileNotFoundError):
            return

        small: list[tuple[Path, int]] = []
        for entry in entries:
            try:
                if entry.is_dir():
                    continue
                size = entry.size
            except Exception:
                continue
            if size < target_bytes:
                small.append((entry, size))

        if len(small) <= min_files:
            return

        # Oldest-first ordering keeps the first compacted file
        # closest to the original write order — useful for time-
        # sortable part names so reads still stream chronologically.
        small.sort(key=lambda pair: pair[0].name)

        buckets: list[list[Path]] = []
        current: list[Path] = []
        current_size = 0
        for entry, size in small:
            if current and current_size + size > target_bytes:
                buckets.append(current)
                current = []
                current_size = 0
            current.append(entry)
            current_size += size
        if current:
            buckets.append(current)

        # A bucket of size 1 is a no-op (renaming a single file to
        # itself doesn't help), so drop those before paying for the
        # rewrite.
        buckets = [b for b in buckets if len(b) > 1]
        if not buckets:
            return

        media_type = self._default_child_media_type()

        for bucket in buckets:
            self._write_compact_bucket(
                leaf, bucket, media_type=media_type,
            )

    def _write_compact_bucket(
        self,
        leaf: Path,
        sources: "list[Path]",
        *,
        media_type: Any,
    ) -> None:
        """Read every file in ``sources`` and write one merged file.

        Lock-free. The ordering guarantee is the only correctness
        primitive: stage + atomic rename of the merged file *first*,
        then unlink the sources. A reader that catches us mid-
        compaction sees ``{merged + sources}`` until the unlinks
        complete — duplicate rows that the response cache dedupes
        on read. A reader that opens a source while we're unlinking
        either succeeds (the open holds an fd through the unlink on
        POSIX) or treats it as a missing-file skip.

        Per-source read failures abort the bucket — better to leave
        the small files in place than land an incomplete merge.
        """
        from yggdrasil.io.buffer.base import TabularIO

        ordered_sources = sorted(sources, key=lambda p: p.name)

        tables: list[pa.Table] = []
        for source in ordered_sources:
            try:
                source_io = TabularIO.from_path(source)
            except Exception:
                return
            with source_io:
                try:
                    tables.append(source_io.read_arrow_table())
                except Exception:
                    return

        if not tables:
            return

        merged = pa.concat_tables(tables, promote_options="default")
        if merged.num_rows == 0:
            return

        staging = leaf.make_staging(media_type=media_type)
        try:
            target_io = TabularIO.from_path(staging, media_type=media_type)
            with target_io:
                target_io.write_arrow_table(merged)
        except Exception:
            try:
                staging.remove(allow_not_found=True)
            except Exception:
                pass
            return

        final_name = self._next_child_name_in(leaf, media_type=media_type)
        final_path = leaf / final_name
        try:
            staging.rename(final_path)
        except Exception:
            try:
                staging.remove(allow_not_found=True)
            except Exception:
                pass
            return

        # Sources are dropped last (post-rename) so a reader has
        # already-published merged rows to fall back on. Best-effort
        # — a missing source is a no-op (someone else won the race).
        for source in ordered_sources:
            try:
                source.remove(allow_not_found=True)
            except Exception:
                pass

    # ``_read_child_batches`` is intentionally NOT overridden. The
    # base implementation reads each child without holding a lock,
    # which is exactly what we want for ``permissive on read/write
    # locks``. The compactor's stage-rename-then-unlink discipline
    # keeps reads correct without per-file ``.rw.lock`` sidecars
    # cluttering the data directory and burning a stat per child.

    # ==================================================================
    # Spark — Arrow → Spark via the dedicated connector
    # ==================================================================

    def spark_connector(self) -> "Any":
        """Build a :class:`YGGFolderSparkConnector` over this IO."""
        from .ygg_folder_spark import YGGFolderSparkConnector

        return YGGFolderSparkConnector(self)

    def _read_spark_frame(self, options: Any) -> "Any":
        """Materialise this folder as a Spark DataFrame.

        Routes through the connector's ``mapInArrow`` pipe so
        predicate / row-size / partition-pruning options on
        :class:`CastOptions` flow through unchanged. Falls back to
        the inherited driver-side ``createDataFrame`` if anything
        Spark-related raises (no SparkSession, mapInArrow not
        supported by the local PySpark, …).
        """
        try:
            return self.spark_connector().read_batch(options=options)
        except Exception:
            return super()._read_spark_frame(options)

    def _scan_spark_frame(self, options: Any) -> "Any":
        """Streaming :class:`DataFrame` over this folder.

        Backed by Spark's native parquet streaming source through
        :meth:`YGGFolderSparkConnector.read_stream`. Falls back to
        the universal ``_scan_spark_frame`` (spill-to-temp parquet)
        on failure — typically when the folder hasn't been written
        yet so there's no committed schema.
        """
        try:
            return self.spark_connector().read_stream(options=options)
        except Exception:
            return super()._scan_spark_frame(options)
