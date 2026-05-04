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

import os
import time
from typing import Any, ClassVar, TYPE_CHECKING

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.io.enums import MimeType, MimeTypes
from yggdrasil.io.fs import Path
from yggdrasil.io.stats import Stats, STATS_FILENAME
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

    def checkpoint(
        self,
        message: "str | None" = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append a checkpoint record to ``.ygg/checkpoints.jsonl``.

        A checkpoint is the streaming-write equivalent of a commit:
        it records that the writer considers everything written up
        to *this point* a coherent unit, with optional caller-supplied
        ``message`` and arbitrary ``**extra`` fields. Concurrent
        ``checkpoint()`` callers serialise on a transient ``.w.lock``
        against the log file so JSON lines never tear.
        """
        record: dict[str, Any] = {
            "id": self._next_checkpoint_id(),
            "ts": time.time(),
            "pid": os.getpid(),
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
        out: list[dict[str, Any]] = []
        for raw in blob.decode("utf-8", errors="replace").splitlines():
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

    def latest_checkpoint(self) -> "dict[str, Any] | None":
        """The most recent checkpoint record, or ``None`` if none."""
        records = self.list_checkpoints()
        return records[-1] if records else None

    def _next_checkpoint_id(self) -> int:
        prev = self.latest_checkpoint()
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
        """
        try:
            entries = self.path.iterdir()
        except (OSError, FileNotFoundError):
            return []
        names: list[str] = []
        for entry in entries:
            if self._is_ignored_path(entry):
                continue
            names.append(entry.name)
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

        per_child: list[Stats] = []
        for child in self.iter_children(options=options):
            try:
                child_stats = Stats.compute(
                    child.read_arrow_table(options=options),
                    name=child.path.name,
                    distinct=distinct,
                )
            except Exception:
                continue
            per_child.append(child_stats)

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
