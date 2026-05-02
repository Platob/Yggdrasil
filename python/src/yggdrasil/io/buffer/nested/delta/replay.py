"""Delta log replay.

Replay reduces the on-disk log to the live state of the table:

- The active :class:`Protocol` (most recent wins).
- The active :class:`Metadata` (most recent wins).
- The set of live :class:`AddFile` instances — every Add minus
  every subsequent Remove of the same path.
- Tracked :class:`DomainMetadata` per-domain.
- The latest commit version.

Replay strategy
---------------

1. Read ``_last_checkpoint`` if present. It points at either a v1
   single-file checkpoint or a v2 manifest.
2. Load the checkpoint:

   - **v1**: read one parquet file whose rows are action-shaped.
     Each row has at most one of {``add``, ``remove``, ``metaData``,
     ``protocol``, ``txn``, ``domainMetadata``} populated.
   - **v2**: read the top-level manifest parquet, which lists
     sidecar parquet references. Read each sidecar in turn; same
     row shape as v1.

3. Apply commit files from ``checkpoint_version + 1`` through
   the latest commit. Newline-delimited JSON, one action per
   line.

4. Validate the final Protocol against our supported feature
   set; refuse on unknown reader features.

Concurrency
-----------

Replay is read-only. Returning a stale view (a writer commits
after we sample :meth:`latest_commit_version`) is acceptable —
the caller decides whether to re-replay. Mid-replay races (a
commit file being renamed in while we walk) are handled by
treating gaps as corruption: a missing version between the
checkpoint and the latest is a real error, not a transient.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import pyarrow as pa

from yggdrasil.io.enums import MediaTypes
from yggdrasil.io.fs import Path

from .actions import (
    AddFile,
    DomainMetadata,
    Metadata,
    Protocol,
    RemoveFile,
    parse_add,
    parse_domain_metadata,
    parse_metadata,
    parse_protocol,
    parse_remove,
)
from .constants import (
    CHECKPOINT_V1_FILE_RE,
    CHECKPOINT_V1_MULTIPART_RE,
    CHECKPOINT_V2_FILE_RE,
    COMMIT_FILE_RE,
    KNOWN_REFUSED_READER_FEATURES,
    READER_VERSION_FEATURES,
    SIDECARS_DIR_NAME,
    SUPPORTED_READER_FEATURES,
    SUPPORTED_READER_VERSION_LEGACY,
)

if TYPE_CHECKING:
    from yggdrasil.io.buffer.primitive import PrimitiveIO


__all__ = [
    "ReplayResult",
    "replay_log",
    "latest_commit_version",
    "read_last_checkpoint",
]


# ---------------------------------------------------------------------------
# Replay result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ReplayResult:
    """Snapshot of a Delta table's live state at a given commit version."""
    version: int
    protocol: Protocol | None
    metadata: Metadata | None
    live_files: tuple[AddFile, ...]
    domain_metadata: Mapping[str, DomainMetadata]

    @classmethod
    def empty(cls) -> "ReplayResult":
        return cls(
            version=-1,
            protocol=None,
            metadata=None,
            live_files=(),
            domain_metadata={},
        )

    @property
    def is_empty(self) -> bool:
        return self.version == -1


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def replay_log(log_dir: Path) -> ReplayResult:
    """Replay the entire log at *log_dir*; return the live state.

    Refuses on:

    - Unknown reader features.
    - Multi-part v1 checkpoints (we don't read them).
    - Unrecognized v2 manifest fields.
    - Gaps in the commit sequence between the latest checkpoint
      and the latest commit.

    Empty / non-existent log returns :meth:`ReplayResult.empty`.
    """
    latest = latest_commit_version(log_dir)
    if latest == -1:
        return ReplayResult.empty()

    return _do_replay(log_dir, up_to=latest)


def latest_commit_version(log_dir: Path) -> int:
    """Highest commit version present, or -1 if no log."""
    if not log_dir.exists():
        return -1
    out = -1
    for child in log_dir.iterdir():
        m = COMMIT_FILE_RE.match(child.name)
        if m is None:
            continue
        v = int(m.group(1))
        if v > out:
            out = v
    return out


def read_last_checkpoint(log_dir: Path) -> Mapping[str, Any] | None:
    """Read ``_last_checkpoint`` JSON, or ``None`` if absent."""
    cp = log_dir / "_last_checkpoint"
    if not cp.exists():
        return None
    try:
        return json.loads(cp.read_text())
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _do_replay(log_dir: Path, *, up_to: int) -> ReplayResult:
    """Internal: walk checkpoint + commits up to *up_to* inclusive."""
    state = _ReplayState()

    start_version = 0
    last_cp = read_last_checkpoint(log_dir)
    if last_cp is not None:
        cp_version = int(last_cp["version"])
        _load_checkpoint_into(state, log_dir, cp_version, last_cp)
        start_version = cp_version + 1

    for version in range(start_version, up_to + 1):
        commit_path = log_dir / f"{version:020d}.json"
        if not commit_path.exists():
            raise FileNotFoundError(
                f"Delta log gap: commit version {version} missing at "
                f"{commit_path!r}. The log may be corrupted, or vacuum "
                "may have removed un-checkpointed commits."
            )
        for action in _read_commit_actions(commit_path):
            _apply_action(state, action)

    if state.protocol is None or state.metadata is None:
        raise ValueError(
            f"Delta log at {log_dir!r} has no Protocol/Metadata after "
            "replay. Either the table was never initialized or the log "
            "is corrupted."
        )

    _validate_protocol(state.protocol)

    return ReplayResult(
        version=up_to,
        protocol=state.protocol,
        metadata=state.metadata,
        live_files=tuple(state.live_files.values()),
        domain_metadata=dict(state.domain_metadata),
    )


# ---------------------------------------------------------------------------
# Mutable replay state
# ---------------------------------------------------------------------------


class _ReplayState:
    """Mutable state during a single replay pass.

    Lives only inside :func:`_do_replay`. Keeping it as an explicit
    class — rather than a tuple of locals — makes the per-action
    apply functions readable.
    """

    __slots__ = ("protocol", "metadata", "live_files", "domain_metadata")

    def __init__(self) -> None:
        self.protocol: Protocol | None = None
        self.metadata: Metadata | None = None
        # path -> AddFile
        self.live_files: dict[str, AddFile] = {}
        # domain -> DomainMetadata
        self.domain_metadata: dict[str, DomainMetadata] = {}


def _apply_action(state: _ReplayState, action: Any) -> None:
    """Fold one parsed action into the running state.

    Skips ``commitInfo`` / ``Txn`` / ``None`` (CDC) — they don't
    affect the live data set. Anything else folds in.
    """
    if action is None:
        return
    if isinstance(action, AddFile):
        state.live_files[action.path] = action
    elif isinstance(action, RemoveFile):
        state.live_files.pop(action.path, None)
    elif isinstance(action, Metadata):
        state.metadata = action
    elif isinstance(action, Protocol):
        state.protocol = action
    elif isinstance(action, DomainMetadata):
        if action.removed:
            state.domain_metadata.pop(action.domain, None)
        else:
            state.domain_metadata[action.domain] = action
    # CommitInfo and Txn are informational; nothing to do.


# ---------------------------------------------------------------------------
# Commit file reading
# ---------------------------------------------------------------------------


def _read_commit_actions(commit_path: Path) -> Iterator[Any]:
    """Yield parsed actions from one commit JSON file.

    Delta commits are tiny (kilobytes); reading the whole file at
    once is fine. We use the same ``parse_action`` dispatch as
    checkpoints so the action dataclass set is consistent.
    """
    from .actions import parse_action

    text = commit_path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        yield parse_action(raw)


# ---------------------------------------------------------------------------
# Protocol validation
# ---------------------------------------------------------------------------


def _validate_protocol(protocol: Protocol) -> None:
    """Refuse the table if its protocol exceeds what we implement."""
    # Reader features (table-features model). When minReaderVersion >=
    # READER_VERSION_FEATURES, the readerFeatures list is authoritative —
    # every entry must be in our supported set.
    if protocol.min_reader_version >= READER_VERSION_FEATURES:
        bad = [
            f for f in protocol.reader_features
            if f not in SUPPORTED_READER_FEATURES
        ]
        if bad:
            known_bad = sorted(set(bad) & KNOWN_REFUSED_READER_FEATURES)
            unknown = sorted(set(bad) - KNOWN_REFUSED_READER_FEATURES)
            msg = ["yggdrasil DeltaIO refuses to read this table."]
            if known_bad:
                msg.append(
                    f"Unsupported reader feature(s): {known_bad!r}."
                )
            if unknown:
                msg.append(
                    f"Unknown reader feature(s): {unknown!r}."
                )
            msg.append(
                f"Supported reader features: {sorted(SUPPORTED_READER_FEATURES)!r}."
            )
            raise ValueError(" ".join(msg))
        return

    # Legacy integer-only protocol. Reader version > 1 without features
    # means the table claims something we can't enumerate.
    if protocol.min_reader_version > SUPPORTED_READER_VERSION_LEGACY:
        raise ValueError(
            f"Delta table declares minReaderVersion="
            f"{protocol.min_reader_version} (legacy protocol). "
            f"yggdrasil DeltaIO supports up to "
            f"{SUPPORTED_READER_VERSION_LEGACY}, or "
            f"{READER_VERSION_FEATURES}+ with declared reader features."
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_checkpoint_into(
    state: _ReplayState,
    log_dir: Path,
    version: int,
    last_cp: Mapping[str, Any],
) -> None:
    """Load the v1 or v2 checkpoint at *version* into *state*.

    The ``_last_checkpoint`` JSON disambiguates v1 vs v2. v2 has
    either ``v2Checkpoint=true`` or non-empty ``sidecars`` /
    ``v2`` keys (writers vary). v1 may have ``parts`` > 1 for
    multi-part — we refuse those.
    """
    if _looks_like_v2(last_cp):
        _load_v2_checkpoint_into(state, log_dir, version, last_cp)
        return

    parts = last_cp.get("parts")
    if parts is not None and int(parts) > 1:
        raise ValueError(
            f"Delta multi-part v1 checkpoint at version {version} "
            f"(parts={parts}) is not supported."
        )

    _load_v1_checkpoint_into(state, log_dir, version)


def _looks_like_v2(last_cp: Mapping[str, Any]) -> bool:
    """Heuristic: does this _last_checkpoint refer to a v2 manifest?

    Reference writers vary in what fields they set. The v2 manifest
    is signaled by ``v2Checkpoint``: true at minimum; some writers
    additionally include ``v2`` (path) or sidecar references.
    """
    if last_cp.get("v2Checkpoint"):
        return True
    if last_cp.get("sidecars"):
        return True
    if "v2" in last_cp:
        return True
    return False


def _load_v1_checkpoint_into(
    state: _ReplayState,
    log_dir: Path,
    version: int,
) -> None:
    """Read ``NN…NN.checkpoint.parquet`` and fold it into *state*."""
    cp_path = log_dir / f"{version:020d}.checkpoint.parquet"

    # Defense in depth: refuse multi-part files even if _last_checkpoint
    # claimed parts=1.
    for child in log_dir.iterdir():
        m = CHECKPOINT_V1_MULTIPART_RE.match(child.name)
        if m is not None and int(m.group(1)) == version:
            raise ValueError(
                f"Found multi-part v1 checkpoint at version {version}; "
                "yggdrasil DeltaIO does not support these."
            )

    if not cp_path.exists():
        raise FileNotFoundError(
            f"v1 checkpoint at {cp_path!r} is missing but "
            "_last_checkpoint references it."
        )

    table = _read_checkpoint_parquet(cp_path)
    _fold_checkpoint_table(state, table)


def _load_v2_checkpoint_into(
    state: _ReplayState,
    log_dir: Path,
    version: int,
    last_cp: Mapping[str, Any],
) -> None:
    """Read a v2 manifest + sidecars; fold the union into *state*.

    The manifest itself is a parquet file at
    ``NN…NN.checkpoint.<uuid>.parquet``. Its rows are either:

    - Action-shaped rows (Protocol, Metadata, etc., live directly
      in the manifest), or
    - SidecarFile rows pointing at parquet files in
      ``_delta_log/_sidecars/<uuid>.parquet`` that contain Add /
      Remove rows.

    Concrete writers split the work between manifest and sidecars
    differently. We treat the union: read every row of the manifest,
    apply non-sidecar action rows, then read every referenced
    sidecar and apply its rows.

    The manifest path. We pick the manifest filename from
    ``_last_checkpoint`` if it includes a ``v2`` field; otherwise
    we list candidates and pick the one matching version.
    """
    manifest_path = _resolve_v2_manifest_path(log_dir, version, last_cp)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            f"v2 checkpoint manifest for version {version} not found "
            f"under {log_dir!r}."
        )

    manifest = _read_checkpoint_parquet(manifest_path)

    # Apply non-sidecar rows from the manifest first (Protocol /
    # Metadata typically live here).
    sidecar_refs = _fold_checkpoint_table(state, manifest, collect_sidecars=True)

    sidecars_dir = log_dir / SIDECARS_DIR_NAME
    for ref in sidecar_refs:
        sidecar_path = sidecars_dir / ref
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"v2 checkpoint sidecar {ref!r} referenced by "
                f"{manifest_path!r} not found at {sidecar_path!r}."
            )
        sidecar_table = _read_checkpoint_parquet(sidecar_path)
        _fold_checkpoint_table(state, sidecar_table, collect_sidecars=False)


def _resolve_v2_manifest_path(
    log_dir: Path,
    version: int,
    last_cp: Mapping[str, Any],
) -> Path | None:
    """Find the v2 manifest file for *version*.

    Strategy: prefer ``last_cp["v2"]["path"]`` if present (some
    writers emit it), otherwise scan the log dir for files
    matching the v2 pattern at this version.
    """
    v2 = last_cp.get("v2")
    if isinstance(v2, Mapping):
        explicit = v2.get("path") or v2.get("v2_path")
        if explicit:
            return log_dir / str(explicit)

    candidates: list[Path] = []
    for child in log_dir.iterdir():
        m = CHECKPOINT_V2_FILE_RE.match(child.name)
        if m is None:
            continue
        if int(m.group(1)) == version:
            candidates.append(child)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Multiple v2 manifests for the same version can coexist
    # transiently (concurrent writers, cleanup pending). Pick the
    # one matching ``_last_checkpoint["sizeInBytes"]`` if available;
    # otherwise pick the most-recently-modified.
    declared_size = last_cp.get("sizeInBytes")
    if declared_size is not None:
        for cand in candidates:
            try:
                if int(cand.size) == int(declared_size):
                    return cand
            except Exception:
                continue

    candidates.sort(key=lambda p: (p.mtime or 0.0), reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Checkpoint parquet reading
# ---------------------------------------------------------------------------


def _read_checkpoint_parquet(path: Path) -> pa.Table:
    """Read a checkpoint or sidecar parquet file as a pyarrow Table.

    Routes through our :class:`PrimitiveIO` so the read benefits
    from our local-fast-path and codec discipline.
    """
    from yggdrasil.io.tabular import TabularIO

    cp_io = TabularIO.from_path(path, media_type=MediaTypes.PARQUET)
    with cp_io:
        return cp_io.read_arrow_table()


def _fold_checkpoint_table(
    state: _ReplayState,
    table: pa.Table,
    *,
    collect_sidecars: bool = False,
) -> list[str]:
    """Apply checkpoint-parquet rows to *state*.

    Each row has at most one of the action-key columns populated;
    we route per-column. ``sidecar`` columns are collected for v2
    manifests and returned; v1 checkpoints / sidecar files use
    ``collect_sidecars=False`` and the returned list stays empty.
    """
    sidecars: list[str] = []
    cols = set(table.column_names)
    n = table.num_rows

    add_col = table["add"] if "add" in cols else None
    remove_col = table["remove"] if "remove" in cols else None
    meta_col = table["metaData"] if "metaData" in cols else None
    proto_col = table["protocol"] if "protocol" in cols else None
    domain_col = table["domainMetadata"] if "domainMetadata" in cols else None
    sidecar_col = table["sidecar"] if "sidecar" in cols and collect_sidecars else None
    # v2 manifests sometimes use "sidecarFile" instead of "sidecar".
    if sidecar_col is None and collect_sidecars and "sidecarFile" in cols:
        sidecar_col = table["sidecarFile"]

    for i in range(n):
        if add_col is not None and add_col[i].is_valid:
            add = parse_add(add_col[i].as_py())
            state.live_files[add.path] = add
            continue
        if remove_col is not None and remove_col[i].is_valid:
            rem = parse_remove(remove_col[i].as_py())
            state.live_files.pop(rem.path, None)
            continue
        if meta_col is not None and meta_col[i].is_valid:
            state.metadata = parse_metadata(meta_col[i].as_py())
            continue
        if proto_col is not None and proto_col[i].is_valid:
            state.protocol = parse_protocol(proto_col[i].as_py())
            continue
        if domain_col is not None and domain_col[i].is_valid:
            dm = parse_domain_metadata(domain_col[i].as_py())
            if dm.removed:
                state.domain_metadata.pop(dm.domain, None)
            else:
                state.domain_metadata[dm.domain] = dm
            continue
        if sidecar_col is not None and sidecar_col[i].is_valid:
            sidecar = sidecar_col[i].as_py()
            # Sidecar rows have at least a ``path`` field. The path
            # is relative to ``_delta_log/_sidecars/`` per spec.
            path = sidecar.get("path") if isinstance(sidecar, Mapping) else None
            if path:
                sidecars.append(str(path))

    return sidecars
