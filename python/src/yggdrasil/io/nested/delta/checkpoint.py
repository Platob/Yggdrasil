"""Checkpoint writers — V1 (single parquet) and V2 (manifest + sidecars).

A checkpoint is "the state of the table at version N, materialized
as one or more parquet files plus a pointer at ``_last_checkpoint``."
Subsequent reads can start from the checkpoint and only apply the
JSON commits with version > N — that's the whole point of the
mechanism.

V1 layout
---------

    _delta_log/
        <N>.checkpoint.parquet      ← one row per action (Protocol,
                                      Metadata, every active AddFile,
                                      live Txns, live DomainMetadata)
        _last_checkpoint            ← ``{"version": N, "size": <count>}``

V2 layout
---------

    _delta_log/
        <N>.checkpoint.<uuid>.json  ← manifest, one action per line
                                      (one or more ``{"sidecar": ...}``
                                      lines plus a ``checkpointMetadata``
                                      tail)
        _sidecars/
            <sc-uuid>.parquet       ← actions, one row per action
        _last_checkpoint            ← ``{"version": N, "size": <count>,
                                          "v2Checkpoint": {"version": N}}``

Both flavors share the per-row encoding: each row has exactly one
populated column, the rest are null. The column name is the action
key (``add``, ``remove``, ``metaData``, ``protocol``, ``txn``,
``domainMetadata``). Pyarrow infers the union schema from the
materialized rows.

The writer here doesn't try to be incremental — the snapshot is
already in memory, and emitting the full action list as one parquet
is fast enough that splitting into per-class sidecars would be
premature work. Modern V2-using engines (Databricks, Spark) do split
on retention-domain boundaries; we leave that for a follow-up when
a real caller needs it.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME,
    SIDECARS_DIR_NAME,
    format_checkpoint_v1_name,
    format_checkpoint_v2_manifest_name,
    format_checkpoint_v2_sidecar_name,
)
from yggdrasil.io.nested.delta.protocol import Txn

if TYPE_CHECKING:
    from yggdrasil.io.nested.delta.snapshot import Snapshot
    from yggdrasil.io.path import Path


__all__ = ["write_checkpoint", "update_last_checkpoint"]


def write_checkpoint(
    snap: "Snapshot",
    *,
    log_path: "Path",
    kind: str = "v1",
) -> Optional[int]:
    """Materialize *snap* as a V1 or V2 checkpoint under *log_path*.

    Returns the number of actions written, or ``None`` when the
    snapshot was empty (no protocol / metadata / files / txns) and
    nothing got written. The ``_last_checkpoint`` pointer is *not*
    updated by this function — :func:`update_last_checkpoint` does
    that, separately, so callers writing multi-step checkpoints can
    decide when the pointer becomes visible.
    """
    actions = _snapshot_to_actions(snap)
    if not actions:
        return None

    if kind == "v2":
        _write_v2(snap.version, actions, log_path=log_path)
    else:
        _write_v1(snap.version, actions, log_path=log_path)
    return len(actions)


def update_last_checkpoint(
    *,
    log_path: "Path",
    version: int,
    size: int,
    kind: str = "v1",
) -> None:
    """Refresh ``_last_checkpoint`` so readers find the new checkpoint
    without scanning the directory.
    """
    payload: Dict[str, Any] = {"version": int(version), "size": int(size)}
    if kind == "v2":
        payload["v2Checkpoint"] = {"version": int(version)}
    body = ygg_json.dumps(payload, separators=(",", ":"))
    if isinstance(body, str):
        body = body.encode("utf-8")
    ptr = log_path / LAST_CHECKPOINT_NAME
    with ptr.open("wb") as bio:
        bio.truncate(0)
        bio.write_bytes(body)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_v1(version: int, actions: List[dict], *, log_path: "Path") -> None:
    """V1 — single ``<version>.checkpoint.parquet`` next to the commits."""
    from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions

    ck_path = log_path / format_checkpoint_v1_name(version)
    table = _actions_to_arrow_table(actions)

    leaf = ParquetFile(holder=ck_path, owns_holder=False)
    with leaf as opened:
        opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))


def _write_v2(version: int, actions: List[dict], *, log_path: "Path") -> None:
    """V2 — one sidecar parquet referenced from a manifest JSON."""
    from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions

    sidecar_uuid = uuid.uuid4().hex
    sidecar_dir = log_path / SIDECARS_DIR_NAME
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    sidecar_name = format_checkpoint_v2_sidecar_name(sidecar_uuid)
    sidecar_path = sidecar_dir / sidecar_name

    table = _actions_to_arrow_table(actions)
    leaf = ParquetFile(holder=sidecar_path, owns_holder=False)
    with leaf as opened:
        opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))

    manifest_uuid = uuid.uuid4().hex
    manifest_path = log_path / format_checkpoint_v2_manifest_name(
        version,
        manifest_uuid,
    )
    manifest_lines = [
        ygg_json.dumps(
            {
                "sidecar": {
                    "path": sidecar_name,
                    "sizeInBytes": int(sidecar_path.size),
                    "modificationTime": int(time.time() * 1000),
                }
            },
            separators=(",", ":"),
            to_bytes=False,
        ),
        ygg_json.dumps(
            {"checkpointMetadata": {"version": int(version), "flavor": "v2"}},
            separators=(",", ":"),
            to_bytes=False,
        ),
    ]
    body = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    with manifest_path.open("wb") as bio:
        bio.truncate(0)
        bio.write_bytes(body)


# ---------------------------------------------------------------------------
# Snapshot → action list
# ---------------------------------------------------------------------------


def _snapshot_to_actions(snap: "Snapshot") -> "List[dict]":
    """Project *snap* into the action list a checkpoint should hold.

    Order matches Delta's reduction-friendly convention: ``protocol``
    first, then ``metaData``, then live ``txn`` / ``domainMetadata``,
    then the active ``add`` files. ``remove`` actions are *not* part
    of a checkpoint — by definition a checkpoint is the active set,
    and removed paths are gone from it. A reader replaying from this
    checkpoint never sees the historical removes; it only sees the
    survivors.
    """
    out: List[dict] = []
    if snap.protocol is not None:
        out.append(snap.protocol.to_action())
    if snap.metadata is not None:
        out.append(snap.metadata.to_action())
    for app, v in snap.txns.items():
        out.append(Txn(app_id=app, version=v).to_action())
    for dm in snap.domain_metadata.values():
        out.append(dm.to_action())
    for add in snap.active_files.values():
        out.append(add.to_action())
    return out


def _actions_to_arrow_table(actions: "List[dict]") -> pa.Table:
    """Lay out checkpoint actions as one row per action.

    Each row has exactly one populated column (the action's key);
    the rest are null. Pyarrow infers the union schema from the rows.

    Empty nested dicts get pruned to ``None`` first — pyarrow's
    type-inference can't represent ``struct<>`` (zero-field struct)
    in parquet, so a Metadata action with ``"options": {}`` would
    blow up the writer. Dropping the empty value to ``None`` lets
    the type land as nullable, which parquet *can* write.
    """
    rows: list[dict] = []
    for entry in actions:
        if not entry:
            continue
        key = next(iter(entry))
        rows.append({key: _drop_empties(entry[key])})
    if not rows:
        return pa.table({})
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        k = next(iter(r))
        if k not in seen:
            seen.add(k)
            keys.append(k)
    flattened = [{k: (r[k] if k in r else None) for k in keys} for r in rows]
    return pa.Table.from_pylist(flattened)


def _drop_empties(value: Any) -> Any:
    """Recursively prune empty ``dict`` / ``list`` values to ``None``.

    Pyarrow refuses to write a ``struct<>`` (zero-field struct) into
    parquet — and that's exactly what an empty ``{}`` infers to. We
    walk the action payload once, dropping empties, before handing it
    to :meth:`pa.Table.from_pylist`. Functional shape: input is JSON-
    safe (dict / list / scalars) and the output is the same shape
    with empty containers replaced by ``None``.
    """
    if isinstance(value, dict):
        cleaned = {k: _drop_empties(v) for k, v in value.items()}
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = [_drop_empties(v) for v in value]
        cleaned_list = [v for v in cleaned_list if v is not None]
        return cleaned_list or None
    return value
