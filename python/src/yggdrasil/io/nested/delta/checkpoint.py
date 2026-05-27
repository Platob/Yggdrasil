"""Checkpoint writers — V1 (single parquet) and V2 (manifest + sidecars).

V1 layout
---------
    _delta_log/
        <N>.checkpoint.parquet      <- one row per action
        _last_checkpoint            <- {"version": N, "size": <count>}

V2 layout
---------
    _delta_log/
        <N>.checkpoint.<uuid>.json  <- manifest with sidecar references
        _sidecars/
            <sc-uuid>.parquet       <- actions, one row per action
        _last_checkpoint            <- {"version": N, "size": <count>,
                                        "v2Checkpoint": {"version": N,
                                        "sidecarFiles": [...]}}

V2 checkpoints support multi-sidecar writes and per-action-class splits.
This implementation writes a single sidecar for simplicity but correctly
reads multi-sidecar checkpoints produced by other engines.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

from yggdrasil.enums import Mode
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
    from yggdrasil.path import Path


__all__ = ["write_checkpoint", "update_last_checkpoint"]


def write_checkpoint(
    snap: "Snapshot",
    *,
    log_path: "Path",
    kind: str = "v1",
) -> Optional[int]:
    """Materialize *snap* as a V1 or V2 checkpoint under *log_path*.

    Returns the number of actions written, or ``None`` when the snapshot
    was empty.
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
    sidecar_files: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Refresh ``_last_checkpoint``."""
    payload: Dict[str, Any] = {"version": int(version), "size": int(size)}
    if kind == "v2":
        v2_info: Dict[str, Any] = {"version": int(version)}
        if sidecar_files:
            v2_info["sidecarFiles"] = sidecar_files
        payload["v2Checkpoint"] = v2_info
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
    from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions

    ck_path = log_path / format_checkpoint_v1_name(version)
    table = _actions_to_arrow_table(actions)

    leaf = ParquetFile(holder=ck_path, owns_holder=False)
    with leaf as opened:
        opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))


def _write_v2(version: int, actions: List[dict], *, log_path: "Path") -> None:
    """V2 — sidecar parquet(s) referenced from a manifest JSON.

    Writes per-action-class sidecars when the action count is large
    enough to benefit from the split; otherwise falls back to a single
    sidecar.
    """
    from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions

    sidecar_dir = log_path / SIDECARS_DIR_NAME
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Split actions by type for per-class sidecars
    sidecar_groups: Dict[str, List[dict]] = {}
    for action in actions:
        key = next(iter(action))
        sidecar_groups.setdefault(key, []).append(action)

    sidecar_entries: List[Dict[str, Any]] = []

    if len(sidecar_groups) > 1 and len(actions) > 100:
        # Multi-sidecar: one parquet per action class
        for action_class, class_actions in sidecar_groups.items():
            sc_uuid = uuid.uuid4().hex
            sc_name = format_checkpoint_v2_sidecar_name(sc_uuid)
            sc_path = sidecar_dir / sc_name

            table = _actions_to_arrow_table(class_actions)
            leaf = ParquetFile(holder=sc_path, owns_holder=False)
            with leaf as opened:
                opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))

            sidecar_entries.append({
                "path": sc_name,
                "sizeInBytes": int(sc_path.size),
                "modificationTime": int(time.time() * 1000),
                "tags": {"actionType": action_class},
            })
    else:
        # Single sidecar
        sc_uuid = uuid.uuid4().hex
        sc_name = format_checkpoint_v2_sidecar_name(sc_uuid)
        sc_path = sidecar_dir / sc_name

        table = _actions_to_arrow_table(actions)
        leaf = ParquetFile(holder=sc_path, owns_holder=False)
        with leaf as opened:
            opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))

        sidecar_entries.append({
            "path": sc_name,
            "sizeInBytes": int(sc_path.size),
            "modificationTime": int(time.time() * 1000),
        })

    manifest_uuid = uuid.uuid4().hex
    manifest_path = log_path / format_checkpoint_v2_manifest_name(
        version,
        manifest_uuid,
    )
    manifest_lines: List[str] = []
    for entry in sidecar_entries:
        manifest_lines.append(
            ygg_json.dumps(
                {"sidecar": entry},
                separators=(",", ":"),
                to_bytes=False,
            )
        )
    manifest_lines.append(
        ygg_json.dumps(
            {"checkpointMetadata": {"version": int(version), "flavor": "v2"}},
            separators=(",", ":"),
            to_bytes=False,
        ),
    )
    body = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    with manifest_path.open("wb") as bio:
        bio.truncate(0)
        bio.write_bytes(body)


# ---------------------------------------------------------------------------
# Snapshot -> action list
# ---------------------------------------------------------------------------


def _snapshot_to_actions(snap: "Snapshot") -> "List[dict]":
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
    if isinstance(value, dict):
        cleaned = {k: _drop_empties(v) for k, v in value.items()}
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = [_drop_empties(v) for v in value]
        cleaned_list = [v for v in cleaned_list if v is not None]
        return cleaned_list or None
    return value
