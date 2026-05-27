"""Checkpoint writers — V1 (single parquet) and V2 (manifest + sidecars)."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import (
    LAST_CHECKPOINT_NAME, SIDECARS_DIR_NAME,
    format_checkpoint_v1_name, format_checkpoint_v2_manifest_name,
    format_checkpoint_v2_sidecar_name,
)
from yggdrasil.io.nested.delta.protocol import Txn

if TYPE_CHECKING:
    from yggdrasil.io.nested.delta.snapshot import Snapshot
    from yggdrasil.path import Path

__all__ = ["write_checkpoint", "update_last_checkpoint"]


def write_checkpoint(snap: "Snapshot", *, log_path: "Path",
                     kind: str = "v1") -> "Optional[tuple[int, Optional[List[Dict[str, Any]]]]]":
    # Build action list from snapshot
    actions: List[dict] = []
    if snap.protocol is not None: actions.append(snap.protocol.to_action())
    if snap.metadata is not None: actions.append(snap.metadata.to_action())
    for app, v in snap.txns.items(): actions.append(Txn(app_id=app, version=v).to_action())
    for dm in snap.domain_metadata.values(): actions.append(dm.to_action())
    for add in snap.active_files.values(): actions.append(add.to_action())
    if not actions:
        return None

    from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions

    if kind == "v2":
        sidecar_dir = log_path / SIDECARS_DIR_NAME
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        # Split by action class for multi-sidecar when large
        groups: Dict[str, List[dict]] = {}
        for a in actions: groups.setdefault(next(iter(a)), []).append(a)

        sidecar_entries: List[Dict[str, Any]] = []
        if len(groups) > 1 and len(actions) > 100:
            for cls, cls_actions in groups.items():
                sc_uuid = uuid.uuid4().hex
                sc_name = format_checkpoint_v2_sidecar_name(sc_uuid)
                sc_path = sidecar_dir / sc_name
                with ParquetFile(holder=sc_path, owns_holder=False) as f:
                    f._write_arrow_table(_to_arrow(cls_actions), ParquetOptions(mode=Mode.OVERWRITE))
                sidecar_entries.append({"path": sc_name, "sizeInBytes": int(sc_path.size),
                                        "modificationTime": int(time.time() * 1000),
                                        "tags": {"actionType": cls}})
        else:
            sc_uuid = uuid.uuid4().hex
            sc_name = format_checkpoint_v2_sidecar_name(sc_uuid)
            sc_path = sidecar_dir / sc_name
            with ParquetFile(holder=sc_path, owns_holder=False) as f:
                f._write_arrow_table(_to_arrow(actions), ParquetOptions(mode=Mode.OVERWRITE))
            sidecar_entries.append({"path": sc_name, "sizeInBytes": int(sc_path.size),
                                    "modificationTime": int(time.time() * 1000)})

        manifest_lines = [ygg_json.dumps({"sidecar": e}, separators=(",", ":"), to_bytes=False)
                          for e in sidecar_entries]
        manifest_lines.append(ygg_json.dumps({"checkpointMetadata": {"version": int(snap.version)}},
                                              separators=(",", ":"), to_bytes=False))
        manifest_path = log_path / format_checkpoint_v2_manifest_name(snap.version, uuid.uuid4().hex)
        with manifest_path.open("wb") as bio:
            bio.truncate(0); bio.write_bytes(("\n".join(manifest_lines) + "\n").encode("utf-8"))
        return len(actions), sidecar_entries
    else:
        ck_path = log_path / format_checkpoint_v1_name(snap.version)
        with ParquetFile(holder=ck_path, owns_holder=False) as f:
            f._write_arrow_table(_to_arrow(actions), ParquetOptions(mode=Mode.OVERWRITE))
        return len(actions), None


def update_last_checkpoint(*, log_path: "Path", version: int, size: int,
                           kind: str = "v1",
                           sidecar_files: Optional[List[Dict[str, Any]]] = None) -> None:
    payload: Dict[str, Any] = {"version": int(version), "size": int(size)}
    if kind == "v2":
        v2: Dict[str, Any] = {"version": int(version)}
        if sidecar_files: v2["sidecarFiles"] = sidecar_files
        payload["v2Checkpoint"] = v2
    body = ygg_json.dumps(payload, separators=(",", ":"))
    if isinstance(body, str): body = body.encode("utf-8")
    with (log_path / LAST_CHECKPOINT_NAME).open("wb") as bio:
        bio.truncate(0); bio.write_bytes(body)


def _to_arrow(actions: "List[dict]") -> pa.Table:
    def _drop(value: Any) -> Any:
        if isinstance(value, dict):
            c = {k: _drop(v) for k, v in value.items()}
            c = {k: v for k, v in c.items() if v is not None}
            return c or None
        if isinstance(value, list):
            c = [_drop(v) for v in value]
            c = [v for v in c if v is not None]
            return c or None
        return value

    rows = [{next(iter(e)): _drop(e[next(iter(e))])} for e in actions if e]
    if not rows: return pa.table({})
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        k = next(iter(r))
        if k not in seen: seen.add(k); keys.append(k)
    return pa.Table.from_pylist([{k: (r[k] if k in r else None) for k in keys} for r in rows])
