"""Benchmark Databricks ``Path`` surface + remote-call counts.

Runs against :class:`DBFSPath`, :class:`VolumePath`, :class:`WorkspacePath`
with stubbed SDK clients (``MagicMock`` shaped like
``DatabricksClient.workspace_client()``) — no real workspace required.
Each scenario captures *both* a microbench time AND the number of
SDK calls the operation makes, so the "fewer round trips" half of
the performance story stays visible.

The companion :mod:`bench_io_remote` script measures real network
behavior; this one measures the in-process overhead and the
per-op SDK-call count, runs deterministic in CI, and lets us
catch regressions where an optimization "looks faster locally"
but quietly issues an extra ``get_status`` / ``head_object`` round
trip.

Usage::

    PYTHONPATH=src python benchmarks/io/path/bench_remote_databricks.py --repeat 3
"""
from __future__ import annotations

import argparse
import io as _stdio
import statistics
import time
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

import pyarrow as pa

from yggdrasil.databricks.fs.dbfs_path import DBFSPath
from yggdrasil.databricks.fs.volume_path import VolumePath
from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.io.path.remote_path import RemotePath
from yggdrasil.io.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.csv_file import CSVFile
from yggdrasil.io.ndjson_file import NDJSONFile
from yggdrasil.io.parquet_file import ParquetFile


# ---------------------------------------------------------------------------
# Timing + call-count helpers
# ---------------------------------------------------------------------------


def _time_one(label: str, fn: Callable[[], None], *, repeat: int, inner: int) -> dict:
    fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {
        "label": label,
        "best": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
    }


def _fmt(r: dict) -> str:
    scale = 1e6
    unit = "us"
    if r["best"] < 1e-6:
        scale, unit = 1e9, "ns"
    elif r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    extra = ""
    if "calls" in r:
        extra = f"  sdk_calls={r['calls']:>3d}"
    return (
        f"{r['label']:<62s}  "
        f"best={r['best']*scale:8.2f} {unit}  "
        f"median={r['median']*scale:8.2f} {unit}  "
        f"mean={r['mean']*scale:8.2f} {unit}{extra}"
    )


def _total_sdk_calls(workspace: MagicMock) -> int:
    """Sum of every ``method.call_count`` under the workspace mock.

    Walks one layer deep into each attribute (e.g.
    ``workspace.dbfs.get_status``) and reads ``.call_count``. The
    mock children for top-level attributes (``dbfs``, ``files``,
    ``workspace``) themselves carry a ``call_count`` (incremented
    every time the test treats them as callables — they're not, but
    MagicMock counts the access if any caller does), so the recursive
    sum overcounts; one layer is the right depth for the actual SDK
    method surface."""
    total = 0
    for attr in vars(type(workspace)):
        continue  # pragma: no cover — vars() over a class is empty
    for name, surface in workspace._mock_children.items():
        # ``surface`` is the mock for e.g. workspace.dbfs — each of
        # its child methods is the actual SDK entry point.
        for method_name, method in surface._mock_children.items():
            total += int(method.call_count)
        total += int(surface.call_count)
    return total


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


def _stub_databricks_service(*, payload: bytes = b"") -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return ``(service, client, workspace)`` shaped like a
    :class:`DatabricksService` / :class:`DatabricksClient` /
    :class:`databricks.sdk.WorkspaceClient` chain.

    :class:`DatabricksPath` reaches the workspace SDK through
    ``self.service.client.workspace_client()`` — wiring a service-
    shaped mock keeps the construction path independent of real
    Databricks credentials. Configures the most common SDK return
    values so the path operations succeed end-to-end:

    * ``dbfs.get_status`` → file with ``len(payload)`` bytes.
    * ``dbfs.read`` → base64 of ``payload`` in a single page (short
      page = EOF signal).
    * ``files.get_metadata`` / ``files.download`` → ``payload``.
    * ``workspace.workspace.get_status`` → NOTEBOOK kind.
    """
    import base64
    service = MagicMock()
    client = service.client
    workspace = client.workspace_client.return_value

    # DBFS surface
    workspace.dbfs.get_status.return_value = SimpleNamespace(
        is_dir=False, file_size=len(payload), modification_time=1_700_000_000_000,
    )
    workspace.dbfs.read.return_value = SimpleNamespace(
        data=base64.b64encode(payload).decode("ascii"),
    )

    workspace.dbfs.list.return_value = [
        SimpleNamespace(path=f"/tmp/file-{i:03d}.parquet", is_dir=False)
        for i in range(32)
    ]

    # Volumes Files API surface
    workspace.files.get_metadata.return_value = SimpleNamespace(
        content_length=len(payload),
        content_type="application/octet-stream",
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )
    workspace.files.get_directory_metadata.side_effect = NotFound()
    workspace.files.download.return_value = SimpleNamespace(
        contents=_ReadOnce(payload),
        content_type="application/octet-stream",
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )
    workspace.files.list_directory_contents.return_value = [
        SimpleNamespace(
            path=f"/Volumes/c/s/v/file-{i:03d}.parquet",
            is_directory=False,
        )
        for i in range(32)
    ]

    # Workspace API surface
    workspace.workspace.get_status.return_value = SimpleNamespace(
        object_type=SimpleNamespace(name="NOTEBOOK"),
        size=len(payload),
        modified_at=1_700_000_000_000,
    )
    return service, client, workspace


class _ReadOnce:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._read = False

    def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._payload


class NotFound(Exception):
    """Duck-typed ``databricks.sdk.NotFound`` so the path's error
    classifier (``_looks_like_not_found``) treats it as missing."""


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def scenarios(repeat: int) -> list[dict]:
    out: list[dict] = []

    parquet_payload = _build_parquet_bytes()

    # -------------------------------------------------------------------
    # DBFS construction + traversal
    # -------------------------------------------------------------------

    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()

    out.append(_time_one(
        "DBFSPath('/dbfs/x') singleton hit",
        lambda: DBFSPath("/dbfs/foo/bar.parquet", service=service),
        repeat=repeat, inner=20_000,
    ))

    deep = DBFSPath("/dbfs/a/b/c/d/e/file.parquet", service=service)
    out.append(_time_one(
        "DBFSPath.parent (5 levels)",
        lambda: _walk_parents(deep, 5),
        repeat=repeat, inner=5_000,
    ))

    base = DBFSPath("/dbfs/base/", service=service)
    out.append(_time_one(
        "DBFSPath.joinpath('sub', 'file.csv')",
        lambda: base.joinpath("sub", "file.csv"),
        repeat=repeat, inner=20_000,
    ))

    # -------------------------------------------------------------------
    # DBFS — stat / read with SDK call counts
    # -------------------------------------------------------------------

    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    p = DBFSPath("/dbfs/x.parquet", service=service)
    p.invalidate_singleton(remove_global=False)
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.read.reset_mock()
    p.exists()
    out.append({
        "label": "DBFSPath.exists() (cold) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.dbfs.get_status.call_count
            + workspace.dbfs.read.call_count
        ),
    })

    # Warm — should drop to 0 SDK calls.
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.read.reset_mock()
    p.exists()
    out.append({
        "label": "DBFSPath.exists() (warm) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.dbfs.get_status.call_count
            + workspace.dbfs.read.call_count
        ),
    })

    # read_bytes — should be one ``dbfs.read`` round trip (no preceding
    # ``get_status`` because :meth:`DatabricksPath.read_mv` short-circuits
    # the whole-file case).
    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    p = DBFSPath("/dbfs/x.parquet", service=service)
    p.invalidate_singleton(remove_global=False)
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.read.reset_mock()
    p.read_bytes()
    out.append({
        "label": "DBFSPath.read_bytes() (cold) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.dbfs.get_status.call_count
            + workspace.dbfs.read.call_count
        ),
    })

    # -------------------------------------------------------------------
    # Volume — stat / read with SDK call counts
    # -------------------------------------------------------------------

    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.parquet", service=service)
    p.invalidate_singleton(remove_global=False)
    workspace.files.get_metadata.reset_mock()
    workspace.files.download.reset_mock()
    p.exists()
    out.append({
        "label": "VolumePath.exists() (cold) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.files.get_metadata.call_count
            + workspace.files.download.call_count
        ),
    })

    workspace.files.get_metadata.reset_mock()
    workspace.files.download.reset_mock()
    p.exists()
    out.append({
        "label": "VolumePath.exists() (warm) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.files.get_metadata.call_count
            + workspace.files.download.call_count
        ),
    })

    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.parquet", service=service)
    p.invalidate_singleton(remove_global=False)
    workspace.files.get_metadata.reset_mock()
    workspace.files.download.reset_mock()
    p.read_bytes()
    out.append({
        "label": "VolumePath.read_bytes() (cold) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.files.get_metadata.call_count
            + workspace.files.download.call_count
        ),
    })

    # -------------------------------------------------------------------
    # Tabular IO ↔ Path interaction — the headline scenario
    # -------------------------------------------------------------------
    # ParquetFile(path=remote).read_arrow_table() is the canonical "give
    # me a frame from this remote object". We want the operation to
    # bottom out in one download — the optimizer should fold the
    # size-probe and the read into a single round trip.

    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    pio = ParquetFile(holder=VolumePath(
        "/Volumes/cat/sch/vol/x.parquet", service=service,
    ))
    pio.holder.invalidate_singleton(remove_global=False)
    workspace.files.get_metadata.reset_mock()
    workspace.files.download.reset_mock()
    pio.read_arrow_table()
    out.append({
        "label": "ParquetFile(VolumePath).read_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.files.get_metadata.call_count
            + workspace.files.download.call_count
        ),
    })

    # Same for DBFS.
    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    pio = ParquetFile(holder=DBFSPath("/dbfs/x.parquet", service=service))
    pio.holder.invalidate_singleton(remove_global=False)
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.read.reset_mock()
    pio.read_arrow_table()
    out.append({
        "label": "ParquetFile(DBFSPath).read_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.dbfs.get_status.call_count
            + workspace.dbfs.read.call_count
        ),
    })

    # collect_schema — should hit the cheap footer read path.
    service, client, workspace = _stub_databricks_service(payload=parquet_payload)
    RemotePath._INSTANCES.clear()
    pio = ParquetFile(holder=VolumePath(
        "/Volumes/cat/sch/vol/x.parquet", service=service,
    ))
    pio.holder.invalidate_singleton(remove_global=False)
    workspace.files.get_metadata.reset_mock()
    workspace.files.download.reset_mock()
    pio.collect_schema()
    out.append({
        "label": "ParquetFile(VolumePath).collect_schema — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": (
            workspace.files.get_metadata.call_count
            + workspace.files.download.call_count
        ),
    })

    # Timings for the same operations
    out.append(_time_one(
        "ParquetFile(VolumePath).read_arrow_table",
        lambda: _read_via_volume_path(service, parquet_payload),
        repeat=repeat, inner=200,
    ))

    # -------------------------------------------------------------------
    # Write SDK-call profiles
    # -------------------------------------------------------------------
    # Measure how many backend calls each write shape costs. The
    # page-buffered truncate override should eliminate the extra
    # download + re-upload that the base Path.truncate(0) issues.

    write_payload = b"x" * 8192

    # VolumePath — write_all (truncate + write_bytes under the hood)
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    workspace.files.delete.reset_mock()
    p.write_all(write_payload)
    out.append({
        "label": "VolumePath.write_all(8 KiB) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # VolumePath — write_bytes (single upload)
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    p.write_bytes(write_payload)
    out.append({
        "label": "VolumePath.write_bytes(8 KiB) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # VolumePath — open("wb") + write + close (cursor path)
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    with p.open("wb") as f:
        f.write(write_payload)
    out.append({
        "label": "VolumePath open('wb') write close — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # DBFSPath — write_all
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = DBFSPath("/dbfs/out.bin", service=service)
    workspace.dbfs.put.reset_mock()
    workspace.dbfs.read.reset_mock()
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.delete.reset_mock()
    p.write_all(write_payload)
    out.append({
        "label": "DBFSPath.write_all(8 KiB) — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # DBFSPath — open("wb") + write + close
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = DBFSPath("/dbfs/out.bin", service=service)
    workspace.dbfs.put.reset_mock()
    workspace.dbfs.read.reset_mock()
    workspace.dbfs.get_status.reset_mock()
    workspace.dbfs.delete.reset_mock()
    with p.open("wb") as f:
        f.write(write_payload)
    out.append({
        "label": "DBFSPath open('wb') write close — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # VolumePath — seek + multi-write (still 1 call)
    service, client, workspace = _stub_databricks_service(payload=b"old")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    with p.open("wb") as f:
        f.write(b"head")
        f.seek(100)
        f.write(b"tail")
    out.append({
        "label": "VolumePath open('wb') seek write write — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # ParquetFile write → VolumePath via write_arrow_io
    service, client, workspace = _stub_databricks_service(payload=b"")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.parquet", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    ParquetFile(holder=p, owns_holder=False).write_arrow_table(
        pa.table({"id": pa.array(range(100), type=pa.int64())}),
    )
    out.append({
        "label": "ParquetFile(VolumePath).write_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    _bench_table = pa.table({"id": pa.array(range(100), type=pa.int64())})

    # ArrowIPCFile write → VolumePath
    service, client, workspace = _stub_databricks_service(payload=b"")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.arrow", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    ArrowIPCFile(holder=p, owns_holder=False).write_arrow_table(_bench_table)
    out.append({
        "label": "ArrowIPCFile(VolumePath).write_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # CSVFile write → VolumePath
    service, client, workspace = _stub_databricks_service(payload=b"")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.csv", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    CSVFile(holder=p, owns_holder=False).write_arrow_table(_bench_table)
    out.append({
        "label": "CSVFile(VolumePath).write_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # NDJSONFile write → VolumePath
    service, client, workspace = _stub_databricks_service(payload=b"")
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/x.ndjson", service=service)
    workspace.files.upload.reset_mock()
    workspace.files.download.reset_mock()
    workspace.files.get_metadata.reset_mock()
    NDJSONFile(holder=p, owns_holder=False).write_arrow_table(_bench_table)
    out.append({
        "label": "NDJSONFile(VolumePath).write_arrow_table — SDK calls",
        "best": 0.0, "median": 0.0, "mean": 0.0,
        "calls": _total_sdk_calls(workspace),
    })

    # Write timing — VolumePath.write_all
    out.append(_time_one(
        "VolumePath.write_all(8 KiB)",
        lambda: _write_volume(write_payload),
        repeat=repeat, inner=500,
    ))

    # Write timing — VolumePath open("wb") + write
    out.append(_time_one(
        "VolumePath open('wb') write close",
        lambda: _write_volume_cursor(write_payload),
        repeat=repeat, inner=500,
    ))

    return out


def _write_volume(payload: bytes) -> None:
    service, _, _ = _stub_databricks_service()
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    p.write_all(payload)


def _write_volume_cursor(payload: bytes) -> None:
    service, _, _ = _stub_databricks_service()
    RemotePath._INSTANCES.clear()
    p = VolumePath("/Volumes/cat/sch/vol/out.bin", service=service)
    with p.open("wb") as f:
        f.write(payload)


def _read_via_volume_path(service, payload):
    RemotePath._INSTANCES.clear()
    workspace = service.client.workspace_client.return_value
    # Re-prime the download response so each call gets a fresh _ReadOnce.
    workspace.files.download.return_value = SimpleNamespace(
        contents=_ReadOnce(payload),
        content_type="application/octet-stream",
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
    )
    return ParquetFile(holder=VolumePath(
        "/Volumes/cat/sch/vol/x.parquet", service=service,
    )).read_arrow_table()


def _walk_parents(p, n: int):
    for _ in range(n):
        p = p.parent


def _build_parquet_bytes() -> bytes:
    """A small Parquet file used by the read scenarios.

    The mock SDK hands the same bytes back on every ``download`` so
    every iteration sees a real, decodable Parquet stream — the
    cost reflects real format parsing, not just byte counting.
    """
    table = pa.table({
        "id": pa.array(range(1024), type=pa.int64()),
        "v": pa.array([1.5] * 1024, type=pa.float64()),
    })
    sink = _stdio.BytesIO()
    import pyarrow.parquet as pq
    pq.write_table(table, sink)
    return sink.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3,
                    help="Outer repeat count per scenario.")
    args = ap.parse_args()

    print(f"# repeat={args.repeat}")
    print(f"# {'label':<62s}  {'best':>14s}  {'median':>16s}  {'mean':>14s}")
    for row in scenarios(args.repeat):
        print(_fmt(row))


if __name__ == "__main__":
    main()
