"""Benchmark the :class:`ParquetFolder` read path.

A :class:`~yggdrasil.parquet.ParquetFolder` reads a directory of parquet
part files as one **dataset**: the engine discovers Hive partitions on the
path, pushes the projection into each leaf, and skips whole files / row
groups whose footer min/max can't satisfy the predicate — in one native
pass. This bench pins the surfaces that matters:

* **collect_schema** — :meth:`ParquetFolder._collect_schema` reads the
  shape from the **first part file's footer** (not the ``.ygg`` sidecar),
  so it's correct for folders written by external tools and never stale.
  The footer read is a metadata-only scan; the row compares it against the
  generic :class:`Folder` sidecar read over the same directory.
* **full read** — arrow + polars, the unfiltered scan baseline.
* **predicate pushdown** — a clustered ``id`` predicate that the dataset
  satisfies from per-file statistics, skipping most parts; contrast with
  the full read to read the skip benefit.
* **partition pruning** — a Hive ``region == r`` predicate that drops whole
  ``region=<v>/`` subtrees without opening their part files.

Fixtures are local temp folders built once at import (cleaned at exit). The
flat folder lays ``id`` out monotonically across parts so the min/max
statistics actually let the reader skip; the Hive folder partitions by a
low-cardinality ``region`` column.

Indicative (200k rows, 20 parts, local tmpfs, median of 3 ``--repeat``,
dev box)::

    collect_schema (first-file footer)           561 us
    collect_schema (generic .ygg sidecar)        292 us
    read_arrow_table full                        9.62 ms
    read_polars_frame full                       4.59 ms
    read_arrow_table predicate id>=190000        5.37 ms   (skips 19/20 parts)
    read_arrow_table hive full                   6.81 ms
    read_arrow_table hive predicate region==r0   2.84 ms   (prunes 7/8)

Two headlines: pushdown / pruning cut the scan roughly in proportion to the
data skipped (predicate ~0.56x, partition ~0.42x of the full read), and the
footer-based schema detection costs ~2x the sidecar read — that's the
deliberate correctness trade (the footer is authoritative and present even
when no sidecar was written), not a speed win, and the bench keeps it
honest by timing both.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_parquet_folder.py
    PYTHONPATH=src python benchmarks/io/bench_parquet_folder.py --repeat 5
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile

# Sibling-module import that works both when run directly and when spawned
# by ``run_all.py`` — the script's directory must be on ``sys.path``.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pyarrow as pa

from yggdrasil.enums import Mode
from yggdrasil.execution.expr import col
from yggdrasil.parquet import ParquetFolder, ParquetFolderOptions
from yggdrasil.path.folder import Folder

from _common import make_cli, time_one  # type: ignore[import-not-found]


#: Cardinality of the Hive ``region`` partition column.
_N_REGIONS = 8
_REGIONS = [f"r{i}" for i in range(_N_REGIONS)]

_TMP = tempfile.mkdtemp(prefix="ygg_pqfolder_read_bench_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


def _flat_folder(rows: int, parts: int) -> str:
    """Write a flat folder of ``parts`` part files with ``id`` laid out
    monotonically across parts (so footer min/max skip works)."""
    d = os.path.join(_TMP, f"flat_{rows}_{parts}")
    if os.path.isdir(d):
        return d
    ParquetFolder._INSTANCES.clear()
    f = ParquetFolder(path=d)
    chunk = rows // parts
    for k in range(parts):
        lo = k * chunk
        sub = pa.table({
            "id": pa.array(range(lo, lo + chunk), pa.int64()),
            "v": pa.array([1.5 + k] * chunk, pa.float64()),
            "name": pa.array([f"n{i % 100}" for i in range(chunk)], pa.string()),
        })
        f.write_arrow_batches(
            sub.to_batches(), options=ParquetFolderOptions(mode=Mode.APPEND),
        )
    return d


def _hive_folder(rows: int) -> str:
    """Write a Hive-partitioned folder keyed on a low-cardinality
    ``region`` column."""
    d = os.path.join(_TMP, f"hive_{rows}")
    if os.path.isdir(d):
        return d
    ParquetFolder._INSTANCES.clear()
    schema = pa.schema([
        pa.field("region", pa.utf8(), metadata={b"t:partition_by": b"True"}),
        pa.field("id", pa.int64()),
        pa.field("v", pa.float64()),
    ])
    regions = [_REGIONS[i % _N_REGIONS] for i in range(rows)]
    batch = pa.record_batch(
        [
            pa.array(regions, pa.utf8()),
            pa.array(range(rows), pa.int64()),
            pa.array([1.5] * rows, pa.float64()),
        ],
        schema=schema,
    )
    ParquetFolder(path=d).write_arrow_batches(
        [batch], options=ParquetFolderOptions(mode=Mode.OVERWRITE),
    )
    return d


def scenarios(repeat: int) -> list[dict]:
    rows, parts = 200_000, 20
    flat_dir = _flat_folder(rows, parts)
    hive_dir = _hive_folder(rows)
    opts = ParquetFolderOptions()

    out: list[dict] = []

    # --- schema detection: first-file footer vs generic sidecar read ----
    # ``_collect_schema`` bypasses the public cache so the footer read is
    # actually measured every call (the public ``collect_schema`` would
    # short-circuit on the warm ``_schema_cache`` after the first hit).
    out.append(time_one(
        f"pqfolder: collect_schema (first-file footer) parts={parts}",
        lambda: ParquetFolder(path=flat_dir)._collect_schema(opts),
        repeat=repeat, inner=500,
    ))
    out.append(time_one(
        f"pqfolder: collect_schema (generic .ygg sidecar) parts={parts}",
        lambda: Folder(path=flat_dir)._collect_schema(opts),
        repeat=repeat, inner=500,
    ))

    # --- full read: arrow + polars -------------------------------------
    flat = ParquetFolder(path=flat_dir)
    out.append(time_one(
        f"pqfolder: read_arrow_table full rows={rows} parts={parts}",
        lambda: flat.read_arrow_table(),
        repeat=repeat, inner=20,
    ))
    try:
        import polars  # noqa: F401
        out.append(time_one(
            f"pqfolder: read_polars_frame full rows={rows} parts={parts}",
            lambda: flat.read_polars_frame(),
            repeat=repeat, inner=20,
        ))
    except ImportError:
        pass

    # --- predicate pushdown: clustered id selects the last part only ---
    hi = rows - (rows // parts)  # keep ~1 of `parts` parts
    out.append(time_one(
        f"pqfolder: read_arrow_table predicate id>={hi} (skips {parts - 1}/{parts} parts)",
        lambda: flat.read_arrow_table(predicate=col("id") >= hi),
        repeat=repeat, inner=20,
    ))

    # --- partition pruning: region == r0 drops the other subtrees ------
    hive = ParquetFolder(path=hive_dir)
    out.append(time_one(
        f"pqfolder: read_arrow_table hive full rows={rows} parts={_N_REGIONS}",
        lambda: hive.read_arrow_table(),
        repeat=repeat, inner=20,
    ))
    out.append(time_one(
        f"pqfolder: read_arrow_table hive predicate region=='r0' (prunes {_N_REGIONS - 1}/{_N_REGIONS})",
        lambda: hive.read_arrow_table(predicate=col("region") == "r0"),
        repeat=repeat, inner=20,
    ))

    return out


main = make_cli(scenarios)


if __name__ == "__main__":
    main()
