"""Benchmark the ExcelFile fast paths: windowed reads and batched cell edits.

- read_range(n_rows=...) vs reading the whole sheet — the viewport fetch
  shouldn't pay for the full sheet.
- apply_edits(batch) (one load + one save) vs one save per cell — batching a
  range of edits into a single workbook open is the whole point.

Usage::

    PYTHONPATH=src python benchmarks/io/bench_xlsx_edit.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import openpyxl

from yggdrasil.path.local_path import LocalPath


def _seed(path: Path, rows: int, cols: int) -> None:
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Data")
    ws.append([f"c{j}" for j in range(cols)])
    for i in range(rows):
        ws.append([i * cols + j for j in range(cols)])
    wb.save(str(path))
    wb.close()


def main() -> None:
    rows, cols = 5000, 12
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "big.xlsx"
        _seed(path, rows, cols)
        print(f"\n  workbook: {rows} rows x {cols} cols\n")

        with LocalPath(str(path)).open("rb") as wb:
            t0 = time.perf_counter()
            for _ in range(5):
                full = wb.read_range("Data")
            full_ms = (time.perf_counter() - t0) / 5 * 1000
            t0 = time.perf_counter()
            for _ in range(5):
                wb.read_range("Data", n_rows=100)
            win_ms = (time.perf_counter() - t0) / 5 * 1000
        print(f"  read full ({full.num_rows} rows):   {full_ms:8.2f} ms")
        print(f"  read window (100 rows):    {win_ms:8.2f} ms   {full_ms / win_ms:5.1f}x faster\n")

        # 50 scattered cell edits: one batched save vs one save per edit.
        edits = [(2 + i, 1 + (i % cols), i * 7) for i in range(50)]

        with LocalPath(str(path)).open("rb") as wb:
            t0 = time.perf_counter()
            wb.apply_edits("Data", edits)
            batched_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for e in edits:
            with LocalPath(str(path)).open("rb") as wb:
                wb.apply_edits("Data", [e])
        per_edit_ms = (time.perf_counter() - t0) * 1000
        print(f"  {len(edits)} edits batched (1 save):    {batched_ms:8.2f} ms")
        print(f"  {len(edits)} edits one-save-each:      {per_edit_ms:8.2f} ms   {per_edit_ms / batched_ms:5.1f}x slower\n")


if __name__ == "__main__":
    main()
