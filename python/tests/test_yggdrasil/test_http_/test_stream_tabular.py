"""Reading tabular payloads (parquet / zip / ndjson) from a streamed
body that is larger than the in-memory window.

A ``stream=True`` response keeps the live socket as its buffer source and
the buffer is a :class:`MemoryStream`: bytes above ``spill_threshold``
(the in-memory window) spill to a tempfile and stay readable until the
total ``byte_size`` retention cap is hit, after which the oldest bytes
are evicted (dropped). These tests pin what that means for real
file formats, several of which need random access (parquet reads its
footer at EOF then seeks back to row groups; zip reads the central
directory at the end):

* raw byte reads (``.content`` / ``.stream`` / ``read_mv``) reconstruct
  the exact payload across the spill boundary, so the bytes can be
  re-parsed with pyarrow / zipfile;
* a backward seek into the spilled (but not evicted) region works;
* once the body exceeds the ``byte_size`` cap the head is evicted and a
  seek behind it raises — a footer-seek format can't be parsed in that
  regime, which is the documented limit, not a silent corruption.
"""
from __future__ import annotations

import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yggdrasil.path.memory_stream import MemoryStream


# ---------------------------------------------------------------------------
# Payload builders — each comfortably larger than the tiny window used below
# ---------------------------------------------------------------------------


def _parquet_bytes(rows: int = 200_000) -> bytes:
    tbl = pa.table({"a": pa.array(range(rows)), "b": ["x" * 16] * rows})
    buf = io.BytesIO()
    pq.write_table(tbl, buf)
    return buf.getvalue()


def _zip_bytes(members: int = 50) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(members):
            zf.writestr(f"f{i}.txt", (f"row {i} " * 5000))
    return buf.getvalue()


def _ndjson_bytes(rows: int = 100_000) -> bytes:
    return ("\n".join(json.dumps({"i": i, "v": "y" * 30}) for i in range(rows))).encode()


def _spilling_stream(data: bytes, *, window_kib: int = 64) -> MemoryStream:
    """MemoryStream whose in-memory window is *window_kib* but whose total
    retention budget covers the whole payload — so the body spills to disk
    but is never evicted.
    """
    return MemoryStream(
        io.BytesIO(data),
        spill_threshold=window_kib * 1024,
        byte_size=len(data) + 4096,
    )


# ---------------------------------------------------------------------------
# Raw byte fidelity across the spill boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [_parquet_bytes, _zip_bytes, _ndjson_bytes])
def test_full_read_matches_across_spill(builder):
    data = builder()
    ms = _spilling_stream(data)
    assert len(data) > 64 * 1024, "payload must exceed the window to exercise spill"
    got = bytes(ms.read_mv(-1, 0))
    assert got == data
    assert ms.has_spill, "body larger than the window should have spilled to disk"


def test_parquet_reparses_from_spilled_stream():
    data = _parquet_bytes()
    ms = _spilling_stream(data)
    got = bytes(ms.read_mv(-1, 0))
    table = pq.read_table(pa.BufferReader(got))
    assert table.num_rows == 200_000


def test_zip_reparses_from_spilled_stream():
    data = _zip_bytes()
    ms = _spilling_stream(data)
    got = bytes(ms.read_mv(-1, 0))
    with zipfile.ZipFile(io.BytesIO(got)) as zf:
        assert len(zf.namelist()) == 50
        assert zf.read("f0.txt").startswith(b"row 0 ")


def test_ndjson_reparses_from_spilled_stream():
    data = _ndjson_bytes()
    ms = _spilling_stream(data)
    got = bytes(ms.read_mv(-1, 0))
    assert got.count(b"\n") == 99_999
    last = json.loads(got.splitlines()[-1])
    assert last["i"] == 99_999


# ---------------------------------------------------------------------------
# Random access: backward seek into the spilled region
# ---------------------------------------------------------------------------


def test_backward_seek_into_spill_region_works():
    data = _parquet_bytes()
    ms = _spilling_stream(data)
    ms.read_mv(-1, 0)  # pull to EOF — most of the body is now on disk
    # Footer-style read at EOF, then a seek all the way back to the magic
    # header that lives behind the in-memory window (in the spill file).
    tail = bytes(ms.read_mv(4, len(data) - 4))
    head = bytes(ms.read_mv(4, 0))
    assert head == b"PAR1"
    assert tail == b"PAR1"


# ---------------------------------------------------------------------------
# The hard limit: beyond byte_size the head is evicted and seeks raise
# ---------------------------------------------------------------------------


def test_seek_behind_evicted_region_raises():
    data = _parquet_bytes()
    # Window 32 KiB, total budget 64 KiB — far below the body, so the head
    # is evicted (dropped) once the tail is pulled in.
    ms = MemoryStream(io.BytesIO(data), spill_threshold=32 * 1024, byte_size=64 * 1024)
    ms.read_mv(-1, 0)
    with pytest.raises(ValueError):
        ms.read_mv(4, 0)  # head is gone — a footer-seek format can't parse here
