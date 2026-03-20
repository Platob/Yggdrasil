# tests/test_parquet_io.py
from __future__ import annotations

from pathlib import Path

import pytest

from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.parquet_io import ParquetIO, ParquetOptions
from yggdrasil.io.config import BufferConfig
from yggdrasil.io.enums import SaveMode, MimeType


def _make_cfg(tmp_path: Path) -> BufferConfig:
    # small spill threshold to also exercise spilled -> pyarrow OSFile / memory_map path
    return BufferConfig(
        spill_bytes=128,
        tmp_dir=tmp_path,
        prefix="test_parquetio_",
        suffix=".parquet",
        keep_spilled_file=False,
    )


def _pa():
    # single import point (your project uses yggdrasil.arrow.lib.pyarrow)
    from yggdrasil.arrow.lib import pyarrow as pa

    return pa


@pytest.fixture()
def cfg(tmp_path: Path) -> BufferConfig:
    return _make_cfg(tmp_path)


@pytest.fixture()
def sample_table():
    pa = _pa()
    return pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "s": pa.array(["a", "b", "c"], type=pa.string()),
            "x": pa.array([1.25, 2.5, None], type=pa.float64()),
        }
    )


def test_write_then_read_roundtrip_memory(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = MediaIO.make(buf, MimeType.PARQUET)

    io_.write_arrow_table(sample_table, options=ParquetOptions(mode=SaveMode.OVERWRITE))

    assert buf.size > 0
    out = io_.read_arrow_table()

    assert out.schema == sample_table.schema
    assert out.to_pylist() == sample_table.to_pylist()


def test_read_with_columns_projection(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = MediaIO.make(buf, MimeType.PARQUET)
    io_.write_arrow_table(sample_table, options=ParquetOptions(mode=SaveMode.OVERWRITE))
    out = io_.read_arrow_table(options=ParquetOptions(columns=["id", "s"]))

    assert out.column_names == ["id", "s"]
    assert out.num_rows == 3


def test_ignore_mode_does_not_overwrite(cfg: BufferConfig, sample_table):
    pa = _pa()
    buf = BytesIO(config=cfg)
    io_ = MediaIO.make(buf, MimeType.PARQUET)

    t1 = sample_table
    t2 = pa.table({"id": pa.array([999], type=pa.int64()), "s": pa.array(["z"]), "x": pa.array([0.0])})

    io_.write_arrow_table(t1, options=ParquetOptions(mode=SaveMode.OVERWRITE))
    size1 = buf.size
    bytes1 = buf.to_bytes()

    # IGNORE: should leave buffer unchanged if already exists
    io_.write_arrow_table(t2, options=ParquetOptions(mode=SaveMode.IGNORE))
    assert buf.size == size1
    assert buf.to_bytes() == bytes1

    out = io_.read_arrow_table()
    assert out.to_pylist() == t1.to_pylist()


def test_error_if_exists_raises(cfg: BufferConfig, sample_table):
    buf = BytesIO(config=cfg)
    io_ = MediaIO.make(buf, MimeType.PARQUET)

    io_.write_arrow_table(sample_table, options=ParquetOptions(mode=SaveMode.OVERWRITE))

    with pytest.raises(IOError):
        io_.write_arrow_table(sample_table, options=ParquetOptions(mode=SaveMode.ERROR_IF_EXISTS))


def test_overwrite_replaces_content(cfg: BufferConfig, sample_table):
    pa = _pa()
    buf = BytesIO(config=cfg)
    io_ = MediaIO.make(buf, MimeType.PARQUET)

    t1 = sample_table
    t2 = pa.table(
        {
            "id": pa.array([10, 20], type=pa.int64()),
            "s": pa.array(["x", "y"], type=pa.string()),
            "x": pa.array([9.0, 8.0], type=pa.float64()),
        }
    )

    io_.write_arrow_table(t1, options=ParquetOptions(mode=SaveMode.OVERWRITE))
    bytes1 = buf.to_bytes()

    io_.write_arrow_table(t2, options=ParquetOptions(mode=SaveMode.OVERWRITE))
    bytes2 = buf.to_bytes()

    assert bytes1 != bytes2
    out = io_.read_arrow_table()
    assert out.to_pylist() == t2.to_pylist()


def test_spilled_buffer_path_read_write(cfg: BufferConfig, sample_table):
    """
    Force spill and ensure parquet read/write still works when BytesIO uses a path-backed file.
    """
    # Force spill aggressively
    cfg2 = BufferConfig(
        spill_bytes=1,
        tmp_dir=cfg.tmp_dir,
        prefix=cfg.prefix,
        suffix=cfg.suffix,
        keep_spilled_file=False,
    )
    buf = BytesIO(config=cfg2)
    io_ = MediaIO.make(buf, MimeType.PARQUET)

    io_.write_arrow_table(sample_table, options=ParquetOptions(mode=SaveMode.OVERWRITE))
    assert buf.spilled is True
    assert buf.path is not None
    assert buf.size > 0

    out = io_.read_arrow_table()
    assert out.to_pylist() == sample_table.to_pylist()
