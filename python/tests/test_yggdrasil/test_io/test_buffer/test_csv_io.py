from __future__ import annotations

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.csv_io import CsvOptions
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.enums import GZIP, MediaType


def test_csvio_read_infers_arrow_types_from_csv():
    buf = BytesIO(
        b"id,price,active,name\n"
        b"1,3.14,true,alice\n"
        b"2,,false,bob\n"
    )

    table = MediaIO.make(buf, MimeTypes.CSV).read_arrow_table()

    assert table.to_pylist() == [
        {"id": 1, "price": 3.14, "active": True, "name": "alice"},
        {"id": 2, "price": None, "active": False, "name": "bob"},
    ]
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("price").type == pa.float64()
    assert table.schema.field("active").type == pa.bool_()
    assert table.schema.field("name").type == pa.string()


def test_csvio_read_infers_semicolon_delimiter():
    buf = BytesIO(
        b"name;count;enabled\n"
        b"alpha;1;true\n"
        b"beta;2;false\n"
    )

    table = MediaIO.make(buf, MimeTypes.CSV).read_arrow_table()

    assert table.to_pylist() == [
        {"name": "alpha", "count": 1, "enabled": True},
        {"name": "beta", "count": 2, "enabled": False},
    ]


def test_csvio_tsv_defaults_to_tab_delimiter():
    buf = BytesIO(b"id\tamount\n1\t10\n2\t20\n")

    table = MediaIO.make(buf, MimeTypes.TSV).read_arrow_table()

    assert table.to_pylist() == [
        {"id": 1, "amount": 10},
        {"id": 2, "amount": 20},
    ]


def test_csvio_read_without_header_generates_column_names():
    buf = BytesIO(b"1,2\n3,4\n")

    table = MediaIO.make(buf, MimeTypes.CSV).read_arrow_table()

    assert table.column_names == ["f0", "f1"]
    assert table.to_pylist() == [
        {"f0": 1, "f1": 2},
        {"f0": 3, "f1": 4},
    ]


def test_csvio_write_and_read_roundtrip():
    buf = BytesIO()
    io_ = MediaIO.make(buf, MimeTypes.CSV)
    table = pa.table({"id": [1, 2], "name": ["x", "y"]})

    io_.write_arrow_table(table)
    out = io_.read_arrow_table()

    assert out.to_pylist() == table.to_pylist()


def test_csvio_gzip_roundtrip():
    io_ = MediaIO.make(media=MediaType(MimeTypes.CSV, codec=GZIP))
    table = pa.table({"id": [1, 2], "score": [10.5, 11.5]})

    io_.write_arrow_table(table)

    raw = io_.holder.to_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert io_.read_arrow_table().to_pylist() == table.to_pylist()


def test_csvio_columns_projection():
    buf = BytesIO(b"id,value,flag\n1,10,true\n2,20,false\n")

    table = MediaIO.make(buf, MimeTypes.CSV).read_arrow_table(
        options=CsvOptions(columns=["id", "flag"])
    )

    assert table.column_names == ["id", "flag"]
    assert table.to_pylist() == [
        {"id": 1, "flag": True},
        {"id": 2, "flag": False},
    ]
