from __future__ import annotations

import uuid

import pyarrow as pa

from yggdrasil.mongoengine import (
    Document,
    IntField,
    StringField,
    UUIDField,
)


class MyDoc(Document):
    name = StringField(required=True)
    qty = IntField(required=True)
    uid = UUIDField(binary=False, required=False)


def test_arrow_schema_uses_field_arrow_type():
    schema = MyDoc.arrow_schema()

    assert schema == pa.schema(
        [
            pa.field("_id", pa.string(), nullable=True),
            pa.field("name", pa.string(), nullable=False),
            pa.field("qty", pa.int64(), nullable=False),
            pa.field("uid", pa.string(), nullable=True),
        ]
    )


def test_to_dict_normalizes_object_id_and_uuid():
    doc = MyDoc(name="nika", qty=7, uid=uuid.uuid4())

    data = doc.to_dict()

    assert isinstance(data, dict)
    assert data["name"] == "nika"
    assert data["qty"] == 7
    assert isinstance(data["_id"], str)
    assert isinstance(data["uid"], str)


def test_to_dict_without_normalize_keeps_raw_types():
    doc = MyDoc(name="nika", qty=7, uid=uuid.uuid4())

    data = doc.to_dict(normalize=False)

    assert isinstance(data, dict)
    assert data["name"] == "nika"
    assert data["qty"] == 7
    assert "_id" in data


def test_to_arrow_table():
    doc = MyDoc(name="nika", qty=7)

    table = doc.to_arrow_table()

    assert table.num_rows == 1
    assert table.schema == MyDoc.arrow_schema()

    row = table.to_pylist()[0]
    assert row["name"] == "nika"
    assert row["qty"] == 7
    assert isinstance(row["_id"], str)


def test_to_arrow_batch():
    doc = MyDoc(name="nika", qty=7)

    batch = doc.to_arrow_batch()

    assert batch.num_rows == 1
    assert batch.schema == MyDoc.arrow_schema()

    row = batch.to_pylist()[0]
    assert row["name"] == "nika"
    assert row["qty"] == 7


def test_to_arrow_batches():
    doc = MyDoc(name="nika", qty=7)

    batches = list(doc.to_arrow_batches())

    assert len(batches) == 1
    assert batches[0].num_rows == 1
    assert batches[0].schema == MyDoc.arrow_schema()


def test_to_pandas():
    doc = MyDoc(name="nika", qty=7)

    df = doc.to_pandas()

    assert list(df.columns) == ["_id", "name", "qty", "uid"]
    assert len(df) == 1
    assert df.loc[0, "name"] == "nika"
    assert df.loc[0, "qty"] == 7
    assert isinstance(df.loc[0, "_id"], str)