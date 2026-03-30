from __future__ import annotations

import uuid

import pytest

from yggdrasil.mongoengine.lib import connect, get_connection, mongoengine


def test_mongoengine_insert_and_query_roundtrip_with_document() -> None:
    mongomock = pytest.importorskip("mongomock")

    alias = f"http_proxy_test_{uuid.uuid4().hex}"
    db_name = f"db_{uuid.uuid4().hex}"

    class ProxyDoc(mongoengine.Document):
        key = mongoengine.StringField(required=True)
        value = mongoengine.IntField(required=True)

        meta = {
            "db_alias": alias,
            "collection": f"proxy_docs_{uuid.uuid4().hex}",
        }

    connect(
        alias=alias,
        db=db_name,
        host="mongodb://localhost",
        mongo_client_class=mongomock.MongoClient,
    )

    try:
        ProxyDoc.drop_collection()

        ProxyDoc(key="alpha", value=1).save()
        ProxyDoc(key="beta", value=2).save()

        found = ProxyDoc.objects(key="alpha").first()
        assert found is not None
        assert found.key == "alpha"
        assert found.value == 1

        values = [doc.value for doc in ProxyDoc.objects.order_by("value")]
        assert values == [1, 2]

        client = get_connection(alias=alias)
        assert client is not None
    finally:
        mongoengine.disconnect(alias=alias)
