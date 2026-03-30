import os
from unittest import TestCase

import pandas as pd
from mongoengine import connect, Document, StringField, FloatField, DateTimeField


class Plants(Document):
    plant_name = StringField(required=True)
    plant_type = StringField(required=True)
    plant_subtype = StringField(required=True)
    capacity = FloatField(required=True)
    lat = FloatField(required=True)
    lon = FloatField(required=True)
    country = StringField(required=True)
    as_of = DateTimeField(required=True)

    meta = {
        "db_alias": "GenCast",
        "indexes": [
            {
                "fields": [
                    "-as_of",
                    "-country",
                    "-plant_type",
                    "-plant_subtype",
                    "lat",
                    "lon",
                    "-capacity",
                    "plant_name",
                ],
                "unique": True,
            }
        ],
    }


class MongoCase(TestCase):
    @classmethod
    def setUpClass(cls):
        mongo_uri = os.environ["MONGODB_URI"]
        cls.client = connect(
            db="GenCast",
            alias="GenCast",
            host=mongo_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
            retryWrites=True,
            retryReads=True,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.client is not None:
            cls.client.close()

    def test_plants(self):
        pipeline = [
            {"$sort": {"as_of": -1}},
            {"$group": {
                "_id": {
                    "plant_name": "$plant_name",
                    "plant_type": "$plant_type",
                    "plant_subtype": "$plant_subtype",
                    "country": "$country",
                },
                "latest_update": {"$first": "$$ROOT"},
            }},
            {"$replaceRoot": {"newRoot": "$latest_update"}},
        ]

        result = Plants.objects().aggregate(*pipeline)
        data = pd.DataFrame(result)
        data["_id"] = data.index
        assert not data.empty