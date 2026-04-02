import os
from unittest import TestCase

import pandas as pd
from mongoengine import Document, StringField, FloatField, DateTimeField

from yggdrasil.mongoengine import with_mongo_connection, connect


def plantss_class():
    class Plantss(Document):
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

    return Plantss

def resolver():
    return connect(
        alias="GenCast",
        db="GenCast",
        host=os.environ["MONGODB_URI"]
    )

@with_mongo_connection(
    aliases=["GenCast"],
    databricks="https://dbc-xxx.cloud.databricks.com/",
    resolver=resolver
)
def get_plant_data():
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

    result = plantss_class().objects().aggregate(*pipeline)
    data = pd.DataFrame(result)
    data["_id"] = data.index
    return data


class MongoCase(TestCase):

    def test_plants(self):
        data = get_plant_data()
        assert not data.empty