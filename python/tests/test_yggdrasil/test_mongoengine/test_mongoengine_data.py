import os
from unittest import TestCase

import pandas as pd
from mongoengine import Document, StringField, FloatField, DateTimeField
from wma import connect_to_db_gencast

from yggdrasil.mongoengine import connect


def plants_class():
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

    return Plants

os.environ["GENCAST_MONGO_HOST"] = "mongodb+srv://mats-weather-prod-rw:q53qqmAjCgcz7NHX@mats-weather-prod-rs0-pl-2.3vnse1.mongodb.net"
def resolver():
    return connect(
        alias="GenCast",
        db="GenCast",
        host=os.environ["MONGODB_URI"]
    )

@connect_to_db_gencast
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

    result = plants_class().objects().aggregate(*pipeline)
    data = pd.DataFrame(result)
    data["_id"] = data.index
    return data


class MongoCase(TestCase):

    def test_plants(self):
        for i in range(3):
            data = get_plant_data()
        assert not data.empty