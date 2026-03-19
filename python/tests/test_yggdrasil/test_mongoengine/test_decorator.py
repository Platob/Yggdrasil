from yggdrasil.mongoengine import *
import pandas
from datamanagement.mongo_gencast.mongo_utils import Plants


def resolver():
    connect(
        alias="GenCast",
        db="GenCast",
        host="mongodb+srv://mats-weather-prod-rw:q53qqmAjCgcz7NHX@mats-weather-prod-rs0-pl-3.3vnse1.mongodb.net",
    )

@with_mongo_connection(
    databricks="https://dbc-82edd6f4-1e97.cloud.databricks.com/",
    resolver=resolver,
)
def decorated(a: int, b: int = None):
    pipeline = [
        {'$sort': {'as_of': -1}},
        {'$group': {'_id': {'plant_name': '$plant_name', 'plant_subtype': '$plant_subtype',
                            'capacity': '$capacity', 'lat': '$lat', 'lon': '$lon', 'country': '$country'},
                    'latest_update': {'$first': '$$ROOT'}}},
        {'$replaceRoot': {'newRoot': '$latest_update'}}
    ]

    result = Plants.objects().aggregate(*pipeline)
    data = pandas.DataFrame(result)
    return data.drop(columns=['as_of']), a, b


class TestDecorator:

    def test_decorator(self):
        result = decorated(1)

        assert result is not None