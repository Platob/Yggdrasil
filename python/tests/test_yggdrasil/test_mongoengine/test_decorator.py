from yggdrasil.mongoengine import *


class Cities(Document):
    city_id = IntField(required=True)
    city_name = StringField(required=True)
    location = PointField(required=True)
    country_name = StringField(required=True)
    country_iso = StringField(required=True)
    population_tot = FloatField(required=True)
    population_perc = FloatField(required=True)

    def __repr__(self):
        return f'{self.city_name} {self.country_name} {self.population_tot} {self.population_perc} ({self.location["coordinates"][1]}, {self.location["coordinates"][0]})'

    def __str__(self):
        return f'{self.city_name} {self.country_name} {self.population_tot} {self.population_perc} ({self.location["coordinates"][1]}, {self.location["coordinates"][0]})'

    meta = {
        'db_alias': 'test_connection',
        'indexes': [
            {'fields': ['-city_name', '-country_iso', 'population_perc'],
             'unique': True}
        ]
    }


connect(
    alias="GenCast",
    host="mongodb+srv://xxx:xxx@xxx/?appName=xxx",
)

@with_mongo_connection(
    aliases="GenCast",
    databricks="xxx"
)
def decorated():
    return Cities.objects().first().to_pandas()

class TestDecorator:

    def test_decorator(self):
        result = decorated(1, __install_modules=["yggdrasil"])

        assert result is not None