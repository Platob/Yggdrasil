from yggdrasil.mongoengine import *


class Plants(Document):
    plant_name = StringField(required=True)
    plant_type = StringField(required=True)
    plant_subtype = StringField(required=True)
    capacity = FloatField(required=True)
    lat = FloatField(required=True)
    lon = FloatField(required=True)
    country = StringField(required=True)
    as_of = DateTimeField(required=True)

    def __repr__(self):
        return f'{self.country} {self.plant_name} {self.plant_type} {self.plant_subtype} {self.capacity} MW ({self.lat} {self.lon})'

    def __str__(self):
        return f'{self.country} {self.plant_name} {self.plant_type} {self.plant_subtype} {self.capacity} MW ({self.lat} {self.lon})'

    meta = {
        'db_alias': 'GenCast',
        'indexes': [
            {'fields': ['-as_of', '-country', '-plant_type', '-plant_subtype', 'lat', 'lon', '-capacity', 'plant_name'],
             'unique': True}
        ]
    }


def resolver():
    connect(
        alias="GenCast",
        db="GenCast",
        host="mongodb+srv://xxx:xxx@xxx",
    )

@with_mongo_connection(
    databricks="https://xxx",
    resolver=resolver,
)
def decorated():
    import pandas
    return pandas.DataFrame(Plants.objects())


class TestDecorator:

    def test_decorator(self):
        result = decorated(__install_modules=["yggdrasil"])

        assert result is not None