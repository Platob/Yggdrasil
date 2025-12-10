import pandas
from mongoengine import DynamicDocument, connect

from yggdrasil.databricks.compute.remote import (
    databricks_remote_compute,
)


class Cities(DynamicDocument):
    meta = {'collection': 'cities'}


@databricks_remote_compute(
    cluster_id="xxx",
    workspace="xxx.cloud.databricks.com",
    force_local=True
)
def dump_os(*args, **kwargs):
    connect(db="dft", alias="default",
            host="mongodb+srv://xxx")

    return Cities.objects.limit(10)


def ignore_test_remote_pyeval_executes_function_and_returns_value():
    r = dump_os()

    assert isinstance(r, pandas.DataFrame)
