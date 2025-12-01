from mongoengine import DynamicDocument, connect

from yggdrasil.databricks.compute.remote import databricks_remote_compute


def dump_os(*args, **kwargs):
    class Cities(DynamicDocument):
        meta = {'collection': 'cities'}

    # Connect to local MongoDB (standalone mode)
    connect(db="xxx",alias="default",host=prod)

    # Test the function
    docs = list(Cities.objects.limit(10))
    return [doc.to_mongo().to_dict() for doc in docs]

def test_remote_pyeval_executes_function_and_returns_value():
    result = databricks_remote_compute(
        "xxx",
        workspace="xxx.cloud.databricks.com",
    )(dump_os)

    assert result() == 5
