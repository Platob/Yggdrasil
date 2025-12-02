import datetime

from yggdrasil.databricks.sql import DBXSQL
from yggdrasil.databricks.workspaces import DBXWorkspace


def test_insert():
    sql = DBXSQL(workspace=DBXWorkspace(host="dbc-e646c5f9-8a44.cloud.databricks.com"))

    data = [
        {"test": 1, "b": datetime.datetime.now()}
    ]
    sql.insert_into(data, location="trading.ba_3mv_polaris__p__volcano_ref_input.test", mode="auto")

def test_read():
    sql = DBXSQL(workspace=DBXWorkspace(host="dbc-e646c5f9-8a44.cloud.databricks.com"))

    data = sql.read_arrow_batches(catalog_name="trading", schema_name="ba_3mv_polaris__p__volcano_ref_input", table_name="test").read_all()

    assert data