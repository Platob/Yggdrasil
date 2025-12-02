import datetime

import pandas
from databricks.sdk.service.sql import Format, Disposition

from yggdrasil.databricks.sql import DBXSQL
from yggdrasil.databricks.workspaces import DBXWorkspace


def test_insert():
    sql = DBXSQL(workspace=DBXWorkspace(host="xxx.cloud.databricks.com"))

    data = pandas.DataFrame([
        {"test": 1, "b": datetime.datetime.now()}
    ])

    sql.insert_into(data, location="xxx.xxx.test", mode="overwrite")

    written = sql.execute(
        catalog_name="xxx", schema_name="xxx", table_name="xxx",
        format=Format.ARROW_STREAM
    ).to_pandas()

    assert written.equals(data)

def test_read():
    sql = DBXSQL(workspace=DBXWorkspace(host="xxx.cloud.databricks.com"))

    data = sql.execute(
        catalog_name="xxx", schema_name="xxx", table_name="xxx",
        format=Format.ARROW_STREAM
    ).arrow_table()

    assert data