import os
from dataclasses import  dataclass

import polars

from yggdrasil.databricks.compute import databricks_remote_compute
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.io.path import LocalDataPath

workspace = Workspace(
    host="dbc-e646c5f9-8a44.cloud.databricks.com", # by default if not given will get os.getenv("DATABRICKS_HOST")
    token=None, # by default if not given will get os.getenv("DATABRICKS_TOKEN")
    client_id=None, # by default if not given will get os.getenv("DATABRICKS_CLIENT_ID")
    client_secret=None, # by default if not given will get os.getenv("DATABRICKS_CLIENT_SECRET")
    product=None, # by default if not given will get os.getenv("DATABRICKS_PRODUCT")
)

current_user = workspace.current_user

data = {
    "id": [1, 2, 3],
    "name": ["test", None, None]
}

## Path

path = LocalDataPath(r"C:\Users\NFILLO\dbx_example")
databricks_path = workspace.dbfs_path("/Volumes/trading/unittest/tmp/dbx_example")
databricks_path = LocalDataPath("dbfs://dbc-e646c5f9-8a44.cloud.databricks.com/Volumes/trading/unittest/tmp/dbx_example")

path.mkdir(exist_ok=True)
databricks_path.mkdir(exist_ok=True)

### Folder
folder = (path / "test_folder")
folder.mkdir(exist_ok=True)

folder.write_table(data)

pandas_df = folder.read_pandas()
polars_df = folder.read_polars()


### File
file = (folder / "file.parquet")
file.write_table(data)

pandas_df = file.read_pandas()
polars_df = file.read_polars()


### FileSystem SQL
df = folder.sql(f"select * from `{folder}`", engine="polars")

assert isinstance(df, polars.DataFrame)

## Clean
folder.rmdir()
file.remove()
folder.remove()

## SQL
print("SQL")
engine = workspace.sql()

## Write
print("SQL Write")

engine.insert_into(
    data, # can be pandas.DataFrame, polars.DataFrame, pyspark.sql.DataFrame, list, dict, or path
    catalog_name="trading",
    schema_name="dbx_example",
    table_name="write_example",
    match_by=None # or match by column names like ["id"]
)

## Read
print("SQL Read")

result = engine.execute(
    "select * from trading.dbx_example.write_example",
    catalog_name=None, schema_name=None, wait=True
)

pandas_df = result.to_pandas()
polars_df = result.to_polars()
# spark_df = result.to_spark()


## Secrets
print("Secrets")

@dataclass
class Config:
    id: int = 0
    client_id: str = "xxx"
    client_secret: str = "xxx"

secrets = workspace.secrets()

secrets["dbx_example/config"] = Config()

config = Config(**secrets["dbx_example/config"].value)

secrets["dbx_example/config"].delete_secret()


## Compute
print("Compute")

os.environ["TEST_ENV"] = "testenv"

@databricks_remote_compute(
    cluster_id=None,
    cluster_name=None,
    env_keys=["TEST_ENV"]
)
def decorated(a: int):
    env = os.environ["TEST_ENV"]

    return {
        "os": os.environ,
        "value": a,
        "env": env
    }

result = decorated(1)

print("Done")