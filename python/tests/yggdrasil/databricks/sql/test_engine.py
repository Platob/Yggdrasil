import os
from typing import Any

import pandas

from yggdrasil.databricks.workspaces import Workspace


def test_fetch():
    from yggdrasil.databricks import Workspace, Cluster

    # infer from local environment
    # running locally will use SSO
    # in databricks will use current identity
    workspace = Workspace()
    workspace = Workspace(host="xxx.cloud.databricks.com")

    read = (
        workspace.sql()
        .execute(statement="SELECT 1")
        .to_pandas() # or .to_polars(), .arrow_batches()
    )

    write = (
        workspace.sql()
        .insert_into(
            data=pandas.DataFrame, # or other polars.DataFrame, pyspark.sql.DataFrame, list
            catalog_name="catalog",
            schema_name="schema",
            table_name="name",
            mode="auto", # set overwrite to overwrite table, or clean update kys before merge
            match_by=[] # Add matching keys to update
        )
    )

    cluster = workspace.clusters().find_cluster(cluster_id="123", cluster_name="abc")
    cluster = Cluster.replicated_current_environment(workspace=workspace)

    @cluster.execution_decorator
    def remote_executed(value: Any):
        return os.environ, value, workspace.current_user

    remote_result = remote_executed(1)
