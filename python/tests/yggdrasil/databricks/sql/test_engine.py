from yggdrasil.databricks.workspaces import Workspace


def test_fetch():
    result = Workspace(host="xxx.cloud.databricks.com").sql().execute(statement="SELECT 1").to_pandas()
    print(result)

    assert result.all()