from yggdrasil.databricks.workspaces import DBXWorkspace


def test_fetch():
    result = DBXWorkspace(host="xxx.cloud.databricks.com").sql().execute(statement="SELECT 1").to_pandas()
    print(result)

    assert result.all()