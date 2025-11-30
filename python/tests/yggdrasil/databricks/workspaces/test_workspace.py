from yggdrasil.databricks.workspaces.workspace import DBXWorkspace


def test_find_in_env_prefers_azure_prefix_for_msal():
    env = {
        "AZURE_CLIENT_ID": "cid",
        "AZURE_CLIENT_SECRET": "secret",
        "AZURE_TENANT_ID": "tenant",
        "AZURE_SCOPES": "scope1,scope2",
        "DATABRICKS_HOST": "https://example.cloud.databricks.com",
        "DATABRICKS_AUTH_TYPE": "external-browser",
    }

    ws = DBXWorkspace.find_in_env(env=env)

    assert ws.host == env["DATABRICKS_HOST"]
    assert ws.auth_type == env["DATABRICKS_AUTH_TYPE"]
    assert ws.msal_auth is not None
    assert ws.msal_auth.client_id == env["AZURE_CLIENT_ID"]
    assert ws.msal_auth.scopes == ["scope1", "scope2"]
