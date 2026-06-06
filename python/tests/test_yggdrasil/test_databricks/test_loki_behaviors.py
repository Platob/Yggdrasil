"""Unit tests for the specialized Databricks Loki behaviors (mocked client)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import yggdrasil.databricks.loki  # noqa: F401 — registers the fleet
from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend


class _Base(unittest.TestCase):
    def _loki_with_client(self, client):
        loki = Loki()
        loki._backends = [Backend("databricks", available=True)]
        patcher = patch("yggdrasil.databricks.DatabricksClient")
        DC = patcher.start()
        self.addCleanup(patcher.stop)
        DC.current.return_value = client
        return loki


class TestDatabricksBehaviors(_Base):
    def test_fleet_is_registered_and_requires_databricks(self):
        names = {b.name for b in Loki().skills()}
        for n in ("databricks-sql", "databricks-catalogs", "databricks-schemas",
                  "databricks-tables", "databricks-warehouses", "databricks-jobs",
                  "databricks-job-runs", "databricks-clusters", "databricks-volumes",
                  "databricks-secrets", "databricks-iam", "databricks-serving"):
            self.assertIn(n, names)
        beh = next(b for b in Loki().skills() if b.name == "databricks-sql")
        self.assertEqual(beh.requires, "databricks")

    def test_unavailable_when_no_session(self):
        loki = Loki()
        loki._backends = [Backend("databricks", available=False)]
        beh = next(b for b in loki.skills() if b.name == "databricks-sql")
        self.assertFalse(beh.available(loki))
        with self.assertRaises(RuntimeError):
            loki.run("databricks-sql", query="select 1")

    def test_genie_is_a_databricks_skill_and_asks_first_space(self):
        # Genie now lives in the Databricks fleet (not the base Loki catalog).
        names = {b.name for b in Loki().skills()}
        self.assertIn("genie", names)
        beh = next(b for b in Loki().skills() if b.name == "genie")
        self.assertEqual(beh.requires, "databricks")

        client = MagicMock()
        space = MagicMock(); space.space_id = "s1"
        answer = MagicMock()
        answer.conversation_id = "c1"; answer.text = "hi"; answer.query = None
        answer.statement_id = None
        space.ask.return_value = answer
        client.genie.spaces.return_value = [space]
        loki = self._loki_with_client(client)
        out = loki.run("genie", question="how many orders?")
        self.assertEqual((out["space_id"], out["text"]), ("s1", "hi"))
        client.genie.spaces.assert_called_once()

    def test_sql_executes_and_returns_tabular_result(self):
        client = MagicMock()
        result = MagicMock(); result.statement_id = "st1"
        result.__len__ = lambda self: 3      # the statement result is a Tabular
        client.sql.execute.return_value = result
        loki = self._loki_with_client(client)
        out = loki.run("databricks-sql", query="select 1")
        client.sql.execute.assert_called_once_with("select 1")
        self.assertEqual(out["statement_id"], "st1")
        self.assertEqual(out["row_count"], 3)
        # The raw Tabular result is returned (not pre-serialized).
        self.assertIs(out["rows"], result)

    def test_catalogs_list_and_schemas(self):
        client = MagicMock()
        c1 = MagicMock(); c1.name = "main"
        c2 = MagicMock(); c2.name = "samples"
        client.catalogs.list_catalogs.return_value = [c1, c2]
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-catalogs")["catalogs"], ["main", "samples"])
        # With a catalog, list its schemas.
        s = MagicMock(); s.schema_name = "sales"
        client.catalogs.catalog.return_value.schemas.return_value = [s]
        out = loki.run("databricks-catalogs", catalog="main")
        client.catalogs.catalog.assert_called_once_with("main")
        self.assertEqual(out["schemas"], ["sales"])

    def test_catalogs_create_and_drop(self):
        client = MagicMock()
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-catalogs", op="create", catalog="cat")["created"], "cat")
        client.sql.execute.assert_called_with("CREATE CATALOG IF NOT EXISTS cat")
        loki.run("databricks-catalogs", op="drop", catalog="cat")
        client.sql.execute.assert_called_with("DROP CATALOG IF EXISTS cat")

    def test_schemas_create_drop_cascade(self):
        client = MagicMock()
        loki = self._loki_with_client(client)
        loki.run("databricks-schemas", op="create", catalog="c", schema="s")
        client.sql.execute.assert_called_with("CREATE SCHEMA IF NOT EXISTS c.s")
        loki.run("databricks-schemas", op="drop", catalog="c", schema="s", cascade=True)
        client.sql.execute.assert_called_with("DROP SCHEMA IF EXISTS c.s CASCADE")

    def test_tables_preview_create_drop(self):
        client = MagicMock()
        client.sql.execute.return_value = "RESULT"
        loki = self._loki_with_client(client)
        prev = loki.run("databricks-tables", catalog="c", schema="s", table="t",
                        op="preview", limit=5)
        client.sql.execute.assert_called_with("SELECT * FROM c.s.t LIMIT 5")
        self.assertEqual(prev["rows"], "RESULT")
        loki.run("databricks-tables", catalog="c", schema="s", table="t",
                 op="create", as_select="SELECT 1 AS x")
        client.sql.execute.assert_called_with("CREATE TABLE c.s.t AS SELECT 1 AS x")
        loki.run("databricks-tables", catalog="c", schema="s", table="t", op="drop")
        client.sql.execute.assert_called_with("DROP TABLE IF EXISTS c.s.t")

    def test_clusters_start_stop_restart(self):
        client = MagicMock()
        target = MagicMock()
        client.compute.clusters.find_cluster.return_value = target
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-clusters", op="start", cluster="cl")["started"], "cl")
        target.start.assert_called_once()
        loki.run("databricks-clusters", op="stop", cluster="cl")
        target.delete.assert_called_once()
        loki.run("databricks-clusters", op="restart", cluster="cl")
        target.restart.assert_called_once()

    def test_warehouses_start_stop(self):
        client = MagicMock()
        target = MagicMock()
        client.warehouses.find_warehouse.return_value = target
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-warehouses", op="start", warehouse="wh")["started"], "wh")
        target.start.assert_called_once()
        loki.run("databricks-warehouses", op="stop", warehouse="wh")
        target.stop.assert_called_once()

    def test_secrets_put_and_delete(self):
        client = MagicMock()
        loki = self._loki_with_client(client)
        out = loki.run("databricks-secrets", op="put", scope="sc", key="k", value="v")
        client.secrets.create_secret.assert_called_once_with("k", "v", scope="sc")
        self.assertEqual(out, {"scope": "sc", "put": "k"})   # value never echoed
        loki.run("databricks-secrets", op="delete", scope="sc", key="k")
        client.secrets.delete_secret.assert_called_once_with("k", scope="sc")

    def test_tables_list_via_uc_api(self):
        client = MagicMock()
        t1 = MagicMock(); t1.name = "orders"
        t2 = MagicMock(); t2.name = "customers"
        client.tables.list_tables.return_value = [t1, t2]
        loki = self._loki_with_client(client)
        out = loki.run("databricks-tables", catalog="main", schema="sales")
        # Uses the Unity Catalog tables accessor (no SQL warehouse).
        client.tables.list_tables.assert_called_once_with(catalog_name="main", schema_name="sales")
        client.sql.execute.assert_not_called()
        self.assertEqual(out["tables"], ["orders", "customers"])

    def test_tables_describe_returns_typed_columns(self):
        client = MagicMock()
        col = MagicMock(); col.name = "amount"; col.field.dtype = "Float64"
        t = MagicMock(); t.full_name.return_value = "main.sales.orders"
        t.table_type = "MANAGED"; t.columns = [col]
        client.tables.get.return_value = t
        loki = self._loki_with_client(client)
        out = loki.run("databricks-tables", catalog="main", schema="sales", table="orders")
        self.assertEqual(out["table"], "main.sales.orders")
        self.assertEqual(out["columns"], [{"name": "amount", "type": "Float64"}])

    def test_warehouses_lists_names(self):
        client = MagicMock()
        w1 = MagicMock(); w1.name = "wh-a"
        w2 = MagicMock(); w2.name = "wh-b"
        client.warehouses.list_warehouses.return_value = [w1, w2]
        loki = self._loki_with_client(client)
        out = loki.run("databricks-warehouses")
        self.assertEqual(out["warehouses"], ["wh-a", "wh-b"])

    def test_jobs_list_and_clusters_list(self):
        client = MagicMock()
        j = MagicMock(); j.name = "nightly"
        client.jobs.list.return_value = [j]
        c = MagicMock(); c.name = "cl-1"
        client.compute.clusters.list.return_value = [c]
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-jobs")["jobs"], ["nightly"])
        self.assertEqual(loki.run("databricks-clusters")["clusters"], ["cl-1"])

    def test_jobs_run_triggers_and_returns_run(self):
        client = MagicMock()
        job = MagicMock(); job.name = "nightly"; job.job_id = 7
        job_run = MagicMock(); job_run.run_id = 99; job_run.url = "https://w/jobs/7/runs/99"
        job.run.return_value = job_run
        client.jobs.get.return_value = job
        loki = self._loki_with_client(client)
        out = loki.run("databricks-jobs", run="nightly", parameters={"date": "2026-01-01"})
        job.run.assert_called_once_with(parameters={"date": "2026-01-01"})
        self.assertEqual((out["job_id"], out["run_id"]), (7, 99))
        self.assertEqual(out["url"], "https://w/jobs/7/runs/99")

    def test_volumes_list_uses_keyword_scope(self):
        client = MagicMock()
        v = MagicMock(); v.name = "contracts"
        client.volumes.list.return_value = [v]
        loki = self._loki_with_client(client)
        out = loki.run("databricks-volumes", catalog="samples")
        # catalog_name / schema_name are keyword-only on Volumes.list.
        client.volumes.list.assert_called_once_with(catalog_name="samples", schema_name=None)
        self.assertEqual(out["volumes"], ["contracts"])

    def test_secrets_lists_scopes(self):
        client = MagicMock()
        # A UC secret Scope identifies by ``.key`` (no ``.name``).
        s = MagicMock(spec=["key"]); s.key = "prod"
        client.secrets.list_scopes.return_value = [s]
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-secrets")["scopes"], ["prod"])

    def test_iam_me(self):
        client = MagicMock()
        # "who am I" resolves through the workspace client's /Me call.
        client.workspace_client.return_value.current_user.me.return_value = MagicMock(
            user_name="me@x.io", id="42")
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-iam")["me"], "me@x.io")

    def test_serving_lists_endpoints(self):
        client = MagicMock()
        ep = MagicMock(); ep.name = "llama"
        client.workspace_client.return_value.serving_endpoints.list.return_value = [ep]
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-serving")["endpoints"], ["llama"])

    def test_serving_queries_endpoint_with_prompt(self):
        client = MagicMock()
        oai = MagicMock()
        msg = MagicMock(); msg.content = "served reply"
        oai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=msg)], model="ep")
        client.base_url = "https://w"
        client.workspace_client.return_value.serving_endpoints.get_open_ai_client.return_value = oai
        loki = self._loki_with_client(client)
        out = loki.run("databricks-serving", endpoint="ep", prompt="hi")
        self.assertEqual(out["reply"], "served reply")
        self.assertEqual(out["endpoint"], "ep")
        # The domain preprompt steers the served model (sent as the system msg).
        sent = oai.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertIn("model serving", sent[0]["content"])


class TestDatabricksMCP(_Base):
    def test_mcp_url_templates(self):
        from yggdrasil.databricks.loki.skills import _mcp_url

        self.assertEqual(
            _mcp_url("https://w/", "functions", catalog="main", schema="sales"),
            "https://w/api/2.0/mcp/functions/main/sales")
        self.assertEqual(
            _mcp_url("https://w", "genie", space="01ef"),
            "https://w/api/2.0/mcp/genie/01ef")
        self.assertEqual(
            _mcp_url("https://w", "vector_search", catalog="c", schema="s"),
            "https://w/api/2.0/mcp/vector-search/c/s")

    def test_mcp_url_validates(self):
        from yggdrasil.databricks.loki.skills import _mcp_url

        with self.assertRaises(ValueError):
            _mcp_url("https://w", "nope")
        with self.assertRaises(ValueError):
            _mcp_url("https://w", "functions", catalog="main")  # missing schema

    def test_mcp_behavior_lists_tools(self):
        client = MagicMock()
        w = client.workspace_client.return_value
        w.config.authenticate.return_value = {"Authorization": "Bearer tok"}
        w.config.host = "https://w"
        loki = self._loki_with_client(client)

        async def fake_tools(url, headers):
            self.assertEqual(headers, {"Authorization": "Bearer tok"})
            return ["uc.fn_a", "uc.fn_b"]

        with patch("yggdrasil.databricks.loki.skills._mcp_tools", fake_tools):
            out = loki.run("databricks-mcp", kind="functions", catalog="main", schema="sales")
        self.assertEqual(out["server"], "https://w/api/2.0/mcp/functions/main/sales")
        self.assertEqual(out["tools"], ["uc.fn_a", "uc.fn_b"])
        w.config.authenticate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
