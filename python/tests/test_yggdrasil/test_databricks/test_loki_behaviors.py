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
        names = {b.name for b in Loki().behaviors()}
        for n in ("databricks-sql", "databricks-tables", "databricks-warehouses",
                  "databricks-jobs", "databricks-clusters", "databricks-volumes",
                  "databricks-secrets", "databricks-iam", "databricks-serving"):
            self.assertIn(n, names)
        beh = next(b for b in Loki().behaviors() if b.name == "databricks-sql")
        self.assertEqual(beh.requires, "databricks")

    def test_unavailable_when_no_session(self):
        loki = Loki()
        loki._backends = [Backend("databricks", available=False)]
        beh = next(b for b in loki.behaviors() if b.name == "databricks-sql")
        self.assertFalse(beh.available(loki))
        with self.assertRaises(RuntimeError):
            loki.run("databricks-sql", query="select 1")

    def test_sql_executes_and_returns_rows(self):
        client = MagicMock()
        result = MagicMock(); result.statement_id = "st1"
        result.to_polars.return_value = MagicMock(height=3)
        client.sql.execute.return_value = result
        loki = self._loki_with_client(client)
        out = loki.run("databricks-sql", query="select 1")
        client.sql.execute.assert_called_once_with("select 1")
        self.assertEqual(out["statement_id"], "st1")
        self.assertEqual(out["row_count"], 3)

    def test_tables_uses_show_tables(self):
        client = MagicMock()
        client.sql.execute.return_value = MagicMock(to_polars=lambda: [])
        loki = self._loki_with_client(client)
        loki.run("databricks-tables", catalog="main", schema="sales")
        client.sql.execute.assert_called_once_with("SHOW TABLES IN main.sales")

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

    def test_secrets_lists_scopes(self):
        client = MagicMock()
        s = MagicMock(); s.name = "prod"
        client.secrets.list_scopes.return_value = [s]
        loki = self._loki_with_client(client)
        self.assertEqual(loki.run("databricks-secrets")["scopes"], ["prod"])

    def test_iam_me(self):
        client = MagicMock()
        client.iam.current_user = MagicMock(user_name="me@x.io")
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


if __name__ == "__main__":
    unittest.main()
