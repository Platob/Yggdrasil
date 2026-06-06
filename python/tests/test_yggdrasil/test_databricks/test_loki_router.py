"""Tests for the NL → databricks-skill router and its plan() integration."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from yggdrasil.databricks.loki.router import route
from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend


class TestRoute(unittest.TestCase):
    def test_catalogs_schemas_tables(self):
        self.assertEqual(route("list the catalogs"), ("databricks-catalogs", {}))
        self.assertEqual(route("schemas in samples"),
                         ("databricks-schemas", {"catalog": "samples"}))
        self.assertEqual(route("what tables are in samples.nyctaxi"),
                         ("databricks-tables", {"catalog": "samples", "schema": "nyctaxi"}))
        self.assertEqual(route("describe samples.nyctaxi.trips"),
                         ("databricks-tables",
                          {"catalog": "samples", "schema": "nyctaxi", "table": "trips"}))

    def test_sql_statement(self):
        skill, kw = route("SELECT count(*) FROM samples.nyctaxi.trips")
        self.assertEqual(skill, "databricks-sql")
        self.assertIn("SELECT", kw["query"])

    def test_services(self):
        self.assertEqual(route("show the warehouses")[0], "databricks-warehouses")
        self.assertEqual(route("list jobs")[0], "databricks-jobs")
        self.assertEqual(route("clusters")[0], "databricks-clusters")
        self.assertEqual(route("volumes in samples")[0], "databricks-volumes")
        self.assertEqual(route("secret scopes")[0], "databricks-secrets")
        self.assertEqual(route("who am i"), ("databricks-iam", {"what": "me"}))
        self.assertEqual(route("list serving endpoints")[0], "databricks-serving")

    def test_no_match_falls_through(self):
        self.assertIsNone(route("explain how delta lake works"))
        self.assertIsNone(route("hello there"))


class TestPlanIntegration(unittest.TestCase):
    def _loki(self, *, databricks: bool):
        loki = Loki()
        loki._backends = [Backend("databricks", databricks), Backend("local", True)]
        return loki

    def test_session_present_routes_to_skill(self):
        plan = self._loki(databricks=True).plan("what tables are in samples.nyctaxi")
        self.assertEqual(plan["action"], "skill")
        self.assertEqual(plan.skill, "databricks-tables")
        self.assertEqual(plan.skill_kwargs, {"catalog": "samples", "schema": "nyctaxi"})
        self.assertEqual(plan["specialist"], "databricks")

    def test_no_session_does_not_route_to_skill(self):
        plan = self._loki(databricks=False).plan("list the catalogs")
        self.assertNotEqual(plan["action"], "skill")

    def test_url_request_is_not_hijacked(self):
        plan = self._loki(databricks=True).plan("fetch https://example.com/data.csv")
        self.assertNotEqual(plan["action"], "skill")


if __name__ == "__main__":
    unittest.main()
