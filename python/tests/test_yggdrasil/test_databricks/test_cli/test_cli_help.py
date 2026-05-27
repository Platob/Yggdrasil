"""Tests for CLI help output and basic dispatching."""
from __future__ import annotations

import unittest

from yggdrasil.databricks.cli import main


class TestMainHelp(unittest.TestCase):

    def test_main_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_command_returns_one(self):
        self.assertEqual(main([]), 1)


class TestClustersHelp(unittest.TestCase):

    def test_clusters_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_list_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_create_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "create", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_start_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "start", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestWarehousesHelp(unittest.TestCase):

    def test_warehouses_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_warehouses_list_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_warehouses_create_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "create", "--help"])
        self.assertEqual(ctx.exception.code, 0)
