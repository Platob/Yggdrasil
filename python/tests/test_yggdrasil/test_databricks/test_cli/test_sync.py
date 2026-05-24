"""Tests for ``yggdrasil.databricks.cli.bundle.sync``."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from yggdrasil.databricks.cli.bundle.sync import (
    _collect_files,
    _resolve_notebook_root,
)


class TestCollectFiles(unittest.TestCase):

    def _make_tree(self, tmpdir: str) -> Path:
        root = Path(tmpdir)
        (root / "ingest.py").write_text("# ingest")
        (root / "transform.py").write_text("# transform")
        (root / "config.yml").write_text("key: val")
        (root / "sub").mkdir()
        (root / "sub" / "helper.py").write_text("# helper")
        return root

    def test_include_filters(self):
        with TemporaryDirectory() as d:
            root = self._make_tree(d)
            files = _collect_files(root, ["*.py"], [])
            names = {f.name for f in files}
            self.assertEqual(names, {"ingest.py", "transform.py"})

    def test_include_empty_collects_all(self):
        with TemporaryDirectory() as d:
            root = self._make_tree(d)
            files = _collect_files(root, [], [])
            names = {f.name for f in files}
            self.assertIn("config.yml", names)
            self.assertIn("helper.py", names)

    def test_exclude_filters(self):
        with TemporaryDirectory() as d:
            root = self._make_tree(d)
            files = _collect_files(root, [], ["*.yml"])
            names = {f.name for f in files}
            self.assertNotIn("config.yml", names)
            self.assertIn("ingest.py", names)

    def test_include_and_exclude_combined(self):
        with TemporaryDirectory() as d:
            root = self._make_tree(d)
            files = _collect_files(root, ["*.py", "*.yml"], ["config.yml"])
            names = {f.name for f in files}
            self.assertIn("ingest.py", names)
            self.assertNotIn("config.yml", names)

    def test_sorted_output(self):
        with TemporaryDirectory() as d:
            root = self._make_tree(d)
            files = _collect_files(root, ["*.py"], [])
            self.assertEqual(files, sorted(files))


class TestResolveNotebookRoot(unittest.TestCase):

    def test_from_variables(self):
        resolved = {
            "variables": {
                "notebook_root": {"default": "/Shared/test"},
            },
            "resources": {},
        }
        self.assertEqual(_resolve_notebook_root(resolved), "/Shared/test")

    def test_from_plain_variable(self):
        resolved = {
            "variables": {"notebook_root": "/Shared/plain"},
            "resources": {},
        }
        self.assertEqual(_resolve_notebook_root(resolved), "/Shared/plain")

    def test_from_task_notebook_path(self):
        resolved = {
            "variables": {},
            "resources": {
                "jobs": {
                    "j": {
                        "tasks": [
                            {
                                "notebook_task": {
                                    "notebook_path": "/Workspace/Users/test/step",
                                },
                            },
                        ],
                    },
                },
            },
        }
        self.assertEqual(
            _resolve_notebook_root(resolved),
            "/Workspace/Users/test",
        )

    def test_returns_none_when_no_clue(self):
        resolved = {"variables": {}, "resources": {}}
        self.assertIsNone(_resolve_notebook_root(resolved))
