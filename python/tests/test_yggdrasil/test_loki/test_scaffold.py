"""Tests for project scaffolding — yggdrasil.loki.scaffold + ScaffoldSkill."""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from yggdrasil.loki import scaffold


class TestResolveLanguages(unittest.TestCase):
    def test_detects_named_languages(self):
        self.assertEqual(scaffold.resolve_languages("a python + rust cli"), ["python", "rust"])

    def test_aliases_map_to_canonical(self):
        self.assertEqual(scaffold.resolve_languages("a TS/node service"), ["typescript"])
        self.assertEqual(scaffold.resolve_languages("golang tool"), ["go"])

    def test_word_boundary_avoids_false_hits(self):
        # "go" inside "good" must not match the Go language.
        self.assertEqual(scaffold.resolve_languages("a good python project"), ["python"])

    def test_empty_when_none_named(self):
        self.assertEqual(scaffold.resolve_languages("just a project"), [])


class TestScaffoldProject(unittest.TestCase):
    def test_python_tree_and_pyproject(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("acme-cli", ["python"], base_dir=d, git=False)
            root = Path(res["path"])
            self.assertTrue((root / "README.md").is_file())
            self.assertTrue((root / ".gitignore").is_file())
            self.assertTrue((root / "python" / "pyproject.toml").is_file())
            self.assertTrue((root / "python" / "src" / "acme_cli" / "__init__.py").is_file())
            self.assertTrue((root / "python" / "tests" / "test_smoke.py").is_file())
            # pyproject is rendered with the project name + identifier-safe pkg.
            pyproj = (root / "python" / "pyproject.toml").read_text()
            self.assertIn('name = "acme-cli"', pyproj)
            self.assertIn('packages = ["src/acme_cli"]', pyproj)

    def test_polyglot_creates_one_folder_per_language(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("poly", ["python", "rust", "go"], base_dir=d, git=False)
            root = Path(res["path"])
            for lang, manifest in (("python", "pyproject.toml"), ("rust", "Cargo.toml"), ("go", "go.mod")):
                self.assertTrue((root / lang / manifest).is_file(), lang)
                self.assertTrue((root / lang / "src").is_dir())
                self.assertTrue((root / lang / "tests").is_dir())
            self.assertEqual(res["languages"], ["python", "rust", "go"])

    def test_gitignore_composes_per_language_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("x", ["python", "typescript"], base_dir=d, git=False)
            gi = (Path(res["path"]) / ".gitignore").read_text()
            self.assertIn("__pycache__/", gi)        # python block
            self.assertIn("node_modules/", gi)        # typescript block
            self.assertIn(".DS_Store", gi)            # common block

    def test_unknown_language_is_dropped_and_reported(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("x", ["python", "cobol"], base_dir=d, git=False)
            self.assertEqual(res["languages"], ["python"])
            self.assertIn("cobol", res["unknown_languages"])

    def test_defaults_to_python_when_no_known_language(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("x", ["cobol"], base_dir=d, git=False)
            self.assertEqual(res["languages"], ["python"])

    def test_git_init_makes_initial_commit_on_main(self):
        if not _has_git():
            self.skipTest("git not available")
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("g", ["python"], base_dir=d, git=True)
            self.assertTrue(res["git"])
            log = subprocess.run(["git", "-C", res["path"], "log", "--oneline"],
                                 capture_output=True, text=True).stdout
            self.assertIn("Initial scaffold", log)


class TestPresets(unittest.TestCase):
    def test_resolve_preset_and_cloud(self):
        self.assertEqual(scaffold.resolve_preset("build a realtime full-stack app"),
                         "fullstack-realtime")
        self.assertEqual(scaffold.resolve_preset("a python library"), "lib")
        self.assertEqual(scaffold.resolve_cloud("deploy on aws and databricks"),
                         ["aws", "databricks"])

    def test_fullstack_realtime_tree_is_runnable(self):
        import ast
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("pulse", preset="fullstack-realtime",
                                            cloud=["aws", "databricks"], base_dir=d, git=False)
            root = Path(res["path"])
            self.assertEqual(res["preset"], "fullstack-realtime")
            self.assertEqual(res["cloud"], ["aws", "databricks"])
            for rel in ("backend/src/app/main.py", "backend/tests/test_app.py",
                        "frontend/index.html", "frontend/app.js", "Dockerfile",
                        ".github/workflows/ci.yml", "deploy/aws.md", "deploy/databricks.md"):
                self.assertTrue((root / rel).is_file(), rel)
            main = (root / "backend" / "src" / "app" / "main.py").read_text()
            ast.parse(main)                              # valid Python
            self.assertNotIn("{{", main)                 # no leaked format braces
            self.assertIn('f"data: {json.dumps(_tick())}', main)   # real SSE f-string
            # No leaked braces in the JS / YAML either.
            self.assertNotIn("{{", (root / "frontend" / "app.js").read_text())
            self.assertNotIn("{{", (root / ".github" / "workflows" / "ci.yml").read_text())

    def test_repl_scaffold_args_infer_name_preset_cloud(self):
        from yggdrasil.loki import cli

        args = cli._scaffold_args("build a realtime full-stack app called pulse on aws and databricks")
        self.assertEqual(args["name"], "pulse")
        self.assertEqual(args["preset"], "fullstack-realtime")
        self.assertEqual(args["cloud"], ["aws", "databricks"])

    def test_repl_scaffold_args_name_from_called(self):
        from yggdrasil.loki import cli

        # "project" is a noun here, not the name — only "called X" names it.
        self.assertEqual(cli._scaffold_args("create a new python project called acme")["name"], "acme")
        self.assertEqual(cli._scaffold_args("a new go service repo")["name"], "new-project")

    def test_unknown_cloud_target_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            res = scaffold.scaffold_project("x", preset="fullstack-realtime",
                                            cloud=["aws", "gcp"], base_dir=d, git=False)
            self.assertEqual(res["cloud"], ["aws"])


class TestScaffoldSkill(unittest.TestCase):
    def test_run_via_loki_with_name_kwarg(self):
        # Regression: Loki.run reserved `name`, clashing with scaffold(name=…).
        from yggdrasil.loki import Loki

        with tempfile.TemporaryDirectory() as d:
            res = Loki().run("scaffold", name="acme", languages=["python"], base_dir=d, git=False)
        self.assertEqual(res["name"], "acme")
        self.assertEqual(res["languages"], ["python"])


def _has_git() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except OSError:
        return False


if __name__ == "__main__":
    unittest.main()
