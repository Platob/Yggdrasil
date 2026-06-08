"""Unit tests for the Loki toolbox — the agent's confined file/shell hands."""
from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest.mock import patch

from yggdrasil.loki.tools import filesystem_toolbox


class TestFilesystemToolbox(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ygg-tools-")
        self.root = pathlib.Path(self.dir)
        (self.root / "a.txt").write_text("hello\nworld\n")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.py").write_text("x = 1\nprint(x)\n")

    def test_read_only_box_has_no_write_tools(self):
        box = filesystem_toolbox(self.dir, read_only=True)
        self.assertIn("read_file", box.names())
        self.assertNotIn("write_file", box.names())
        self.assertNotIn("edit_file", box.names())
        self.assertNotIn("run", box.names())

    def test_shell_and_python_are_default_write_tools(self):
        names = filesystem_toolbox(self.dir).names()
        self.assertIn("run", names)         # shell on by default now
        self.assertIn("run_python", names)
        self.assertNotIn("run", filesystem_toolbox(self.dir, read_only=True).names())

    def test_shell_runs_a_command(self):
        out = filesystem_toolbox(self.dir).call("run", {"command": "echo hi && pwd"})
        self.assertIn("exit=0", out)
        self.assertIn("hi", out)

    def test_run_python_is_a_default_write_tool(self):
        box = filesystem_toolbox(self.dir)
        self.assertIn("run_python", box.names())
        out = box.call("run_python", {"code": "print(6 * 7)"})
        self.assertIn("exit=0", out)
        self.assertIn("42", out)

    def test_run_python_absent_when_read_only(self):
        self.assertNotIn("run_python", filesystem_toolbox(self.dir, read_only=True).names())

    def test_mesh_tool_present_only_when_in_a_mesh(self):
        import os
        from unittest.mock import patch

        self.assertNotIn("mesh", filesystem_toolbox(self.dir).names())   # not in a mesh
        mesh_path = os.path.join(self.dir, "mesh-kv.json")
        with patch.dict(os.environ, {"LOKI_MESH": mesh_path}):
            box = filesystem_toolbox(self.dir)
            self.assertIn("mesh", box.names())
            self.assertIn("published", box.call("mesh", {"action": "put", "key": "k", "value": "v"}))
            self.assertIn("v", box.call("mesh", {"action": "get", "key": "k"}))
            self.assertIn("k", box.call("mesh", {"action": "list"}))

    def test_smoke_and_bench_are_default_validation_tools(self):
        names = filesystem_toolbox(self.dir).names()
        self.assertIn("smoke", names)
        self.assertIn("bench", names)
        ro = filesystem_toolbox(self.dir, read_only=True).names()
        self.assertNotIn("smoke", ro)        # pair with the write tools

    def test_smoke_code_runs_assertion(self):
        box = filesystem_toolbox(self.dir)
        ok = box.call("smoke", {"code": "assert 2 + 2 == 4; print('green')"})
        self.assertIn("exit=0", ok)
        self.assertIn("green", ok)
        bad = box.call("smoke", {"code": "assert 1 == 2, 'boom'"})
        self.assertNotIn("exit=0", bad)      # a failing checkpoint is visible

    def test_smoke_auto_discovers_pytest_with_src_importable(self):
        import pathlib

        root = pathlib.Path(self.dir)
        (root / "src").mkdir(exist_ok=True)
        (root / "src" / "mod.py").write_text("def add(a, b):\n    return a + b\n")
        (root / "tests").mkdir(exist_ok=True)
        (root / "tests" / "test_mod.py").write_text(
            "from mod import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
        out = filesystem_toolbox(self.dir).call("smoke", {})
        self.assertIn("exit=0", out)
        self.assertIn("passed", out)

    def test_smoke_byte_compiles_when_no_tests(self):
        import pathlib

        (pathlib.Path(self.dir) / "thing.py").write_text("x = 1\n")
        out = filesystem_toolbox(self.dir).call("smoke", {})
        self.assertIn("compiled OK", out)

    def test_bench_runs_a_script(self):
        import pathlib

        (pathlib.Path(self.dir) / "b.py").write_text(
            'import argparse\np = argparse.ArgumentParser()\n'
            'p.add_argument("--repeat", type=int, default=5)\n'
            'print("reps", p.parse_args().repeat)\n')
        out = filesystem_toolbox(self.dir).call("bench", {"path": "b.py", "repeat": 3})
        self.assertIn("exit=0", out)
        self.assertIn("reps 3", out)

    def test_confirm_gates_overwrite_of_existing_nontemp_file(self):
        (self.root / "keep.txt").write_text("original")
        calls = []

        def confirm(action):
            calls.append(action)
            return False

        # Make the temp dir look non-temporary so the gate engages.
        with patch("yggdrasil.loki.tools.tempfile.gettempdir", return_value="/nowhere"):
            box = filesystem_toolbox(self.dir, confirm=confirm)
            out = box.call("write_file", {"path": "keep.txt", "content": "NEW"})
        self.assertIn("REFUSED", out)
        self.assertEqual((self.root / "keep.txt").read_text(), "original")
        self.assertTrue(calls and "overwrite keep.txt" in calls[0])

    def test_confirm_not_asked_for_new_file(self):
        with patch("yggdrasil.loki.tools.tempfile.gettempdir", return_value="/nowhere"):
            box = filesystem_toolbox(self.dir, confirm=lambda a: False)  # would refuse
            out = box.call("write_file", {"path": "fresh.txt", "content": "x"})
        self.assertIn("created", out)                    # new file → not gated

    def test_confirm_skipped_for_temporary_root(self):
        # The real temp root (under the system temp dir) is scratch → no confirm.
        (self.root / "k.txt").write_text("a")
        box = filesystem_toolbox(self.dir, confirm=lambda a: False)
        out = box.call("write_file", {"path": "k.txt", "content": "b"})
        self.assertIn("overwrote", out)

    def test_web_tools_are_opt_in(self):
        base = filesystem_toolbox(self.dir)
        self.assertNotIn("web_fetch", base.names())
        web = filesystem_toolbox(self.dir, allow_web=True)
        self.assertIn("web_fetch", web.names())
        self.assertIn("web_table", web.names())
        self.assertIn("web_image", web.names())

    def test_read_table_tool_parses_local_csv(self):
        try:
            import polars  # noqa: F401
        except Exception:
            self.skipTest("requires the polars/io data stack")
        (self.root / "t.csv").write_text("a,b\n1,2\n3,4\n")
        out = filesystem_toolbox(self.dir).call("read_table", {"path": "t.csv"})
        self.assertIn("shape=(2, 2)", out)
        self.assertIn("a", out)

    def test_list_dir_and_read_file(self):
        box = filesystem_toolbox(self.dir)
        listing = box.call("list_dir", {"path": "."})
        self.assertIn("a.txt", listing)
        self.assertIn("sub/", listing)
        body = box.call("read_file", {"path": "a.txt"})
        self.assertIn("1\thello", body)
        self.assertIn("2\tworld", body)

    def test_read_file_line_range(self):
        box = filesystem_toolbox(self.dir)
        body = box.call("read_file", {"path": "sub/b.py", "start": 2, "end": 2})
        self.assertIn("2\tprint(x)", body)
        self.assertNotIn("x = 1", body)

    def test_find_and_grep(self):
        box = filesystem_toolbox(self.dir)
        self.assertIn("sub/b.py", box.call("find", {"pattern": "*.py"}))
        hits = box.call("grep", {"pattern": r"print", "glob": "*.py"})
        self.assertIn("sub/b.py:2", hits)

    def test_write_records_change(self):
        box = filesystem_toolbox(self.dir)
        out = box.call("write_file", {"path": "new/c.txt", "content": "hi"})
        self.assertIn("created", out)
        self.assertEqual((self.root / "new" / "c.txt").read_text(), "hi")
        self.assertEqual(box.changed, ["new/c.txt"])

    def test_edit_unique_replacement(self):
        box = filesystem_toolbox(self.dir)
        out = box.call("edit_file", {"path": "a.txt", "old": "world", "new": "loki"})
        self.assertIn("edited", out)
        self.assertEqual((self.root / "a.txt").read_text(), "hello\nloki\n")
        self.assertIn("a.txt", box.changed)

    def test_edit_missing_text_is_a_readable_error(self):
        box = filesystem_toolbox(self.dir)
        out = box.call("edit_file", {"path": "a.txt", "old": "nope", "new": "x"})
        self.assertIn("not found", out)
        self.assertEqual(box.changed, [])

    def test_edit_ambiguous_text_refused(self):
        box = filesystem_toolbox(self.dir)
        (self.root / "dup.txt").write_text("z\nz\n")
        out = box.call("edit_file", {"path": "dup.txt", "old": "z", "new": "q"})
        self.assertIn("2×", out)

    def test_path_confinement_refused(self):
        box = filesystem_toolbox(self.dir)
        out = box.call("read_file", {"path": "../../etc/passwd"})
        self.assertIn("escapes the agent root", out)
        out = box.call("write_file", {"path": "../escape.txt", "content": "x"})
        self.assertIn("escapes the agent root", out)

    def test_unknown_tool_and_bad_args(self):
        box = filesystem_toolbox(self.dir)
        self.assertIn("unknown tool", box.call("nope", {}))
        self.assertIn("bad arguments", box.call("read_file", {"wrong": 1}))

    def test_shell_runs_in_root(self):
        box = filesystem_toolbox(self.dir, allow_shell=True)
        out = box.call("run", {"command": "echo hi && ls"})
        self.assertIn("exit=0", out)
        self.assertIn("hi", out)
        self.assertIn("a.txt", out)


if __name__ == "__main__":
    unittest.main()
