"""Tests for PyEnv environment-variable CRUD and run application."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from yggdrasil.node.api.schemas.excel import ExcelQueryRequest
from yggdrasil.node.api.schemas.pyenv import PyEnvCreate, PyEnvUpdate
from yggdrasil.node.api.services.excel import ExcelService
from yggdrasil.node.api.services.fs import FsService
from yggdrasil.node.api.services.pyenv import PyEnvService
from yggdrasil.node.config import Settings
from yggdrasil.node.exceptions import NotFoundError


def _settings(home: Path) -> Settings:
    return Settings(node_id="t", node_home=home, front_home=home)


def _register(svc: PyEnvService, name: str, env_vars: dict | None = None) -> int:
    """Insert a ready env without building a venv."""
    import datetime as dt
    from yggdrasil.node.api.schemas.pyenv import PyEnvEntry
    from yggdrasil.node.ids import make_id
    eid = make_id(name)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    svc._envs.set(eid, PyEnvEntry(
        id=eid, name=name, python_version="3.11", dependencies=[],
        env_vars=env_vars or {}, path=str(svc._envs_root / str(eid)),
        status="ready", created_at=now, updated_at=now,
    ))
    svc._name_to_id[name] = eid
    return eid


class TestPyEnvEnvVarsCrud(unittest.TestCase):
    def _svc(self, home):
        return PyEnvService(_settings(home))

    def test_create_stores_env_vars(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            # create() builds a venv; skip that and test the CRUD surface
            # against a registered entry instead.
            eid = _register(svc, "e1", {"A": "1"})
            got = asyncio.run(svc.get_env_vars(eid))
            self.assertEqual(got.env_vars, {"A": "1"})

    def test_set_merges_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1"})
            res = asyncio.run(svc.set_env_vars(eid, {"B": "2"}))
            self.assertEqual(res.env_vars, {"A": "1", "B": "2"})

    def test_set_replace_swaps_whole_map(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1", "B": "2"})
            res = asyncio.run(svc.set_env_vars(eid, {"C": "3"}, replace=True))
            self.assertEqual(res.env_vars, {"C": "3"})

    def test_set_coerces_values_to_str(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e")
            res = asyncio.run(svc.set_env_vars(eid, {"N": 5}))
            self.assertEqual(res.env_vars, {"N": "5"})

    def test_delete_var(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1", "B": "2"})
            res = asyncio.run(svc.delete_env_var(eid, "A"))
            self.assertEqual(res.env_vars, {"B": "2"})

    def test_delete_missing_var_raises(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1"})
            with self.assertRaises(NotFoundError):
                asyncio.run(svc.delete_env_var(eid, "NOPE"))

    def test_env_vars_for_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1"})
            self.assertEqual(svc.env_vars_for(eid), {"A": "1"})
            self.assertEqual(svc.env_vars_for_name("e"), {"A": "1"})
            self.assertEqual(svc.env_vars_for(None), {})
            self.assertEqual(svc.env_vars_for_name("missing"), {})

    def test_update_sets_env_vars(self):
        with tempfile.TemporaryDirectory() as d:
            svc = self._svc(Path(d))
            eid = _register(svc, "e", {"A": "1"})
            asyncio.run(svc.update(eid, PyEnvUpdate(env_vars={"X": "9"})))
            self.assertEqual(svc.env_vars_for(eid), {"X": "9"})


class TestEnvVarsAppliedToExcelRun(unittest.TestCase):
    def test_env_stored_vars_reach_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            pyenv = PyEnvService(_settings(home))
            # point a "ready" env at the live interpreter so the run works
            import sys
            eid = _register(pyenv, "withvars", {"YGG_FROM_ENV": "from-pyenv"})
            pyenv._envs[eid] = pyenv._envs.get(eid).model_copy(
                update={"path": str(Path(sys.executable).parent.parent)}
            )
            # get_python_path checks bin/python under path — point at the
            # real venv-like prefix of the running interpreter.
            excel = ExcelService(_settings(home), fs=FsService(_settings(home)), pyenv=pyenv)
            # request var overrides; stored var also present
            table = asyncio.run(excel.run_python(ExcelQueryRequest(
                code="import os\ndf = {'a': [os.environ.get('YGG_FROM_ENV','?')], "
                     "'b': [os.environ.get('YGG_FROM_REQ','?')]}",
                env="withvars",
                env_vars={"YGG_FROM_REQ": "from-req"},
            )))
            self.assertEqual(table.column("a").to_pylist(), ["from-pyenv"])
            self.assertEqual(table.column("b").to_pylist(), ["from-req"])


if __name__ == "__main__":
    unittest.main()
