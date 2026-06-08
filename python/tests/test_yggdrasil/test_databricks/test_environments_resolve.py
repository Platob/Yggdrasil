"""Unit tests for the centralized base-environment discovery on
:class:`~yggdrasil.databricks.environments.service.Environments` —
``resolve`` (direct path / named stem / auto) and ``client_project``.

Discovery used to live in ``job/service.py``; it now belongs to the dedicated
``dbc.environments`` service. These exercise the resolution branches directly
against mocked ``list`` / ``get`` / workspace lookups.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from yggdrasil.databricks.environments.environment import Environment
from yggdrasil.databricks.environments.service import Environments
from yggdrasil.version import VersionInfo


def _service() -> Environments:
    return Environments(client=MagicMock())


class TestResolveDirectPath:
    def test_existing_yml_path_wraps_as_environment(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as from_:
            path = from_.return_value
            path.exists.return_value = True
            path.full_path.return_value = (
                "/Workspace/Shared/environment/ygg/0.8.58/ygg-0.8.58-py311.yml"
            )
            env = svc.resolve("/Workspace/Shared/environment/ygg/0.8.58/ygg-0.8.58-py311.yml")
        assert isinstance(env, Environment)
        assert env.serverless.endswith("ygg-0.8.58-py311.yml")
        assert env.name == "ygg-0.8.58-py311"

    def test_missing_path_returns_none(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as from_:
            from_.return_value.exists.return_value = False
            assert svc.resolve("/Workspace/Shared/environment/nope.yml") is None


class TestResolveNamedStem:
    def test_matches_deployed_stem(self) -> None:
        svc = _service()
        target = Environment(name="meteologica-1.0.0-py311",
                             serverless="/ws/meteologica.yml")
        with patch.object(svc, "list", return_value=[
            Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml"),
            target,
        ]):
            assert svc.resolve("meteologica-1.0.0-py311") is target

    def test_unknown_stem_returns_none(self) -> None:
        svc = _service()
        with patch.object(svc, "list", return_value=[]):
            assert svc.resolve("ghost-9.9.9-py311") is None


class TestResolveByProjectName:
    def test_project_name_resolves_latest_version(self) -> None:
        # ``environment="meteologica"`` (a project name, not a version-tagged stem)
        # → the latest deployed version for the current Python.
        from yggdrasil.databricks.environments.service import environment_stem
        from yggdrasil.databricks.wheels.service import environment_key_for

        svc = _service()
        key = environment_key_for()                     # the local interpreter's py3XX
        old = Environment(name=f"meteologica-1.0.0-{key}",
                          project="meteologica", version=VersionInfo(1, 0, 0),
                          serverless="/ws/meteologica-1.0.0.yml")
        new = Environment(name=f"meteologica-1.2.0-{key}",
                          project="meteologica", version=VersionInfo(1, 2, 0),
                          serverless="/ws/meteologica-1.2.0.yml")
        with patch.object(svc, "list", return_value=[old, new]):
            # The bare project name isn't an exact stem, so it falls through to a
            # by-folder lookup that returns the highest version.
            resolved = svc.resolve("meteologica")
        assert resolved is new
        assert str(environment_stem("meteologica", version="1.2.0")) == new.name

    def test_project_name_filters_to_current_python(self) -> None:
        from yggdrasil.databricks.wheels.service import environment_key_for

        svc = _service()
        key = environment_key_for()
        other = "py399" if key != "py399" else "py398"   # a Python that isn't local
        match = Environment(name=f"meteologica-2.0.0-{key}", project="meteologica",
                            version=VersionInfo(2, 0, 0), serverless="/ws/m2.yml")
        with patch.object(svc, "list", return_value=[
            # A newer version built for a *different* Python must not be picked.
            Environment(name=f"meteologica-9.9.9-{other}", project="meteologica",
                        version=VersionInfo(9, 9, 9), serverless="/ws/m999.yml"),
            match,
        ]):
            assert svc.resolve("meteologica") is match


class TestResolveAuto:
    def test_prefers_client_project(self) -> None:
        svc = _service()
        svc.client.project = "ygg"
        proj = Environment(name="myproj-2.0.0-py311", serverless="/ws/myproj.yml")
        with patch.object(svc, "client_project", return_value=proj) as cp, \
                patch.object(svc, "get") as get:
            assert svc.resolve() is proj
            cp.assert_called_once()
            get.assert_not_called()

    def test_falls_back_to_client_project_env(self) -> None:
        svc = _service()
        svc.client.project = "ygg"
        ygg = Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml")
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", return_value=ygg) as get:
            assert svc.resolve() is ygg
            get.assert_called_once_with("ygg", None, python=None,
                                        workspace_dir=None, refresh=False)

    def test_returns_none_when_nothing_deployed(self) -> None:
        svc = _service()
        svc.client.project = "ygg"
        with patch.object(svc, "client_project", return_value=None), \
                patch.object(svc, "get", return_value=None):
            assert svc.resolve() is None


class TestListCache:
    _WALK = [
        "/ws/env/ygg/0.8.58/ygg-0.8.58-py311.yml",
        "/ws/env/ygg/0.8.58/ygg-0.8.58-py311.requirements.txt",
    ]

    def test_list_served_from_cache_within_ttl(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.deployed_environments",
                   return_value=self._WALK) as walk:
            first = svc.list()
            second = svc.list()
        assert walk.call_count == 1            # second call hit the cache
        assert second is first                 # same cached snapshot object
        assert [e.name for e in first] == ["ygg-0.8.58-py311"]

    def test_refresh_bypasses_cache(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.deployed_environments",
                   return_value=self._WALK) as walk:
            svc.list()
            svc.list(refresh=True)
        assert walk.call_count == 2

    def test_invalidate_cache_forces_rewalk(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.deployed_environments",
                   return_value=self._WALK) as walk:
            svc.list()
            svc.invalidate_cache()      # e.g. after an out-of-band deploy
            svc.list()
        assert walk.call_count == 2

    def test_get_and_resolve_share_one_walk(self) -> None:
        # get → find → list, and resolve(stem) → list: both ride one cached walk.
        # Build the stem for the live interpreter so find()'s py3XX filter matches.
        from yggdrasil.databricks.wheels.service import environment_key_for

        stem = f"ygg-0.8.58-{environment_key_for()}"
        walk_paths = [f"/ws/env/ygg/0.8.58/{stem}.yml",
                      f"/ws/env/ygg/0.8.58/{stem}.requirements.txt"]
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.deployed_environments",
                   return_value=walk_paths) as walk:
            got = svc.get("ygg")
            resolved = svc.resolve(stem)
        assert walk.call_count == 1
        assert got is not None and got.name == stem
        assert resolved is not None and resolved.name == stem


class TestClientProject:
    def test_reads_local_pyproject_and_looks_up(self) -> None:
        svc = _service()
        proj = Environment(name="ygg-0.8.58-py311", serverless="/ws/ygg.yml")
        with patch("yggdrasil.databricks.environments.service.find_pyproject",
                   return_value="/repo/pyproject.toml"), \
                patch("yggdrasil.databricks.environments.service.read_pyproject",
                      return_value={"name": "ygg", "version": "0.8.58"}), \
                patch.object(svc, "get", return_value=proj) as get:
            assert svc.client_project() is proj
            get.assert_called_once_with("ygg", "0.8.58", workspace_dir=None, refresh=False)

    def test_no_pyproject_returns_none(self) -> None:
        svc = _service()
        with patch("yggdrasil.databricks.environments.service.find_pyproject",
                   side_effect=OSError):
            assert svc.client_project() is None
