"""Unit tests for the uniform wheel registry service (``dbc.wheels``).

Covers version parsing/comparison (:class:`VersionInfo`), wheel-filename parsing,
``registry_upload``, ``runtime_dependencies``, and the CRUD surface
(create/find/get/update/delete/list) with the fetch + workspace layers mocked.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.wheels.service import (
    Wheels,
    parse_version,
    registry_upload,
    runtime_dependencies,
    wheel_parts,
)
from yggdrasil.databricks.wheels.wheel import Wheel
from yggdrasil.version import VersionInfo


# ── version + filename parsing ──────────────────────────────────────────────
class TestParsing:
    def test_parse_version_forms(self):
        assert parse_version("0.8.57") == VersionInfo(0, 8, 57)
        assert parse_version("0_8_57") == VersionInfo(0, 8, 57)         # wheel escaping
        assert parse_version("1.2.3+host.box") == VersionInfo(1, 2, 3)  # local segment
        assert parse_version(VersionInfo(1, 0, 0)) == VersionInfo(1, 0, 0)
        assert parse_version("not-a-version") is None
        assert parse_version(None) is None

    def test_version_ordering(self):
        assert parse_version("0.8.57") < parse_version("0.9.0")
        assert max(parse_version(v) for v in ("1.0.0", "1.2.0", "1.1.9")) == VersionInfo(1, 2, 0)

    def test_wheel_parts(self):
        dist, version, tag = wheel_parts("/ws/pypi/ygg/ygg-0.8.57-py3-none-any.whl")
        assert dist == "ygg"
        assert version == VersionInfo(0, 8, 57)
        assert tag == "py3-none-any"
        assert wheel_parts("databricks_sdk-0.1.0-py3-none-any.whl")[0] == "databricks-sdk"


# ── registry_upload + runtime_dependencies ──────────────────────────────────
class TestHelpers:
    def test_registry_upload_lands_in_dist_folder(self, tmp_path):
        local = tmp_path / "ygg-1.0-py3-none-any.whl"
        local.write_bytes(b"wheel")
        node = MagicMock(); node.exists.return_value = False
        dbp = MagicMock(); dbp.from_.return_value = node
        with patch("yggdrasil.databricks.path.DatabricksPath", dbp):
            dest = registry_upload(MagicMock(), local, workspace_dir="/ws/pypi")
        assert dest == "/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"
        node.write_bytes.assert_called_once()

    def test_registry_upload_reuses_existing(self, tmp_path):
        local = tmp_path / "ygg-1.0-py3-none-any.whl"
        local.write_bytes(b"wheel")
        node = MagicMock(); node.exists.return_value = True
        dbp = MagicMock(); dbp.from_.return_value = node
        with patch("yggdrasil.databricks.path.DatabricksPath", dbp):
            dest = registry_upload(MagicMock(), local, workspace_dir="/ws/pypi")
        assert dest.endswith("ygg-1.0-py3-none-any.whl")
        node.write_bytes.assert_not_called()              # immutable → skipped

    def test_runtime_dependencies_pins_bare_names(self):
        import importlib.metadata as ilmd

        with patch.object(ilmd, "requires", return_value=["polars", "httpx>=0.2",
                                                          'pytest; extra == "test"']), \
             patch.object(ilmd, "version", return_value="1.5.0"), \
             patch.object(ilmd, "packages_distributions", return_value={}):
            out = runtime_dependencies("ygg", extras=())
        assert "polars==1.5.0" in out                     # bare name pinned
        assert "httpx>=0.2" in out                        # spec kept
        assert all("pytest" not in d for d in out)        # other-extra dep dropped


# ── CRUD ────────────────────────────────────────────────────────────────────
def _svc():
    return Wheels(client=MagicMock())


def _wheel(svc, path):
    return Wheel(svc, path=path)


class TestCreate:
    def test_create_fetches_and_uploads(self):
        svc = _svc()
        with patch("yggdrasil.databricks.wheels.service.fetch_wheels",
                   return_value=[Path("/tmp/ygg-1.0-py3-none-any.whl")]) as fetch, \
             patch("yggdrasil.databricks.wheels.service.registry_upload",
                   side_effect=lambda c, w, **k: f"/ws/pypi/ygg/{Path(w).name}"):
            out = svc.create("ygg", "1.0")
        assert fetch.call_args.args[0] == "ygg"
        assert fetch.call_args.args[1] == "1.0.0"          # normalized via VersionInfo
        assert [w.path for w in out] == ["/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"]

    def test_update_forces_rebuild_and_overwrite(self):
        svc = _svc()
        with patch.object(Wheels, "create", return_value=[]) as create:
            svc.update("ygg")
        assert create.call_args.kwargs["overwrite"] is True
        assert create.call_args.kwargs["rebuild"] is True


class TestFindGet:
    def test_find_selects_by_version(self):
        svc = _svc()
        wheels = [_wheel(svc, "/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"),
                  _wheel(svc, "/ws/pypi/ygg/ygg-2.0-py3-none-any.whl")]
        with patch.object(Wheels, "list", return_value=wheels):
            hit = svc.find("ygg", "1.0", install=False)
        assert str(hit.version) == "1.0.0"

    def test_find_latest_when_no_version(self):
        svc = _svc()
        wheels = [_wheel(svc, "/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"),
                  _wheel(svc, "/ws/pypi/ygg/ygg-2.0-py3-none-any.whl")]
        with patch.object(Wheels, "list", return_value=wheels):
            hit = svc.find("ygg", install=False)
        assert str(hit.version) == "2.0.0"

    def test_find_builds_on_miss(self):
        svc = _svc()
        built = _wheel(svc, "/ws/pypi/ygg/ygg-1.0-py3-none-any.whl")
        with patch.object(Wheels, "list", return_value=[]), \
             patch.object(Wheels, "create", return_value=[built]) as create:
            hit = svc.find("ygg", "1.0")
        create.assert_called_once()
        assert hit is built

    def test_get_never_builds(self):
        svc = _svc()
        with patch.object(Wheels, "list", return_value=[]), \
             patch.object(Wheels, "create") as create:
            assert svc.get("ygg") is None
        create.assert_not_called()


class TestListDelete:
    def test_list_distributions_and_wheels(self):
        svc = _svc()
        folder = MagicMock(); folder.exists.return_value = True
        whl = MagicMock(); whl.name = "ygg-1.0-py3-none-any.whl"
        whl.full_path.return_value = "/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"
        readme = MagicMock(); readme.name = "README.md"
        folder.iterdir.return_value = [whl, readme]
        dbp = MagicMock(); dbp.from_.return_value = folder
        with patch("yggdrasil.databricks.path.DatabricksPath", dbp):
            wheels = svc.list("ygg")
        assert [w.path for w in wheels] == ["/ws/pypi/ygg/ygg-1.0-py3-none-any.whl"]

    def test_delete_removes_matching_version(self):
        svc = _svc()
        w1 = MagicMock(spec=Wheel); w1.version = VersionInfo(1, 0, 0)
        w2 = MagicMock(spec=Wheel); w2.version = VersionInfo(2, 0, 0)
        with patch.object(Wheels, "list", return_value=[w1, w2]):
            removed = svc.delete("ygg", "1.0")
        assert removed == [w1]
        w1.delete.assert_called_once()
        w2.delete.assert_not_called()
