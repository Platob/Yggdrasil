"""Unit tests for project-name aliasing in
:func:`yggdrasil.databricks.wheels.service.distribution_for` — the explicit
:data:`PROJECT_ALIASES` map (``yggdrasil`` → ``ygg``) and its
``YGG_DATABRICKS_PROJECT_ALIASES`` env override, plus the downstream effect on
environment folders/stems so a deploy named for either lands in one place.
"""
from __future__ import annotations

import pytest

from yggdrasil.databricks.environments.service import (
    environment_folder,
    environment_stem,
)
from yggdrasil.databricks.wheels.service import PROJECT_ALIASES, distribution_for


class TestProjectAliases:
    def test_yggdrasil_aliases_to_ygg(self) -> None:
        assert PROJECT_ALIASES["yggdrasil"] == "ygg"
        assert distribution_for("yggdrasil") == "ygg"

    def test_alias_is_case_insensitive(self) -> None:
        assert distribution_for("Yggdrasil") == "ygg"
        assert distribution_for("YGGDRASIL") == "ygg"

    def test_canonical_name_passes_through(self) -> None:
        assert distribution_for("ygg") == "ygg"

    def test_env_override_extends_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YGG_DATABRICKS_PROJECT_ALIASES", "foo=bar, baz = qux")
        assert distribution_for("foo") == "bar"
        assert distribution_for("baz") == "qux"
        # The built-in alias still applies alongside env-provided ones.
        assert distribution_for("yggdrasil") == "ygg"


class TestAliasFlowsToEnvironmentNaming:
    def test_folder_and_stem_collapse_to_canonical(self) -> None:
        # Deploying a project named "yggdrasil" must land in the "ygg" folder.
        assert environment_folder("yggdrasil") == "ygg"
        assert environment_folder("ygg") == "ygg"
        stem = environment_stem("yggdrasil", python="3.11", version="0.8.58")
        assert stem == "ygg-0.8.58-py311"
