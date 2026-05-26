from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from yggdrasil.node.app import create_app
from yggdrasil.node.config import Settings


@pytest.fixture
def tmp_home(tmp_path):
    return tmp_path / "ygg_home"


@pytest.fixture
def settings(tmp_home):
    return Settings(
        node_home=tmp_home,
        node_id="test-node-integration",
        max_python_timeout=30.0,
        max_cmd_timeout=10.0,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)
