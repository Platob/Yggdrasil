import pytest

from yggdrasil.databricks.compute.pyeval import remote_pyeval
from yggdrasil.databricks.workspaces import DBXWorkspace


def test_remote_pyeval_executes_function_and_returns_value():
    workspace = DBXWorkspace(
        host=""
    )

    result = remote_pyeval(
        "",
        lambda a, b: a + b,
        2,
        3,
        workspace=workspace,
    )

    assert result == 5
