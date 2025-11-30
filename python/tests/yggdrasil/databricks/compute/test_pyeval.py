import io
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
from databricks.sdk.service.compute import CommandStatus, Language, ResultType, Results, CommandStatusResponse

from yggdrasil.databricks.compute.pyeval import remote_pyeval


class StubCommandExecutionAPI:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.executed = []

    def create_and_wait(self, *, cluster_id: str, language: Language):
        self.created.append((cluster_id, language))
        return SimpleNamespace(id="ctx-1", status=CommandStatus.RUNNING)

    def execute_and_wait(
        self,
        *,
        cluster_id: str,
        command: str,
        context_id: str,
        language: Language,
        timeout=None,
    ):
        self.executed.append((cluster_id, context_id, language, timeout))

        # Simulate the remote command output locally
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exec(command, {})
        encoded_output = buffer.getvalue().strip()

        return CommandStatusResponse(
            id="cmd-1",
            status=CommandStatus.FINISHED,
            results=Results(result_type=ResultType.TEXT, data=encoded_output),
        )

    def destroy(self, *, cluster_id: str, context_id: str):
        self.destroyed.append((cluster_id, context_id))


class StubWorkspaceClient:
    def __init__(self):
        self.command_execution = StubCommandExecutionAPI()


class StubDBXWorkspace:
    def __init__(self):
        self.sdk_calls = 0
        self.client = StubWorkspaceClient()

    def sdk(self):
        self.sdk_calls += 1
        return self.client


def test_remote_pyeval_executes_function_and_returns_value():
    workspace = StubWorkspaceClient()

    result = remote_pyeval(
        "cluster-123",
        lambda a, b: a + b,
        2,
        3,
        workspace=workspace,
    )

    assert result == 5
    assert workspace.command_execution.created[0][0] == "cluster-123"
    assert workspace.command_execution.destroyed == [("cluster-123", "ctx-1")]


def test_remote_pyeval_uses_dbxworkspace_and_bubbles_remote_error():
    workspace = StubDBXWorkspace()

    with pytest.raises(RuntimeError) as excinfo:
        remote_pyeval(
            "cluster-123",
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            workspace=workspace,
        )

    assert "boom" in str(excinfo.value)
    assert workspace.sdk_calls == 1
    assert workspace.client.command_execution.destroyed == [("cluster-123", "ctx-1")]
