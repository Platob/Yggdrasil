from yggdrasil.databricks.ai.genie import Genie, GenieAnswer, GenieSpace


class DummyGenieAPI:
    def __init__(self):
        self._counter = 0

    def start_conversation_and_wait(self, space_id: str, content: str):
        return {
            "conversation_id": "conv-1",
            "space_id": space_id,
            "id": "msg-1",
            "status": "COMPLETED",
            "content": content,
            "attachments": [{"text": "hello from genie", "attachment_id": "att-1", "query": "SELECT 1"}],
        }

    def create_message_and_wait(self, space_id: str, conversation_id: str, content: str):
        return {
            "id": "msg-2",
            "status": "COMPLETED",
            "content": content,
            "attachments": [{"text": "follow up", "attachment_id": "att-2", "query": "SELECT 2"}],
        }

    def get_message(self, space_id: str, conversation_id: str, message_id: str):
        self._counter += 1
        return {
            "id": message_id,
            "status": "COMPLETED",
            "content": "done",
            "attachments": [{"text": "done", "attachment_id": "att-3", "query": "SELECT 3"}],
        }

    def create_space(self, **kwargs):
        return {"id": "space-1", **kwargs}

    def get_space(self, space_id: str, **kwargs):
        return {"id": space_id, **kwargs}

    def update_space(self, **kwargs):
        return {"updated": True, **kwargs}

    def trash_space(self, space_id: str):
        return {"trashed": space_id}


class DummyWorkspaceClient:
    def __init__(self):
        self.genie = DummyGenieAPI()


class DummyDatabricksClient:
    def workspace_client(self):
        return DummyWorkspaceClient()


def test_start_conversation_returns_ids():
    service = Genie(client=DummyDatabricksClient(), default_space_id="space-1")

    conversation = service.start_conversation("hello")

    assert isinstance(conversation, GenieSpace)
    assert conversation.space_id == "space-1"
    assert conversation.conversation_id == "conv-1"


def test_ask_starts_new_conversation_and_returns_answer():
    service = Genie(client=DummyDatabricksClient(), default_space_id="space-1")

    answer = service.ask("hello")

    assert isinstance(answer, GenieAnswer)
    assert answer.space_id == "space-1"
    assert answer.conversation_id == "conv-1"
    assert answer.message_id == "msg-1"
    assert answer.text == "hello from genie"
    assert answer.query == "SELECT 1"


def test_ask_follow_up_uses_existing_conversation():
    service = Genie(client=DummyDatabricksClient(), default_space_id="space-1")

    answer = service.ask("next", conversation_id="conv-abc")

    assert answer.conversation_id == "conv-abc"
    assert answer.message_id == "msg-2"
    assert answer.text == "follow up"


def test_space_crud_calls():
    service = Genie(client=DummyDatabricksClient(), default_space_id="space-1")

    created = service.create_space(warehouse_id="wh-1", serialized_space="{}")
    fetched = service.get_space("space-1")
    updated = service.update_space(space_id="space-1", title="New")
    deleted = service.delete_space("space-1")

    assert created["id"] == "space-1"
    assert fetched["id"] == "space-1"
    assert updated["updated"] is True
    assert deleted["trashed"] == "space-1"
