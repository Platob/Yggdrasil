"""Unit tests for the Databricks Genie service (mocked SDK GenieAPI)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from yggdrasil.databricks.genie import Genie, GenieAnswer, GenieSpace


def _client_with_api():
    """A mock DatabricksClient whose workspace_client().genie is controllable."""
    client = MagicMock()
    api = client.workspace_client.return_value.genie
    return client, api


def _space_info(space_id="01ef", title="Sales", warehouse_id="wh1"):
    info = MagicMock()
    info.space_id = space_id
    info.title = title
    info.warehouse_id = warehouse_id
    info.description = "desc"
    info.etag = "etag-1"
    return info


def _message(*, text=None, query=None, statement_id=None, conversation_id="c1",
             message_id="m1", status="COMPLETED"):
    msg = MagicMock()
    msg.conversation_id = conversation_id
    msg.message_id = message_id
    msg.id = message_id
    msg.status = MagicMock(value=status)
    msg.error = None
    attachments = []
    if text is not None:
        a = MagicMock(); a.text = MagicMock(content=text); a.query = None
        attachments.append(a)
    if query is not None or statement_id is not None:
        a = MagicMock(); a.text = None
        a.query = MagicMock(query=query, statement_id=statement_id)
        attachments.append(a)
    msg.attachments = attachments
    return msg


class TestGenieSpaces(unittest.TestCase):
    def test_list_spaces_pages(self):
        client, api = _client_with_api()
        api.list_spaces.side_effect = [
            MagicMock(spaces=[_space_info("s1", "A")], next_page_token="tok"),
            MagicMock(spaces=[_space_info("s2", "B")], next_page_token=None),
        ]
        spaces = Genie(client=client).spaces()
        self.assertEqual([s.space_id for s in spaces], ["s1", "s2"])
        self.assertEqual(api.list_spaces.call_count, 2)

    def test_find_by_title(self):
        client, api = _client_with_api()
        api.list_spaces.return_value = MagicMock(
            spaces=[_space_info("s1", "A"), _space_info("s2", "Sales")],
            next_page_token=None,
        )
        found = Genie(client=client).find("Sales")
        self.assertIsInstance(found, GenieSpace)
        self.assertEqual(found.space_id, "s2")
        self.assertIsNone(Genie(client=client).find("Nope"))

    def test_getitem_returns_lazy_space(self):
        client, _ = _client_with_api()
        space = Genie(client=client)["01ef"]
        self.assertIsInstance(space, GenieSpace)
        self.assertEqual(space.space_id, "01ef")

    def test_create_space(self):
        client, api = _client_with_api()
        api.create_space.return_value = _space_info("new", "Built")
        space = Genie(client=client).create_space(
            "wh1", "{...}", title="Built", description="d",
        )
        api.create_space.assert_called_once_with(
            "wh1", "{...}", title="Built", description="d", parent_path=None,
        )
        self.assertEqual(space.space_id, "new")

    def test_update_and_trash(self):
        client, api = _client_with_api()
        api.get_space.return_value = _space_info("s1")
        api.update_space.return_value = _space_info("s1", title="New")
        space = Genie(client=client).space("s1")
        space.refresh()                       # populates etag
        space.update(title="New")
        self.assertEqual(api.update_space.call_args.kwargs["etag"], "etag-1")
        space.trash()
        api.trash_space.assert_called_once_with("s1")


class TestGenieAsk(unittest.TestCase):
    def test_ask_returns_text_answer(self):
        client, api = _client_with_api()
        api.get_space.return_value = _space_info("s1", warehouse_id="wh1")
        api.start_conversation_and_wait.return_value = _message(text="Here you go")
        answer = Genie(client=client).ask("s1", "hi")
        self.assertIsInstance(answer, GenieAnswer)
        self.assertEqual(answer.text, "Here you go")
        self.assertIsNone(answer.query)
        api.start_conversation_and_wait.assert_called_once()

    def test_answer_to_polars_reattaches_statement(self):
        client, api = _client_with_api()
        api.get_space.return_value = _space_info("s1", warehouse_id="wh9")
        api.start_conversation_and_wait.return_value = _message(
            text="Top rows", query="SELECT 1", statement_id="stmt-42",
        )
        # The answer re-attaches to the generated statement through the SQL
        # engine, on the space's warehouse — not by re-running the query.
        result = MagicMock()
        result.to_polars.return_value = "polars-frame"
        client.sql.statement_result.return_value = result

        answer = Genie(client=client).ask("s1", "top rows")
        self.assertEqual(answer.query, "SELECT 1")
        self.assertEqual(answer.statement_id, "stmt-42")
        self.assertEqual(answer.to_polars(), "polars-frame")
        client.sql.statement_result.assert_called_once_with(
            "stmt-42", warehouse_id="wh9",
        )

    def test_answer_without_query_raises_on_result(self):
        client, api = _client_with_api()
        api.get_space.return_value = _space_info("s1")
        api.start_conversation_and_wait.return_value = _message(text="just text")
        answer = Genie(client=client).ask("s1", "hi")
        with self.assertRaises(ValueError):
            answer.result()

    def test_follow_up_uses_create_message(self):
        client, api = _client_with_api()
        api.get_space.return_value = _space_info("s1")
        api.create_message_and_wait.return_value = _message(text="more")
        space = Genie(client=client).space("s1")
        ans = space.follow_up("conv-1", "and then?")
        self.assertEqual(ans.text, "more")
        args = api.create_message_and_wait.call_args
        self.assertEqual(args.args[:3], ("s1", "conv-1", "and then?"))


class TestClientProperty(unittest.TestCase):
    def test_dbc_genie_is_cached_service(self):
        from yggdrasil.databricks import DatabricksClient

        dbc = DatabricksClient(host="https://x.cloud.databricks.com", token="dapi-x")
        self.assertIsInstance(dbc.genie, Genie)
        self.assertIs(dbc.genie, dbc.genie)   # cached
        self.assertIs(dbc.genie.client, dbc)


if __name__ == "__main__":
    unittest.main()
