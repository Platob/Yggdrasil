from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from yggdrasil.databricks.client import DatabricksService

from .resources import GenieAnswer, GenieSpace

__all__ = ["Genie"]


@dataclass(frozen=True)
class Genie(DatabricksService):
    """High-level wrapper around Databricks Workspace Genie APIs."""

    default_space_id: str | None = None

    @property
    def api(self):
        return self.client.workspace_client().genie

    def ask(
        self,
        content: str,
        *,
        space_id: str | None = None,
        conversation_id: str | None = None,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> GenieAnswer:
        resolved_space_id = space_id or self.default_space_id
        if not resolved_space_id:
            raise ValueError("space_id is required unless default_space_id is set")

        if conversation_id:
            message = self._create_message(
                space_id=resolved_space_id,
                conversation_id=conversation_id,
                content=content,
            )
        else:
            conversation, message = self._start_conversation(
                space_id=resolved_space_id,
                content=content,
            )
            conversation_id = conversation.get("id")
            if not conversation_id:
                raise ValueError("Unable to infer conversation_id from start-conversation response")

        message = self._wait_for_message(
            space_id=resolved_space_id,
            conversation_id=conversation_id,
            message=message,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

        return self._to_answer(
            space_id=resolved_space_id,
            conversation_id=conversation_id,
            message=message,
        )

    def start_conversation(self, content: str, *, space_id: str | None = None) -> GenieSpace:
        resolved_space_id = space_id or self.default_space_id
        if not resolved_space_id:
            raise ValueError("space_id is required unless default_space_id is set")

        conversation, _ = self._start_conversation(space_id=resolved_space_id, content=content)

        conversation_id = conversation.get("id")
        if not conversation_id:
            raise ValueError("Unable to infer conversation_id from start-conversation response")

        return GenieSpace(space_id=resolved_space_id, conversation_id=conversation_id)

    def create_message(
        self,
        content: str,
        *,
        space_id: str,
        conversation_id: str,
        timeout_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> GenieAnswer:
        message = self._create_message(space_id=space_id, conversation_id=conversation_id, content=content)
        message = self._wait_for_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message=message,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return self._to_answer(space_id=space_id, conversation_id=conversation_id, message=message)

    def get_message(self, *, space_id: str, conversation_id: str, message_id: str) -> GenieAnswer:
        message = self._as_mapping(
            self.api.get_message(
                space_id=space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        )
        return self._to_answer(space_id=space_id, conversation_id=conversation_id, message=message)

    def list_spaces(self, **kwargs):
        return self.api.list_spaces(**kwargs)

    def create_space(
        self,
        *,
        warehouse_id: str,
        serialized_space: str,
        description: str | None = None,
        parent_path: str | None = None,
        title: str | None = None,
    ):
        return self.api.create_space(
            warehouse_id=warehouse_id,
            serialized_space=serialized_space,
            description=description,
            parent_path=parent_path,
            title=title,
        )

    def get_space(self, space_id: str, **kwargs):
        return self.api.get_space(space_id=space_id, **kwargs)

    def update_space(
        self,
        *,
        space_id: str,
        title: str | None = None,
        description: str | None = None,
        serialized_space: str | None = None,
        warehouse_id: str | None = None,
    ):
        return self.api.update_space(
            space_id=space_id,
            title=title,
            description=description,
            serialized_space=serialized_space,
            warehouse_id=warehouse_id,
        )

    def delete_space(self, space_id: str):
        trash = getattr(self.api, "trash_space", None)
        if trash is not None:
            return trash(space_id=space_id)
        return self.api.delete_space(space_id=space_id)

    def trash_space(self, space_id: str):
        return self.api.trash_space(space_id=space_id)

    def list_conversations(self, space_id: str, **kwargs):
        return self.api.list_conversations(space_id=space_id, **kwargs)

    def delete_conversation(self, *, space_id: str, conversation_id: str):
        return self.api.delete_conversation(space_id=space_id, conversation_id=conversation_id)

    def list_conversation_messages(self, *, space_id: str, conversation_id: str, **kwargs):
        return self.api.list_conversation_messages(space_id=space_id, conversation_id=conversation_id, **kwargs)

    def delete_conversation_message(self, *, space_id: str, conversation_id: str, message_id: str):
        return self.api.delete_conversation_message(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )

    def send_message_feedback(self, *, space_id: str, conversation_id: str, message_id: str, rating: Any):
        return self.api.send_message_feedback(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            rating=rating,
        )

    def execute_message_query(self, *, space_id: str, conversation_id: str, message_id: str):
        return self.api.execute_message_query(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )

    def execute_message_attachment_query(
        self,
        *,
        space_id: str,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
    ):
        return self.api.execute_message_attachment_query(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id,
        )

    def get_message_query_result(self, *, space_id: str, conversation_id: str, message_id: str):
        return self.api.get_message_query_result(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
        )

    def get_message_query_result_by_attachment(
        self,
        *,
        space_id: str,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
    ):
        return self.api.get_message_query_result_by_attachment(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id,
        )

    def generate_download_full_query_result(
        self,
        *,
        space_id: str,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
    ):
        return self.api.generate_download_full_query_result(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id,
        )

    def get_download_full_query_result(
        self,
        *,
        space_id: str,
        conversation_id: str,
        message_id: str,
        attachment_id: str,
        download_id: str,
        download_id_signature: str,
    ):
        return self.api.get_download_full_query_result(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message_id,
            attachment_id=attachment_id,
            download_id=download_id,
            download_id_signature=download_id_signature,
        )

    def create_eval_run(self, *, space_id: str, benchmark_question_ids: Optional[list[str]] = None):
        return self.api.genie_create_eval_run(
            space_id=space_id,
            benchmark_question_ids=benchmark_question_ids,
        )

    def get_eval_run(self, *, space_id: str, eval_run_id: str):
        return self.api.genie_get_eval_run(space_id=space_id, eval_run_id=eval_run_id)

    def list_eval_runs(self, *, space_id: str, page_size: int | None = None, page_token: str | None = None):
        return self.api.genie_list_eval_runs(
            space_id=space_id,
            page_size=page_size,
            page_token=page_token,
        )

    def list_eval_results(
        self,
        *,
        space_id: str,
        eval_run_id: str,
        page_size: int | None = None,
        page_token: str | None = None,
    ):
        return self.api.genie_list_eval_results(
            space_id=space_id,
            eval_run_id=eval_run_id,
            page_size=page_size,
            page_token=page_token,
        )

    def get_eval_result_details(self, *, space_id: str, eval_run_id: str, result_id: str):
        return self.api.genie_get_eval_result_details(
            space_id=space_id,
            eval_run_id=eval_run_id,
            result_id=result_id,
        )

    def _start_conversation(self, *, space_id: str, content: str):
        response = self._call_first(
            ("start_conversation_and_wait", "start_conversation"),
            space_id=space_id,
            content=content,
        )
        response_obj = self._wait_if_needed(response)
        response_map = self._as_mapping(response_obj)

        if "conversation" in response_map and "message" in response_map:
            conversation = self._as_mapping(response_map.get("conversation"))
            message = self._as_mapping(response_map.get("message"))
            return conversation, message

        message = response_map
        conversation = {
            "id": response_map.get("conversation_id"),
            "space_id": response_map.get("space_id", space_id),
        }
        return conversation, message

    def _create_message(self, *, space_id: str, conversation_id: str, content: str):
        response = self._call_first(
            ("create_message_and_wait", "create_message"),
            space_id=space_id,
            conversation_id=conversation_id,
            content=content,
        )
        return self._as_mapping(self._wait_if_needed(response))

    def _call_first(self, names: tuple[str, ...], **kwargs):
        missing = []
        for name in names:
            method = getattr(self.api, name, None)
            if method is None:
                missing.append(name)
                continue
            return method(**kwargs)

        raise AttributeError(f"Genie API method not found. Tried: {', '.join(missing)}")

    def _wait_if_needed(self, value: Any):
        if hasattr(value, "result") and callable(value.result):
            return value.result()
        return value

    def _wait_for_message(
        self,
        *,
        space_id: str,
        conversation_id: str,
        message: Mapping[str, Any],
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> Mapping[str, Any]:
        message_id = message.get("id")
        if not message_id:
            return message

        deadline = time.monotonic() + timeout_seconds
        current = dict(message)

        while True:
            status = (current.get("status") or "").upper()
            if status in {"COMPLETED", "FAILED", "ERROR", "CANCELED"}:
                return current

            if time.monotonic() >= deadline:
                return current

            time.sleep(max(poll_interval_seconds, 0.0))
            current = self._as_mapping(
                self.api.get_message(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
            )

    def _to_answer(self, *, space_id: str, conversation_id: str, message: Mapping[str, Any]) -> GenieAnswer:
        attachments = message.get("attachments") or []
        first_attachment = attachments[0] if attachments else None
        attachment_map = self._as_mapping(first_attachment)

        return GenieAnswer(
            space_id=space_id,
            conversation_id=conversation_id,
            message_id=message.get("id") or "",
            status=message.get("status"),
            content=message.get("content"),
            text=attachment_map.get("text"),
            attachment_id=attachment_map.get("attachment_id"),
            query=attachment_map.get("query"),
            query_result=message.get("query_result"),
            raw_message=dict(message),
        )

    @staticmethod
    def _as_mapping(obj: Any) -> Mapping[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, Mapping):
            return obj

        as_dict = getattr(obj, "as_dict", None)
        if callable(as_dict):
            return as_dict()

        data: dict[str, Any] = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            value = getattr(obj, key, None)
            if callable(value):
                continue
            data[key] = value
        return data
