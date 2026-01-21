from typing import Optional
from dataclasses import field, dataclass

from ..workspaces.workspace import WorkspaceService

try:
    from openai import OpenAI

    def make_openai_client(
        api_key: str,
        base_url: str
    ):
        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )
except ImportError:
    class OpenAI:
        pass

    def make_openai_client(
        api_key: str,
        base_url: str
    ):
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=base_url
        )

__all__ = [
    "Loki"
]


@dataclass
class Loki(WorkspaceService):
    model: str = "databricks-gemini-2-5-flash"

    _openai_client: Optional[OpenAI] = field(repr=False, hash=False, default=None)

    @property
    def openai_client(self):
        if self._openai_client is None:
            self._openai_client = self.make_openai_client()
        return self._openai_client

    def make_openai_client(self):
        return make_openai_client(
            api_key=self.workspace.current_token(),
            base_url=self.workspace.host + "/serving-endpoints"
        )
