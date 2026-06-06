"""Concrete :class:`~yggdrasil.loki.engine.TokenEngine` backends.

Remote (hosted APIs — fast, capable, metered):

- :class:`OpenAIEngine` — the OpenAI API.
- :class:`ClaudeEngine` — the Anthropic (Claude) API.
- :class:`DatabricksServingEngine` — a Databricks model-serving endpoint.

Local (run on this workstation — free, private, resource-bound):

- :class:`TransformersEngine` — an open HuggingFace model via ``transformers``.
- :class:`OllamaEngine` — a model served by a local Ollama server.
"""
from .claude_engine import ClaudeEngine
from .databricks_engine import DatabricksServingEngine
from .ollama_engine import OllamaEngine
from .openai_engine import OpenAIEngine
from .transformers_engine import TransformersEngine

__all__ = [
    "OpenAIEngine",
    "ClaudeEngine",
    "DatabricksServingEngine",
    "TransformersEngine",
    "OllamaEngine",
]
