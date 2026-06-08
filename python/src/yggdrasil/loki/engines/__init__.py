"""Concrete :class:`~yggdrasil.loki.engine.TokenEngine` backends.

Remote (hosted APIs — fast, capable, metered):

- :class:`OpenAIEngine` — the OpenAI API.
- :class:`ClaudeEngine` — the Anthropic (Claude) API.
- :class:`DatabricksServingEngine` — a Databricks model-serving endpoint.

Local (run on this workstation — free, private, resource-bound):

- :class:`TransformersEngine` — an open HuggingFace model via ``transformers``
  (CPU, or an Intel GPU through the XPU torch build).
- :class:`OpenVINOEngine` — a model on the **Intel NPU** (AI Boost) via
  OpenVINO / optimum-intel, falling back to the Intel GPU then CPU.
- :class:`OllamaEngine` — a model served by a local Ollama server.
"""
from .claude_engine import ClaudeEngine
from .databricks_engine import DatabricksServingEngine
from .ollama_engine import OllamaEngine
from .openai_engine import OpenAIEngine
from .openvino_engine import OpenVINOEngine
from .transformers_engine import TransformersEngine

__all__ = [
    "OpenAIEngine",
    "ClaudeEngine",
    "DatabricksServingEngine",
    "TransformersEngine",
    "OpenVINOEngine",
    "OllamaEngine",
]
