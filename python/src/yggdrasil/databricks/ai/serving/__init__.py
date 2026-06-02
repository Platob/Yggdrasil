"""Databricks model-serving service wrappers (Foundation Models, external + custom)."""

from .resources import (
    DEFAULT_SERVING_ENDPOINT,
    ChatResult,
    MessageLike,
    ServingDefaults,
    ServingEndpoint,
)
from .service import ModelServing

__all__ = [
    "DEFAULT_SERVING_ENDPOINT",
    "ChatResult",
    "MessageLike",
    "ModelServing",
    "ServingDefaults",
    "ServingEndpoint",
]
