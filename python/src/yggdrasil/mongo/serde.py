from __future__ import annotations

from typing import Any

from bson import json_util


class BsonJsonSerde:
    """Serialize BSON-capable Python objects through MongoDB Extended JSON."""

    @staticmethod
    def dumps(value: Any) -> str:
        return json_util.dumps(value)

    @staticmethod
    def loads(payload: str) -> Any:
        return json_util.loads(payload)

    @staticmethod
    def normalize(value: Any) -> Any:
        return json_util.loads(json_util.dumps(value))
