from __future__ import annotations

import logging
import os

from ..config import Settings
from ..schemas.env import EnvGetResponse, EnvSetRequest, EnvSetResponse

LOGGER = logging.getLogger(__name__)


class EnvService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_env(self, keys: list[str] | None = None) -> EnvGetResponse:
        if keys:
            variables = {k: os.environ.get(k) for k in keys}
        else:
            variables = dict(os.environ)
        return EnvGetResponse(
            node_id=self.settings.node_id,
            variables=variables,
        )

    async def set_env(self, req: EnvSetRequest) -> EnvSetResponse:
        applied: dict[str, str | None] = {}
        for key, value in req.variables.items():
            if value is None:
                os.environ.pop(key, None)
                LOGGER.info("Unset env var %r", key)
            else:
                os.environ[key] = value
                LOGGER.info("Set env var %r", key)
            applied[key] = value
        return EnvSetResponse(
            node_id=self.settings.node_id,
            applied=applied,
        )
