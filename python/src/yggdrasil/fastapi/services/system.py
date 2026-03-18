from __future__ import annotations

import sys

from ..config import Settings
from ..schemas.system import HealthResponse, SystemInfoResponse


class SystemService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def info(self) -> SystemInfoResponse:
        api = self.settings.api_prefix
        sys_prefix = self.settings.system_prefix
        py_prefix = self.settings.python_prefix
        xl_prefix = self.settings.excel_prefix

        return SystemInfoResponse(
            name=self.settings.app_name,
            version=self.settings.app_version,
            docs=self.settings.docs_url,
            openapi=self.settings.openapi_url,
            routes={
                "system_info": f"{api}{sys_prefix}/info",
                "health": f"{api}{sys_prefix}/healthz",
                "current_env": f"{api}{py_prefix}/envs/current",
                "list_envs": f"{api}{py_prefix}/envs",
                "resolve_env": f"{api}{py_prefix}/envs/resolve",
                "create_env": f"{api}{py_prefix}/envs",
                "delete_env": f"{api}{py_prefix}/envs",
                "requirements": f"{api}{py_prefix}/requirements",
                "install": f"{api}{py_prefix}/packages/install",
                "update": f"{api}{py_prefix}/packages/update",
                "uninstall": f"{api}{py_prefix}/packages/uninstall",
                "execute": f"{api}{py_prefix}/execute",
                "excel_execute": f"{api}{py_prefix}{xl_prefix}/execute",
                "excel_prepare": f"{api}{py_prefix}{xl_prefix}/prepare",
            },
        )

    async def health(self) -> HealthResponse:
        return HealthResponse(
            ok=True,
            env_home=str(self.settings.env_home),
            python_executable=sys.executable,
        )
