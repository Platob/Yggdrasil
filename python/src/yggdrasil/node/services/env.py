from __future__ import annotations

import logging
import os

from ..config import Settings
from ..schemas.env import EnvGetResponse, EnvSetRequest, EnvSetResponse

LOGGER = logging.getLogger(__name__)

# Well-known OS / shell / runtime plumbing variables, dropped from a
# full environment listing so callers see the application-meaningful
# vars (what a user actually set) rather than the platform's furniture.
# Explicit names cover POSIX + Windows; prefixes catch the families
# (locale, XDG, SSH, Windows PROCESSOR_*/PROGRAM*, …).
_SYSTEM_ENV_NAMES = frozenset({
    # POSIX / shell
    "PATH", "HOME", "PWD", "OLDPWD", "SHELL", "SHLVL", "TERM", "USER",
    "LOGNAME", "LANG", "LANGUAGE", "HOSTNAME", "HOSTTYPE", "MACHTYPE",
    "TMPDIR", "TMP", "TEMP", "MAIL", "EDITOR", "VISUAL", "PAGER",
    "DISPLAY", "LS_COLORS", "PS1", "PS2", "PROMPT_COMMAND", "_", "COLORTERM",
    "MANPATH", "INFOPATH", "SSH_AUTH_SOCK", "SSH_AGENT_PID",
    # Python / runtime
    "PYTHONPATH", "PYTHONHOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
    # Windows
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "OS",
    "APPDATA", "LOCALAPPDATA", "USERPROFILE", "USERNAME", "USERDOMAIN",
    "COMPUTERNAME", "HOMEDRIVE", "HOMEPATH", "ALLUSERSPROFILE", "PUBLIC",
    "SESSIONNAME", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
    "DRIVERDATA", "COMMONPROGRAMFILES", "PROGRAMDATA",
})
_SYSTEM_ENV_PREFIXES = (
    "LC_", "XDG_", "SSH_", "DBUS_", "GDM", "GNOME", "KDE", "DESKTOP_",
    "PROCESSOR_", "PROGRAMFILES", "COMMONPROGRAM", "PROGRAMW",
)


def _is_system_env(name: str) -> bool:
    upper = name.upper()
    if upper in _SYSTEM_ENV_NAMES:
        return True
    return any(upper.startswith(p) for p in _SYSTEM_ENV_PREFIXES)


class EnvService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def get_env(
        self,
        keys: list[str] | None = None,
        *,
        include_system: bool = False,
    ) -> EnvGetResponse:
        if keys:
            # Explicit keys are returned verbatim — the caller asked for
            # exactly these, system or not.
            variables = {k: os.environ.get(k) for k in keys}
        else:
            # Whole-environment listing drops the OS/shell/runtime
            # plumbing unless explicitly asked to keep it.
            variables = {
                k: v for k, v in os.environ.items()
                if include_system or not _is_system_env(k)
            }
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
