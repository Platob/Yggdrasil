"""Backend detection — what can Loki reach from where it's running.

Loki is the global yggdrasil agent. It adapts to wherever it wakes up: a
Databricks notebook/job, a workstation with a configured Databricks
session, a running yggdrasil node, or just a bare shell. :func:`detect`
sniffs the environment **offline** (no network, never raises) and reports
the backends Loki can lean on, so the agent can reason about its own
capabilities before it acts.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Backend", "detect", "detect_databricks", "detect_aws", "detect_local"]


@dataclass
class Backend:
    """A capability surface Loki can drive (e.g. Databricks, a node)."""

    name: str
    available: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.available

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available, "detail": self.detail}


def detect_databricks() -> Backend:
    """Detect a usable Databricks session **without** a network round-trip.

    Looks for the signals that mean "credentials are resolvable": the
    Databricks runtime, ``DATABRICKS_*`` env vars, a ``~/.databrickscfg``,
    or a session remembered by ``ygg databricks configure``. Only when one
    is present does it resolve a client (offline — host/auth from config),
    so it never pollutes the process-global client with a credential-less
    instance.
    """
    home = pathlib.Path.home()
    signals = {
        "runtime": os.getenv("DATABRICKS_RUNTIME_VERSION") is not None,
        "env_host": bool(os.getenv("DATABRICKS_HOST")),
        "env_token": bool(os.getenv("DATABRICKS_TOKEN")),
        "env_oauth": bool(os.getenv("DATABRICKS_CLIENT_ID")),
        "cfg": (home / ".databrickscfg").exists(),
        "session": _has_remembered_session(home),
    }
    if not any(signals.values()):
        return Backend("databricks", available=False, detail={"signals": signals})

    detail: dict[str, Any] = {"signals": signals}
    try:
        from yggdrasil.databricks import DatabricksClient

        client = DatabricksClient.current()
        detail["host"] = client.base_url
        detail["auth_type"] = getattr(client, "auth_type", None)
        detail["catalog"] = client.catalog_name
        detail["schema"] = client.schema_name
        available = bool(detail.get("host"))
    except Exception as exc:  # offline detection must never raise
        detail["error"] = f"{type(exc).__name__}: {exc}"
        available = False
    return Backend("databricks", available=available, detail=detail)


def _has_remembered_session(home: pathlib.Path) -> bool:
    sessions = home / ".config" / "databricks-sdk-py" / "sessions"
    try:
        return sessions.is_dir() and any(sessions.glob("*.json"))
    except OSError:
        return False


def detect_aws() -> Backend:
    """Detect a usable AWS session **without** a network round-trip.

    Looks for the signals that mean "boto3 can resolve credentials": the
    ``AWS_*`` env vars, an ``AWS_PROFILE``, a web-identity token (IRSA), or a
    ``~/.aws`` config/credentials file. Offline only — never calls STS, so it
    never blocks or raises.
    """
    home = pathlib.Path.home()
    signals = {
        "env_key": bool(os.getenv("AWS_ACCESS_KEY_ID")),
        "env_profile": bool(os.getenv("AWS_PROFILE")),
        "env_region": bool(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")),
        "web_identity": bool(os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")),
        "config": (home / ".aws" / "credentials").exists() or (home / ".aws" / "config").exists(),
    }
    if not any(signals.values()):
        return Backend("aws", available=False, detail={"signals": signals})
    return Backend(
        "aws",
        available=True,
        detail={
            "signals": signals,
            "region": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            "profile": os.getenv("AWS_PROFILE"),
        },
    )


def detect_local() -> Backend:
    """The local machine — always available."""
    import getpass
    import socket

    return Backend(
        "local",
        available=True,
        detail={"user": _safe(getpass.getuser), "host": _safe(socket.gethostname)},
    )


def _safe(fn) -> str:
    try:
        return fn()
    except Exception:
        return "unknown"


def detect() -> list[Backend]:
    """Every backend Loki can see from here, in priority order."""
    return [detect_databricks(), detect_aws(), detect_local()]
