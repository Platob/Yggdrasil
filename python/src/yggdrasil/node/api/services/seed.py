"""Default assets seeded on node startup so a fresh node is useful at once.

A brand-new node has no PyEnv and no PyFunc — the dashboard shows empty
panels and there is nothing to run. This seeds a ``default`` PyEnv plus a
handful of starter PyFuncs the moment the server boots.

Design notes:
- **Idempotent.** ``PyEnvService.create`` / ``PyFuncService.create`` upsert
  by name, so re-running on every boot is safe (the registry is in-memory
  and resets per process).
- **Non-blocking.** The env build shells out to ``uv venv`` + ``pip
  install``, which can take a while — it runs in a background task so node
  readiness is never gated on it. The functions are registered instantly.
- **Graceful before the env is ready.** Seeded functions are bound to the
  default env, but ``PyFuncRunService`` falls back to the node's own
  interpreter while the env is still ``creating``, so they run immediately.

The starter functions lean on the standard library + ``psutil`` (a
yggdrasil dependency, present in the node interpreter), so they work even
with no third-party install.
"""
from __future__ import annotations

import logging
import sys

from ..schemas.pyenv import PyEnvCreate
from ..schemas.pyfunc import PyFuncCreate

LOGGER = logging.getLogger(__name__)

#: Default environment — broadly useful data/HTTP stack. ``pandas`` /
#: ``pyarrow`` arrive transitively with the base yggdrasil install; ``requests``
#: is the one explicit extra. Built in the background on first boot.
DEFAULT_ENV_NAME = "default"
DEFAULT_ENV_DEPS = ["pandas", "pyarrow", "requests"]

# ── Starter functions ───────────────────────────────────────────────────────
# Each ``code`` runs as a standalone script: the run harness injects ``_args``
# / ``_kwargs`` globals and exposes ``__ygg_outputs_file__`` for the structured
# result (read back as the run's ``result``). Keep them dependency-light.

_HELLO_CODE = '''\
import os, json, platform

name = _kwargs.get("name") if isinstance(_kwargs, dict) else None
if not name and _args:
    name = _args[0]
name = name or "world"

node_id = os.environ.get("YGG_NODE_ID", "?")
message = f"Hello, {name}! You are talking to Yggdrasil node {node_id}."
print(message)

result = {
    "message": message,
    "node_id": node_id,
    "python": platform.python_version(),
}
_out = os.environ.get("__ygg_outputs_file__")
if _out:
    with open(_out, "w", encoding="utf-8") as f:
        json.dump({"result": result}, f)
'''

_NODE_INFO_CODE = '''\
import os, json, platform, shutil

info = {
    "node_id": os.environ.get("YGG_NODE_ID", "?"),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "cpu_percent": None,
    "memory": {},
}
try:
    import psutil
    info["cpu_percent"] = psutil.cpu_percent(interval=0.3)
    vm = psutil.virtual_memory()
    info["memory"] = {
        "total_mb": round(vm.total / 1_000_000),
        "used_mb": round(vm.used / 1_000_000),
        "percent": vm.percent,
    }
    info["cpu_count"] = psutil.cpu_count()
except Exception as exc:  # psutil missing or unavailable — report and continue
    info["cpu_error"] = str(exc)

try:
    du = shutil.disk_usage(os.path.abspath(os.sep))
    info["disk"] = {
        "total_gb": round(du.total / 1_000_000_000, 1),
        "used_gb": round(du.used / 1_000_000_000, 1),
        "free_gb": round(du.free / 1_000_000_000, 1),
    }
except Exception:
    pass

print(json.dumps(info, indent=2))
_out = os.environ.get("__ygg_outputs_file__")
if _out:
    with open(_out, "w", encoding="utf-8") as f:
        json.dump({"result": info}, f)
'''

_ECHO_CODE = '''\
import os, json

payload = {"args": list(_args), "kwargs": dict(_kwargs)}
print(json.dumps(payload))

_out = os.environ.get("__ygg_outputs_file__")
if _out:
    with open(_out, "w", encoding="utf-8") as f:
        json.dump({"result": payload}, f)
'''

DEFAULT_FUNCS = [
    {
        "name": "hello",
        "description": "Smoke-test: greet a name and report node identity.",
        "code": _HELLO_CODE,
    },
    {
        "name": "node_info",
        "description": "Report this node's platform, Python, CPU, memory and disk.",
        "code": _NODE_INFO_CODE,
    },
    {
        "name": "echo",
        "description": "Return the args/kwargs it was called with — wiring check.",
        "code": _ECHO_CODE,
    },
]


async def seed_defaults(pyenv_service, pyfunc_service) -> None:
    """Register the default PyFuncs (instant) and build the default PyEnv.

    Idempotent and resilient: any single failure is logged and skipped so a
    bad seed never crashes startup. Intended to be launched as a background
    task from the app's startup event.
    """
    # Match the env's Python to the interpreter the node already runs on, so
    # ``uv`` reuses it instead of downloading another version.
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # 1. Functions first — code-only, near-instant, bound to the default env.
    env_id: int | None = None
    try:
        env_resp = await pyenv_service.create(
            PyEnvCreate(
                name=DEFAULT_ENV_NAME,
                python_version=python_version,
                dependencies=list(DEFAULT_ENV_DEPS),
            )
        )
        env_id = env_resp.env.id
        LOGGER.info(
            "Seeded default PyEnv %r (id=%d, status=%s)",
            DEFAULT_ENV_NAME, env_id, env_resp.env.status,
        )
    except Exception:
        # Env build failed (no uv, offline, etc.) — functions still run on the
        # node interpreter, so press on without an env binding.
        LOGGER.warning("Default PyEnv seed failed; functions will use the node interpreter", exc_info=True)

    seeded = 0
    for spec in DEFAULT_FUNCS:
        try:
            await pyfunc_service.create(
                PyFuncCreate(
                    name=spec["name"],
                    code=spec["code"],
                    description=spec["description"],
                    python_version=python_version,
                    env_id=env_id,
                )
            )
            seeded += 1
        except Exception:
            LOGGER.warning("Failed to seed default function %r", spec["name"], exc_info=True)
    LOGGER.info("Seeded %d/%d default functions", seeded, len(DEFAULT_FUNCS))
