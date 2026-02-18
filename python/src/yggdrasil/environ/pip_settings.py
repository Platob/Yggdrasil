"""Module dependency and pip index inspection utilities."""

# modules.py
from __future__ import annotations

import dataclasses as dc
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Iterable, MutableMapping

__all__ = [
    "PipIndexSettings",
    "get_pip_index_settings",
]


DEFAULT_PIP_INDEX_SETTINGS = None
_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _run_pip(*args: str) -> Tuple[int, str, str]:
    """Run pip with arguments and return (returncode, stdout, stderr).

    Args:
        *args: Pip arguments.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    p = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


@dc.dataclass(frozen=True)
class PipIndexSettings:
    """Resolved pip index configuration from env and config sources."""
    index_url: Optional[str] = None
    extra_index_urls: List[str] = dc.field(default_factory=list)
    sources: Dict[str, Dict[str, Any]] = dc.field(default_factory=dict)  # {"env": {...}, "config": {...}}

    @classmethod
    def current(cls):
        """Return the cached default pip index settings.

        Returns:
            Default PipIndexSettings instance.
        """
        global DEFAULT_PIP_INDEX_SETTINGS

        if DEFAULT_PIP_INDEX_SETTINGS is None:
            DEFAULT_PIP_INDEX_SETTINGS = get_pip_index_settings()

        return DEFAULT_PIP_INDEX_SETTINGS

    @property
    def extra_index_url(self):
        """Return extra index URLs as a space-separated string.

        Returns:
            Space-separated extra index URLs or None.
        """
        if self.extra_index_urls:
            return " ".join(self.extra_index_urls)
        return None

    def setenv(
        self,
        env: MutableMapping[str, str] | None = None,
        *,
        keys: Iterable[str] | None = None
    ):
        keys = keys or ["UV_EXTRA_INDEX_URL", "PIP_EXTRA_INDEX_URL"]

        url = self.extra_index_url

        if env is None:
            env = os.environ

        for key in keys:
            env[key] = url

    def as_dict(self) -> dict:
        """Return a dict representation of the settings.

        Returns:
            Dict representation of settings.
        """
        return dc.asdict(self)


def get_pip_index_settings() -> PipIndexSettings:
    """
    Inspect pip settings:
      - env (PIP_INDEX_URL / PIP_EXTRA_INDEX_URL)
      - pip config (merged view from `pip config list`)

    Precedence:
      env overrides config.
    """
    sources: Dict[str, Dict[str, Any]] = {"env": {}, "config": {}}

    env_index = os.environ.get("PIP_INDEX_URL")
    env_extra = os.environ.get("PIP_EXTRA_INDEX_URL")

    if env_index:
        sources["env"]["PIP_INDEX_URL"] = env_index
    if env_extra:
        sources["env"]["PIP_EXTRA_INDEX_URL"] = env_extra

    env_extra_urls: List[str] = shlex.split(env_extra) if env_extra else []

    # Read pip config (best-effort)
    rc, out, _err = _run_pip("config", "list", "--format=json")
    config_index_url: Optional[str] = None
    config_extra_raw: List[Any] = []

    if rc == 0 and out:
        cfg = json.loads(out)
        for k, v in cfg.items():
            lk = k.lower()
            if lk.endswith("index-url"):
                sources["config"][k] = v
                if lk.endswith("index-url") and not lk.endswith("extra-index-url"):
                    config_index_url = str(v)
                elif lk.endswith("extra-index-url"):
                    if isinstance(v, list):
                        config_extra_raw.extend(v)
                    else:
                        config_extra_raw.append(v)
    else:
        rc2, out2, _ = _run_pip("config", "list")
        if rc2 == 0 and out2:
            for line in out2.splitlines():
                if "=" not in line:
                    continue
                k, v = [x.strip() for x in line.split("=", 1)]
                lk = k.lower()
                if lk.endswith("extra-index-url"):
                    sources["config"][k] = v
                    config_extra_raw.append(v)
                elif lk.endswith("index-url") and not lk.endswith("extra-index-url"):
                    sources["config"][k] = v
                    config_index_url = v

    # Apply precedence
    index_url = env_index or config_index_url

    # extras: if env is set, it replaces config (pip behavior)
    if env_extra_urls:
        candidates = list(env_extra_urls)  # already tokenized; do NOT split again
    else:
        # config entries might contain multiple URLs in a single string => split them
        candidates: List[str] = []
        for item in config_extra_raw:
            if item is None:
                continue
            candidates.extend(shlex.split(str(item)))

    # Dedup preserving order
    seen = set()
    extra_index_urls: List[str] = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            extra_index_urls.append(u)

    return PipIndexSettings(index_url=index_url, extra_index_urls=extra_index_urls, sources=sources)
