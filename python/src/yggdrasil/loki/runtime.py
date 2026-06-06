"""Auto-installing optional-dependency loader for Loki.

Loki reaches for heavy optional packages — an engine SDK (``anthropic`` /
``openai``), a local-model runtime (``transformers`` + ``torch``), a headless
browser (``playwright``) — only when a feature actually needs one. Rather than
fail with an ``ImportError`` three frames deep, it **installs the missing
package into the running interpreter on first use** and continues.

The install is routed through the project's own
:meth:`~yggdrasil.environ.PyEnv.runtime_import_module` (via
:func:`yggdrasil.lazy_imports._lazy_import` with ``install=True``), which
anchors on ``sys.executable`` so the package lands in the same site-packages
the process already reads from. Set ``YGG_LOKI_AUTO_INSTALL=0`` to turn this
off — then a missing package raises the normal ``ImportError``.
"""
from __future__ import annotations

import os
from typing import Any

__all__ = ["load", "auto_install_enabled"]


def auto_install_enabled() -> bool:
    """True unless ``YGG_LOKI_AUTO_INSTALL`` is set to a falsey value."""
    return os.getenv("YGG_LOKI_AUTO_INSTALL", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def load(module_name: str, pip_name: str | None = None, *, install: bool | None = None) -> Any:
    """Import *module_name*, installing *pip_name* on miss when allowed.

    ``install`` overrides the global :func:`auto_install_enabled` setting for
    one call (``True`` forces a runtime install, ``False`` forbids it). Returns
    the live module — identical to ``import module_name``. Imports through
    :func:`importlib.import_module` (so it honors ``sys.modules``), and only on
    a real miss hands off to :meth:`~yggdrasil.environ.PyEnv.runtime_import_module`,
    which installs into the running interpreter and imports again.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        do_install = auto_install_enabled() if install is None else install
        if not do_install:
            raise
        from yggdrasil.environ import PyEnv

        return PyEnv.runtime_import_module(
            module_name=module_name, pip_name=pip_name or module_name, install=True,
        )
