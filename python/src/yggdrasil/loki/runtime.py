"""Auto-installing optional-dependency loader for Loki.

Loki reaches for heavy optional packages — an engine SDK (``anthropic`` /
``openai``), a local-model runtime (``transformers`` + ``torch``), a headless
browser (``playwright``) — only when a feature actually needs one. Rather than
fail with an ``ImportError`` three frames deep, it **installs the missing
package into the running interpreter on first use** and continues.

This is a thin, default-on wrapper over the project's own optional-dependency
guard, :func:`yggdrasil.lazy_imports._lazy_import`: it imports a module and, on
a miss, installs *pip_name* via
:meth:`~yggdrasil.environ.PyEnv.runtime_import_module` — which anchors on
``sys.executable`` so the package **persists in the interpreter Loki is running
in**. The only thing Loki adds is the default: ``_lazy_import`` defaults to
``install=False`` (project-wide hygiene), while Loki's :func:`load` defaults to
``install=True`` so a feature just works on first use. Set
``YGG_LOKI_AUTO_INSTALL=0`` to turn that off — then a missing package raises the
normal ``ImportError``.
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
    """Import *module_name*, installing *pip_name* into the current env on miss.

    Delegates to :func:`yggdrasil.lazy_imports._lazy_import` — the project's
    one import-or-install guard. ``install`` overrides the default for a single
    call (``True`` forces a runtime install, ``False`` forbids it); left
    ``None`` it follows :func:`auto_install_enabled` (default ``True``). Returns
    the live module — identical to ``import module_name``.
    """
    from yggdrasil.lazy_imports import _lazy_import

    do_install = auto_install_enabled() if install is None else install
    return _lazy_import(module_name, pip_name, install=do_install)
