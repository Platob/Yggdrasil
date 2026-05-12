"""Bridge between `yggdrasil` and the `yggcy` Cython extension.

The compiled module lives at the top level as `_yggcy` because two
wheels cannot both own the `yggdrasil/` package directory (`ygg`
already ships `yggdrasil/__init__.py` as a regular package). This
bridge exposes the `_yggcy.<sub>` submodules as plain attributes on
``yggdrasil.cy`` so call sites read like
``from yggdrasil import cy as _cy; _cy.io_url.parse_url(...)``.

`HAS_CY` exists so call sites can short-circuit cleanly when the
optional wheel isn't installed; the Python fallbacks in the matching
`yggdrasil.*` modules cover every kernel exposed here.
"""

from __future__ import annotations

from types import ModuleType

__all__ = ["HAS_CY", "cy", "io_url"]

HAS_CY: bool = False
_IMPORT_ERROR: BaseException | None = None
cy: ModuleType | None = None
io_url: ModuleType | None = None

try:
    import _yggcy as _cy

    cy = _cy

    _io_mod = getattr(_cy, "io", None)
    if _io_mod is not None:
        io_url = getattr(_io_mod, "url", None)

    HAS_CY = io_url is not None
except ImportError as exc:
    # Quiet at import time — the wheel is optional. Surface the
    # original exception via ``_IMPORT_ERROR`` for diagnostics.
    _IMPORT_ERROR = exc
