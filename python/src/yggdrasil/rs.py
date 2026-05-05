"""Bridge between `yggdrasil` and the `yggrs` Rust extension.

The compiled module lives at the top level as `_yggrs` because two wheels
cannot both own the `yggdrasil/` package directory (`ygg` already ships
`yggdrasil/__init__.py` as a regular package). This bridge re-publishes
the `_yggrs.<sub>` submodules under the `yggdrasil.rust` namespace so
call sites can stay on the conceptual layout
(`yggdrasil.rust.io.url.parse_url`).

`HAS_RS` exists for legacy call sites; since `ygg` now hard-depends on
`yggrs`, missing-extension fallbacks should not trigger in practice. The
flag stays `False` when the import fails so test environments without a
compiled wheel still report the failure cleanly via the `ImportError`
captured in `_IMPORT_ERROR`.
"""

from __future__ import annotations

import sys
from types import ModuleType

__all__ = ["HAS_RS", "rust", "io_url"]

HAS_RS: bool = False
_IMPORT_ERROR: BaseException | None = None
rust: ModuleType | None = None
io_url: ModuleType | None = None

try:
    import _yggrs as _rs

    rust = _rs

    # Re-publish `_yggrs.io.url` as `yggdrasil.rust.io.url` so call sites
    # don't need to know the underlying top-level name. Shimming via
    # sys.modules is enough — pyo3 builds plain `PyModule` instances, no
    # fancy import machinery to fight.
    _rust_pkg = ModuleType("yggdrasil.rust")
    _rust_pkg.__doc__ = "Bridge re-export of the _yggrs Rust extension."
    for _name in ("data", "io"):
        _sub = getattr(_rs, _name, None)
        if _sub is not None:
            setattr(_rust_pkg, _name, _sub)
            sys.modules[f"yggdrasil.rust.{_name}"] = _sub
    sys.modules["yggdrasil.rust"] = _rust_pkg

    _io_mod = getattr(_rs, "io", None)
    if _io_mod is not None:
        _url_mod = getattr(_io_mod, "url", None)
        if _url_mod is not None:
            io_url = _url_mod
            sys.modules["yggdrasil.rust.io.url"] = _url_mod

    HAS_RS = True
except ImportError as exc:
    # Stay quiet at import time — yggrs is declared as a hard dep, so a
    # missing module here means a packaging issue. Surface the original
    # exception via `_IMPORT_ERROR` for diagnostic call sites.
    _IMPORT_ERROR = exc
