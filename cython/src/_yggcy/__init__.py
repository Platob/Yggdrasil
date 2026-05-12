"""Top-level compiled package for the `yggcy` wheel.

The `ygg` runtime imports this package through `yggdrasil.cy`, which
re-publishes the compiled submodules under the `yggdrasil.cy.<sub>`
namespace. Two wheels can't co-own `yggdrasil/`, so the C extensions
live here and the bridge handles the rename.
"""

from . import io

__all__ = ["io"]
