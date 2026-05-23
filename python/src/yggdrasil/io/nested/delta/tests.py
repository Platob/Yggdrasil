"""Unittest base class for Delta tests.

Sub-class :class:`DeltaTestCase` instead of importing
``yggdrasil.io.nested.delta`` at module level — it skips cleanly when
pyarrow isn't available, and gives every test a fresh empty table
directory plus a few convenience constructors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from yggdrasil.arrow.tests import ArrowTestCase

if TYPE_CHECKING:
    from yggdrasil.io.nested.delta.delta_folder import DeltaFolder


__all__ = ["DeltaTestCase"]


class DeltaTestCase(ArrowTestCase):
    """:class:`ArrowTestCase` with Delta convenience helpers.

    Adds:

    - :meth:`delta_io` — return a fresh :class:`DeltaFolder` over a
      sub-directory of the per-test :attr:`tmp_path`.
    - :meth:`new_table` — convenience for "create a brand-new
      :class:`DeltaFolder` and seed it with a pyarrow Table in one call."
    """

    require_parquet: ClassVar[bool] = True

    def delta_io(self, name: str = "delta") -> "DeltaFolder":
        from yggdrasil.io.nested.delta.delta_folder import DeltaFolder

        sub = self.tmp_path / name
        sub.mkdir(parents=True, exist_ok=True)
        return DeltaFolder(path=str(sub))

    def new_table(
        self,
        table: Any,
        *,
        name: str = "delta",
        options: Any = None,
    ) -> "DeltaFolder":
        d = self.delta_io(name)
        d.write_arrow_table(table, options=options)
        return d
