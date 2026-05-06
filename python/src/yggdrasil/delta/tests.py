"""Unittest base class for Delta tests.

Sub-class :class:`DeltaTestCase` instead of importing
``yggdrasil.delta`` at module level — it skips cleanly when pyarrow
isn't available, and gives every test a fresh empty table directory
plus a few convenience constructors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from yggdrasil.arrow.tests import ArrowTestCase

if TYPE_CHECKING:
    from yggdrasil.delta.io import DeltaIO


__all__ = ["DeltaTestCase"]


class DeltaTestCase(ArrowTestCase):
    """:class:`ArrowTestCase` with Delta convenience helpers.

    Adds:

    - :attr:`delta_root` — the ``tmp_path`` cast to a string, ready to
      pass to :class:`DeltaIO`.
    - :meth:`delta_io` — return a fresh :class:`DeltaIO` over a
      sub-directory of the per-test :attr:`tmp_path`.
    - :meth:`new_table` — convenience for "create a brand-new
      :class:`DeltaIO` and seed it with a pyarrow Table in one call."
    """

    require_parquet: ClassVar[bool] = True

    def delta_io(self, name: str = "delta") -> "DeltaIO":
        from yggdrasil.delta.io import DeltaIO

        sub = self.tmp_path / name
        sub.mkdir(parents=True, exist_ok=True)
        return DeltaIO(path=str(sub))

    def new_table(
        self,
        table: Any,
        *,
        name: str = "delta",
        options: Any = None,
    ) -> "DeltaIO":
        d = self.delta_io(name)
        d.write_arrow_table(table, options=options)
        return d
