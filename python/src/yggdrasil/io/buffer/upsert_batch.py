"""Origin-tagged carrier for an UPSERT payload staged against a TabularIO.

UPSERT writes split incoming rows into two sides:

- ``match`` — rows whose match-by key already exists in ``parent``;
  on apply they overwrite the matched destination rows.
- ``new``   — rows whose match-by key is not yet in ``parent``; on
  apply they're appended.

:class:`TabularUpsertBatch` keeps that split visible end-to-end, so
callers that staged an UPSERT from multiple sources (e.g. several
remote-cache tables, several local-cache folders) can preserve which
source contributed which rows — the same convention
:class:`yggdrasil.io.response_batch.ResponseBatch` uses for local /
remote hits. Both sides are insertion-ordered ``dict`` mappings keyed
by source identity (caller's choice); values are :class:`TabularIO`
holders so Python (Arrow) and Spark inputs share one read contract.

The class is a pure data carrier — no IO state, no engine logic.
``parent.write_arrow_batches`` and friends do the actual writing; the
:meth:`apply` helper is a thin convenience that walks the buckets in
order and routes them through the parent.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

from yggdrasil.data.options import CastOptions

from .base import TabularIO
from yggdrasil.io.enums import Mode

if TYPE_CHECKING:
    pass


__all__ = ["TabularUpsertBatch"]


O = TypeVar("O", bound=CastOptions)


@dataclasses.dataclass(slots=True)
class TabularUpsertBatch(Generic[O]):
    """Match/new split staged against a target :class:`TabularIO`.

    Three fields, in apply order:

    - ``parent`` — the destination holder being updated. All writes
      land here; reads of the existing state come from here too.
    - ``match``  — keyed dict of holders carrying rows whose match-by
      key already exists in ``parent``. On :meth:`apply` each holder
      is fed to ``parent`` with ``Mode.UPSERT`` so the engine routes
      it through the configured merge path (native MERGE on Delta /
      Spark, read/merge/overwrite rewrite on Arrow-IPC and friends).
    - ``new``    — keyed dict of holders carrying rows whose match-by
      key is not yet in ``parent``. On :meth:`apply` each holder is
      fed to ``parent`` with ``Mode.APPEND`` — there's nothing to
      merge against, so we skip the more expensive UPSERT path.

    Both keyed dicts are insertion-ordered so iteration over the
    contributed sources stays deterministic. Keys are caller-defined
    strings (a URL, a cache-table name, a partition value, …) — the
    batch itself doesn't interpret them, it just preserves them.
    """

    parent: TabularIO[O]
    match: dict[str, TabularIO[O]] = dataclasses.field(default_factory=dict)
    new: dict[str, TabularIO[O]] = dataclasses.field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # ``Mapping`` inputs (e.g. ``MappingProxyType``) get copied into a
        # plain dict so the batch owns its container — callers can mutate
        # their original mapping without leaking through, and our slotted
        # dataclass attribute stays a real dict (helpful for `setdefault`,
        # ordered insertion, etc.).
        if not isinstance(self.match, dict):
            self.match = dict(self.match)
        if not isinstance(self.new, dict):
            self.new = dict(self.new)
        self._check_holders(self.match, "match")
        self._check_holders(self.new, "new")

    @staticmethod
    def _check_holders(
        holders: Mapping[str, Any],
        side: str,
    ) -> None:
        for key, holder in holders.items():
            if not isinstance(holder, TabularIO):
                raise TypeError(
                    f"TabularUpsertBatch.{side}[{key!r}] must be a TabularIO, "
                    f"got {type(holder).__name__}."
                )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[TabularIO[O]]:
        """Walk every staged holder, match side first then new side."""
        yield from self.match.values()
        yield from self.new.values()

    def is_empty(self) -> bool:
        """True if neither side has any keyed holder."""
        return not self.match and not self.new

    @property
    def match_keys(self) -> list[str]:
        return list(self.match)

    @property
    def new_keys(self) -> list[str]:
        return list(self.new)

    # ------------------------------------------------------------------
    # Apply — route every staged holder through the parent
    # ------------------------------------------------------------------

    def apply(self, options: "O | None" = None) -> None:
        """Run the staged upsert against :attr:`parent`.

        ``match`` holders go through ``parent`` with ``Mode.UPSERT``
        so the destination engine picks its native path (MERGE on
        Delta / Spark, read/merge/overwrite rewrite on Arrow-IPC and
        friends — see :meth:`TabularIO._upsert`). ``new`` holders go
        through with ``Mode.APPEND`` — there are no existing rows to
        merge against on the new side, so the cheaper append path is
        always correct and avoids reading the destination back.

        ``options`` is forwarded to ``parent.write_arrow_batches`` —
        callers typically pass the same options they would have
        threaded through a direct write (carrying
        ``match_by_names``, ``update_column_names``, target schema,
        engine knobs, etc.). When ``None``, ``parent`` falls back to
        its default options.
        """
        for holder in self.match.values():
            self.parent.write_arrow_batches(
                holder.read_arrow_batches(options),
                options=self._with_mode(options, Mode.UPSERT),
            )
        for holder in self.new.values():
            self.parent.write_arrow_batches(
                holder.read_arrow_batches(options),
                options=self._with_mode(options, Mode.APPEND),
            )

    @staticmethod
    def _with_mode(options: "O | None", mode: Mode) -> "O | None":
        """Return ``options`` with ``mode`` set, preserving ``None``.

        ``None`` propagates through so the parent's own defaults
        apply — ``Mode.UPSERT`` and ``Mode.APPEND`` do still need to
        land somewhere, so callers passing ``None`` are responsible
        for ensuring the parent's defaults are right. The common
        case (caller passes a real ``CastOptions``) gets the explicit
        mode override.
        """
        if options is None:
            return None
        return options.copy(mode=mode)
