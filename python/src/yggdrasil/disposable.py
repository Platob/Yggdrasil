from __future__ import annotations

from abc import ABC


class Disposable(ABC):
    """Transactional resource with binary open/close, with-stack
    nesting, and an owned-children graph."""

    __slots__ = (
        '_acquired',
        '_dirty',
        '_depth',
        '_ctx_owns_close',
    )

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        self._acquired: bool = False
        self._dirty: bool = False
        self._depth: int = 0
        self._ctx_owns_close: bool = False

    # ------------------------------------------------------------------
    # State predicates
    # ------------------------------------------------------------------

    def _init_from_disposable(self, obj: "Disposable"):
        self._acquired = obj._acquired
        self._dirty = obj._dirty
        self._depth = obj._depth
        self._ctx_owns_close = obj._ctx_owns_close

    @property
    def opened(self) -> bool:
        """True iff :meth:`_acquire` has run and :meth:`_release` hasn't."""
        return self._acquired

    @property
    def closed(self) -> bool:
        """Inverse of :attr:`opened`."""
        return not self._acquired

    # ------------------------------------------------------------------
    # Binary open/close — not reentrant
    # ------------------------------------------------------------------

    def open(self) -> "Disposable":
        """Acquire the resource and cascade into owned children.

        Order:

        1. Run our own :meth:`_acquire` (subclass body).
        2. Flip :attr:`opened` to True and mark ``_self_opened``.
        3. For each owned child, in registration order:

           - If the child is already opened, just :meth:`_claim` it.
             It stays self-opened — the existing self-open is what
             keeps it alive after we let go.
           - Otherwise, call :meth:`open` on the child (which
             recursively cascades into ITS owned children), then
             *clear* the child's ``_self_opened`` flag so the child
             knows its open is parent-driven, then :meth:`_claim` it.
             Without that flag clear, the eventual :meth:`_unclaim`
             would refuse to close — it would see "I'm self-opened,
             someone explicitly opened me, leave me alone."

           Both branches record the child in our per-frame scratch
           list so :meth:`_release` knows what to unclaim.

        Transactional rollback: if any child's open or claim raises,
        we walk back through the children we already touched (in
        reverse), unclaim each, then call our own :meth:`_release`
        with ``committed=False`` and re-raise the original exception.
        From the caller's view, the open atomically either succeeded
        with the whole graph live, or failed with nothing changed.

        Not reentrant: raises :class:`RuntimeError` if already opened.
        Nesting is expressed via ``with self:`` blocks, not via paired
        :meth:`open` calls.
        """
        if self._acquired:
            if self._depth == 0:
                return self
            raise RuntimeError("open() called on an already-opened Disposable")
        self._dirty = False
        self._acquire()
        self._acquired = True
        return self

    def commit(self):
        """Commit current state"""
        if self.is_dirty():
            try:
                self._commit()
            except Exception:
                self.rollback()
                raise

            self.clear_dirty()

    def _commit(self):
        """Commit current state"""

    def rollback(self):
        """Rollback current state"""

    def _rollback(self):
        """Rollback current state"""

    def close(self, force: bool = False) -> None:
        """Release the resource and cascade into owned children.

        Order:

        1. If currently held open by an outside parent claim
           (``_claim_count > 0``) AND we are *not* in self-opened
           state, this is a no-op — the parents that opened us
           still need us live. (Handled inside :meth:`_do_close`.)
        2. Walk our scratch list of acquired children in REVERSE
           registration order; :meth:`_unclaim` each. A child whose
           claim count hits zero and isn't otherwise self-opened
           closes itself.
        3. Run :meth:`_before_release`, then :meth:`_release` —
           with ``committed`` reflecting the dirty bit (cleared on
           exception by ``__exit__``).

        Idempotent: no-op when already closed, unless *force*.

        ``force=True`` runs teardown even when :attr:`closed`.
        Intended for error-recovery paths where subclass state
        might be inconsistent.

        Does NOT touch :attr:`depth` — the ``with``-stack counter
        belongs to :meth:`__enter__`/:meth:`__exit__` exclusively.
        If a caller calls :meth:`close` inside an active ``with``
        block, the outer :meth:`__exit__` will harmlessly skip the
        now-no-op close on unwind.
        """
        self._close(force=force)

    def _close(self, *, force: bool) -> None:
        """Shared close path, used by both :meth:`close` and the
        last-parent-leaves path in :meth:`_unclaim`.

        ``bypass_claims=True`` is the internal signal from
        :meth:`_unclaim` that the claim count just transitioned to
        zero and the close should proceed even though we're not
        flagged as self-opened — the path is "no parent is holding
        us anymore, so finish teardown."
        """
        if not self._acquired and not force:
            return

        self._acquired = False
        self.commit()

        try:
            self._before_release()
        finally:
            self._release()

    def is_dirty(self):
        return self._dirty

    def mark_dirty(self) -> None:
        """Signal pending mutations — commit on next clean :meth:`close`."""
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False

    # ------------------------------------------------------------------
    # Context manager — with-stack nesting
    # ------------------------------------------------------------------

    def __enter__(self) -> "Disposable":
        """Enter a context. Opens the resource only if not already open.

        The first ``__enter__`` that finds the resource closed flips
        :attr:`_ctx_owns_close` — that frame's matching ``__exit__``
        is the one that closes. Nested ``__enter__``s see the resource
        already open and leave the flag alone.
        """
        if not self._acquired:
            self.open()
            self._ctx_owns_close = True
        elif self._depth == 0:
            self._ctx_owns_close = True
        self._depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit a context. Closes only from the outermost frame that
        owns the close.

        On that frame, if an exception is propagating we clear the
        dirty bit before :meth:`close` so :meth:`_release` sees
        ``committed=False`` — half-applied mutations don't reach
        storage.
        """
        self._depth -= 1
        if self._depth > 0:
            return
        if not self._ctx_owns_close:
            return
        self._ctx_owns_close = False
        if exc_type is not None:
            self._dirty = False
        self.close()

    # ------------------------------------------------------------------
    # Template methods — subclasses override
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Acquire the transactional resource.

        Called once per :meth:`open`, before :attr:`opened` flips
        True and before owned children are cascaded into.
        Default no-op for subclasses that only need bookkeeping.
        """

    def _before_release(self) -> None:
        """Drop nested handles before the main teardown.

        Called once on :meth:`close`, AFTER owned-child claims have
        been dropped, BEFORE :meth:`_release`. Exceptions propagate
        but do not prevent :meth:`_release`.
        """

    def _release(self) -> None:
        """Tear down the transactional resource.

        ``committed`` reflects the dirty bit at close time, with
        exception-suppression applied by ``__exit__`` if relevant.
        """

    # ------------------------------------------------------------------
    # GC safety net
    # ------------------------------------------------------------------

    def __del__(self):
        # Best-effort only; never raise from __del__. Do NOT commit on
        # GC — data integrity depends on explicit close.
        try:
            if self.opened:
                self.close(force=True)
        except Exception:
            pass
