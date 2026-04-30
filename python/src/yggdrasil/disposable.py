"""Transactional :class:`Disposable` base class with with-stack nesting
and an owned-children graph.

Extracted from :class:`yggdrasil.io.buffer.media_io.MediaIO` — the
MediaIO open/close/dirty/flush dance turned out to be a reusable
pattern.

Two orthogonal concepts
-----------------------

This class splits two things the original Disposable conflated:

1. **open/close** — a binary state: the resource is either acquired or
   it isn't. :meth:`open` is NOT reentrant; calling it twice raises.
   :meth:`close` is idempotent (no-op when already closed).

2. **``with`` stack depth** — an orthogonal nesting counter that tracks
   how many context frames are currently active. Only the ``__enter__``
   that actually called :meth:`open` (i.e. the outermost frame that
   found the resource closed) is responsible for calling :meth:`close`
   on its matching ``__exit__``. Inner frames just bump the depth so
   they don't trigger teardown when they exit.

Owned-children graph
--------------------

A :class:`Disposable` can register OTHER :class:`Disposable` instances
as **owned children** via :meth:`add_owned`. The graph is bidirectional:

- Parent → child reference is **strong** (lives in ``_owned``, a list
  in insertion order).
- Child → parent reference is **weak** (lives in ``_parents``, a
  :class:`weakref.WeakSet`). If a parent is GC'd, its claim on the
  child auto-prunes.

When a parent's :meth:`_acquire` runs, the children-cascade fires
AFTER the subclass's own ``_acquire`` body — the parent acquires its
own resource first, then walks owned children in registration order
and either opens them (if closed) or just records a "claim" on them
(if already open). On :meth:`_release`, the parent unclaims those
children in reverse order; a child only actually closes when its
claim count hits zero AND no parent currently holds it open.

This means a child can be wired into multiple parents safely:

- Parent A acquires → opens child, claim=1.
- Parent B acquires → child already open, claim=2.
- Parent A releases → unclaim, claim=1; child stays open (B still
  holds it).
- Parent B releases → unclaim, claim=0; child closes.

Acquire is **transactional across siblings**: if child N's open()
raises, the parent walks back through children 0..N-1 (in reverse),
unclaims each, then re-raises. The parent itself is also rolled back
— ``_acquired`` stays False and the caller sees the original
exception. This means if you can ``open()``, you got the whole graph;
if it raised, nothing was left half-open.

Why split open/close from the with-stack
----------------------------------------

You can safely write methods like ::

    def do_work(self):
        with self:                     # nested if caller already opened
            for batch in self._iter_batches(...):
                ...

and get both shapes right:

* Caller has already opened ``self`` (``with bio: bio.do_work()``):
  the inner ``__enter__`` sees :attr:`opened` True, bumps depth, does
  not flip the "owns close" flag. The inner ``__exit__`` just
  decrements depth. The caller's outer ``__exit__`` handles the close.
* Caller hasn't opened anything (``bio.do_work()``): the inner
  ``__enter__`` opens, depth becomes 1, flips "owns close". The
  inner ``__exit__`` decrements to 0, sees "owns close", calls
  :meth:`close`.

Nested ``with self:`` frames within the same caller also work:

* Outer ``__enter__`` opens, depth 0→1, owns close.
* Inner ``__enter__`` sees already-open, depth 1→2, does not flip ownership.
* Inner ``__exit__`` depth 2→1, not outermost — skip close.
* Outer ``__exit__`` depth 1→0, owns close — calls :meth:`close`.

Three template methods
----------------------

Subclasses override:

* :meth:`_acquire` — called exactly once per :meth:`open`, to
  materialize the resource. Runs before :attr:`opened` flips True
  and before owned children are cascaded into.
* :meth:`_before_release` — called once on :meth:`close`, before
  :meth:`_release`. Hook for unwinding nested handles. Exceptions
  propagate but don't prevent :meth:`_release` — the ``finally``
  guarantees teardown.
* :meth:`_release(committed)` — called once on :meth:`close`.
  ``committed=True`` means "the dirty bit was set and no exception
  suppressed it"; commit the mutations. ``committed=False`` means
  "drop without committing".

Dirty bit / exception suppression
---------------------------------

Subclasses call :meth:`mark_dirty` to signal "pending mutations
exist". On a clean outermost ``__exit__``, the bit is preserved and
``_release(committed=True)`` runs. If an exception is propagating
through the outermost ``__exit__`` that owns the close, the bit is
cleared first so ``_release`` sees ``committed=False`` — preventing a
half-applied transaction from reaching storage. Subclasses that want
to commit partial progress on failure must catch the exception inside
their own transaction body.

Subclassing notes
-----------------

``Disposable`` declares ``__slots__`` and does not carry a ``__dict__``.
Subclasses that want to stay ``__dict__``-free should declare their
own ``__slots__``. Subclasses needing init should call
``super().__init__()``.
"""

from __future__ import annotations

import weakref
from abc import ABC
from typing import Any, Iterable, List


class Disposable(ABC):
    """Transactional resource with binary open/close, with-stack
    nesting, and an owned-children graph."""

    def __init__(self) -> None:
        # Binary open/close flag. Flipped by open()/close() only.
        self._acquired: bool = False
        # Pending mutations — committed by _release if clean.
        self._dirty: bool = False
        # Number of currently-active `with self:` frames. Managed
        # entirely by __enter__/__exit__; open()/close() do NOT touch it.
        self._depth: int = 0
        # Set by the __enter__ that actually called open(); cleared by
        # the __exit__ that actually calls close(). Lets nested frames
        # skip teardown when they exit.
        self._ctx_owns_close: bool = False
        # Owned-children graph state.
        self._owned: List["Disposable"] = []
        self._parents: "weakref.WeakSet[Disposable]" = weakref.WeakSet()
        self._claim_count: int = 0
        self._self_opened: bool = False
        self._last_acquired_owned: List["Disposable"] = []

    # ------------------------------------------------------------------
    # State predicates
    # ------------------------------------------------------------------

    @property
    def opened(self) -> bool:
        """True iff :meth:`_acquire` has run and :meth:`_release` hasn't."""
        return self._acquired

    @property
    def closed(self) -> bool:
        """Inverse of :attr:`opened`."""
        return not self._acquired

    @property
    def depth(self) -> int:
        """Number of currently-active ``with self:`` frames.

        Zero outside any context manager, regardless of :attr:`opened`.
        Not the same as "have I been opened" — a caller who did
        ``bio.open()`` has ``opened=True``, ``depth=0``.
        """
        return self._depth

    @property
    def claim_count(self) -> int:
        """Number of parents currently holding an open frame that
        includes this Disposable.

        Incremented when a parent's :meth:`_acquire` cascades into
        us; decremented on the matching :meth:`_release`. Independent
        of :attr:`depth` (which counts ``with self:`` frames on this
        instance) — both can be non-zero simultaneously.
        """
        return self._claim_count

    @property
    def owned(self) -> List["Disposable"]:
        """Snapshot of currently-owned children in registration order.

        Returned as a fresh list so callers can iterate without
        races against concurrent :meth:`add_owned` /
        :meth:`remove_owned`.
        """
        return list(self._owned)

    @property
    def parents(self) -> List["Disposable"]:
        """Snapshot of live parents that own this Disposable.

        Parents that have been GC'd without explicit
        :meth:`remove_owned` are pruned from the WeakSet
        automatically and won't appear here.
        """
        return list(self._parents)

    # ------------------------------------------------------------------
    # Owned-children graph — registration
    # ------------------------------------------------------------------

    def manage_object(self, child: "Disposable") -> "Disposable":
        """Register *child* as an owned dependency.

        On future :meth:`_acquire` calls of self, *child* will be
        opened (or claimed, if already open) AFTER our own
        ``_acquire`` body runs. On :meth:`_release` the claim is
        dropped in reverse-registration order.

        **Mid-frame add**: if *self* is currently in an open frame
        (``self.opened`` is True), the child is claimed immediately
        (and opened first, if closed) so the rest of the open frame
        sees it live and the matching :meth:`_release` will unclaim
        it correctly. Without this, a child added mid-frame would
        sit in ``_owned`` but not in ``_last_acquired_owned``, and
        the eventual close cascade would skip it — its claim would
        never balance back to zero.

        Idempotent: adding the same child twice is a no-op. The
        bidirectional edge (child→parent) is set up automatically
        with a weak parent reference.

        Returns *child* for chaining.

        :raises TypeError: if *child* is not a :class:`Disposable`.
        :raises ValueError: if *child* is *self* (would form a self
            loop in the cascade).
        """
        if not isinstance(child, Disposable):
            raise TypeError(
                f"add_owned expects a Disposable, got {type(child).__name__}"
            )
        if child is self:
            raise ValueError("Disposable cannot own itself")
        if child in self._owned:
            return child
        self._owned.append(child)
        child._parents.add(self)

        # Mid-frame add: claim the child immediately so the open
        # cascade is consistent. Mirrors what _acquire's cascade
        # does for children registered before _acquire ran.
        if self._acquired:
            if not child._acquired:
                child.open()
                # Cascade-opened children are not self-opened — the
                # parent (we) own their lifecycle. Clear the flag
                # so the matching unclaim can fire close.
                child._self_opened = False
            child._claim()
            self._last_acquired_owned.append(child)
        return child

    def unmanage_object(self, child: "Disposable") -> bool:
        """Unregister *child* from our owned set.

        If we currently hold a claim on *child* (i.e. self is opened
        and *child* was acquired during our cascade), the claim is
        dropped; the child may close as a result if no other parent
        is still holding it.

        Returns True if *child* was registered and is now removed,
        False if it wasn't registered.

        Does NOT raise on unknown children — disconnection is
        idempotent.
        """
        try:
            self._owned.remove(child)
        except ValueError:
            return False
        # Drop the back-reference. WeakSet.discard is no-raise.
        child._parents.discard(self)
        # If we were currently holding a claim on this child, drop it.
        # Use the scratch list — that's the authoritative record of
        # what we actually claimed during this acquire frame.
        if child in self._last_acquired_owned:
            self._last_acquired_owned.remove(child)
            child._unclaim()
        return True

    # ------------------------------------------------------------------
    # Owned-children graph — claim refcount (internal protocol between
    # parent's _acquire/_release and the child's open/close decision)
    # ------------------------------------------------------------------

    def _claim(self) -> None:
        """Increment the parent-claim count.

        Called on a child by its parent's acquire cascade. The child
        treats this as "one more parent currently has an open frame
        that includes me"; while ``_claim_count > 0``, :meth:`close`
        defers the actual teardown until the count drains.
        """
        self._claim_count += 1

    def _unclaim(self) -> None:
        """Decrement the parent-claim count and possibly close.

        Called on a child by its parent's release cascade. If the
        count reaches zero AND we are not currently held open by our
        own :meth:`open` (i.e. ``_self_opened`` is False), close
        immediately — the last parent has let go and no one else is
        holding us.
        """
        if self._claim_count <= 0:
            # Defensive: should not happen if claim/unclaim are
            # paired through normal acquire/release. Guard against
            # double-unclaim drifting the count negative.
            self._claim_count = 0
            return
        self._claim_count -= 1
        if self._claim_count == 0 and self._acquired and not self._self_opened:
            # We were only open because a parent claimed us. Last
            # parent is leaving — perform the actual close now,
            # bypassing the claim guard since we're the one
            # signalling that all claims are gone.
            self._do_close(force=False, bypass_claims=True)

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
        self._self_opened = True
        # Cascade into owned children. Track what we actually
        # claimed so _release / rollback knows the inverse.
        self._last_acquired_owned = []
        try:
            for child in self._owned:
                if not child._acquired:
                    # Cascade-opened: the child wasn't open before
                    # we touched it, so it's NOT in self-opened
                    # state — only this parent (and any later
                    # parents that claim it) hold it. Clearing the
                    # flag immediately after open() lets the final
                    # unclaim cascade actually close the child.
                    # open() on the child would have set
                    # _self_opened=True; we override.
                    child.open()  # recursive cascade
                    child._self_opened = False
                child._claim()
                self._last_acquired_owned.append(child)
        except BaseException:
            # Roll back any siblings we already claimed, in reverse.
            self._rollback_owned()
            # Roll back our own state. _release with committed=False
            # so subclass teardown gets a chance, but mutations are
            # explicitly NOT committed.
            self._self_opened = False
            self._acquired = False
            try:
                self._release(False)
            except Exception:
                # Don't mask the original failure with a teardown
                # error — best-effort.
                pass
            self._dirty = False
            raise
        return self

    def commit(self):
        """Commit current state"""

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
        self._do_close(force=force, bypass_claims=False)

    def _do_close(self, *, force: bool, bypass_claims: bool) -> None:
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

        # Defer if a parent still has us claimed and the caller
        # didn't go through the unclaim path. The parent's eventual
        # _unclaim will call back here when it's safe.
        if (
            not bypass_claims
            and not force
            and self._claim_count > 0
        ):
            # We're a child in someone else's cascade. The last
            # parent's _unclaim will close us. Mark that we're no
            # longer in self-opened state so when the last claim
            # drops, the close actually fires.
            self._self_opened = False
            return

        committed = self._dirty
        # Flip flags first so any callbacks seeing us mid-teardown
        # observe a closed Disposable rather than an inconsistent
        # half-open one.
        self._acquired = False
        self._self_opened = False
        # Drop owned-children claims (LIFO), regardless of whether
        # this is a normal close or a forced one. Walk a snapshot so
        # _unclaim's potential close-cascade doesn't disturb the
        # iteration.
        owned_to_drop = list(reversed(self._last_acquired_owned))
        self._last_acquired_owned = []
        for child in owned_to_drop:
            try:
                child._unclaim()
            except Exception:
                # Best-effort; don't let one child's teardown
                # cascade-break siblings.
                pass
        try:
            self._before_release()
        finally:
            try:
                self._release(committed)
            finally:
                self._dirty = False

    def _rollback_owned(self) -> None:
        """Unclaim every child we touched during a partial acquire.

        Walks ``_last_acquired_owned`` in reverse and calls
        :meth:`_unclaim`. Each error is swallowed individually —
        rollback is best-effort across siblings, since the caller
        is already about to see an exception from the operation
        that triggered the rollback.
        """
        owned_to_drop = list(reversed(self._last_acquired_owned))
        self._last_acquired_owned = []
        for child in owned_to_drop:
            try:
                child._unclaim()
            except Exception:
                pass

    @property
    def dirty(self):
        return self._dirty

    def is_dirty(self):
        return self._dirty

    def mark_dirty(self) -> None:
        """Signal pending mutations — commit on next clean :meth:`close`."""
        if not self._acquired:
            raise RuntimeError("mark_dirty() called on a closed Disposable")
        self._dirty = True

    def _clear_dirty(self) -> None:
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

    def _release(self, committed: bool) -> None:
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
            if self._acquired:
                self._dirty = False
                # Use force=True to bypass the claim deferral; if we
                # made it to __del__ with _acquired=True, no parent
                # is reachable to ever come back and unclaim us.
                self._do_close(force=True, bypass_claims=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _close_quietly(obj: Any) -> None:
        """Call ``obj.close()`` swallowing exceptions. Useful in
        :meth:`_release` when unwinding multiple resources and a
        failure on one must not prevent the rest from being released.
        """
        if obj is None:
            return
        close = getattr(obj, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            pass

    @staticmethod
    def _close_all_quietly(objs: Iterable[Any]) -> None:
        """Close an iterable of objects, swallowing per-item errors."""
        for obj in objs:
            Disposable._close_quietly(obj)