"""Tests for Disposable with owned-children graph."""

from __future__ import annotations

import gc
import sys
import traceback
from typing import List

from yggdrasil.disposable import Disposable


class Tracked(Disposable):
    """Disposable that records lifecycle events into a shared log."""

    __slots__ = ("name", "log", "fail_on_acquire", "fail_on_release")

    def __init__(self, name: str, log: List[str] = None,
                 fail_on_acquire: bool = False, fail_on_release: bool = False) -> None:
        super().__init__()
        self.name = name
        self.log = log if log is not None else []
        self.fail_on_acquire = fail_on_acquire
        self.fail_on_release = fail_on_release

    def _acquire(self) -> None:
        self.log.append(f"{self.name}:acquire")
        if self.fail_on_acquire:
            raise RuntimeError(f"{self.name} acquire failure")

    def _release(self, committed: bool) -> None:
        self.log.append(f"{self.name}:release(committed={committed})")
        if self.fail_on_release:
            raise RuntimeError(f"{self.name} release failure")


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        print(f"FAIL: {msg}")
        print(f"  expected: {expected!r}")
        print(f"  actual:   {actual!r}")
        raise AssertionError(msg)


def test_basic_open_close():
    """Smoke test: simple open/close still works without children."""
    log = []
    d = Tracked("a", log)
    d.open()
    assert d.opened
    d.close()
    assert d.closed
    assert_eq(log, ["a:acquire", "a:release(committed=False)"], "basic open/close")
    print("PASS: test_basic_open_close")


def test_with_block_no_children():
    """`with` block opens and closes a childless Disposable."""
    log = []
    d = Tracked("a", log)
    with d:
        assert d.opened
    assert d.closed
    assert_eq(log, ["a:acquire", "a:release(committed=False)"], "with block")
    print("PASS: test_with_block_no_children")


def test_dirty_committed():
    """mark_dirty causes committed=True on clean exit."""
    log = []
    d = Tracked("a", log)
    with d:
        d.mark_dirty()
    assert_eq(log, ["a:acquire", "a:release(committed=True)"], "dirty committed")
    print("PASS: test_dirty_committed")


def test_dirty_dropped_on_exception():
    """Exception clears dirty before close."""
    log = []
    d = Tracked("a", log)
    try:
        with d:
            d.mark_dirty()
            raise ValueError("boom")
    except ValueError:
        pass
    assert_eq(log, ["a:acquire", "a:release(committed=False)"], "dirty dropped")
    print("PASS: test_dirty_dropped_on_exception")


def test_owned_cascade_open_close():
    """Parent open opens children in registration order; close in reverse."""
    log = []
    parent = Tracked("p", log)
    c1 = Tracked("c1", log)
    c2 = Tracked("c2", log)
    parent.manage_object(c1)
    parent.manage_object(c2)

    parent.open()
    assert parent.opened
    assert c1.opened
    assert c2.opened
    assert_eq(c1.claim_count, 1, "c1 claimed once")
    assert_eq(c2.claim_count, 1, "c2 claimed once")

    parent.close()
    assert parent.closed
    assert c1.closed
    assert c2.closed

    assert_eq(log, [
        "p:acquire",
        "c1:acquire",
        "c2:acquire",
        # release: parent flips to closed first, then unclaims in
        # reverse (c2 first, then c1), then runs its own _release.
        "c2:release(committed=False)",
        "c1:release(committed=False)",
        "p:release(committed=False)",
    ], "owned cascade order")
    print("PASS: test_owned_cascade_open_close")


def test_already_open_child_is_just_claimed():
    """A child already open when parent acquires is claimed, not re-opened."""
    log = []
    parent = Tracked("p", log)
    child = Tracked("c", log)
    parent.manage_object(child)

    child.open()
    assert child.opened
    assert_eq(child.claim_count, 0, "no parent claim yet")

    parent.open()
    # Child should NOT have been re-acquired.
    assert_eq(log, ["c:acquire", "p:acquire"], "child not re-opened")
    assert_eq(child.claim_count, 1, "parent claimed pre-opened child")

    # Parent close drops the claim. Child stays open because it's
    # self-opened.
    parent.close()
    assert child.opened, "child stays open when parent leaves"
    assert_eq(child.claim_count, 0, "claim dropped to 0")

    child.close()
    assert child.closed
    print("PASS: test_already_open_child_is_just_claimed")


def test_two_parents_share_child():
    """Two parents both claim the same child; child closes only when both leave."""
    log = []
    p1 = Tracked("p1", log)
    p2 = Tracked("p2", log)
    child = Tracked("c", log)
    p1.manage_object(child)
    p2.manage_object(child)

    p1.open()
    # p1 opened child as part of its cascade.
    assert child.opened
    assert_eq(child.claim_count, 1)

    p2.open()
    # p2 sees child already open, just claims.
    assert_eq(child.claim_count, 2)
    assert_eq(log, ["p1:acquire", "c:acquire", "p2:acquire"], "p2 didn't re-acquire c")

    p1.close()
    # p1 unclaims; p2 still holds, child stays open.
    assert child.opened, "child stays open while p2 holds"
    assert_eq(child.claim_count, 1)

    p2.close()
    # p2 unclaims; claim hits 0, child not self-opened, child closes.
    assert child.closed, "child closes when last parent leaves"
    assert_eq(child.claim_count, 0)

    # Verify the close happened at the right time (during p2's release).
    expected_tail = [
        "p1:release(committed=False)",  # p1 closes; child's claim only goes from 2→1, no close
        "p2:release(committed=False)",  # p2 closes; child's claim 1→0, child closes — but actually
        "c:release(committed=False)",   # the child release happens as part of the unclaim cascade
    ]
    # The actual ordering: when p2._do_close runs, it flips _acquired
    # first, then unclaims c (which closes c), then runs p2's _release.
    # So c's release happens BEFORE p2's release in the log.
    assert "c:release(committed=False)" in log
    p1_idx = log.index("p1:release(committed=False)")
    c_idx = log.index("c:release(committed=False)")
    p2_idx = log.index("p2:release(committed=False)")
    assert p1_idx < c_idx, "c didn't close prematurely (p1's release came first)"
    assert c_idx < p2_idx, "c closed during p2's unclaim, before p2's _release body"
    print("PASS: test_two_parents_share_child")


def test_acquire_failure_rolls_back_siblings():
    """If a child's acquire raises, already-opened siblings are unclaimed."""
    log = []
    parent = Tracked("p", log)
    c1 = Tracked("c1", log)
    c_bad = Tracked("c_bad", log, fail_on_acquire=True)
    c3 = Tracked("c3", log)
    parent.manage_object(c1)
    parent.manage_object(c_bad)
    parent.manage_object(c3)  # never reached

    try:
        parent.open()
    except RuntimeError as e:
        assert "c_bad acquire failure" in str(e)
    else:
        raise AssertionError("expected RuntimeError")

    # Parent rolled back; c1 should be closed; c3 was never touched.
    assert parent.closed, "parent rolled back to closed"
    assert c1.closed, "c1 rolled back"
    assert c_bad.closed, "c_bad never opened"
    assert c3.closed, "c3 never reached"
    assert_eq(c1.claim_count, 0)

    # Log should show: p acquire, c1 acquire, c_bad acquire (raises),
    # then rollback: c1 unclaim → c1 close, then p's _release.
    assert "p:acquire" in log
    assert "c1:acquire" in log
    assert "c_bad:acquire" in log  # the call started, then raised
    assert "c3:acquire" not in log, "c3 should never have been touched"
    assert "c1:release(committed=False)" in log, "c1 was rolled back via unclaim"
    assert "p:release(committed=False)" in log, "p ran its own _release on rollback"
    print("PASS: test_acquire_failure_rolls_back_siblings")


def test_remove_owned_drops_claim():
    """remove_owned on an open parent unclaims the child."""
    log = []
    parent = Tracked("p", log)
    child = Tracked("c", log)
    parent.manage_object(child)

    parent.open()
    assert_eq(child.claim_count, 1)

    removed = parent.unmanage_object(child)
    assert removed
    # Claim dropped → child closes (was not self-opened).
    assert child.closed
    assert_eq(child.claim_count, 0)
    # Child no longer in parent's owned list.
    assert child not in parent.owned
    assert parent not in child.parents

    # Parent close should NOT try to close child again.
    parent.close()
    assert parent.closed
    print("PASS: test_remove_owned_drops_claim")


def test_remove_owned_idempotent():
    """remove_owned on unknown child returns False without error."""
    parent = Tracked("p")
    child = Tracked("c")
    assert_eq(parent.unmanage_object(child), False)
    print("PASS: test_remove_owned_idempotent")


def test_add_owned_idempotent():
    """add_owned twice is a no-op the second time."""
    parent = Tracked("p")
    child = Tracked("c")
    parent.manage_object(child)
    parent.manage_object(child)
    assert_eq(parent.owned, [child])
    assert_eq(child.parents, [parent])
    # And opening only claims once.
    parent.open()
    assert_eq(child.claim_count, 1)
    parent.close()
    print("PASS: test_add_owned_idempotent")


def test_add_owned_self_raises():
    parent = Tracked("p")
    try:
        parent.manage_object(parent)
    except ValueError:
        print("PASS: test_add_owned_self_raises")
        return
    raise AssertionError("expected ValueError")


def test_add_owned_non_disposable_raises():
    parent = Tracked("p")
    try:
        parent.manage_object("not a disposable")
    except TypeError:
        print("PASS: test_add_owned_non_disposable_raises")
        return
    raise AssertionError("expected TypeError")


def test_nested_with_does_not_close_early():
    """Inner `with self:` doesn't close on exit if outer is still active."""
    log = []
    d = Tracked("a", log)
    with d:
        with d:
            assert d.opened
            assert_eq(d.depth, 2)
        assert d.opened, "inner exit didn't close outer-owned frame"
        assert_eq(d.depth, 1)
    assert d.closed
    assert_eq(log, ["a:acquire", "a:release(committed=False)"], "single open/close pair")
    print("PASS: test_nested_with_does_not_close_early")


def test_grandchild_cascade():
    """3-level graph: parent → child → grandchild."""
    log = []
    p = Tracked("p", log)
    c = Tracked("c", log)
    g = Tracked("g", log)
    c.manage_object(g)
    p.manage_object(c)

    p.open()
    assert p.opened and c.opened and g.opened
    assert_eq(g.claim_count, 1, "g claimed by c")
    assert_eq(c.claim_count, 1, "c claimed by p")

    p.close()
    assert p.closed and c.closed and g.closed
    print("PASS: test_grandchild_cascade")


def test_weak_parent_refs():
    """If a parent is GC'd without remove_owned, the weak ref auto-prunes."""
    child = Tracked("c")
    parent = Tracked("p")
    parent.manage_object(child)
    assert_eq(len(child.parents), 1)
    # Drop the parent; forced GC.
    del parent
    gc.collect()
    assert_eq(len(child.parents), 0, "weak parent auto-pruned")
    print("PASS: test_weak_parent_refs")


def test_self_close_while_parent_holds_defers():
    """Calling close() on a child while a parent has it claimed defers."""
    log = []
    parent = Tracked("p", log)
    child = Tracked("c", log)
    parent.manage_object(child)

    parent.open()
    assert_eq(child.claim_count, 1)

    # Try to close the child directly. Parent still holds it.
    child.close()
    # Should be a deferred close — child is still _acquired.
    assert child.opened, "child close deferred while parent claims it"

    # Parent leaves.
    parent.close()
    # Now child should actually be closed (last parent let go).
    assert child.closed, "child closes when parent finally releases"
    print("PASS: test_self_close_while_parent_holds_defers")


def test_force_close_overrides_claims():
    """force=True closes even when claims are outstanding."""
    log = []
    parent = Tracked("p", log)
    child = Tracked("c", log)
    parent.manage_object(child)

    parent.open()
    assert_eq(child.claim_count, 1)

    child.close(force=True)
    assert child.closed, "force close bypassed claim guard"
    print("PASS: test_force_close_overrides_claims")


def test_release_failure_in_one_child_doesnt_break_siblings():
    """If a child's release raises, sibling release still proceeds."""
    log = []
    parent = Tracked("p", log)
    bad = Tracked("bad", log, fail_on_release=True)
    good = Tracked("good", log)
    parent.manage_object(bad)
    parent.manage_object(good)

    parent.open()
    parent.close()
    # Both children should have had their release attempted.
    assert "bad:release(committed=False)" in log
    assert "good:release(committed=False)" in log
    # Parent release still ran.
    assert "p:release(committed=False)" in log
    print("PASS: test_release_failure_in_one_child_doesnt_break_siblings")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception:
            failed += 1
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
    print()
    print(f"=== {len(tests) - failed}/{len(tests)} tests passed ===")
    sys.exit(1 if failed else 0)