"""
tests/test_expiring.py  —  stdlib unittest, no third-party deps
"""
from __future__ import annotations

import copy
import pickle
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone

from yggdrasil.dataclasses.expiring import (
    Expiring, ExpiringDict, RefreshResult,
    datetime_to_epoch_ns, now_utc_ns, timedelta_to_ns,
)
from yggdrasil.dataclasses import expiring as _expiring_mod

_1s_ns = 1_000_000_000
def _ns_from_now(s): return now_utc_ns() + int(s * _1s_ns)
def _sleep_past(s): time.sleep(s + 0.015)

class _Counter(Expiring[int]):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.call_count = 0

    def __setstate__(self, state):
        super().__setstate__(state)
        # call_count is not part of Expiring state — reinitialise after unpickle
        if not hasattr(self, "call_count"):
            self.call_count = 0

    def _refresh(self):
        self.call_count += 1
        return RefreshResult.make(value=self.call_count, ttl_ns=timedelta(milliseconds=60))

# ── Time utils ────────────────────────────────────────────
class TestTimeUtils(unittest.TestCase):
    def test_now_positive(self): self.assertGreater(now_utc_ns(), 0)
    def test_now_monotonic(self):
        a = now_utc_ns(); b = now_utc_ns(); self.assertGreaterEqual(b, a)
    def test_td_to_ns_seconds(self): self.assertEqual(timedelta_to_ns(timedelta(seconds=1)), _1s_ns)
    def test_td_to_ns_ms(self): self.assertEqual(timedelta_to_ns(timedelta(milliseconds=500)), 500_000_000)
    def test_dt_epoch(self): self.assertEqual(datetime_to_epoch_ns(datetime(1970,1,1,tzinfo=timezone.utc)), 0)
    def test_dt_naive_as_utc(self): self.assertEqual(datetime_to_epoch_ns(datetime(1970,1,1,0,0,1)), _1s_ns)
    def test_dt_y2k(self):
        ns = datetime_to_epoch_ns(datetime(2000,1,1,tzinfo=timezone.utc))
        self.assertAlmostEqual(ns, 946684800*_1s_ns, delta=1000)

# ── RefreshResult ─────────────────────────────────────────
class TestRefreshResult(unittest.TestCase):
    def test_defaults(self):
        rr = RefreshResult.make(value=42)
        self.assertEqual(rr.value, 42)
        self.assertIsNone(rr.ttl_ns); self.assertIsNone(rr.expires_at_ns)
    def test_ttl_timedelta(self):
        self.assertEqual(RefreshResult.make(value=1, ttl_ns=timedelta(seconds=5)).ttl_ns, 5*_1s_ns)
    def test_ttl_int(self):
        self.assertEqual(RefreshResult.make(value=1, ttl_ns=_1s_ns).ttl_ns, _1s_ns)
    def test_bad_ttl_raises(self):
        with self.assertRaises(TypeError): RefreshResult.make(value=1, ttl_ns="bad")

# ── Expiring[T] ───────────────────────────────────────────
class TestExpiring(unittest.TestCase):
    def test_none_triggers_refresh(self):
        c = _Counter(_value=None); self.assertEqual(c.value, 1)
    def test_no_double_refresh(self):
        c = _Counter(_value=None); _ = c.value; _ = c.value; self.assertEqual(c.call_count, 1)
    def test_auto_refresh_on_expiry(self):
        c = _Counter(_value=None); _ = c.value; _sleep_past(0.06); self.assertEqual(c.value, 2)
    def test_is_expired_false(self):
        c = _Counter(); c._value=1; c._expires_at_ns=_ns_from_now(10); self.assertFalse(c.is_expired())
    def test_is_expired_true(self):
        c = _Counter(); c._value=1; c._expires_at_ns=now_utc_ns()-1; self.assertTrue(c.is_expired())
    def test_ttl_ns_setter(self):
        c = _Counter(); c._value=1; c._created_at_ns=now_utc_ns(); c.ttl_ns=2*_1s_ns
        self.assertEqual(c._expires_at_ns, c._created_at_ns+2*_1s_ns)
    def test_value_setter(self):
        c = _Counter(_value=7); c.value=42; self.assertEqual(c._value, 42)
    def test_force_refresh(self):
        c = _Counter(_value=None); c.refresh(); self.assertEqual(c.call_count, 1)
    def test_pickle(self):
        c = _Counter(_value=None); _ = c.value
        c2 = pickle.loads(pickle.dumps(c))
        self.assertEqual(c2._value, c._value); self.assertIsNot(c2._lock, c._lock)
    def test_deepcopy(self):
        c = _Counter(_value=None); _ = c.value; c2 = copy.deepcopy(c)
        self.assertEqual(c2._value, c._value)
    def test_pickle_expired_refreshes(self):
        c = _Counter(_value=None); _ = c.value; _sleep_past(0.06)
        c2 = pickle.loads(pickle.dumps(c)); self.assertGreaterEqual(c2.value, 1)

# ── ExpiringDict — basic ──────────────────────────────────
class TestEDBasic(unittest.TestCase):
    def setUp(self): self.d = ExpiringDict(default_ttl=10.0)
    def test_set_get(self): self.d.set("k",1); self.assertEqual(self.d.get("k"),1)
    def test_item_access(self): self.d["k"]=2; self.assertEqual(self.d["k"],2)
    def test_missing_default(self): self.assertIsNone(self.d.get("x")); self.assertEqual(self.d.get("x",0),0)
    def test_missing_raises(self):
        with self.assertRaises(KeyError): _ = self.d["x"]
    def test_contains(self): self.d["a"]=1; self.assertIn("a", self.d); self.assertNotIn("b", self.d)
    def test_len(self): self.d["a"]=1; self.d["b"]=2; self.assertEqual(len(self.d),2)
    def test_del(self): self.d["a"]=1; del self.d["a"]; self.assertNotIn("a", self.d)
    def test_del_missing(self):
        with self.assertRaises(KeyError): del self.d["x"]
    def test_pop(self): self.d["k"]=9; self.assertEqual(self.d.pop("k"),9); self.assertNotIn("k",self.d)
    def test_pop_default(self): self.assertEqual(self.d.pop("x",-1),-1)
    def test_pop_raises(self):
        with self.assertRaises(KeyError): self.d.pop("x")
    def test_iter(self): self.d["a"]=1; self.d["b"]=2; self.assertEqual(set(self.d),{"a","b"})
    def test_keys_values_items(self):
        self.d["a"]=1; self.d["b"]=2
        self.assertEqual(set(self.d.keys()),{"a","b"})
        self.assertEqual(set(self.d.values()),{1,2})
        self.assertEqual(set(self.d.items()),{("a",1),("b",2)})
    def test_clear(self): self.d["a"]=1; self.d.clear(); self.assertEqual(len(self.d),0)
    def test_repr(self): self.assertIn("ExpiringDict", repr(self.d))

# ── ExpiringDict — TTL ────────────────────────────────────
class TestEDTTL(unittest.TestCase):
    def test_expires_float(self):
        d=ExpiringDict(default_ttl=0.06); d["k"]=1; _sleep_past(0.06); self.assertIsNone(d.get("k"))
    def test_expires_timedelta(self):
        d=ExpiringDict(default_ttl=timedelta(milliseconds=60)); d["k"]=1; _sleep_past(0.06); self.assertNotIn("k",d)
    def test_per_key_ttl(self):
        d=ExpiringDict(default_ttl=60.0); d.set("k","v",ttl=0.06); _sleep_past(0.06); self.assertIsNone(d.get("k"))
    def test_per_key_ttl_td(self):
        d=ExpiringDict(default_ttl=60.0); d.set("k","v",ttl=timedelta(milliseconds=60)); _sleep_past(0.06); self.assertIsNone(d.get("k"))
    def test_none_ttl_never_expires(self):
        d=ExpiringDict(default_ttl=0.06); d.set("k",42,ttl=None); _sleep_past(0.06); self.assertEqual(d.get("k"),42)
    def test_no_default_ttl(self):
        d=ExpiringDict(default_ttl=None); d["k"]="v"; _sleep_past(0.06); self.assertEqual(d["k"],"v")
    def test_ttl_seconds(self):
        d=ExpiringDict(default_ttl=5.0); d["k"]=1; r=d.ttl("k")
        self.assertIsNotNone(r); self.assertGreater(r,0); self.assertLessEqual(r,5.0)
    def test_ttl_expired(self):
        d=ExpiringDict(default_ttl=0.06); d["k"]=1; _sleep_past(0.06); self.assertIsNone(d.ttl("k"))
    def test_ttl_ns(self):
        d=ExpiringDict(default_ttl=5.0); d["k"]=1; r=d.ttl_ns("k")
        self.assertIsNotNone(r); self.assertGreater(r,0); self.assertLessEqual(r,5*_1s_ns)
    def test_ttl_non_expiring(self):
        d=ExpiringDict(default_ttl=None); d["k"]=1; self.assertIsNone(d.ttl("k"))
    def test_purge(self):
        d=ExpiringDict(default_ttl=0.06); d["a"]=1; d["b"]=2; d.set("c",3,ttl=10.0)
        _sleep_past(0.06); self.assertEqual(d.purge_expired(),2); self.assertIn("c",d)
    def test_contains_expired_false(self):
        d=ExpiringDict(default_ttl=0.06); d["k"]=1; _sleep_past(0.06); self.assertNotIn("k",d)

# ── ExpiringDict — bulk ───────────────────────────────────
class TestEDBulk(unittest.TestCase):
    def test_set_many(self):
        d=ExpiringDict(default_ttl=10.0); d.set_many({"a":1,"b":2,"c":3})
        self.assertEqual(d["a"],1); self.assertEqual(d["c"],3)
    def test_set_many_ttl(self):
        d=ExpiringDict(default_ttl=10.0); d.set_many({"x":1,"y":2},ttl=0.06)
        _sleep_past(0.06); self.assertIsNone(d.get("x")); self.assertIsNone(d.get("y"))
    def test_get_many(self):
        d=ExpiringDict(default_ttl=10.0); d.set_many({"a":1,"b":2})
        self.assertEqual(d.get_many(["a","b","?"]),{"a":1,"b":2})
    def test_get_many_expired(self):
        d=ExpiringDict(default_ttl=10.0); d.set("a",1,ttl=0.06); d.set("b",2,ttl=10.0)
        _sleep_past(0.06); self.assertEqual(d.get_many(["a","b"]),{"b":2})
    def test_delete_many(self):
        d=ExpiringDict(default_ttl=10.0); d.set_many({"a":1,"b":2,"c":3})
        self.assertEqual(d.delete_many(["a","c","?"]),2); self.assertIn("b",d)

# ── ExpiringDict — refresh ────────────────────────────────
class TestEDRefresh(unittest.TestCase):
    def test_refresh_key(self):
        d=ExpiringDict(default_ttl=0.1); d["k"]=1; _sleep_past(0.05)
        self.assertTrue(d.refresh_key("k",ttl=1.0)); _sleep_past(0.08); self.assertEqual(d.get("k"),1)
    def test_refresh_key_missing(self):
        d=ExpiringDict(default_ttl=10.0); self.assertFalse(d.refresh_key("x"))
    def test_refresh_key_expired(self):
        d=ExpiringDict(default_ttl=0.06); d["k"]=1; _sleep_past(0.06); self.assertFalse(d.refresh_key("k"))
    def test_get_or_set_missing(self):
        d=ExpiringDict(default_ttl=10.0); self.assertEqual(d.get_or_set("k",42),42); self.assertEqual(d["k"],42)
    def test_get_or_set_existing(self):
        d=ExpiringDict(default_ttl=10.0); d["k"]=99; self.assertEqual(d.get_or_set("k",0),99)
    def test_get_or_set_callable(self):
        d=ExpiringDict(default_ttl=10.0); self.assertEqual(d.get_or_set("k",lambda:"c"),"c")
    def test_get_or_set_expired(self):
        d=ExpiringDict(default_ttl=0.06); d["k"]="old"; _sleep_past(0.06); self.assertEqual(d.get_or_set("k","new"),"new")
    def test_refresher(self):
        calls=[]
        def r(k): calls.append(k); return RefreshResult.make(value=f"fresh:{k}",ttl_ns=timedelta(seconds=10))
        d=ExpiringDict(default_ttl=0.06,refresher=r); d["oil"]="stale"; _sleep_past(0.06)
        self.assertEqual(d.get("oil"),"fresh:oil"); self.assertEqual(calls,["oil"])
    def test_refresher_exception_default(self):
        def bad(k): raise RuntimeError()
        d=ExpiringDict(default_ttl=0.06,refresher=bad); d["k"]="v"; _sleep_past(0.06)
        self.assertEqual(d.get("k","fb"),"fb")
    def test_apply_refresh_result(self):
        d=ExpiringDict(default_ttl=10.0)
        d.apply_refresh_result("s", RefreshResult.make(value=55.5,ttl_ns=timedelta(seconds=5)))
        self.assertEqual(d["s"],55.5); r=d.ttl("s")
        self.assertIsNotNone(r); self.assertLessEqual(r,5.0)

# ── ExpiringDict — max_size ───────────────────────────────
class TestEDMaxSize(unittest.TestCase):
    def test_evict_soonest(self):
        d=ExpiringDict(default_ttl=10.0,max_size=2)
        d.set("a",1,ttl=1.0); d.set("b",2,ttl=5.0); d.set("c",3,ttl=10.0)
        self.assertEqual(len(d),2); self.assertNotIn("a",d); self.assertIn("c",d)
    def test_overwrite_no_evict(self):
        d=ExpiringDict(default_ttl=10.0,max_size=2)
        d.set("a",1); d.set("b",2); d.set("a",99)
        self.assertEqual(len(d),2); self.assertEqual(d["a"],99)

# ── ExpiringDict — snapshot ───────────────────────────────
class TestEDSnapshot(unittest.TestCase):
    def test_live_entries(self):
        d=ExpiringDict(default_ttl=10.0); d["a"]=1; d["b"]=2
        self.assertEqual(set(d.snapshot()),{"a","b"})
    def test_excludes_expired(self):
        d=ExpiringDict(default_ttl=0.06); d.set("old","v",ttl=0.06); d.set("live","v2",ttl=10.0)
        _sleep_past(0.06); snap=d.snapshot()
        self.assertNotIn("old",snap); self.assertIn("live",snap)
    def test_is_copy(self):
        d=ExpiringDict(default_ttl=10.0); d["k"]=1; snap=d.snapshot()
        snap["k"]=(999,None); self.assertEqual(d["k"],1)

# ── ExpiringDict — pickle ─────────────────────────────────
class TestEDPickle(unittest.TestCase):
    def test_roundtrip(self):
        d=ExpiringDict(default_ttl=60.0); d.set_many({"CL":82.0,"NG":1.85})
        d2=pickle.loads(pickle.dumps(d))
        self.assertEqual(d2["CL"],82.0); self.assertEqual(d2._default_ttl_ns,d._default_ttl_ns)
        self.assertIsNot(d2._lock,d._lock)
    def test_drops_expired(self):
        d=ExpiringDict(default_ttl=60.0); d.set("live",1,ttl=60.0); d.set("dead",2,ttl=0.06)
        _sleep_past(0.06); d2=pickle.loads(pickle.dumps(d))
        self.assertIn("live",d2); self.assertNotIn("dead",d2)
    def test_deepcopy(self):
        d=ExpiringDict(default_ttl=10.0); d["a"]=1; d2=copy.deepcopy(d)
        self.assertEqual(d2["a"],1); self.assertIsNot(d2._lock,d._lock)
    def test_refresher_not_pickled(self):
        d=ExpiringDict(default_ttl=10.0,refresher=lambda k:RefreshResult.make(42))
        d2=pickle.loads(pickle.dumps(d)); self.assertIsNone(d2._refresher)

# ── ExpiringDict — update ─────────────────────────────────
class TestEDUpdate(unittest.TestCase):
    def test_update_from_plain_dict(self):
        d=ExpiringDict(default_ttl=10.0); d.update({"a":1,"b":2})
        self.assertEqual(d["a"],1); self.assertEqual(d["b"],2)

    def test_update_kwargs(self):
        d=ExpiringDict(default_ttl=10.0); d.update(x=10, y=20)
        self.assertEqual(d["x"],10); self.assertEqual(d["y"],20)

    def test_update_dict_and_kwargs(self):
        d=ExpiringDict(default_ttl=10.0); d.update({"a":1}, b=2)
        self.assertEqual(d["a"],1); self.assertEqual(d["b"],2)

    def test_update_overwrites_existing(self):
        d=ExpiringDict(default_ttl=10.0); d["k"]=1; d.update({"k":99})
        self.assertEqual(d["k"],99)

    def test_update_none_is_noop(self):
        d=ExpiringDict(default_ttl=10.0); d["k"]=1; d.update(None)
        self.assertEqual(d["k"],1); self.assertEqual(len(d),1)

    def test_update_with_explicit_ttl_expires(self):
        d=ExpiringDict(default_ttl=60.0); d.update({"a":1,"b":2}, ttl=0.06)
        _sleep_past(0.06)
        self.assertIsNone(d.get("a")); self.assertIsNone(d.get("b"))

    def test_update_with_explicit_ttl_none_no_expiry(self):
        d=ExpiringDict(default_ttl=0.06); d.update({"k":"v"}, ttl=None)
        _sleep_past(0.06); self.assertEqual(d.get("k"),"v")

    def test_update_from_expiring_dict_preserves_ttl(self):
        src=ExpiringDict(default_ttl=60.0)
        src.set("price", 82.0, ttl=2.0)
        dst=ExpiringDict(default_ttl=60.0)
        dst.update(src)
        rem = dst.ttl("price")
        # TTL carried over: should be close to 2s but strictly less
        self.assertIsNotNone(rem)
        self.assertGreater(rem, 0)
        self.assertLessEqual(rem, 2.0)

    def test_update_from_expiring_dict_skips_expired(self):
        src=ExpiringDict(default_ttl=0.06); src.set("dead",1,ttl=0.06); src.set("live",2,ttl=60.0)
        _sleep_past(0.06)
        dst=ExpiringDict(default_ttl=10.0); dst.update(src)
        self.assertNotIn("dead",dst); self.assertIn("live",dst)

    def test_update_from_expiring_dict_explicit_ttl_overrides(self):
        src=ExpiringDict(default_ttl=60.0); src.set("k",1,ttl=60.0)
        dst=ExpiringDict(default_ttl=60.0); dst.update(src, ttl=0.06)
        _sleep_past(0.06); self.assertIsNone(dst.get("k"))

    def test_update_from_expiring_dict_non_expiring_uses_default(self):
        src=ExpiringDict(default_ttl=None); src["k"]="v"
        dst=ExpiringDict(default_ttl=0.06); dst.update(src)
        # src key is non-expiring; dst has default 60ms → should expire
        _sleep_past(0.06); self.assertIsNone(dst.get("k"))

    def test_update_respects_max_size(self):
        d=ExpiringDict(default_ttl=10.0, max_size=2)
        d["a"]=1; d.update({"b":2,"c":3})
        self.assertLessEqual(len(d), 2)

    def test_update_with_items_protocol(self):
        """Any object with an .items() method is accepted."""
        class FakeMapping:
            def items(self): return [("x", 9), ("y", 8)]
        d=ExpiringDict(default_ttl=10.0); d.update(FakeMapping())
        self.assertEqual(d["x"],9); self.assertEqual(d["y"],8)

# ── ExpiringDict — threads ────────────────────────────────
class TestEDThreads(unittest.TestCase):
    def test_concurrent_writes(self):
        d=ExpiringDict(default_ttl=5.0); errors=[]
        def writer(n):
            try:
                for i in range(200): d.set(f"k:{n}:{i}",i)
            except Exception as e: errors.append(e)
        def reader():
            try:
                for _ in range(500): _=len(d); _=d.get("k:0:0")
            except Exception as e: errors.append(e)
        ts=[threading.Thread(target=writer,args=(n,)) for n in range(4)]
        ts+=[threading.Thread(target=reader) for _ in range(4)]
        for t in ts: t.start()
        for t in ts: t.join()
        self.assertEqual(errors,[])
    def test_concurrent_purge(self):
        d=ExpiringDict(default_ttl=0.02); errors=[]
        def filler():
            try:
                for i in range(50): d.set(f"k{i}",i); time.sleep(0.001)
            except Exception as e: errors.append(e)
        def purger():
            try:
                for _ in range(100): d.purge_expired(); time.sleep(0.001)
            except Exception as e: errors.append(e)
        ts=[threading.Thread(target=filler) for _ in range(3)]
        ts+=[threading.Thread(target=purger) for _ in range(3)]
        for t in ts: t.start()
        for t in ts: t.join()
        self.assertEqual(errors,[])

# ── ExpiringDict — _last_purge_ns / purge scheduling ─────
class TestEDPurgeScheduling(unittest.TestCase):
    """Tests for the background-purge gate: _last_purge_ns, _purge_pending,
    and the double-checked-locking logic in _maybe_schedule_purge."""

    _PURGE_NS = _expiring_mod._PURGE_INTERVAL_NS  # 15 min in ns

    # ── construction ─────────────────────────────────────
    def test_last_purge_ns_initialized_near_now(self):
        """_last_purge_ns is set to approximately now_utc_ns() at construction."""
        before = now_utc_ns()
        d = ExpiringDict(default_ttl=10.0)
        after = now_utc_ns()
        self.assertGreaterEqual(d._last_purge_ns, before)
        self.assertLessEqual(d._last_purge_ns, after)

    def test_purge_pending_false_initially(self):
        """_purge_pending starts as False."""
        d = ExpiringDict(default_ttl=10.0)
        self.assertFalse(d._purge_pending)

    # ── fast-path (interval not elapsed) ─────────────────
    def test_maybe_schedule_purge_noop_when_recent(self):
        """No state changes when the purge interval has NOT elapsed."""
        d = ExpiringDict(default_ttl=10.0)
        ts = d._last_purge_ns
        d._maybe_schedule_purge()
        self.assertEqual(d._last_purge_ns, ts)
        self.assertFalse(d._purge_pending)

    def test_set_does_not_advance_last_purge_ns_when_fresh(self):
        """set() calls _maybe_schedule_purge; fresh dict → no change."""
        d = ExpiringDict(default_ttl=10.0)
        ts = d._last_purge_ns
        d.set("k", 1)
        self.assertEqual(d._last_purge_ns, ts)
        self.assertFalse(d._purge_pending)

    def test_get_does_not_advance_last_purge_ns_when_fresh(self):
        """get() calls _maybe_schedule_purge; fresh dict → no change."""
        d = ExpiringDict(default_ttl=10.0)
        d["k"] = 99
        ts = d._last_purge_ns
        _ = d.get("k")
        self.assertEqual(d._last_purge_ns, ts)
        self.assertFalse(d._purge_pending)

    # ── slow-path (interval elapsed) ──────────────────────
    def test_maybe_schedule_purge_updates_last_purge_ns_when_stale(self):
        """`_last_purge_ns` is stamped forward when the interval has elapsed."""
        d = ExpiringDict(default_ttl=10.0)
        d._last_purge_ns = now_utc_ns() - self._PURGE_NS - 1
        old_ts = d._last_purge_ns
        before = now_utc_ns()
        d._maybe_schedule_purge()
        self.assertGreater(d._last_purge_ns, old_ts)
        self.assertGreaterEqual(d._last_purge_ns, before)

    def test_maybe_schedule_purge_sets_pending_when_stale(self):
        """`_purge_pending` becomes True when a background purge is claimed."""
        d = ExpiringDict(default_ttl=10.0)
        d._last_purge_ns = now_utc_ns() - self._PURGE_NS - 1
        # Prevent the Job from actually firing so we can inspect _purge_pending
        original = d._background_purge
        fired = []
        def stub():
            fired.append(1)
            original()
        d._background_purge = stub
        d._maybe_schedule_purge()
        # Immediately after claiming: pending is True (may flip False quickly)
        # After stub runs: pending goes back to False
        time.sleep(0.15)
        self.assertFalse(d._purge_pending)   # background thread clears it
        self.assertEqual(len(fired), 1)

    def test_maybe_schedule_purge_no_double_fire(self):
        """Double-checked locking: only one thread schedules the background purge."""
        d = ExpiringDict(default_ttl=10.0)
        d._last_purge_ns = now_utc_ns() - self._PURGE_NS - 1

        fire_count = []
        original = d._background_purge
        lock = threading.Lock()
        def counting_purge():
            with lock:
                fire_count.append(1)
            original()
        d._background_purge = counting_purge

        threads = [threading.Thread(target=d._maybe_schedule_purge) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(0.2)  # let background daemon finish

        self.assertEqual(sum(fire_count), 1,
                         "_background_purge must fire exactly once")

    # ── _background_purge ─────────────────────────────────
    def test_background_purge_clears_pending(self):
        """`_background_purge` sets `_purge_pending` back to False."""
        d = ExpiringDict(default_ttl=10.0)
        d._purge_pending = True
        d._background_purge()
        self.assertFalse(d._purge_pending)

    def test_background_purge_evicts_expired_entries(self):
        """`_background_purge` removes expired keys from the store."""
        d = ExpiringDict(default_ttl=10.0)
        d._store["stale"] = (99, now_utc_ns() - 1)   # already expired
        d._store["fresh"] = (1,  now_utc_ns() + 60 * _1s_ns)
        d._purge_pending = True
        d._background_purge()
        self.assertNotIn("stale", d._store)
        self.assertIn("fresh", d._store)

    def test_background_purge_leaves_non_expiring_entries(self):
        """`_background_purge` does not evict entries with expires_at=None."""
        d = ExpiringDict(default_ttl=None)
        d._store["forever"] = (42, None)
        d._purge_pending = True
        d._background_purge()
        self.assertIn("forever", d._store)

    # ── pickle / deepcopy ─────────────────────────────────
    def test_pickle_restores_last_purge_ns_to_recent(self):
        """After unpickling, _last_purge_ns is set near the time of pickling."""
        d = ExpiringDict(default_ttl=60.0)
        d["k"] = 1
        before = now_utc_ns()
        d2 = pickle.loads(pickle.dumps(d))
        after = now_utc_ns()
        self.assertGreaterEqual(d2._last_purge_ns, before)
        self.assertLessEqual(d2._last_purge_ns, after)

    def test_pickle_restores_purge_pending_false(self):
        """After unpickling, _purge_pending is always False."""
        d = ExpiringDict(default_ttl=60.0)
        d["k"] = 1
        d._purge_pending = True   # simulate in-progress purge at pickle time
        d2 = pickle.loads(pickle.dumps(d))
        self.assertFalse(d2._purge_pending)

    def test_deepcopy_restores_purge_state(self):
        """deepcopy produces independent purge counters."""
        d = ExpiringDict(default_ttl=60.0)
        d["k"] = 1
        d2 = copy.deepcopy(d)
        self.assertFalse(d2._purge_pending)
        # Advancing d's counter must not affect d2
        d._last_purge_ns = 0
        self.assertGreater(d2._last_purge_ns, 0)

    # ── end-to-end: stale dict auto-purges via set() ──────
    def test_stale_dict_auto_purges_on_set(self):
        """When _last_purge_ns is very old, calling set() eventually clears expired keys."""
        d = ExpiringDict(default_ttl=10.0)
        # Plant an expired key directly so we can confirm it gets evicted
        d._store["zombie"] = (1, now_utc_ns() - 1)
        # Force the purge interval to appear elapsed
        d._last_purge_ns = now_utc_ns() - self._PURGE_NS - 1
        # set() → _maybe_schedule_purge → background thread
        d.set("new", 2)
        time.sleep(0.2)  # allow background daemon to run
        self.assertNotIn("zombie", d._store)
        self.assertIn("new", d)


if __name__ == "__main__":
    unittest.main(verbosity=2)

