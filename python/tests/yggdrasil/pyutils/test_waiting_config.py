# test_waiting_config.py
import datetime as dt
import inspect
import time

import pytest

# Be resilient to project layout differences
from yggdrasil.pyutils.waiting_config import WaitingConfig, DEFAULT_WAITING_CONFIG


def test_default_instance_matches_constant():
    assert WaitingConfig.default() is DEFAULT_WAITING_CONFIG
    assert DEFAULT_WAITING_CONFIG == WaitingConfig()


def test_timeout_timedelta():
    wc = WaitingConfig(timeout=12.5)
    assert wc.timeout_timedelta == dt.timedelta(seconds=12.5)


@pytest.mark.parametrize(
    "arg, expected_timeout",
    [
        (5, 5.0),
        (5.5, 5.5),
        (dt.timedelta(seconds=7), 7.0),
    ],
)
def test_check_arg_scalar_sets_timeout(arg, expected_timeout):
    wc = WaitingConfig.check_arg(arg)
    assert isinstance(wc, WaitingConfig)
    assert wc.timeout == expected_timeout
    # other defaults in your check_arg() fallback path
    assert wc.interval == 2.0
    assert wc.backoff == 1.0
    assert wc.max_interval == 15.0


def test_check_arg_instance_returns_equivalent_config():
    wc0 = WaitingConfig(timeout=1.0, interval=3.0, backoff=2.0, max_interval=9.0)
    wc = WaitingConfig.check_arg(wc0)
    # your implementation returns a new instance; equality is what matters
    assert wc == wc0
    assert wc is not wc0


def test_check_arg_dict_timeout():
    wc = WaitingConfig.check_arg({"timeout": 11, "interval": 0.5, "backoff": 2, "max_interval": 4})
    assert wc.timeout == 11.0
    assert wc.interval == 0.5
    assert wc.backoff == 2.0
    assert wc.max_interval == 4.0


def test_check_arg_dict_deadline(monkeypatch):
    fixed_now = dt.datetime(2026, 1, 26, 12, 0, 0)

    class _FixedDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is None else fixed_now.replace(tzinfo=tz)

    # Patch the module where WaitingConfig is defined (no hard-coded module name)
    mod = inspect.getmodule(WaitingConfig)
    assert mod is not None, "Could not resolve WaitingConfig module for monkeypatching"
    monkeypatch.setattr(mod.dt, "datetime", _FixedDT)

    # IMPORTANT: create deadline as _FixedDT (since mod.dt.datetime is now _FixedDT)
    deadline = _FixedDT(2026, 1, 26, 12, 0, 10)

    wc = WaitingConfig.check_arg({"deadline": deadline})
    assert wc.timeout == 10.0


def test_check_arg_dict_deadline_and_timeout_raises():
    with pytest.raises(ValueError):
        WaitingConfig.check_arg({"deadline": dt.datetime.now(), "timeout": 1})


@pytest.mark.parametrize(
    "arg, expected_timeout",
    [
        # IMPORTANT: in Python, bool is a subclass of int.
        # Your check_arg checks (int, float, timedelta) BEFORE bool,
        # so True becomes 1.0 and False becomes 0.0 here.
        (True, 1.0),
        (False, 0.0),
    ],
)
def test_check_arg_bool_is_treated_as_int_due_to_type_order(arg, expected_timeout):
    wc = WaitingConfig.check_arg(arg)
    assert wc.timeout == expected_timeout
    assert wc.interval == 2.0
    assert wc.backoff == 1.0
    assert wc.max_interval == 15.0


def test_check_arg_kwargs_override_arg():
    wc = WaitingConfig.check_arg({"timeout": 10, "interval": 1}, timeout=3, max_interval=8)
    assert wc.timeout == 3.0
    assert wc.interval == 1.0
    assert wc.max_interval == 8.0


def test_check_arg_negative_timeout_clamped():
    wc = WaitingConfig.check_arg(timeout=-5)
    assert wc.timeout == 0.0


def test_sleep_iteration_negative_raises():
    wc = WaitingConfig(timeout=10, interval=1, backoff=1, max_interval=10)
    with pytest.raises(ValueError):
        wc.sleep(-1)


def test_sleep_interval_zero_no_sleep(monkeypatch):
    wc = WaitingConfig(timeout=10, interval=0, backoff=2, max_interval=10)

    called = {"n": 0}

    def fake_sleep(_):
        called["n"] += 1

    monkeypatch.setattr(time, "sleep", fake_sleep)
    wc.sleep(0)
    assert called["n"] == 0


def test_sleep_computes_exponential_backoff(monkeypatch):
    wc = WaitingConfig(timeout=10, interval=2, backoff=3, max_interval=0)

    seen = []

    def fake_sleep(s):
        seen.append(s)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    wc.sleep(0)  # 2 * 3^0 = 2
    wc.sleep(1)  # 2 * 3^1 = 6
    wc.sleep(2)  # 2 * 3^2 = 18

    assert seen == [2.0, 6.0, 18.0]


def test_sleep_caps_to_max_interval(monkeypatch):
    wc = WaitingConfig(timeout=10, interval=2, backoff=10, max_interval=7)

    seen = []

    def fake_sleep(s):
        seen.append(s)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    wc.sleep(0)  # 2
    wc.sleep(1)  # 20 -> cap 7
    wc.sleep(2)  # 200 -> cap 7

    assert seen == [2.0, 7.0, 7.0]


def test_sleep_with_start_caps_to_remaining(monkeypatch):
    wc = WaitingConfig(timeout=5, interval=4, backoff=1, max_interval=0)

    base = 1000.0

    def fake_time():
        return base + 3.0  # elapsed=3 => remaining=2

    seen = []

    def fake_sleep(s):
        seen.append(s)

    monkeypatch.setattr(time, "time", fake_time)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    wc.sleep(iteration=0, start=base)  # computed 4, capped to remaining 2
    assert seen == [2.0]


def test_sleep_with_start_raises_timeout_when_out_of_time(monkeypatch):
    wc = WaitingConfig(timeout=5, interval=1, backoff=1, max_interval=0)
    base = 1000.0

    def fake_time():
        return base + 5.0001  # elapsed > timeout

    monkeypatch.setattr(time, "time", fake_time)

    with pytest.raises(TimeoutError):
        wc.sleep(iteration=0, start=base)
