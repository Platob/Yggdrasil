# test_retry.py

import importlib
import pytest

from yggdrasil.pyutils.retry import retry, retry_fixed, random_jitter


def _get_retry_module():
    """Return the actual module object where `retry` is defined."""
    return importlib.import_module(retry.__module__)


def test_retry_success_after_failures_sync():
    calls = {"n": 0}

    @retry(tries=4, delay=0, backoff=1)  # no sleeping in tests
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert calls["n"] == 3  # failed twice, succeeded on third


def test_retry_exhaust_raises_sync():
    calls = {"n": 0}

    @retry(tries=3, delay=0, backoff=1)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        always_fails()

    # 3 tries total
    assert calls["n"] == 3


def test_retry_reraise_false_returns_none():
    calls = {"n": 0}

    @retry(tries=3, delay=0, backoff=1, reraise=False)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    result = always_fails()
    assert result is None
    assert calls["n"] == 3


def test_retry_with_specific_exceptions_only():
    calls = {"n": 0}

    @retry(exceptions=(ValueError,), tries=3, delay=0, backoff=1)
    def mixed_failures():
        calls["n"] += 1
        if calls["n"] == 1:
            # Not in exceptions -> should not retry
            raise RuntimeError("unexpected")
        raise ValueError("expected")

    # First call raises RuntimeError and should NOT be retried
    with pytest.raises(RuntimeError):
        mixed_failures()
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_success_after_failures_async():
    calls = {"n": 0}

    @retry(tries=4, delay=0, backoff=1)
    async def async_flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"

    result = await async_flaky()
    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_async_exhaust_raises():
    calls = {"n": 0}

    @retry(tries=3, delay=0, backoff=1)
    async def async_always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await async_always_fails()

    assert calls["n"] == 3


def test_retry_fixed_behavior():
    calls = {"n": 0}

    @retry_fixed(tries=4, delay=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 4:
            raise ValueError("boom")
        return "ok"

    result = flaky()
    assert result == "ok"
    assert calls["n"] == 4  # should use all attempts


def test_random_jitter_range():
    jitter = random_jitter(scale=0.2)
    base_delay = 10.0

    # Run multiple times to make sure the jitter stays within expected bounds
    for _ in range(100):
        j = jitter(base_delay)
        # With scale=0.2 -> +/- 20%
        assert 8.0 <= j <= 12.0


def test_random_jitter_zero_delay_passthrough():
    jitter = random_jitter(scale=0.5)
    assert jitter(0.0) == 0.0
    assert jitter(-1.0) == -1.0


# -----------------
# Timeout behaviour
# -----------------


class FakeMonotonic:
    """
    Simple fake monotonic clock that increases by step on each call.
    Lets us test timeout logic without real sleeping.
    """

    def __init__(self, start: float = 0.0, step: float = 1.0) -> None:
        self.value = start
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def test_retry_timeout_stops_before_max_tries_sync(monkeypatch):
    calls = {"n": 0}
    fake_clock = FakeMonotonic(start=0.0, step=1.0)
    retry_module = _get_retry_module()

    # Patch time.monotonic inside the actual retry module
    monkeypatch.setattr(retry_module.time, "monotonic", fake_clock, raising=True)

    @retry(tries=5, delay=0, backoff=1, timeout=0.5)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    # timeout is checked after the first failure:
    # start_time = 1.0, after first failure elapsed = 2.0 - 1.0 = 1.0 >= 0.5
    with pytest.raises(RuntimeError):
        always_fails()

    # Should stop after first attempt due to timeout, not use all 5 tries
    assert calls["n"] == 1


def test_retry_timeout_reraise_false_returns_none_sync(monkeypatch):
    calls = {"n": 0}
    fake_clock = FakeMonotonic(start=0.0, step=1.0)
    retry_module = _get_retry_module()
    monkeypatch.setattr(retry_module.time, "monotonic", fake_clock, raising=True)

    @retry(tries=5, delay=0, backoff=1, timeout=0.5, reraise=False)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    result = always_fails()

    assert result is None
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_timeout_stops_before_max_tries_async(monkeypatch):
    calls = {"n": 0}
    fake_clock = FakeMonotonic(start=0.0, step=1.0)
    retry_module = _get_retry_module()
    monkeypatch.setattr(retry_module.time, "monotonic", fake_clock, raising=True)

    @retry(tries=5, delay=0, backoff=1, timeout=0.5)
    async def async_always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await async_always_fails()

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retry_timeout_reraise_false_returns_none_async(monkeypatch):
    calls = {"n": 0}
    fake_clock = FakeMonotonic(start=0.0, step=1.0)
    retry_module = _get_retry_module()
    monkeypatch.setattr(retry_module.time, "monotonic", fake_clock, raising=True)

    @retry(tries=5, delay=0, backoff=1, timeout=0.5, reraise=False)
    async def async_always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    result = await async_always_fails()

    assert result is None
    assert calls["n"] == 1
