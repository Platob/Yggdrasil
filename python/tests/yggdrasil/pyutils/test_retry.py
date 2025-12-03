# test_retry.py

import pytest

from yggdrasil.pyutils.retry import retry, retry_fixed, random_jitter


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
