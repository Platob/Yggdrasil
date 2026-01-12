"""
retry.py

Lightweight, dependency-free retry decorators for sync and async Python code.

Features:
- Configurable max attempts
- Fixed or exponential backoff
- Optional jitter
- Optional total timeout
- Works for both sync and async functions
- Optional logging hook
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    TypeVar,
    ParamSpec,
    Union,
    Tuple,
)

P = ParamSpec("P")
R = TypeVar("R")

ExceptionTypes = Union[Type[BaseException], Tuple[Type[BaseException], ...]]


def _ensure_exception_tuple(exc: ExceptionTypes) -> Tuple[Type[BaseException], ...]:
    """Normalize exception inputs into a tuple of exception classes.

    Args:
        exc: Exception class or tuple of classes.

    Returns:
        Tuple of exception classes.
    """
    if isinstance(exc, type) and issubclass(exc, BaseException):
        return (exc,)
    return tuple(exc)


def retry(
    exceptions: ExceptionTypes = Exception,
    tries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    max_delay: Optional[float] = None,
    jitter: Optional[Callable[[float], float]] = None,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    timeout: Optional[float] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Retry decorator that works for both sync and async functions.

    Parameters
    ----------
    exceptions:
        Exception class or tuple of exception classes to catch and retry on.
    tries:
        Total number of attempts (including the first one).
    delay:
        Initial sleep delay between retries in seconds.
    backoff:
        Multiplier for exponential backoff (e.g. 2.0 doubles the delay each retry).
    max_delay:
        Upper bound for the delay. If None, delay is unbounded.
    jitter:
        Optional function taking the current delay and returning adjusted delay
        (e.g. lambda d: d * random.uniform(0.8, 1.2)).
    logger:
        Optional logger for debug/info messages.
    reraise:
        If True, re-raise the last exception after exhausting retries or timeout.
        If False, returns None after all retries fail or timeout is hit.
    timeout:
        Optional max total time in seconds across all attempts. The timeout is
        checked after each failed attempt. In-flight attempts are not forcibly
        cancelled; we just stop scheduling new retries when the timeout is hit.

    Usage
    -----
    @retry(exceptions=(ValueError,), tries=5, delay=0.1, backoff=2)
    def flaky():
        ...

    @retry(tries=5, timeout=2.0)
    async def async_flaky():
        ...
    """

    if tries < 1:
        raise ValueError("tries must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if backoff < 1:
        raise ValueError("backoff must be >= 1")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be > 0 when provided")

    exc_types = _ensure_exception_tuple(exceptions)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        """Wrap a callable with retry behavior.

        Args:
            func: Callable to wrap.

        Returns:
            Wrapped callable with retry semantics.
        """
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
                """Execute an async callable with retry behavior.

                Args:
                    *args: Positional args for the callable.
                    **kwargs: Keyword args for the callable.

                Returns:
                    Callable result.
                """
                _delay = delay
                attempt = 1
                start_time = time.monotonic() if timeout is not None else None

                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exc_types as e:  # type: ignore[misc]
                        # Check timeout after a failed attempt
                        if start_time is not None:
                            elapsed = time.monotonic() - start_time
                            if elapsed >= timeout:  # type: ignore[operator]
                                if logger:
                                    logger.error(
                                        "Retry timeout (%.3fs) reached on async %s "
                                        "after %s attempts: %r",
                                        elapsed,
                                        func.__name__,
                                        attempt,
                                        e,
                                    )
                                if reraise:
                                    raise
                                return None  # type: ignore[return-value]

                        if attempt >= tries:
                            if logger:
                                logger.error(
                                    "Retry failed after %s attempts on async %s: %r",
                                    attempt,
                                    func.__name__,
                                    e,
                                )
                            if reraise:
                                raise
                            return None  # type: ignore[return-value]

                        if logger:
                            logger.warning(
                                "Retry %s/%s for async %s after %r",
                                attempt,
                                tries,
                                func.__name__,
                                e,
                            )

                        sleep_for = _delay
                        if jitter:
                            sleep_for = jitter(sleep_for)
                        if max_delay is not None:
                            sleep_for = min(sleep_for, max_delay)

                        if sleep_for > 0:
                            await asyncio.sleep(sleep_for)

                        _delay *= backoff
                        attempt += 1

            return async_wrapper  # type: ignore[return-value]

        else:

            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
                """Execute a sync callable with retry behavior.

                Args:
                    *args: Positional args for the callable.
                    **kwargs: Keyword args for the callable.

                Returns:
                    Callable result.
                """
                _delay = delay
                attempt = 1
                start_time = time.monotonic() if timeout is not None else None

                while True:
                    try:
                        return func(*args, **kwargs)
                    except exc_types as e:  # type: ignore[misc]
                        # Check timeout after a failed attempt
                        if start_time is not None:
                            elapsed = time.monotonic() - start_time
                            if elapsed >= timeout:  # type: ignore[operator]
                                if logger:
                                    logger.error(
                                        "Retry timeout (%.3fs) reached on %s "
                                        "after %s attempts: %r",
                                        elapsed,
                                        func.__name__,
                                        attempt,
                                        e,
                                    )
                                if reraise:
                                    raise
                                return None  # type: ignore[return-value]

                        if attempt >= tries:
                            if logger:
                                logger.error(
                                    "Retry failed after %s attempts on %s: %r",
                                    attempt,
                                    func.__name__,
                                    e,
                                )
                            if reraise:
                                raise
                            return None  # type: ignore[return-value]

                        if logger:
                            logger.warning(
                                "Retry %s/%s for %s after %r",
                                attempt,
                                tries,
                                func.__name__,
                                e,
                            )

                        sleep_for = _delay
                        if jitter:
                            sleep_for = jitter(sleep_for)
                        if max_delay is not None:
                            sleep_for = min(sleep_for, max_delay)

                        if sleep_for > 0:
                            time.sleep(sleep_for)

                        _delay *= backoff
                        attempt += 1

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def retry_fixed(
    exceptions: ExceptionTypes = Exception,
    tries: int = 3,
    delay: float = 0.5,
    **kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Convenience wrapper: fixed delay, no backoff.

    Example
    -------
    @retry_fixed(tries=5, delay=1.0)
    def call_api():
        ...
    """
    return retry(
        exceptions=exceptions,
        tries=tries,
        delay=delay,
        backoff=1.0,
        **kwargs,
    )


def random_jitter(scale: float = 0.1) -> Callable[[float], float]:
    """
    Returns a jitter function to add +/- (scale * delay) randomness.

    Example
    -------
    @retry(jitter=random_jitter(0.2))
    def flaky():
        ...
    """

    def _jitter(d: float) -> float:
        """Apply random jitter to a delay value.

        Args:
            d: Base delay value.

        Returns:
            Jittered delay value.
        """
        if d <= 0:
            return d
        delta = d * scale
        return d + random.uniform(-delta, delta)

    return _jitter


# Example usage / quick smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("retry-demo")

    counter = {"n": 0}

    @retry(tries=4, delay=0.1, backoff=2, logger=log, timeout=5.0)
    def flaky_function() -> str:
        """Demonstrate retry behavior for a flaky sync function.

        Returns:
            String result.
        """
        counter["n"] += 1
        if counter["n"] < 3:
            raise ValueError("boom")
        return "ok"

    print("Result:", flaky_function())

    async def main():
        """Run an async retry demonstration.

        Returns:
            None.
        """
        async_counter = {"n": 0}

        @retry(tries=4, delay=0.1, backoff=2, logger=log, timeout=5.0)
        async def async_flaky() -> str:
            """Demonstrate retry behavior for a flaky async function.

            Returns:
                String result.
            """
            async_counter["n"] += 1
            if async_counter["n"] < 3:
                raise RuntimeError("async boom")
            return "async ok"

        print("Async result:", await async_flaky())

    asyncio.run(main())
