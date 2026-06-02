from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.enums.state import State

__all__ = ["Awaitable", "AwaitableBatch"]


class Awaitable(ABC):
    _state: State = State.IDLE
    _attempts: int = 0

    @property
    def _sleeper(self) -> threading.Event:
        try:
            return self.__dict__["_sleeper"]
        except KeyError:
            evt = threading.Event()
            evt.set()
            self.__dict__["_sleeper"] = evt
            return evt

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self._state = State.from_(value)

    @property
    def attempts(self) -> int:
        return self._attempts

    @property
    def is_idle(self) -> bool:
        return self._state.is_idle

    @property
    def is_active(self) -> bool:
        return self._state.is_active

    @property
    def is_running(self) -> bool:
        return self._state.is_running

    @property
    def is_done(self) -> bool:
        return self._state.is_done

    @property
    def is_succeeded(self) -> bool:
        return self._state.is_succeeded

    @property
    def is_failed(self) -> bool:
        return self._state.is_failed

    @property
    def is_paused(self) -> bool:
        return not self._sleeper.is_set()

    @property
    def is_canceled(self) -> bool:
        return self._state.is_canceled

    @property
    def started(self) -> bool:
        return self._state.is_started

    @property
    def retryable(self) -> bool:
        return False

    @abstractmethod
    def _poll(self) -> None:
        ...

    @abstractmethod
    def _start(self) -> None:
        ...

    @abstractmethod
    def _error_for_status(self) -> BaseException | None:
        ...

    def _cancel(self) -> None:
        self._state = State.CANCELED
        self._sleeper.set()

    def _pause(self) -> None:
        self._sleeper.clear()

    def pause(
        self,
        *,
        wait: WaitingConfigArg = False,
        raise_error: bool = True,
    ) -> "Awaitable":
        if not self.is_active:
            return self
        self._pause()
        if wait is not False:
            wc = WaitingConfig.from_(wait)
            self._wait(wc, raise_error=raise_error)
        return self

    def _resume(self) -> None:
        self._sleeper.set()

    def resume(self) -> "Awaitable":
        if not self.is_paused:
            return self
        self._resume()
        return self

    @property
    def error(self) -> BaseException | None:
        if self.is_failed:
            return self._error_for_status()
        return None

    def raise_for_status(self) -> None:
        err = self.error
        if err is not None:
            raise err

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Awaitable":
        if self.started and not reset:
            return self
        if reset and self.started and not self.is_done:
            self.cancel(wait=False, raise_error=False)
        self._state = State.PENDING
        self._attempts += 1
        self._start()
        if wait is not False:
            wc = WaitingConfig.from_(wait)
            self._wait(wc, raise_error=raise_error)
        return self

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Awaitable":
        if wait is False:
            self._poll()
            if self.is_failed and raise_error:
                self.raise_for_status()
            return self
        wc = WaitingConfig.from_(wait)
        return self._wait(wc, raise_error=raise_error)

    def _retry_backoff(self, wait: "WaitingConfig", start: float) -> float:
        """Seconds to sleep before re-submitting a retryable failure.

        Zero by default — most awaitables retry promptly. Subclasses whose
        retries contend (e.g. a warehouse statement hitting
        ``DELTA_CONCURRENT_APPEND`` under N concurrent writers) override
        this with a jittered backoff so retries don't collide in lockstep.
        """
        del wait, start
        return 0.0

    @staticmethod
    def _jittered_backoff(
        attempts: int, wait: "WaitingConfig", start: float, *, cap: float = 5.0,
    ) -> float:
        """Exponential backoff with full jitter, bounded by *cap* and the
        remaining wait budget — the building block subclasses use in
        :meth:`_retry_backoff`."""
        backoff = min(0.1 * (2 ** max(0, attempts - 1)), cap)
        backoff += random.uniform(0.0, backoff or 0.1)
        if wait.timeout > 0:
            backoff = min(backoff, max(0.0, wait.timeout - (time.time() - start)))
        return max(0.0, backoff)

    def _wait(
        self,
        wait: WaitingConfig,
        raise_error: bool = True,
    ) -> "Awaitable":
        start = time.time()
        iteration = 0
        next_log_at = 120.0
        while True:
            if not self._sleeper.is_set():
                if wait.timeout > 0:
                    remaining = wait.timeout - (time.time() - start)
                    if remaining > 0:
                        self._sleeper.wait(timeout=remaining)
                else:
                    self._sleeper.wait()
            self._poll()
            if not self.is_done and logger.isEnabledFor(logging.INFO):
                elapsed = time.time() - start
                if elapsed >= next_log_at:
                    logger.info(
                        "%s still waiting after %.0fs (state=%s)",
                        type(self).__name__, elapsed, self._state,
                    )
                    next_log_at = elapsed + 900.0
            if self.is_done:
                if self.is_failed and self.retryable and not wait.is_expired(start):
                    if wait.max_attempts is not None and self._attempts >= wait.max_attempts:
                        pass
                    else:
                        logger.warning(
                            "%s retry %d: %s",
                            type(self).__name__, self._attempts, self.error,
                        )
                        delay = self._retry_backoff(wait, start)
                        if delay > 0:
                            time.sleep(delay)
                        self.start(reset=True, wait=False)
                        iteration = 0
                        continue
                if self.is_failed and raise_error:
                    if wait.is_expired(start):
                        raise TimeoutError(
                            f"{type(self).__name__} timed out after {wait.timeout:.1f}s "
                            f"(state={self._state})"
                        )
                    self.raise_for_status()
                return self
            if wait.is_expired(start):
                if raise_error:
                    raise TimeoutError(
                        f"{type(self).__name__} timed out after {wait.timeout:.1f}s "
                        f"(state={self._state})"
                    )
                return self
            wait.sleep(iteration, start)
            iteration += 1

    def cancel(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Awaitable":
        if self.is_done or self.is_idle:
            return self
        self._cancel()
        if wait is not False:
            wc = WaitingConfig.from_(wait)
            self._wait(wc, raise_error=raise_error)
        return self

    @classmethod
    def as_completed(
        cls,
        awaitables: Iterable["Awaitable"],
        *,
        wait: WaitingConfigArg = True,
    ) -> Iterator["Awaitable"]:
        if wait is False:
            for a in awaitables:
                a.wait(wait=False, raise_error=False)
                if a.is_done:
                    yield a
            return
        pending = list(awaitables)
        wc = WaitingConfig.from_(wait)
        start = time.time()
        iteration = 0
        while pending:
            still_pending = []
            for a in pending:
                a.wait(wait=False, raise_error=False)
                if a.is_done:
                    yield a
                else:
                    still_pending.append(a)
            pending = still_pending
            if pending:
                if wc.is_expired(start):
                    return
                wc.sleep(iteration, start)
                iteration += 1

    # ── async / await ────────────────────────────────────────────────────

    def __await__(self):
        return self._async_wait().__await__()

    async def _async_wait(
        self,
        wait: WaitingConfig | None = None,
        raise_error: bool = True,
    ) -> "Awaitable":
        import asyncio
        if wait is None:
            wait = WaitingConfig.from_(True)
        start = time.time()
        iteration = 0
        next_log_at = 120.0
        loop = asyncio.get_running_loop()
        while True:
            if not self._sleeper.is_set():
                if wait.timeout > 0:
                    remaining = wait.timeout - (time.time() - start)
                    if remaining > 0:
                        await loop.run_in_executor(
                            None, self._sleeper.wait, remaining,
                        )
                else:
                    await loop.run_in_executor(None, self._sleeper.wait)
            self._poll()
            if not self.is_done and logger.isEnabledFor(logging.INFO):
                elapsed = time.time() - start
                if elapsed >= next_log_at:
                    logger.info(
                        "%s still waiting after %.0fs (state=%s)",
                        type(self).__name__, elapsed, self._state,
                    )
                    next_log_at = elapsed + 900.0
            if self.is_done:
                if self.is_failed and self.retryable and not wait.is_expired(start):
                    if wait.max_attempts is not None and self._attempts >= wait.max_attempts:
                        pass
                    else:
                        logger.warning(
                            "%s retry %d: %s",
                            type(self).__name__, self._attempts, self.error,
                        )
                        delay = self._retry_backoff(wait, start)
                        if delay > 0:
                            time.sleep(delay)
                        self.start(reset=True, wait=False)
                        iteration = 0
                        continue
                if self.is_failed and raise_error:
                    if wait.is_expired(start):
                        raise TimeoutError(
                            f"{type(self).__name__} timed out after {wait.timeout:.1f}s "
                            f"(state={self._state})"
                        )
                    self.raise_for_status()
                return self
            if wait.is_expired(start):
                if raise_error:
                    raise TimeoutError(
                        f"{type(self).__name__} timed out after {wait.timeout:.1f}s "
                        f"(state={self._state})"
                    )
                return self
            delay = wait.get_delay(iteration, start)
            if delay > 0:
                await asyncio.sleep(delay)
            iteration += 1

    def __repr__(self) -> str:
        return f"<{type(self).__name__} state={self._state}>"


class AwaitableBatch(Awaitable):

    @abstractmethod
    def awaitables(self) -> Iterator[Awaitable]:
        ...

    @property
    def max_concurrency(self) -> int:
        return 1

    def _error_for_status(self) -> BaseException | None:
        children = getattr(self, "_children", ())
        errors = [c.error for c in children if c.is_failed and c.error is not None]
        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        try:
            return BaseExceptionGroup(
                f"{type(self).__name__}: {len(errors)} failures", errors
            )
        except NameError:
            return RuntimeError(
                f"{type(self).__name__}: {len(errors)} failures"
            )

    def _start(self) -> None:
        self._children: list[Awaitable] = list(self.awaitables())
        if not self._children:
            self._state = State.SUCCEEDED
            return
        self._state = State.RUNNING
        concurrency = self.max_concurrency
        if concurrency <= 1:
            self._seq_index = 0
            self._children[0].start(wait=False)
        else:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=concurrency)
            for child in self._children:
                self._executor.submit(child.start, raise_error=False)

    def _poll(self) -> None:
        if self.is_done:
            return
        concurrency = self.max_concurrency
        if concurrency <= 1:
            self._poll_sequential()
        else:
            self._poll_concurrent()

    def _poll_sequential(self) -> None:
        if self._seq_index >= len(self._children):
            return
        current = self._children[self._seq_index]
        current.wait(wait=False, raise_error=False)
        if current.is_done:
            self._seq_index += 1
            if self._seq_index < len(self._children):
                self._children[self._seq_index].start(wait=False)
            else:
                self._resolve()

    def _poll_concurrent(self) -> None:
        if all(c.is_done for c in self._children):
            self._resolve()

    def _resolve(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)
        if any(c.is_failed for c in self._children):
            self._state = State.FAILED
        else:
            self._state = State.SUCCEEDED

    def _cancel(self) -> None:
        for child in getattr(self, "_children", ()):
            if not child.is_done and not child.is_idle:
                child.cancel(wait=False, raise_error=False)
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        self._state = State.CANCELED
        self._sleeper.set()

    def _pause(self) -> None:
        for child in getattr(self, "_children", ()):
            if child.is_active:
                child.pause()
        self._sleeper.clear()

    def _resume(self) -> None:
        for child in getattr(self, "_children", ()):
            if child.is_paused:
                child.resume()
        self._sleeper.set()
