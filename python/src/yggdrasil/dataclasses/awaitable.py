from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.enums.state import State

__all__ = ["Awaitable"]


class Awaitable(ABC):
    _state: State = State.IDLE
    _attempts: int = 0

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

    def _wait(
        self,
        wait: WaitingConfig,
        raise_error: bool = True,
    ) -> "Awaitable":
        start = time.time()
        iteration = 0
        while True:
            self._poll()
            if self.is_done:
                if self.is_failed and self.retryable and not wait.is_expired(start):
                    if wait.max_attempts is not None and self._attempts >= wait.max_attempts:
                        pass
                    else:
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

    def __repr__(self) -> str:
        return f"<{type(self).__name__} state={self._state}>"
