from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.enums.state import State

__all__ = ["Awaitable"]


class Awaitable(ABC):
    _state: State = State.IDLE

    @property
    def state(self) -> State:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self._state = State.from_(value)

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

    @abstractmethod
    def _poll(self) -> None:
        ...

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
        self._start()
        if wait is not False:
            self.wait(wait=wait, raise_error=raise_error)
        return self

    @abstractmethod
    def _start(self) -> None:
        ...

    def wait(
        self,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Awaitable":
        if wait is False:
            self._poll()
            if self.is_done and self.is_failed and raise_error:
                self.raise_for_status()
            return self

        wc = WaitingConfig.from_(wait)
        start = time.time()
        iteration = 0
        while True:
            self._poll()
            if self.is_done:
                if self.is_failed and raise_error:
                    self.raise_for_status()
                return self
            if wc.is_expired(start):
                if raise_error:
                    raise TimeoutError(
                        f"{type(self).__name__} timed out after {wc.timeout:.1f}s "
                        f"(state={self._state})"
                    )
                return self
            wc.sleep(iteration, start)
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
            self.wait(wait=wait, raise_error=raise_error)
        return self

    def _cancel(self) -> None:
        self._state = State.CANCELED

    def raise_for_status(self) -> None:
        if self.is_failed:
            raise RuntimeError(
                f"{type(self).__name__} {self._state}: {self._state.name}"
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} state={self._state}>"
