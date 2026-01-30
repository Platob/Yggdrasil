import datetime as dt
import time
from dataclasses import dataclass
from typing import Optional, Union

__all__ = ["WaitingConfig", "WaitingConfigArg"]


def _safe_seconds_tick(ticks: Union[int, float, dt.timedelta]):
    if isinstance(ticks, dt.timedelta):
        return ticks.total_seconds()
    return ticks


DEFAULT_TIMEOUT_TICKS = float(20 * 60) # 20 minutes
WaitingConfigArg = Union["WaitingConfig", dict, int, float, dt.datetime, bool]


@dataclass(frozen=True)
class WaitingConfig:
    timeout: float = DEFAULT_TIMEOUT_TICKS
    interval: float = 2.0
    backoff: float = 1.0
    max_interval: float = 10.0

    @property
    def timeout_timedelta(self) -> dt.timedelta:
        return dt.timedelta(seconds=self.timeout)

    @classmethod
    def default(cls):
        return DEFAULT_WAITING_CONFIG

    @staticmethod
    def _to_seconds(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dt.timedelta):
            return float(value.total_seconds())
        if isinstance(value, (int, float)):
            return float(value)
        raise TypeError(f"Expected seconds as int/float/timedelta, got {type(value)!r}")

    @staticmethod
    def _deadline_to_timeout(deadline: dt.datetime) -> float:
        if not isinstance(deadline, dt.datetime):
            raise TypeError(f"deadline must be datetime, got {type(deadline)!r}")
        now = dt.datetime.now(tz=deadline.tzinfo) if deadline.tzinfo else dt.datetime.now()
        return (deadline - now).total_seconds()

    @classmethod
    def check_arg(
        cls,
        arg: Optional[WaitingConfigArg] = None,
        timeout: Optional[Union[int, float, dt.timedelta]] = None,
        interval: Optional[Union[int, float, dt.timedelta]] = None,
        backoff: Optional[Union[int, float, dt.timedelta]] = None,
        max_interval: Optional[Union[int, float, dt.timedelta]] = None,
    ) -> Optional["WaitingConfig"]:
        base_timeout: Optional[float] = None
        base_interval: Optional[float] = None
        base_backoff: Optional[float] = None
        base_max_interval: Optional[float] = None

        if arg is not None:
            if isinstance(arg, cls):
                if timeout is None and interval is None and backoff is None and max_interval is None:
                    return arg

                base_timeout = arg.timeout
                base_interval = arg.interval
                base_backoff = arg.backoff
                base_max_interval = arg.max_interval

            elif isinstance(arg, bool):
                base_timeout = DEFAULT_TIMEOUT_TICKS if arg else 0.0
                base_interval = 2.0
                base_backoff = 2.0
                base_max_interval = 15.0

            elif isinstance(arg, (int, float, dt.timedelta)):
                base_timeout = cls._to_seconds(arg)

            elif isinstance(arg, dt.datetime):
                base_timeout = float(cls._deadline_to_timeout(arg))

            elif isinstance(arg, dict):
                if "deadline" in arg and "timeout" in arg:
                    raise ValueError("Provide only one of 'deadline' or 'timeout' in WaitingOptions dict.")

                if "deadline" in arg and arg["deadline"] is not None:
                    base_timeout = float(cls._deadline_to_timeout(arg["deadline"]))
                else:
                    base_timeout = cls._to_seconds(arg.get("timeout"))

                base_interval = cls._to_seconds(arg.get("interval"))
                base_backoff = cls._to_seconds(arg.get("backoff"))
                base_max_interval = cls._to_seconds(arg.get("max_interval"))

            else:
                raise TypeError(f"Unsupported WaitingOptions arg type: {type(arg)!r}")

        # explicit kwargs win
        final_timeout = cls._to_seconds(timeout) if timeout is not None else base_timeout
        final_interval = cls._to_seconds(interval) if interval is not None else base_interval
        final_backoff = cls._to_seconds(backoff) if backoff is not None else base_backoff
        final_max_interval = cls._to_seconds(max_interval) if max_interval is not None else base_max_interval

        # defaults to match non-Optional signature
        if final_timeout is None:
            final_timeout = 0.0
        elif final_timeout < 0:
            final_timeout = 0.0

        if final_interval is None:
            final_interval = 2.0

        if final_backoff is None:
            final_backoff = 2.0
        elif final_backoff < 1:
            final_backoff = 2.0

        if final_max_interval is None:
            final_max_interval = 10.0

        return cls(
            timeout=float(final_timeout),
            interval=float(final_interval),
            backoff=float(final_backoff),
            max_interval=float(final_max_interval),
        )

    def sleep(self, iteration: int, start: float | None = None) -> None:
        """
        iteration is 0-based (first wait => iteration=0)

        - interval == 0 => no sleep
        - backoff >= 1 => interval * backoff**iteration
        - max_interval == 0 => no cap, else cap sleep to max_interval
        - if start is provided and timeout > 0:
            * raise TimeoutError if already out of time
            * cap sleep so we don't oversleep past timeout
        """
        if iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {iteration}")

        if self.interval == 0:
            return

        sleep_s = self.interval * (self.backoff ** int(iteration))

        if self.max_interval > 0:
            sleep_s = min(sleep_s, self.max_interval)

        if sleep_s <= 0:
            return

        if start is not None and self.timeout > 0:
            elapsed = time.time() - float(start)
            remaining = self.timeout - elapsed
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting after {self.timeout:.3f}s")
            sleep_s = min(sleep_s, remaining)

        if sleep_s <= 0:
            return

        time.sleep(sleep_s)


DEFAULT_WAITING_CONFIG = WaitingConfig()