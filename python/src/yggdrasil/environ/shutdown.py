"""
yggdrasil.environ.shutdown

Shutdown callback registry for process exit and termination signals.

Callbacks are run on:
- normal interpreter exit via atexit
- termination-related signals such as SIGINT / SIGTERM / SIGHUP / SIGBREAK
  when available on the current platform and installable from the current thread

Features:
- decorator or direct registration
- unregister support
- priority ordering
- run-once guarantee
- restores previous signal handlers on uninstall
- one failing callback does not block the others

Typical usage:

    from yggdrasil.environ.shutdown import on_shutdown, shutdown_registry

    @on_shutdown
    def cleanup():
        print("cleanup called")

    @on_shutdown(priority=100)
    def flush_metrics():
        print("flush first")

    shutdown_registry.unregister(cleanup)

For bound instance methods, unregistering only works reliably if you pass the
same bound method object back, or you keep the handle returned by register().
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Callable, Iterable, TypeVar, overload

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
Callback = Callable[..., Any]


def _callback_key(callback: Callback) -> tuple[Any, ...]:
    """
    Produce a stable identity key for callables, including bound methods.

    Plain functions:
        (<function object>,)

    Bound methods:
        ("bound_method", id(instance), function_object)

    This lets unregister(obj.method) work even though each attribute access
    produces a new bound method object.
    """
    self_obj = getattr(callback, "__self__", None)
    func_obj = getattr(callback, "__func__", None)
    if self_obj is not None and func_obj is not None:
        return ("bound_method", id(self_obj), func_obj)
    return (callback,)


@dataclass(order=True, slots=True)
class _Entry:
    sort_key: tuple[int, int] = field(init=False, repr=False)
    priority: int
    order: int
    key: tuple[Any, ...] = field(compare=False)
    callback: Callback = field(compare=False)
    pass_reason: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        # Higher priority first, earlier registration breaks ties.
        self.sort_key = (-self.priority, self.order)


class ShutdownRegistry:
    """
    Registry for cleanup callbacks triggered by atexit and termination signals.
    """

    def __init__(
        self,
        *,
        handled_signals: Iterable[int] | None = None,
        raise_on_callback_error: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[Any, ...], _Entry] = {}
        self._order = 0
        self._ran = False
        self._installed = False
        self._atexit_registered = False
        self._raise_on_callback_error = raise_on_callback_error

        requested = handled_signals if handled_signals is not None else self._default_signals()
        self._handled_signals = tuple(sig for sig in requested if self._supports_signal(sig))
        self._previous_handlers: dict[int, Any] = {}

    @staticmethod
    def _default_signals() -> tuple[int, ...]:
        candidates = [
            getattr(signal, "SIGINT", None),
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGBREAK", None),  # Windows
        ]
        return tuple(sig for sig in candidates if sig is not None)

    @staticmethod
    def _supports_signal(sig: int) -> bool:
        try:
            signal.getsignal(sig)
            return True
        except (AttributeError, OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _can_install_signal_handlers() -> bool:
        # CPython requires this from the main thread of the main interpreter.
        return threading.current_thread() is threading.main_thread()

    def install(self) -> None:
        """
        Install the atexit hook and signal handlers.

        Safe to call multiple times. If called outside the main thread, atexit is
        still registered but signal handlers are skipped.
        """
        with self._lock:
            if not self._atexit_registered:
                atexit.register(self._atexit_runner)
                self._atexit_registered = True

            if self._installed:
                return

            if self._can_install_signal_handlers():
                for sig in self._handled_signals:
                    try:
                        self._previous_handlers[sig] = signal.getsignal(sig)
                        signal.signal(sig, self._make_signal_handler(sig))
                    except (OSError, RuntimeError, ValueError):
                        logger.debug(
                            "Failed to install shutdown signal handler for %s",
                            sig,
                            exc_info=True,
                        )
            else:
                logger.debug("Skipping shutdown signal handler install outside main thread")

            self._installed = True

    def uninstall(self) -> None:
        """
        Restore previous signal handlers.

        Note: atexit registration is intentionally left in place; stdlib support for
        atexit.unregister exists in modern Python, but leaving the runner registered
        is harmless because run() is idempotent.
        """
        with self._lock:
            if not self._installed:
                return

            for sig, previous in self._previous_handlers.items():
                try:
                    signal.signal(sig, previous)
                except (OSError, RuntimeError, ValueError):
                    logger.exception("Failed to restore previous handler for signal %s", sig)

            self._previous_handlers.clear()
            self._installed = False

    @overload
    def register(
        self,
        callback: F,
        *,
        priority: int = 0,
        pass_reason: bool = False,
        install: bool = True,
    ) -> F: ...

    @overload
    def register(
        self,
        callback: None = None,
        *,
        priority: int = 0,
        pass_reason: bool = False,
        install: bool = True,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        callback: Callback | None = None,
        *,
        priority: int = 0,
        pass_reason: bool = False,
        install: bool = True,
    ):
        """
        Register a callback.

        Supports:
            registry.register(func)
            registry.register(func, priority=100)

            @registry.register
            def func(): ...

            @registry.register(priority=100)
            def func(): ...

        If pass_reason=True, the callback will receive one positional argument:
            reason: str
        """
        if callback is None:
            def decorator(fn: F) -> F:
                self.register(
                    fn,
                    priority=priority,
                    pass_reason=pass_reason,
                    install=install,
                )
                return fn

            return decorator

        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)!r}")

        key = _callback_key(callback)

        with self._lock:
            self._order += 1
            self._entries[key] = _Entry(
                priority=priority,
                order=self._order,
                key=key,
                callback=callback,
                pass_reason=pass_reason,
            )

            if install:
                self.install()

        return callback

    def unregister(self, callback: Callback) -> bool:
        """
        Unregister a callback. Returns True if removed.
        """
        key = _callback_key(callback)
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """
        Remove all registered callbacks.
        """
        with self._lock:
            self._entries.clear()

    def is_registered(self, callback: Callback) -> bool:
        key = _callback_key(callback)
        with self._lock:
            return key in self._entries

    def callbacks(self) -> tuple[Callback, ...]:
        """
        Return currently registered callbacks in execution order.
        """
        with self._lock:
            entries = sorted(self._entries.values())
            return tuple(entry.callback for entry in entries)

    def run(self, reason: str = "manual") -> None:
        """
        Run registered callbacks once.

        Subsequent calls are no-ops until reset_run_state() is called.
        """
        with self._lock:
            if self._ran:
                return
            self._ran = True
            entries = sorted(self._entries.values())

        errors: list[BaseException] = []

        for entry in entries:
            try:
                if entry.pass_reason:
                    entry.callback(reason)
                else:
                    entry.callback()
            except BaseException as exc:
                errors.append(exc)
                logger.exception(
                    "Shutdown callback failed: %r (reason=%s)",
                    entry.callback,
                    reason,
                )

        if errors and self._raise_on_callback_error:
            raise RuntimeError(f"{len(errors)} shutdown callback(s) failed") from errors[0]

    def reset_run_state(self) -> None:
        """
        Allow callbacks to run again. Mainly useful for tests.
        """
        with self._lock:
            self._ran = False

    def _atexit_runner(self) -> None:
        self.run(reason="atexit")

    def _make_signal_handler(self, sig: int) -> Callable[[int, FrameType | None], None]:
        def handler(signum: int, frame: FrameType | None) -> None:
            try:
                signame = signal.Signals(signum).name
            except ValueError:
                signame = str(signum)

            self.run(reason=f"signal:{signame}")

            previous = self._previous_handlers.get(sig, signal.SIG_DFL)

            if previous is signal.SIG_IGN:
                raise SystemExit(128 + signum)

            if previous in (signal.SIG_DFL, None):
                if signum == getattr(signal, "SIGINT", None):
                    raise KeyboardInterrupt
                raise SystemExit(128 + signum)

            if previous is handler:
                raise SystemExit(128 + signum)

            try:
                previous(signum, frame)
            except TypeError:
                # Some odd handlers may not match the full signal signature.
                try:
                    previous(signum)
                except Exception:
                    raise SystemExit(128 + signum)
            except BaseException:
                raise

            raise SystemExit(128 + signum)

        return handler


shutdown_registry = ShutdownRegistry()


@overload
def on_shutdown(callback: F) -> F: ...


@overload
def on_shutdown(
    callback: None = None,
    *,
    priority: int = 0,
    pass_reason: bool = False,
) -> Callable[[F], F]: ...


def on_shutdown(
    callback: Callback | None = None,
    *,
    priority: int = 0,
    pass_reason: bool = False,
):
    """
    Module-level decorator/helper backed by the shared registry.

    Usage:
        @on_shutdown
        def cleanup(): ...

        @on_shutdown(priority=100, pass_reason=True)
        def cleanup(reason: str): ...
    """
    return shutdown_registry.register(
        callback,
        priority=priority,
        pass_reason=pass_reason,
        install=True,
    )


def register(
    callback: Callback | None = None,
    *,
    priority: int = 0,
    pass_reason: bool = False,
):
    """
    Convenience alias for the shared registry register().
    """
    return shutdown_registry.register(
        callback,
        priority=priority,
        pass_reason=pass_reason,
        install=True,
    )


def unregister(callback: Callback) -> bool:
    """
    Convenience alias for the shared registry unregister().
    """
    return shutdown_registry.unregister(callback)


def unregister_shutdown(callback: Callback) -> bool:
    """
    Backward-compatible alias for unregister().
    """
    return shutdown_registry.unregister(callback)