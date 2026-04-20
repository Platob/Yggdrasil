"""
yggdrasil.environ.shutdown

Shutdown callback registry for process exit and termination signals.

Callbacks are run on:
- normal interpreter exit via atexit
- termination-related signals such as SIGINT / SIGTERM / SIGHUP / SIGBREAK
  when available on the current platform and installable from the current thread

Design notes
------------
- Signal handlers do NOT raise synthetic SystemExit into arbitrary stack
  frames. Instead they run callbacks, restore the previous handler, and
  re-raise the signal to the process so the OS / previous handler decides
  what happens next. This avoids injecting exceptions into finally blocks
  and C extensions.
- run() is both run-once and re-entrancy safe: a signal arriving while
  atexit callbacks are executing will NOT start a parallel run, and will
  NOT truncate the in-flight run.
- Bound methods are tracked via weakref.WeakMethod so registering an
  instance method does not extend the instance's lifetime, and does not
  suffer from id() reuse after GC.
- atexit is registered eagerly at module import so ordering relative to
  other libraries is predictable (LIFO at import time).

Typical usage
-------------

    from yggdrasil.environ.shutdown import on_shutdown, shutdown_registry

    @on_shutdown
    def cleanup():
        print("cleanup called")

    @on_shutdown(priority=100)
    def flush_metrics():
        print("flush first")

    shutdown_registry.unregister(cleanup)
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import threading
import weakref
from dataclasses import dataclass
from types import FrameType
from typing import Any, Callable, Iterable, TypeVar, overload

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
Callback = Callable[..., Any]


# ---------------------------------------------------------------------------
# Callback identity / weak references
# ---------------------------------------------------------------------------

def _is_bound_method(callback: Callback) -> bool:
    return (
        getattr(callback, "__self__", None) is not None
        and getattr(callback, "__func__", None) is not None
    )


def _callback_key(callback: Callback) -> tuple[Any, ...]:
    """
    Produce a stable identity key for callables.

    Plain functions:
        (<function object>,)

    Bound methods:
        ("bound_method", <func>, id(instance))

    id(instance) is stable while the instance is alive. Since the registry
    keeps a WeakMethod, dead entries are pruned at run time and their keys
    become irrelevant even if id() is later reused.
    """
    if _is_bound_method(callback):
        return ("bound_method", callback.__func__, id(callback.__self__))
    return (callback,)


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _Entry:
    priority: int
    order: int
    key: tuple[Any, ...]
    # Either a strong reference (plain callable) or a WeakMethod (bound method).
    # Resolve via _resolve() to get the actual callable, or None if dead.
    ref: Any
    is_weak: bool
    pass_reason: bool = False

    def _resolve(self) -> Callback | None:
        if self.is_weak:
            return self.ref()  # WeakMethod() returns None if dead
        return self.ref

    @property
    def sort_key(self) -> tuple[int, int]:
        # Higher priority first, earlier registration breaks ties.
        return (-self.priority, self.order)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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

        # Run-once + re-entrancy guards.
        self._ran = False
        self._running = False

        self._installed = False
        self._atexit_registered = False
        self.raise_on_callback_error = raise_on_callback_error

        requested = (
            handled_signals if handled_signals is not None else self._default_signals()
        )
        self._handled_signals: tuple[int, ...] = tuple(
            sig for sig in requested if self._supports_signal(sig)
        )
        self._previous_handlers: dict[int, Any] = {}

    # --- platform helpers -------------------------------------------------

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

    # --- install / uninstall ---------------------------------------------

    def install(self) -> None:
        """
        Install the atexit hook and signal handlers.

        Safe to call multiple times. If called outside the main thread, atexit
        is still registered but signal handlers are skipped.
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
                logger.debug(
                    "Skipping shutdown signal handler install outside main thread"
                )

            self._installed = True

    def uninstall(self) -> None:
        """
        Restore previous signal handlers.

        atexit registration is intentionally left in place; run() is idempotent
        so leaving the runner registered is harmless.
        """
        with self._lock:
            if not self._installed:
                return

            for sig, previous in list(self._previous_handlers.items()):
                try:
                    signal.signal(sig, previous)
                except (OSError, RuntimeError, ValueError):
                    logger.exception(
                        "Failed to restore previous handler for signal %s", sig
                    )

            self._previous_handlers.clear()
            self._installed = False

    # --- register / unregister -------------------------------------------

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

        If pass_reason=True, the callback will receive one positional argument:
            reason: str

        Bound methods are held via weakref so the registry does not extend
        the instance's lifetime.
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

        if _is_bound_method(callback):
            # When the instance dies, drop the entry automatically.
            def _on_dead(_ref: Any, _key: tuple[Any, ...] = key) -> None:
                with self._lock:
                    self._entries.pop(_key, None)

            ref: Any = weakref.WeakMethod(callback, _on_dead)
            is_weak = True
        else:
            ref = callback
            is_weak = False

        with self._lock:
            self._order += 1
            self._entries[key] = _Entry(
                priority=priority,
                order=self._order,
                key=key,
                ref=ref,
                is_weak=is_weak,
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
        Return currently live callbacks in execution order.

        Dead weak references are skipped (but not pruned here; pruning happens
        in run() or via the WeakMethod finalizer).
        """
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.sort_key)
            resolved: list[Callback] = []
            for entry in entries:
                cb = entry._resolve()
                if cb is not None:
                    resolved.append(cb)
            return tuple(resolved)

    # --- execution --------------------------------------------------------

    def run(self, reason: str = "manual") -> None:
        """
        Run registered callbacks once.

        Re-entrancy safe: if a signal arrives while this is already running,
        the nested call is a no-op and does NOT truncate the in-flight run.

        Subsequent top-level calls are also no-ops until reset_run_state() is
        called.
        """
        with self._lock:
            if self._ran or self._running:
                return
            self._running = True
            # Snapshot and resolve weakrefs under the lock.
            entries = sorted(self._entries.values(), key=lambda e: e.sort_key)
            resolved: list[tuple[_Entry, Callback]] = []
            dead_keys: list[tuple[Any, ...]] = []
            for entry in entries:
                cb = entry._resolve()
                if cb is None:
                    dead_keys.append(entry.key)
                else:
                    resolved.append((entry, cb))
            for k in dead_keys:
                self._entries.pop(k, None)

        errors: list[BaseException] = []

        try:
            for entry, cb in resolved:
                try:
                    if entry.pass_reason:
                        cb(reason)
                    else:
                        cb()
                except BaseException as exc:
                    errors.append(exc)
                    logger.exception(
                        "Shutdown callback failed: %r (reason=%s)",
                        cb,
                        reason,
                    )
        finally:
            with self._lock:
                self._ran = True
                self._running = False

        if errors and self.raise_on_callback_error:
            raise RuntimeError(
                f"{len(errors)} shutdown callback(s) failed"
            ) from errors[0]

    def reset_run_state(self) -> None:
        """
        Allow callbacks to run again. Mainly useful for tests.
        """
        with self._lock:
            self._ran = False
            self._running = False

    # --- atexit / signal plumbing ----------------------------------------

    def _atexit_runner(self) -> None:
        # Swallow everything here: raising out of atexit produces noisy
        # "Error in atexit._run_exitfuncs" output that obscures real errors.
        try:
            self.run(reason="atexit")
        except BaseException:
            logger.exception("Error running shutdown callbacks at interpreter exit")

    def _make_signal_handler(
        self, sig: int
    ) -> Callable[[int, FrameType | None], None]:
        """
        Build a signal handler that:
          1. runs registered callbacks (once, re-entrancy safe)
          2. restores the previous handler
          3. re-raises the signal to the process so the OS / previous handler
             decides the final fate

        This avoids raising synthetic SystemExit into arbitrary stack frames,
        which is the main cause of ugly "errors at end of process".
        """
        def handler(signum: int, frame: FrameType | None) -> None:
            try:
                signame = signal.Signals(signum).name
            except ValueError:
                signame = str(signum)

            try:
                self.run(reason=f"signal:{signame}")
            except BaseException:
                logger.exception(
                    "Error running shutdown callbacks for signal %s", signame
                )

            # Restore the previous handler and re-raise. On POSIX and Windows
            # (Python 3.8+) signal.raise_signal exists and delivers the signal
            # to the current process.
            previous = self._previous_handlers.get(signum, signal.SIG_DFL)

            try:
                signal.signal(signum, previous)
            except (OSError, RuntimeError, ValueError):
                logger.debug(
                    "Failed to restore previous handler for %s", signame,
                    exc_info=True,
                )

            # If the previous handler explicitly ignored the signal, we do
            # the same: do not force a termination.
            if previous is signal.SIG_IGN:
                return

            # Prefer re-raising via the OS so whatever handler is now installed
            # (default or user's previous) decides what happens.
            try:
                signal.raise_signal(signum)
                return
            except (AttributeError, OSError, ValueError):
                # Very old Python or exotic platform: fall through to a
                # best-effort exit.
                pass

            # Last-resort fallback. os._exit skips atexit (we already ran
            # callbacks above, so that's fine) and does not raise an exception
            # into user code.
            os._exit(128 + signum)

        return handler


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

shutdown_registry = ShutdownRegistry()

# Register atexit eagerly so ordering relative to other libraries is
# predictable (LIFO at import time) instead of depending on when the first
# callback happens to be registered.
shutdown_registry.install()


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