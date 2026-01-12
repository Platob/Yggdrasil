"""Parallel execution decorator utilities."""

from __future__ import annotations

import concurrent.futures as cf
from contextlib import nullcontext
from functools import wraps
from typing import (
    Callable,
    Iterator,
    Optional,
    Type,
    TypeVar,
    ParamSpec,
)

import dill

from multiprocessing.reduction import ForkingPickler

ForkingPickler.loads = dill.loads
ForkingPickler.dumps = dill.dumps

P = ParamSpec("P")
R = TypeVar("R")


def parallelize(
    executor_cls: Type[cf.Executor] = cf.ThreadPoolExecutor,
    *,
    max_workers: Optional[int] = None,
    arg_index: int = 0,
    timeout: Optional[float] = None,
    return_exceptions: bool = False,
    show_progress: bool = False,  # 👈 new flag
) -> Callable[[Callable[P, R]], Callable[P, Iterator[R]]]:
    """
    Decorator to parallelize a function/method over one iterable argument
    using a concurrent.futures.Executor.

    Returns
    -------
    A wrapper that returns an iterator (generator) of results, not a list.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, Iterator[R]]:
        """Wrap a callable to execute over an iterable using an executor."""
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[R]:  # type: ignore[misc]
            """Execute the wrapped function across items in parallel."""
            # Basic sanity checks
            if arg_index < 0 or arg_index >= len(args):
                raise ValueError(
                    f"arg_index {arg_index} out of range for {len(args)} positional args"
                )

            iterable_arg = args[arg_index]

            # Try to know total upfront for proper progress bar
            try:
                total = len(iterable_arg)  # type: ignore[arg-type]
            except TypeError:
                total = None

            try:
                iterator = iter(iterable_arg)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError(
                    f"Argument at position {arg_index} must be iterable "
                    f"for parallel execution, got {type(iterable_arg)!r}"
                )

            # Split args into prefix / iterable / suffix so we can rebuild per task
            prefix = args[:arg_index]
            suffix = args[arg_index + 1 :]

            # Allow caller to provide existing executor via kwarg
            executor: Optional[cf.Executor] = kwargs.pop("executor", None)
            owns_executor = executor is None

            if executor is None:
                executor = executor_cls(max_workers=max_workers)

            # Generator that will actually submit tasks and yield results
            def gen() -> Iterator[R]:
                """Yield results from parallel execution in input order."""
                futures: list[cf.Future[R]] = []
                processed = 0  # for progress

                # If we created the executor, manage its lifetime with a context manager.
                # If executor is external, use nullcontext so we don't close it.
                ctx = executor if owns_executor else nullcontext(executor)

                def _print_progress() -> None:
                    """Emit progress output when enabled."""
                    if not show_progress:
                        return

                    nonlocal processed
                    processed += 1

                    # Known total: real progress bar
                    if total is not None and total > 0:
                        width = 40
                        frac = processed / total
                        filled = int(width * frac)
                        bar = "#" * filled + "-" * (width - filled)
                        print(
                            f"\r[{bar}] {processed}/{total} ({frac:6.1%})",
                            end="",
                            flush=True,
                        )
                        if processed == total:
                            print()  # newline at end
                    else:
                        # Unknown total: simple counter
                        print(f"\rProcessed {processed} items", end="", flush=True)

                with ctx:
                    # Submit all tasks first so they can run in parallel
                    for item in iterator:
                        call_args = (*prefix, item, *suffix)
                        futures.append(
                            executor.submit(  # type: ignore[arg-type]
                                func,
                                *call_args,
                                **kwargs,
                            )
                        )

                    # Yield results in input order
                    for fut in futures:
                        try:
                            res = fut.result(timeout=timeout)
                        except Exception as e:
                            if return_exceptions:
                                _print_progress()
                                # type: ignore[list-item]
                                yield e  # type: ignore[misc]
                                continue
                            # Best-effort cancel of remaining futures if we own the executor
                            if owns_executor:
                                for f2 in futures:
                                    if not f2.done():
                                        f2.cancel()
                            # ensure progress line ends cleanly
                            if show_progress:
                                print()
                            raise
                        else:
                            _print_progress()
                            yield res

                # if we printed a counter with unknown total, finish with newline
                if show_progress and (total is None or total == 0):
                    print()

            # Return generator; execution happens when iterated
            return gen()

        return wrapper

    return decorator
