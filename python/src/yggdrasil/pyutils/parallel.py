"""
parallel.py

Utilities to easily parallelize function/method calls using concurrent.futures.

Core feature:
- @parallelize(...) decorator that turns a function into a "parallel map"
  over one of its arguments, executed via a concurrent.futures.Executor.

Now returns a generator instead of a list, so results are not all collected
into memory at once.

Example
-------

from concurrent.futures import ThreadPoolExecutor
from yggdrasil.pyutils.parallel import parallelize


@parallelize()  # default: ThreadPoolExecutor, arg_index=0
def square(x: int) -> int:
    return x * x

# Result is a generator:
result_iter = square(range(10))
result = list(result_iter)  # -> [0, 1, 4, 9, ...]

# For instance methods (self at index 0, data at index 1):

class Foo:
    @parallelize(arg_index=1)
    def double(self, x: int) -> int:
        return x * 2

foo = Foo()
out = list(foo.double(range(5)))  # -> [0, 2, 4, 6, 8]
"""

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

P = ParamSpec("P")
R = TypeVar("R")


def parallelize(
    executor_cls: Type[cf.Executor] = cf.ThreadPoolExecutor,
    *,
    max_workers: Optional[int] = None,
    arg_index: int = 0,
    timeout: Optional[float] = None,
    return_exceptions: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, Iterator[R]]]:
    """
    Decorator to parallelize a function/method over one iterable argument
    using a concurrent.futures.Executor.

    Returns
    -------
    A wrapper that returns an iterator (generator) of results, not a list.

    Parameters
    ----------
    executor_cls:
        Executor class to use (ThreadPoolExecutor, ProcessPoolExecutor, or
        a custom subclass of concurrent.futures.Executor).
    max_workers:
        Max workers to pass to the executor when created internally.
        If an executor is explicitly provided via `executor=` at call time,
        this is ignored.
    arg_index:
        Index of the argument that should be treated as an iterable of tasks.
        For a plain function `f(xs)`, use arg_index=0.
        For a method `obj.f(xs)`, use arg_index=1 (since self is at 0).
    timeout:
        Optional per-future timeout (seconds) passed to `Future.result()`.
        This does NOT limit total wall-clock time, only each individual task.
    return_exceptions:
        If False (default), the first exception will be raised and remaining
        futures will be cancelled (if we own the executor).
        If True, exceptions are yielded in the results stream in place of the
        value for that task.

    Call-time behaviour
    -------------------
    The decorated function returns an iterator of results, preserving the
    original order of the iterable.

    You may optionally pass an existing executor at call time:

        @parallelize()
        def work(x: int) -> int:
            return x * 2

        with ThreadPoolExecutor(max_workers=8) as ex:
            results_iter = work(range(100), executor=ex)
            results = list(results_iter)

    The executor is then NOT closed by the decorator.

    Notes
    -----
    - The argument at `arg_index` must be iterable.
    - All tasks share the same remaining args/kwargs.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, Iterator[R]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[R]:  # type: ignore[misc]
            # Basic sanity checks
            if arg_index < 0 or arg_index >= len(args):
                raise ValueError(
                    f"arg_index {arg_index} out of range for {len(args)} positional args"
                )

            iterable_arg = args[arg_index]

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
                futures: list[cf.Future[R]] = []

                # If we created the executor, manage its lifetime with a context manager.
                # If executor is external, use nullcontext so we don't close it.
                ctx = executor if owns_executor else nullcontext(executor)

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
                                # type: ignore[list-item]
                                yield e  # type: ignore[misc]
                                continue
                            # Best-effort cancel of remaining futures if we own the executor
                            if owns_executor:
                                for f2 in futures:
                                    if not f2.done():
                                        f2.cancel()
                            raise
                        else:
                            yield res

            # Return generator; execution happens when iterated
            return gen()

        return wrapper

    return decorator
