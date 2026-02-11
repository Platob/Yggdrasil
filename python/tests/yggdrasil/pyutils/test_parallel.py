# test_parallel.py

import concurrent.futures as cf
import time

import pytest

from yggdrasil.pyutils.parallel import parallelize


def test_parallelize_basic_function():
    @parallelize()
    def square(x: int) -> int:
        return x * x

    data = list(range(10))
    result_iter = square(data)

    result = list(result_iter)
    assert result == [i * i for i in data]


def test_parallelize_preserves_order():
    @parallelize()
    def echo(x: int) -> int:
        # vary sleep to scramble completion order
        time.sleep(0.01 * (x % 3))
        return x

    data = [5, 1, 7, 3, 3, 9]
    result_iter = echo(data)

    result = list(result_iter)
    # must match input order 1:1
    assert result == data


def test_parallelize_on_instance_method_arg_index_1():
    class Multiplier:
        def __init__(self, factor: int) -> None:
            self.factor = factor

        @parallelize(arg_index=1)
        def mul(self, x: int) -> int:
            return self.factor * x

    m = Multiplier(3)
    data = [1, 2, 3, 4]
    result_iter = m.mul(data)

    result = list(result_iter)
    assert result == [3, 6, 9, 12]


def test_parallelize_uses_existing_executor_and_does_not_shutdown():
    @parallelize()
    def inc(x: int) -> int:
        return x + 1

    data = [1, 2, 3]

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        # use external executor
        result_iter = inc(data, executor=ex)
        result = list(result_iter)

        assert result == [2, 3, 4]

        # executor should still be usable (not shut down by decorator)
        fut = ex.submit(lambda: 42)
        assert fut.result(timeout=1) == 42


def test_parallelize_arg_index_out_of_range():
    @parallelize(arg_index=2)  # but we will only pass 2 positional args
    def f(a, b):
        return a + b

    # error happens at call time (before generator is created)
    with pytest.raises(ValueError):
        f(1, 2)


def test_parallelize_non_iterable_argument():
    @parallelize(arg_index=0)
    def f(x):
        return x

    # x must be iterable for parallelization -> TypeError at call
    with pytest.raises(TypeError):
        f(123)  # not iterable


def test_parallelize_timeout_raises():
    @parallelize(timeout=0.01)  # tiny per-task timeout
    def slow(x: int) -> int:
        time.sleep(0.1)
        return x

    data = [1, 2, 3]

    with pytest.raises(cf.TimeoutError):
        # consuming the iterator should trigger the timeout
        list(slow(data))


def test_parallelize_return_exceptions_collects_timeout():
    @parallelize(timeout=0.01, return_exceptions=True)
    def slow(x: int) -> int:
        time.sleep(0.1)
        return x

    data = [1, 2, 3]
    results = list(slow(data))

    assert len(results) == len(data)
    assert all(isinstance(r, cf.TimeoutError) for r in results)


def test_parallelize_with_process_pool_executor():
    @parallelize(executor_cls=cf.ProcessPoolExecutor, show_progress=True)
    def cube(x: int) -> int:
        return x * x * x

    data = [0, 1, 2, 3, 4]
    result_iter = cube(data)
    result = list(result_iter)

    assert result == [i ** 3 for i in data]
