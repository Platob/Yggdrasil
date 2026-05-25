"""Decorator-based distributed function framework.

Decorating a function with @function:
1. Extracts source code via inspect.getsource
2. Infers dependencies from imports (AST analysis)
3. Infers python version from sys.version_info
4. Registers the function on the local node (upsert by name)
5. Returns a FunctionHandle that is callable

Calling the handle returns a FunctionRun (Future-like) that:
- Uses WaitingConfig for timeouts
- Uses State enum for lifecycle tracking
- Polls the node API for completion
- Returns the deserialized result

The handle has methods:
- .with_env(env_name_or_id) -- specify environment
- .on(node_url) -- target a remote node
- .with_args(**kwargs) -- partial application

Usage::

    from yggdrasil.node.fn import function, dag

    @function
    def process_data(input_path: str, threshold: float = 0.5) -> dict:
        import pandas as pd
        df = pd.read_csv(input_path)
        return {"rows": len(df), "filtered": len(df[df.score > threshold])}

    # Call like a normal function -- returns FunctionRun (Future-like)
    run = process_data("data.csv", threshold=0.7)
    result = run.wait()  # blocks until done, returns result

    # Or with explicit environment
    run = process_data.with_env("ml-env")("data.csv")

    # Or on a remote node
    run = process_data.on("http://node-2:8100")("data.csv")

    # DAG chaining
    @function
    def extract(source: str) -> list:
        return [1, 2, 3]

    @function
    def transform(data: list) -> list:
        return [x * 2 for x in data]

    @function
    def load(data: list) -> int:
        return sum(data)

    pipeline = dag("etl", extract >> transform >> load)
    run = pipeline()
    result = run.wait()  # waits for entire DAG
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import json
import logging
import os
import sys
import textwrap
import time
import urllib.request
from typing import Any, Callable, TypeVar, overload

__all__ = ["function", "dag", "FunctionHandle", "FunctionRun", "DagHandle", "get_input", "set_output"]

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Standard library module names — used to exclude stdlib from inferred deps.
_STDLIB_MODULES = frozenset({
    "abc", "argparse", "ast", "asyncio", "base64", "binascii", "builtins",
    "collections", "concurrent", "contextlib", "copy", "csv", "ctypes",
    "dataclasses", "datetime", "decimal", "difflib", "email", "enum",
    "errno", "fnmatch", "fractions", "functools", "gc", "getpass", "glob",
    "gzip", "hashlib", "heapq", "hmac", "html", "http", "importlib",
    "inspect", "io", "itertools", "json", "locale", "logging", "lzma",
    "math", "mmap", "multiprocessing", "numbers", "operator", "os",
    "pathlib", "pickle", "platform", "pprint", "queue", "random", "re",
    "secrets", "shlex", "shutil", "signal", "socket", "sqlite3",
    "statistics", "string", "struct", "subprocess", "sys", "tempfile",
    "textwrap", "threading", "time", "timeit", "traceback", "types",
    "typing", "unittest", "urllib", "uuid", "warnings", "weakref",
    "xml", "zipfile", "zlib",
})


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, no external deps)
# ---------------------------------------------------------------------------


def _local_url() -> str:
    """Get the local node URL from environment."""
    port = os.environ.get("YGG_NODE_PORT", "8100")
    return f"http://127.0.0.1:{port}"


def _post(url: str, data: dict) -> dict:
    """HTTP POST with JSON body, returns parsed response."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get(url: str) -> dict:
    """HTTP GET, returns parsed response."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Dependency inference
# ---------------------------------------------------------------------------


def _infer_dependencies(code: str, own_package: str = "yggdrasil") -> list[str]:
    """Extract third-party dependency names from function source via AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _STDLIB_MODULES and root != own_package:
                    modules.add(root)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root not in _STDLIB_MODULES and root != own_package:
                modules.add(root)

    return sorted(modules)


# ---------------------------------------------------------------------------
# FunctionRun — Future-like result handle
# ---------------------------------------------------------------------------


class FunctionRun:
    """Future-like object for a function execution.

    Integrates with:
    - WaitingConfig for timeout/retry semantics
    - State enum for lifecycle tracking
    """

    __slots__ = (
        "run_id", "node_url", "function_id",
        "_state", "_result", "_exception", "_entry",
    )

    def __init__(self, run_id: int, node_url: str, function_id: int) -> None:
        self.run_id = run_id
        self.node_url = node_url
        self.function_id = function_id
        # Lazy import to avoid circular deps
        from yggdrasil.data.enums.state import State
        self._state: State = State.PENDING
        self._result: Any = None
        self._exception: Exception | None = None
        self._entry: dict | None = None

    @classmethod
    def submit(
        cls,
        function_id: int,
        args: dict[str, Any],
        environment_id: int | None = None,
        node_url: str | None = None,
    ) -> "FunctionRun":
        """Submit a run to the node API."""
        url = node_url or _local_url()
        data: dict[str, Any] = {"function_id": function_id, "args": args}
        if environment_id is not None:
            data["environment_id"] = environment_id
        # POST /api/function/{id}/run
        resp = _post(f"{url}/api/function/{function_id}/run", data)
        run = resp["run"]
        return cls(run["id"], url, function_id)

    @property
    def state(self) -> Any:
        """Current lifecycle state (State enum)."""
        return self._state

    @property
    def is_done(self) -> bool:
        """True when the run has reached a terminal state."""
        return self._state.is_done

    def wait(
        self,
        wait: Any = None,
        raise_error: bool = True,
    ) -> Any:
        """Block until the run completes. Uses WaitingConfig for timeouts.

        Args:
            wait: timeout config -- None=block forever, False=poll once,
                  float=seconds, WaitingConfig for full control
            raise_error: if True, raise on failure

        Returns:
            The function result, or None if not done and raise_error=False

        Raises:
            TimeoutError: When timed wait elapses before completion.
            RuntimeError: When the run failed (if raise_error=True).
        """
        from yggdrasil.dataclasses.waiting import WaitingConfig

        # Non-blocking poll
        if wait is False:
            self._poll()
            if self._state.is_done:
                if self._state.is_failed and raise_error and self._exception:
                    raise self._exception
                return self._result
            return None

        wc = WaitingConfig.from_(wait if wait is not None else True)

        start = time.monotonic()
        iteration = 0
        while True:
            self._poll()
            if self._state.is_done:
                if self._state.is_failed and raise_error and self._exception:
                    raise self._exception
                return self._result
            if wc.is_expired(start):
                if raise_error:
                    raise TimeoutError(
                        f"Run {self.run_id} timed out after {wc.timeout:.1f}s"
                    )
                return None
            wc.sleep(iteration, start)
            iteration += 1

    def _poll(self) -> None:
        """Fetch current run state from the node API.

        Optimization: if the run has already reached a terminal state,
        skip the network call entirely -- the result cannot change.
        """
        if self._state.is_done:
            return  # already settled, skip network call

        from yggdrasil.data.enums.state import State

        try:
            resp = _get(f"{self.node_url}/api/run/{self.run_id}")
        except Exception as exc:
            LOGGER.debug("Poll failed for run %d: %s", self.run_id, exc)
            return

        entry = resp["run"]
        self._entry = entry
        status = entry.get("status", "pending")

        # Map status string to State using the enum's alias table
        self._state = State.from_(status, default=State.PENDING)

        if self._state == State.SUCCEEDED:
            self._result = entry.get("result")
        elif self._state.is_failed:
            error_msg = entry.get("stderr") or entry.get("error") or "Run failed"
            self._exception = RuntimeError(error_msg)
            self._result = None

    @property
    def result(self) -> Any:
        """Get the result if done, None otherwise. Polls if not done."""
        if not self.is_done:
            self._poll()
        return self._result

    @property
    def stdout(self) -> str | None:
        """Captured stdout from the run, if available."""
        return self._entry.get("stdout") if self._entry else None

    @property
    def stderr(self) -> str | None:
        """Captured stderr from the run, if available."""
        return self._entry.get("stderr") if self._entry else None

    def __repr__(self) -> str:
        return f"FunctionRun(id={self.run_id}, state={self._state.name})"


# ---------------------------------------------------------------------------
# FunctionHandle — callable wrapper for decorated functions
# ---------------------------------------------------------------------------


class FunctionHandle:
    """Wraps a decorated function. Callable, returns FunctionRun.

    Acts as a transparent proxy: calling the handle submits the function
    to the node for execution and returns a Future-like FunctionRun.
    """

    def __init__(
        self,
        func: Callable,
        *,
        name: str,
        code: str,
        dependencies: list[str],
        python_version: str,
        description: str,
    ) -> None:
        self._func = func
        self.name = name
        self.code = code
        self.dependencies = dependencies
        self.python_version = python_version
        self.description = description
        self._node_url: str | None = None  # None = local
        self._environment_id: int | None = None
        self._env_name: str | None = None
        self._registered: bool = False
        self._function_id: int | None = None
        # Make it look like the original function
        functools.update_wrapper(self, func)

    def __call__(self, *args: Any, **kwargs: Any) -> FunctionRun:
        """Execute the function, returns a Future-like FunctionRun.

        If running inside a ygg runtime (YGG_RUNTIME_VERSION is set) and
        no explicit remote node is targeted, executes locally in-process
        without network round-trips.
        """
        # If we're inside a ygg runtime, execute locally without network
        if os.environ.get("YGG_RUNTIME_VERSION") and self._node_url is None:
            return self._execute_local(*args, **kwargs)
        self._ensure_registered()
        # Build args dict from function signature
        sig = inspect.signature(self._func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        run_args = dict(bound.arguments)

        # Submit to node (local or remote)
        return FunctionRun.submit(
            function_id=self._function_id,
            args=run_args,
            environment_id=self._environment_id,
            node_url=self._node_url,
        )

    def _execute_local(self, *args: Any, **kwargs: Any) -> FunctionRun:
        """Execute directly in-process when running inside ygg runtime.

        Bypasses network registration and HTTP submission for lower latency
        when the function is already running on a ygg node.
        """
        from yggdrasil.data.enums.state import State

        run = FunctionRun.__new__(FunctionRun)
        run.run_id = 0
        run.node_url = ""
        run.function_id = 0
        run._entry = {}
        try:
            result = self._func(*args, **kwargs)
            run._state = State.SUCCEEDED
            run._result = result
            run._exception = None
        except Exception as e:
            run._state = State.FAILED
            run._result = None
            run._exception = e
        return run

    def with_env(self, env: str | int) -> "FunctionHandle":
        """Return a copy targeting a specific environment.

        Args:
            env: Environment name (str) or ID (int).

        Returns:
            A new FunctionHandle configured for the given environment.
        """
        clone = copy.copy(self)
        clone._registered = False  # Force re-registration with new env
        if isinstance(env, str):
            clone._env_name = env
            clone._environment_id = None
        else:
            clone._environment_id = env
            clone._env_name = None
        return clone

    def on(self, node_url: str) -> "FunctionHandle":
        """Return a copy targeting a remote node.

        Args:
            node_url: Full URL of the remote node (e.g. "http://node-2:8100").

        Returns:
            A new FunctionHandle configured for the remote node.
        """
        clone = copy.copy(self)
        clone._node_url = node_url.rstrip("/")
        clone._registered = False  # Force re-registration on new node
        return clone

    def with_args(self, **kwargs: Any) -> "_PartialHandle":
        """Return a partial-application wrapper with some args pre-bound.

        Args:
            **kwargs: Arguments to pre-bind.

        Returns:
            A _PartialHandle that fills remaining args on call.
        """
        return _PartialHandle(self, kwargs)

    def __rshift__(self, other: "FunctionHandle") -> "_DagChain":
        """Enable f1 >> f2 >> f3 chaining for DAG creation."""
        if isinstance(other, _DagChain):
            return _DagChain([self] + other._handles)
        return _DagChain([self, other])

    def _ensure_registered(self) -> None:
        """Register/upsert the function on the target node."""
        if self._registered:
            return

        url = self._node_url or _local_url()

        # Resolve environment name to ID if needed
        if self._env_name and self._environment_id is None:
            self._environment_id = self._resolve_env_name(url, self._env_name)

        data: dict[str, Any] = {
            "name": self.name,
            "code": self.code,
            "language": "python",
            "description": self.description,
            "python_version": self.python_version,
            "dependencies": self.dependencies,
        }
        if self._environment_id is not None:
            data["environment_id"] = self._environment_id

        resp = _post(f"{url}/api/function", data)
        self._function_id = resp["function"]["id"]
        self._registered = True
        LOGGER.debug(
            "Registered function %r (id=%d) on %s",
            self.name, self._function_id, url,
        )

    @staticmethod
    def _resolve_env_name(node_url: str, env_name: str) -> int | None:
        """Resolve an environment name to its numeric ID."""
        try:
            resp = _get(f"{node_url}/api/environment")
            for env in resp.get("environments", []):
                if env.get("name") == env_name:
                    return env["id"]
        except Exception as exc:
            LOGGER.warning("Failed to resolve environment %r: %s", env_name, exc)
        return None

    def __repr__(self) -> str:
        target = self._node_url or "local"
        return f"FunctionHandle({self.name!r}, target={target})"


class _PartialHandle:
    """Partial application of a FunctionHandle with some args pre-bound."""

    __slots__ = ("_handle", "_bound_kwargs")

    def __init__(self, handle: FunctionHandle, bound_kwargs: dict[str, Any]) -> None:
        self._handle = handle
        self._bound_kwargs = bound_kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> FunctionRun:
        merged = {**self._bound_kwargs, **kwargs}
        return self._handle(*args, **merged)

    def __repr__(self) -> str:
        return f"_PartialHandle({self._handle.name!r}, bound={list(self._bound_kwargs)})"


# ---------------------------------------------------------------------------
# DAG support
# ---------------------------------------------------------------------------


class _DagChain:
    """Accumulates >> chained functions for DAG creation."""

    __slots__ = ("_handles",)

    def __init__(self, handles: list[FunctionHandle]) -> None:
        self._handles = handles

    def __rshift__(self, other: "FunctionHandle | _DagChain") -> "_DagChain":
        if isinstance(other, _DagChain):
            return _DagChain(self._handles + other._handles)
        return _DagChain(self._handles + [other])


class DagHandle:
    """Callable DAG -- executes a pipeline of functions in sequence.

    Each step receives the output of the previous step as input.
    The DAG is registered on the node and executed as a unit.
    """

    def __init__(
        self,
        name: str,
        handles: list[FunctionHandle],
        description: str = "",
    ) -> None:
        self.name = name
        self.handles = handles
        self.description = description
        self._dag_id: int | None = None
        self._registered: bool = False

    def __call__(self, **initial_args: Any) -> FunctionRun:
        """Execute the DAG. Returns a FunctionRun for the whole pipeline.

        Args:
            **initial_args: Arguments passed to the first step.

        Returns:
            A FunctionRun that tracks the entire DAG execution.
        """
        self._ensure_registered()
        url = _local_url()
        resp = _post(f"{url}/api/dag/{self._dag_id}/run", initial_args or {})
        run = resp["run"]
        return FunctionRun(run["id"], url, self._dag_id)

    def _ensure_registered(self) -> None:
        """Register all constituent functions and the DAG itself."""
        if self._registered:
            return

        # Register all functions first
        for h in self.handles:
            h._ensure_registered()

        # Build DAG steps and edges
        steps: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for i, h in enumerate(self.handles):
            step_id = h.name
            depends = [self.handles[i - 1].name] if i > 0 else []
            steps.append({
                "id": step_id,
                "ref": {
                    "function_id": h._function_id,
                    "environment_id": h._environment_id,
                    "node_url": h._node_url,
                    "args": {},
                },
                "depends_on": depends,
            })
            if i > 0:
                edges.append({
                    "from_step": self.handles[i - 1].name,
                    "to_step": step_id,
                    "output_key": "result",
                    "input_key": "input",
                })

        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "steps": steps,
            "edges": edges,
        }
        url = _local_url()
        resp = _post(f"{url}/api/dag", data)
        self._dag_id = resp["dag"]["id"]
        self._registered = True
        LOGGER.debug("Registered DAG %r (id=%d)", self.name, self._dag_id)

    def __repr__(self) -> str:
        step_names = " >> ".join(h.name for h in self.handles)
        return f"DagHandle({self.name!r}, steps=[{step_names}])"


def dag(
    name: str,
    chain: _DagChain | list[FunctionHandle],
    *,
    description: str = "",
) -> DagHandle:
    """Create a DAG from a chain of functions.

    Args:
        name: Name for the DAG pipeline.
        chain: Either a _DagChain (from >> operator) or a list of FunctionHandles.
        description: Optional description of the pipeline.

    Returns:
        A callable DagHandle.

    Usage::

        pipeline = dag("etl", extract >> transform >> load)
        run = pipeline()
        result = run.wait()
    """
    handles = chain._handles if isinstance(chain, _DagChain) else chain
    return DagHandle(name=name, handles=handles, description=description)


# ---------------------------------------------------------------------------
# The @function decorator
# ---------------------------------------------------------------------------


@overload
def function(func: F) -> FunctionHandle: ...


@overload
def function(
    *,
    name: str | None = ...,
    description: str | None = ...,
    environment: str | int | None = ...,
    dependencies: list[str] | None = ...,
) -> Callable[[F], FunctionHandle]: ...


def function(
    func: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    environment: str | int | None = None,
    dependencies: list[str] | None = None,
) -> FunctionHandle | Callable[[F], FunctionHandle]:
    """Decorator that turns a Python function into a distributed callable.

    Infers:
    - name from func.__name__ (or explicit)
    - code from inspect.getsource(func)
    - dependencies from AST import analysis
    - python_version from sys.version_info
    - description from func.__doc__ or explicit

    The decorated function works without a running node -- registration
    is deferred to the first call. When called, it returns a FunctionRun
    (Future-like) that can be awaited for the result.

    Usage::

        @function
        def my_func(x: int) -> int:
            return x * 2

        @function(name="custom_name", environment="ml-env")
        def my_func(x):
            import numpy as np
            return np.mean(x)

        run = my_func(42)
        result = run.wait()  # blocks until done
    """

    def decorator(fn: F) -> FunctionHandle:
        # Extract code
        try:
            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
            # Remove the decorator lines from source
            lines = source.split("\n")
            func_start = next(
                (i for i, line in enumerate(lines) if line.strip().startswith("def ")),
                0,
            )
            code = "\n".join(lines[func_start:])
        except (OSError, TypeError):
            # Fallback: reconstruct a minimal stub from the signature
            sig = inspect.signature(fn)
            code = f"def {fn.__name__}{sig}:\n    pass  # source unavailable"

        # Infer name
        fn_name = name or fn.__name__

        # Infer description
        desc = description or fn.__doc__ or ""
        if desc:
            desc = desc.strip().split("\n")[0]  # first line only

        # Infer dependencies from imports in the function body
        deps = dependencies
        if deps is None:
            own_pkg = (fn.__module__ or "").split(".")[0] or "yggdrasil"
            deps = _infer_dependencies(code, own_package=own_pkg)

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Resolve environment
        env_id: int | None = None
        env_name: str | None = None
        if environment is not None:
            if isinstance(environment, int):
                env_id = environment
            else:
                env_name = str(environment)

        handle = FunctionHandle(
            fn,
            name=fn_name,
            code=code,
            dependencies=deps,
            python_version=py_version,
            description=desc,
        )
        if env_id is not None:
            handle._environment_id = env_id
        if env_name is not None:
            handle._env_name = env_name

        return handle

    if func is not None:
        # Called without arguments: @function
        return decorator(func)
    # Called with arguments: @function(name="foo")
    return decorator


# ---------------------------------------------------------------------------
# DAG communication helpers
# ---------------------------------------------------------------------------


def get_input(key: str = "input", default: Any = None) -> Any:
    """Get input passed from a previous DAG step.

    When running as part of a DAG, previous step outputs are injected
    as JSON in the ``__ygg_inputs__`` env var.

    Args:
        key: The key to look up in the inputs dict.
        default: Value to return if the key is not found.

    Returns:
        The value for the given key, or *default*.
    """
    raw = os.environ.get("__ygg_inputs__")
    if not raw:
        return default
    inputs = json.loads(raw)
    return inputs.get(key, default)


def set_output(key: str = "result", value: Any = None) -> None:
    """Set output for the next DAG step.

    Writes to the file specified by ``__ygg_outputs_file__`` which the
    runtime reads after execution to collect structured outputs.

    Args:
        key: The output key name.
        value: The value to store (must be JSON-serializable).
    """
    outputs_file = os.environ.get("__ygg_outputs_file__")
    if outputs_file:
        existing: dict[str, Any] = {}
        if os.path.exists(outputs_file):
            try:
                with open(outputs_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        existing[key] = value
        with open(outputs_file, "w") as f:
            json.dump(existing, f)
