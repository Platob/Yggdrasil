"""
JobTask — single-task lifecycle within a parent :class:`Job`.

Databricks doesn't expose tasks as a first-class API; tasks live inside
the parent job's settings. :class:`JobTask` round-trips every CRUD
operation through :meth:`Job.update` so the parent's task list stays
the source of truth.

For Python callables, :meth:`JobTask.from_callable` extracts the raw
source via :func:`inspect.getsource`, drops a self-contained ``.py``
script under the user's personal workspace
(``/Workspace/Users/me/.yggdrasil/jobs/``), and wraps it in a
:class:`SparkPythonTask`. No pickling — the source is what runs.
:meth:`JobTask.decorate` (chained off :meth:`Job.task`) is the
decorator form.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import logging
import secrets
import textwrap
from dataclasses import replace as _dc_replace
from typing import Any, Callable, List, Mapping, Optional, TYPE_CHECKING

from databricks.sdk.service.jobs import SparkPythonTask, Task

if TYPE_CHECKING:
    from .job import Job


__all__ = [
    "JobTask",
    "DEFAULT_STAGING_ROOT",
    "describe_signature",
    "format_signature",
    "coerce_kwargs",
]

LOGGER = logging.getLogger(__name__)

#: Default staging area for :meth:`JobTask.from_callable`. Lands under
#: the bound user's workspace home.
DEFAULT_STAGING_ROOT = "/Workspace/Users/me/.yggdrasil/jobs"


class JobTask:
    """A single :class:`Task` bound to a parent :class:`Job`.

    Construction is cheap and never hits the API; pass ``details=None``
    when you only have a ``task_key`` and intend to :meth:`refresh`
    against an existing job-side task. All mutating operations
    (:meth:`create` / :meth:`update` / :meth:`delete`) push the parent
    job's full task list back through :meth:`Job.update`.
    """

    def __init__(
        self,
        job: "Job",
        task_key: str,
        details: Optional[Task] = None,
        *,
        order: Optional[int] = None,
    ) -> None:
        self.job = job
        self.task_key = task_key
        self._details = details
        #: Optional position to place this task at on :meth:`create` /
        #: :meth:`create`. ``None`` keeps the existing position
        #: (or appends when new). Honors Python list-slice indexing, so
        #: ``0`` lands first and ``-1`` lands second-to-last (insert
        #: semantics: ``lst[:order] + [t] + lst[order:]``).
        self.order = order

    def __repr__(self) -> str:
        return (
            f"JobTask(job_id={self.job.job_id!r}, "
            f"task_key={self.task_key!r})"
        )

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    @property
    def details(self) -> Optional[Task]:
        """Return the cached :class:`Task`, fetching from the job on miss."""
        if self._details is None:
            self.refresh()
        return self._details

    def refresh(self) -> "JobTask":
        """Reload this task from the parent job's latest settings."""
        self.job.refresh()
        for t in self._existing_tasks():
            if t.task_key == self.task_key:
                self._details = t
                return self
        self._details = None
        return self

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def create(self) -> "JobTask":
        """Append this task to the parent job — or update the existing entry.

        Idempotent: if a task with the same ``task_key`` already lives
        on the job, its entry is replaced in place (or moved when
        :attr:`order` is set); otherwise the task is inserted. Used by
        :meth:`JobTask.decorate` so re-decorating the same function
        during development doesn't raise — the staged source on the
        second pass overwrites the first task entry.
        """
        if self._details is None:
            raise ValueError(
                f"Cannot create {self!r}: details is None. Construct with a "
                "Task or build through :meth:`from_callable`."
            )
        existing = self._existing_tasks()
        replaced = any(t.task_key == self.task_key for t in existing)
        new_tasks = self._place(existing, self._details)

        LOGGER.debug(
            "%s job task %r on %r",
            "Updating" if replaced else "Creating", self, self.job,
        )
        self.job.update(tasks=new_tasks)
        LOGGER.info(
            "%s job task %r",
            "Updated" if replaced else "Created", self,
        )
        return self

    def update(self, **fields: Any) -> "JobTask":
        """Replace fields on this task and push the new task list back."""
        if self._details is None:
            self.refresh()
        if self._details is None:
            raise ValueError(
                f"Cannot update {self!r}: task not found on {self.job!r}."
            )
        new_details = _dc_replace(self._details, **fields)
        existing = self._existing_tasks()
        updated: List[Task] = [
            new_details if t.task_key == self.task_key else t
            for t in existing
        ]
        LOGGER.debug(
            "Updating job task %r (fields=%r)", self, list(fields),
        )
        self.job.update(tasks=updated)
        self._details = new_details
        LOGGER.info("Updated job task %r", self)
        return self

    def delete(self) -> None:
        """Remove this task from the parent job (no-op if already absent)."""
        existing = self._existing_tasks()
        remaining = [t for t in existing if t.task_key != self.task_key]
        if len(remaining) == len(existing):
            LOGGER.debug("Job task %r already absent from %r", self, self.job)
            return
        LOGGER.debug("Deleting job task %r", self)
        self.job.update(tasks=remaining)
        LOGGER.info("Deleted job task %r", self)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _existing_tasks(self) -> List[Task]:
        settings = self.job.settings
        return list((settings.tasks if settings is not None else None) or [])

    def _place(self, existing: List[Task], new_details: Task) -> List[Task]:
        """Build the new task list with *new_details* placed honoring ``self.order``.

        ``order is None`` keeps the existing task's position (replace
        in place) or appends when the key is new. An integer ``order``
        first strips any prior entry for ``self.task_key`` and inserts
        *new_details* at that slice index (``lst[:order] + [t] +
        lst[order:]``), so the same call both creates and reorders.
        """
        if self.order is None:
            replaced = False
            new_tasks: List[Task] = []
            for t in existing:
                if t.task_key == self.task_key:
                    new_tasks.append(new_details)
                    replaced = True
                else:
                    new_tasks.append(t)
            if not replaced:
                new_tasks.append(new_details)
            return new_tasks
        others = [t for t in existing if t.task_key != self.task_key]
        return [*others[:self.order], new_details, *others[self.order:]]

    # ------------------------------------------------------------------ #
    # Decorator: stage a Python callable onto this task
    # ------------------------------------------------------------------ #
    def decorate(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Stage *func*'s source onto this task and persist it on the job.

        Designed to be chained off :meth:`Job.task` as a decorator::

            @job.task("step_one", description="…").decorate
            def step_one(): ...

        Stages *func*'s raw source under the user's workspace via
        :meth:`from_callable`, then back-fills its derived defaults
        (``spark_python_task``, ``description`` from the docstring)
        onto :attr:`_details` *only where the caller didn't already
        set that field* through :meth:`Job.task`. Anything pre-set on
        the handle — ``spark_python_task=…``, ``description=…``,
        compute, dependencies, retries, environment_key — wins.
        Pushes the result through :meth:`create` so a re-decoration
        replaces the previous entry in place.

        Returns the original callable so the function stays usable
        in-process; the :class:`JobTask` handle is attached as
        ``func._job_task`` for downstream access.
        """
        staged = type(self).from_callable(
            self.job, func, task_key=self.task_key,
        )
        staged_details = staged._details
        assert staged_details is not None, (
            "JobTask.from_callable should always populate _details"
        )
        if self._details is None:
            self._details = staged_details
        else:
            # Caller-supplied fields win; decorate only fills in slots
            # the caller left as None on the pre-built Task.
            defaults = {
                k: v for k, v in vars(staged_details).items()
                if v is not None and getattr(self._details, k, None) is None
            }
            if defaults:
                self._details = _dc_replace(self._details, **defaults)
        self.create()
        func._job_task = self  # type: ignore[attr-defined]
        return func

    # ------------------------------------------------------------------ #
    # Factory: from a Python callable
    # ------------------------------------------------------------------ #
    @classmethod
    def from_callable(
        cls,
        job: "Job",
        func: Callable[..., Any],
        *args: Any,
        task_key: Optional[str] = None,
        staging_root: str = DEFAULT_STAGING_ROOT,
        **kwargs: Any,
    ) -> "JobTask":
        """Stage *func*'s source + bound *args*/*kwargs* as a Python script.

        Extracts the source via :func:`inspect.getsource`, strips any
        decorator lines (the runner side has no ``@job.task(...).decorate``
        in scope), appends an invocation that passes *args* / *kwargs*
        as Python literals, and writes the result to a single ``.py``
        file under *staging_root* (default:
        ``/Workspace/Users/me/.yggdrasil/jobs/<task_key>-<rand>.py``).
        No pickling involved — the script Databricks runs is the exact
        source of the function.

        *args* / *kwargs* are rendered via :func:`repr`, so they must be
        types whose ``repr`` round-trips through ``eval`` (built-in
        scalars, strings, tuples / lists / dicts of the same). Pass
        nothing at decoration time and let the function read its inputs
        from job parameters at run time when that's not enough.

        Limitations: ``inspect.getsource`` needs the function to live in
        an importable source file (no REPL-defined lambdas) and the body
        must be self-contained — closures, module-level globals, and
        decorators other than ``@job.task(...).decorate`` are NOT
        carried over.

        The returned :class:`JobTask` is not persisted on the job yet
        — call :meth:`create` (or use :meth:`JobTask.decorate`, which
        does it for you). :meth:`create` is idempotent — same key
        replaces in place. Compute
        stays caller-owned: layer ``new_cluster`` / ``existing_cluster_id``
        / ``job_cluster_key`` via :meth:`update` once the task is
        registered.
        """
        from yggdrasil.databricks.fs.workspace_path import WorkspacePath

        key = task_key or func.__name__
        suffix = secrets.token_hex(4)

        script = _render_callable_script(func, args, kwargs)

        path = WorkspacePath(
            f"{staging_root.rstrip('/')}/{key}-{suffix}.py",
            client=job.client,
        )

        LOGGER.debug(
            "Staging callable %r as raw source at %r",
            func.__qualname__, path,
        )
        path.write_bytes(script.encode())

        # Description carries the formatted signature so the Databricks
        # UI surfaces "qualname(x: int = 5) -> str" without cracking the
        # script open; the docstring's first line is prepended when set.
        signature_str = format_signature(describe_signature(func))
        doc_line = (func.__doc__ or "").strip().splitlines()[0:1]
        description = (
            f"{doc_line[0]} — {signature_str}" if doc_line else signature_str
        )[:1000]

        details = Task(
            task_key=key,
            description=description,
            spark_python_task=SparkPythonTask(
                python_file=path.full_path(),
            ),
        )
        return cls(job=job, task_key=key, details=details)


def _annotation_to_str(ann: Any) -> Optional[str]:
    """Render a parameter annotation as a short, JSON-friendly string."""
    if ann is inspect.Parameter.empty or ann is None:
        return None
    if isinstance(ann, str):
        # ``from __future__ import annotations`` and PEP 604 stringified
        # forms arrive as a plain string already — keep them verbatim.
        return ann
    if isinstance(ann, type):
        mod = getattr(ann, "__module__", "builtins")
        return ann.__qualname__ if mod == "builtins" else f"{mod}.{ann.__qualname__}"
    return repr(ann)


def _resolved_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    """Resolve string annotations on *func* into real types when possible.

    ``from __future__ import annotations`` (PEP 563) keeps every
    annotation as a string at runtime; :func:`inspect.get_annotations`
    with ``eval_str=True`` evaluates them in the function's own
    module / locals. Failures fall back to the raw string so the
    caller can still log / display the annotation.
    """
    try:
        return inspect.get_annotations(func, eval_str=True)
    except Exception:
        return dict(getattr(func, "__annotations__", {}) or {})


def _default_to_str(default: Any) -> tuple[bool, Optional[str]]:
    """Return ``(has_default, repr(default))`` for a parameter default."""
    if default is inspect.Parameter.empty:
        return False, None
    try:
        return True, repr(default)
    except Exception:
        return True, "<unrepresentable>"


def describe_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Capture *func*'s signature as a JSON-serializable dict.

    Returns ``{"qualname", "module", "parameters": [...], "return"}``
    where each parameter entry carries ``name``, ``kind`` (the
    :class:`inspect.Parameter.kind` name), ``annotation`` (dotted path
    when the annotation is a class), and ``default`` (``repr`` of the
    default) where present. Used to stamp signature metadata onto the
    staged script and the task description so a reader doesn't have
    to crack the source open to know how to call the function.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return {
            "qualname": getattr(func, "__qualname__", repr(func)),
            "module": getattr(func, "__module__", None),
            "parameters": [],
            "return": None,
        }
    resolved = _resolved_annotations(func)
    params: list[dict[str, Any]] = []
    for name, p in sig.parameters.items():
        entry: dict[str, Any] = {"name": name, "kind": p.kind.name}
        ann = _annotation_to_str(resolved.get(name, p.annotation))
        if ann is not None:
            entry["annotation"] = ann
        has_default, default_repr = _default_to_str(p.default)
        if has_default:
            entry["default"] = default_repr
        params.append(entry)
    return {
        "qualname": func.__qualname__,
        "module": getattr(func, "__module__", None),
        "parameters": params,
        "return": _annotation_to_str(resolved.get("return", sig.return_annotation)),
    }


def format_signature(sig_meta: Mapping[str, Any]) -> str:
    """Render :func:`describe_signature` output as ``qualname(x: int = 5) -> str``."""
    parts: list[str] = []
    for p in sig_meta.get("parameters", []):
        token = str(p["name"])
        if "annotation" in p:
            token += f": {p['annotation']}"
        if "default" in p:
            token += f" = {p['default']}"
        parts.append(token)
    qual = sig_meta.get("qualname") or "<unknown>"
    out = f"{qual}({', '.join(parts)})"
    return_ann = sig_meta.get("return")
    if return_ann:
        out += f" -> {return_ann}"
    return out


def coerce_kwargs(
    func: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Coerce *kwargs* to *func*'s annotated parameter types.

    Walks ``inspect.signature(func).parameters`` and routes each value
    whose parameter carries a non-empty annotation through
    :func:`yggdrasil.data.cast.convert`. Unannotated parameters pass
    through untouched. Designed for the Databricks runtime side, where
    parameters arrive as strings (``sys.argv``, ``dbutils.widgets``,
    job parameters) and need typing before reaching the function.
    """
    if not kwargs:
        return dict(kwargs)
    from yggdrasil.data.cast import convert

    sig = inspect.signature(func)
    resolved = _resolved_annotations(func)
    coerced: dict[str, Any] = {}
    for name, value in kwargs.items():
        param = sig.parameters.get(name)
        if param is None:
            coerced[name] = value
            continue
        annotation = resolved.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            coerced[name] = value
            continue
        coerced[name] = convert(value, annotation)
    return coerced


# Embedded runtime checker. Lives at module level inside every staged
# script so the runner can coerce widget / argv kwargs into the
# function's annotated types without re-staging. Mirrors
# :func:`coerce_kwargs` above; kept in sync as one short block.
_COERCE_RUNTIME = '''\
def _yggdrasil_coerce_kwargs(_func, _kwargs):
    """Coerce string-shaped kwargs to *_func*'s annotated types.

    Resolves string annotations (``from __future__ import annotations``)
    via :func:`inspect.get_annotations` with ``eval_str=True`` and routes
    each value through :func:`yggdrasil.data.cast.convert`. Short-circuits
    when *_kwargs* is empty so the yggdrasil import only fires when
    there's actually something to coerce.
    """
    if not _kwargs:
        return dict(_kwargs)
    import inspect as _inspect
    from yggdrasil.data.cast import convert as _convert
    _sig = _inspect.signature(_func)
    try:
        _resolved = _inspect.get_annotations(_func, eval_str=True)
    except Exception:
        _resolved = dict(getattr(_func, "__annotations__", {}) or {})
    _coerced = {}
    for _name, _value in _kwargs.items():
        _p = _sig.parameters.get(_name)
        if _p is None:
            _coerced[_name] = _value
            continue
        _ann = _resolved.get(_name, _p.annotation)
        if _ann is _inspect.Parameter.empty:
            _coerced[_name] = _value
        else:
            _coerced[_name] = _convert(_value, _ann)
    return _coerced
'''


def _render_callable_script(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> str:
    """Render *func* + bound *args* / *kwargs* as a runnable ``.py`` script.

    Embeds a ``__yggdrasil_task__`` metadata block (signature, module,
    yggdrasil version, staging timestamp) and the runtime
    ``_yggdrasil_coerce_kwargs`` helper so the runner side can type
    arbitrary kwargs against the function's annotations via
    :func:`yggdrasil.data.cast.convert`. Returns a UTF-8 ``str``;
    caller encodes for the workspace write.
    """
    from yggdrasil.version import __version__ as ygg_version

    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as exc:  # built-ins, REPL-defined lambdas
        raise ValueError(
            f"Cannot stage {func!r} as a JobTask: inspect.getsource failed "
            f"({exc!s}). from_callable needs a function defined in an "
            "importable source file."
        ) from exc

    # Drop decorator lines preceding ``def`` — the runner has no
    # ``@job.task(...).decorate`` (or any other decorator from this
    # scope) available.
    lines = source.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    body = "\n".join(lines).rstrip() + "\n"

    sig_meta = describe_signature(func)
    meta_payload = {
        **sig_meta,
        "yggdrasil_version": str(ygg_version),
        "staged_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    # JSON-encode for stable shape, then load via ``json.loads`` at
    # module import — keeps the metadata block readable as JSON
    # without Python tripping on ``null`` / ``true`` / ``false``.
    meta_json = json.dumps(meta_payload, indent=2, sort_keys=True)

    positional = [repr(a) for a in args]
    if kwargs:
        kw_literal = "{" + ", ".join(f"{k!r}: {v!r}" for k, v in kwargs.items()) + "}"
        call_args = positional + [
            f"**_yggdrasil_coerce_kwargs({func.__name__}, {kw_literal})"
        ]
    else:
        call_args = positional
    invocation = f"{func.__name__}({', '.join(call_args)})"

    return (
        "# Auto-generated by yggdrasil.databricks.jobs.JobTask.from_callable.\n"
        f"# Function: {func.__qualname__}\n"
        f"# Signature: {format_signature(sig_meta)}\n"
        "# The function body below is the verbatim source of the decorated\n"
        "# callable; signature metadata + the kwargs coercion helper are\n"
        "# embedded so the runner can introspect and type-check inputs.\n"
        "\n"
        "import json as _yggdrasil_json\n"
        f"__yggdrasil_task__ = _yggdrasil_json.loads(r\"\"\"{meta_json}\"\"\")\n"
        "\n"
        f"{_COERCE_RUNTIME}"
        "\n"
        f"{body}"
        "\n"
        'if __name__ == "__main__":\n'
        f"    {invocation}\n"
    )
