"""Signature-driven argument coercion for Python callables.

Wraps :func:`yggdrasil.data.cast.convert` against the function's own
signature so callers can pass string-shaped inputs (CLI argv,
Databricks job parameters, HTTP form fields, environment variables)
and have them reach the function as the annotated Python type — no
manual ``int(os.environ[...])`` / ``datetime.fromisoformat(...)``
sprinkled at every call site.

Two entry points:

* :func:`check_function_args` is the pure utility: takes a function
  and a raw ``(args, kwargs)`` pair, returns coerced
  ``(args, kwargs)`` ready to splat back into the call. Use it when
  you have the raw inputs in hand and want to coerce once before
  dispatching.
* :func:`checkargs` is the decorator built on top: wraps a function so
  every call is routed through :func:`check_function_args` before the
  real call. Use it on functions whose inputs cross a string boundary
  every time.

PEP 563 (``from __future__ import annotations``) and string-quoted
annotations are resolved to real types via
:func:`inspect.get_annotations` with ``eval_str=True``, so the
coercion sees ``int`` rather than the string literal ``"int"``.
Unannotated parameters pass through untouched — there's nothing to
target. ``*args`` and ``**kwargs`` annotations are honored per
element.

Also bundles :func:`describe_signature` / :func:`format_signature` so
callers that want to display or stamp signature metadata (job task
descriptions, OpenAPI shapes, debug logs) reach for the same surface.
"""
from __future__ import annotations

import functools
import importlib
import inspect
import logging
from typing import Any, Callable, Mapping, Optional, TypeVar

__all__ = [
    "check_function_args",
    "checkargs",
    "describe_signature",
    "format_signature",
]

F = TypeVar("F", bound=Callable[..., Any])

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast alias prefixes — recognize ``pa.Table`` / ``pl.DataFrame`` /
# ``pd.DataFrame`` / ``np.ndarray`` shapes when the function's own globals
# don't have the short alias imported. Best-effort: if the dotted import
# fails the annotation stays a string and coercion is skipped at the
# call site.
# ---------------------------------------------------------------------------

_FAST_ALIAS_PREFIXES: dict[str, str] = {
    "pa.": "pyarrow.",
    "pl.": "polars.",
    "pd.": "pandas.",
    "np.": "numpy.",
    "ps.": "pyspark.",
    "ddf.": "dask.dataframe.",
}


def _expand_alias(name: str) -> str:
    """Expand a known short alias prefix (``pa.Table`` → ``pyarrow.Table``)."""
    for short, full in _FAST_ALIAS_PREFIXES.items():
        if name.startswith(short):
            return full + name[len(short):]
    return name


def _resolve_str_annotation(s: str, func_globals: Optional[dict[str, Any]] = None) -> Any:
    """Best-effort: parse a string annotation into a real type.

    Tries, in order:

    1. ``eval`` in *func_globals* + builtins — picks up local imports
       and aliases declared in the function's own module.
    2. ``eval`` against ``typing`` for generic shapes like
       ``Optional[int]`` / ``list[int]`` when *func_globals* doesn't
       have them in scope.
    3. Fast alias prefix expansion (``pa.`` → ``pyarrow.``, …) +
       dotted ``importlib.import_module`` lookup so short forms work
       even when the function's globals never imported the short
       alias.

    Returns the original string when every path fails — callers
    treat that as "unresolved, skip coercion".
    """
    candidate = s.strip()

    if func_globals is not None:
        try:
            return eval(candidate, func_globals, None)
        except Exception:
            pass

    # Generic typing shapes (``Optional[int]``, ``list[int]``, …) live in
    # the ``typing`` namespace; let them resolve without dragging the
    # module into every caller's globals.
    try:
        import typing as _typing
        return eval(candidate, dict(_typing.__dict__), None)
    except Exception:
        pass

    expanded = _expand_alias(candidate)
    if "." in expanded:
        mod_path, _, attr = expanded.rpartition(".")
        mod: Any = None
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            # Module not on the path — try yggdrasil's runtime
            # auto-install before giving up. Respects whatever install
            # policy the active :class:`PyEnv` has configured.
            try:
                from yggdrasil.environ import PyEnv
                mod = PyEnv.runtime_import_module(mod_path, warn=False)
            except Exception:
                mod = None
        except Exception:
            mod = None
        if mod is not None:
            obj = getattr(mod, attr, None)
            if obj is not None:
                return obj

    return s


# ---------------------------------------------------------------------------
# Signature description (small, JSON-friendly)
# ---------------------------------------------------------------------------

def _canonical_module_path(cls: type) -> str:
    """Return the shortest module path that re-exposes *cls* by qualname.

    Type ``__module__`` often points at the internal C-extension /
    private submodule (``pyarrow.lib.Table``,
    ``polars.dataframe.frame.DataFrame``,
    ``pandas.core.frame.DataFrame``) rather than the canonical import
    path users actually type (``pyarrow.Table``, ``polars.DataFrame``,
    ``pandas.DataFrame``). Walk the dotted module hierarchy from the
    shortest prefix down and pick the first one whose attribute
    lookup along ``__qualname__`` yields the same class. Falls back to
    ``cls.__module__`` when nothing matches.
    """
    mod = getattr(cls, "__module__", None)
    if not mod or mod == "builtins":
        return ""
    qualname = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None)
    if not qualname:
        return mod
    parts = mod.split(".")
    for i in range(1, len(parts) + 1):
        candidate = ".".join(parts[:i])
        try:
            module = importlib.import_module(candidate)
        except Exception:
            continue
        obj: Any = module
        for seg in qualname.split("."):
            obj = getattr(obj, seg, None)
            if obj is None:
                break
        if obj is cls:
            return candidate
    return mod


def _annotation_to_str(ann: Any) -> Optional[str]:
    """Render a parameter annotation as a short, JSON-friendly string.

    Builtins (``int`` / ``str`` / ``bool`` / ``list`` / …) render as
    their bare name. Non-builtin classes render with their full
    canonical module path via :func:`_canonical_module_path`, so
    ``pyarrow.Table`` / ``polars.DataFrame`` / ``pandas.DataFrame``
    survive into the staged task metadata as the dotted import names
    a reader would type. PEP 563 strings and typing generics
    (``Optional[int]``, ``list[int]``) round-trip via :func:`repr`.
    """
    if ann is inspect.Parameter.empty or ann is None:
        return None
    if isinstance(ann, str):
        # ``from __future__ import annotations`` and PEP 604 stringified
        # forms arrive as a plain string already — keep them verbatim.
        return ann
    if isinstance(ann, type):
        qualname = getattr(ann, "__qualname__", None) or ann.__name__
        mod = _canonical_module_path(ann)
        return qualname if not mod else f"{mod}.{qualname}"
    return repr(ann)


def _default_to_str(default: Any) -> tuple[bool, Optional[str]]:
    """Return ``(has_default, repr(default))`` for a parameter default."""
    if default is inspect.Parameter.empty:
        return False, None
    try:
        return True, repr(default)
    except Exception:
        return True, "<unrepresentable>"


def _resolved_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    """Resolve string annotations on *func* into real types when possible.

    Two-pass best effort so a single unresolvable annotation doesn't
    blow up the rest:

    1. :func:`inspect.get_annotations` with ``eval_str=True`` —
       evaluates every annotation in the function's globals + builtins
       in one shot. Fast path when the function's own module imports
       all the referenced types.
    2. Per-annotation :func:`_resolve_str_annotation` fallback — tries
       func globals, then ``typing``, then fast alias-prefix expansion
       (``pa.Table`` → ``pyarrow.Table``) + dotted module import.

    Anything still left as a string after both passes is returned
    as-is. The caller (:func:`check_function_args`) treats string
    annotations as "couldn't resolve, skip coercion" rather than
    raising — the goal is to coerce what we can and let the rest
    flow through.
    """
    raw = dict(getattr(func, "__annotations__", {}) or {})
    if not raw:
        return raw

    # Fast path — inspect.get_annotations with eval_str.
    try:
        return dict(inspect.get_annotations(func, eval_str=True))
    except Exception as exc:
        LOGGER.debug(
            "_resolved_annotations: get_annotations(eval_str=True) failed "
            "for %r — falling back to per-annotation resolution (%s)",
            getattr(func, "__qualname__", func), exc,
        )

    # Per-annotation fallback — partial wins still help.
    func_globals = getattr(func, "__globals__", None)
    resolved: dict[str, Any] = {}
    for name, ann in raw.items():
        if isinstance(ann, str):
            resolved[name] = _resolve_str_annotation(ann, func_globals=func_globals)
        else:
            resolved[name] = ann
    return resolved


def describe_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Capture *func*'s signature as a JSON-serializable dict.

    Returns ``{"qualname", "module", "parameters": [...], "return"}``
    where each parameter entry carries ``name``, ``kind`` (the
    :class:`inspect.Parameter.kind` name), ``annotation`` (dotted path
    when the annotation is a class), and ``default`` (``repr`` of the
    default) where present.
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


# ---------------------------------------------------------------------------
# Coercion — pure utility + decorator
# ---------------------------------------------------------------------------

def check_function_args(
    func: Callable[..., Any],
    args: tuple = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[tuple, dict[str, Any]]:
    """Coerce *args* / *kwargs* to match *func*'s annotated signature.

    Walks :func:`inspect.signature` and routes each value whose
    parameter has a non-empty annotation through
    :func:`yggdrasil.data.cast.convert`. Returns coerced
    ``(args, kwargs)`` ready to splat back into the call — positional
    inputs stay positional, keyword inputs stay keyword (no
    pos-to-kw rewrite).

    ``*args`` (``VAR_POSITIONAL``) and ``**kwargs`` (``VAR_KEYWORD``)
    annotations apply per element, so ``def f(*xs: int)`` coerces
    every element of ``xs`` to ``int``. Excess positional or
    unknown keyword inputs (no matching parameter, no ``**kwargs``
    catch-all) pass through untouched so the natural ``TypeError``
    fires on call rather than being silently swallowed here.

    Empty input short-circuits — no yggdrasil import fires.
    """
    kwargs_dict: dict[str, Any] = dict(kwargs or {})
    if not args and not kwargs_dict:
        return args, kwargs_dict

    from yggdrasil.data.cast import convert

    sig = inspect.signature(func)
    resolved = _resolved_annotations(func)

    positional_params: list[inspect.Parameter] = []
    var_positional: Optional[inspect.Parameter] = None
    var_keyword: Optional[inspect.Parameter] = None
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = p
        elif p.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = p
        elif p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_params.append(p)

    def _coerce(name: str, value: Any, fallback: Any) -> Any:
        ann = resolved.get(name, fallback)
        if ann is inspect.Parameter.empty:
            return value
        if isinstance(ann, str):
            # Couldn't resolve the annotation to a real type — pass
            # the value through. The function decides what to do with
            # it; we'd rather be permissive than raise a NameError-shaped
            # surprise from inside the coercion path.
            LOGGER.debug(
                "check_function_args: unresolved annotation %r=%r — "
                "skipping coercion", name, ann,
            )
            return value
        try:
            return convert(value, ann)
        except (TypeError, ValueError) as exc:
            # No converter registered (e.g. ``Union[int, str]``) or the
            # converter itself raised — pass through so the function
            # gets the chance to handle it, rather than failing here.
            LOGGER.debug(
                "check_function_args: convert(%s -> %r) failed for %r — "
                "skipping coercion (%s)",
                type(value).__name__, ann, name, exc,
            )
            return value

    coerced_args: list[Any] = []
    for i, value in enumerate(args):
        if i < len(positional_params):
            p = positional_params[i]
            coerced_args.append(_coerce(p.name, value, p.annotation))
        elif var_positional is not None:
            coerced_args.append(
                _coerce(var_positional.name, value, var_positional.annotation)
            )
        else:
            # Excess positional — let the real call raise TypeError.
            coerced_args.append(value)

    coerced_kwargs: dict[str, Any] = {}
    for name, value in kwargs_dict.items():
        p = sig.parameters.get(name)
        if p is None:
            if var_keyword is not None:
                coerced_kwargs[name] = _coerce(
                    var_keyword.name, value, var_keyword.annotation,
                )
            else:
                coerced_kwargs[name] = value
            continue
        coerced_kwargs[name] = _coerce(p.name, value, p.annotation)

    return tuple(coerced_args), coerced_kwargs


def checkargs(func: F) -> F:
    """Wrap *func* so every call has its args coerced to the annotated types.

    Built on :func:`check_function_args` — every invocation routes
    incoming ``args`` / ``kwargs`` through the coercion pass before
    the real call. Annotated parameters receive values converted via
    :func:`yggdrasil.data.cast.convert`; unannotated parameters pass
    through. :func:`functools.wraps` preserves ``__name__``,
    ``__qualname__``, ``__doc__``, ``__annotations__``, and the
    underlying ``__wrapped__`` so :func:`inspect.signature` still
    reports the original signature.

    Coroutine functions (``async def``) get an ``async`` wrapper that
    awaits the underlying call; sync functions get a plain wrapper.
    Re-wrapping is idempotent — applying ``@checkargs`` twice unwraps
    the inner ``__wrapped__`` so the second decoration doesn't add a
    second coercion pass.
    """
    # Idempotent: if the caller stacks ``@checkargs`` twice, peel back
    # to the original so we don't double-wrap.
    if getattr(func, "__checkargs_wrapped__", False):
        func = func.__wrapped__  # type: ignore[attr-defined]

    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            coerced_args, coerced_kwargs = check_function_args(func, args, kwargs)
            return await func(*coerced_args, **coerced_kwargs)

        _async_wrapper.__checkargs_wrapped__ = True  # type: ignore[attr-defined]
        return _async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        coerced_args, coerced_kwargs = check_function_args(func, args, kwargs)
        return func(*coerced_args, **coerced_kwargs)

    _wrapper.__checkargs_wrapped__ = True  # type: ignore[attr-defined]
    return _wrapper  # type: ignore[return-value]
