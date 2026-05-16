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
import inspect
from typing import Any, Callable, Mapping, Optional, TypeVar

__all__ = [
    "check_function_args",
    "checkargs",
    "describe_signature",
    "format_signature",
]

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Signature description (small, JSON-friendly)
# ---------------------------------------------------------------------------

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
        return convert(value, ann)

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
    """
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        coerced_args, coerced_kwargs = check_function_args(func, args, kwargs)
        return func(*coerced_args, **coerced_kwargs)

    return _wrapper  # type: ignore[return-value]
