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
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, TypeVar

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "build_batch_invoker",
    "build_row_invoker",
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

class _SignaturePlan:
    """Pre-computed dispatch plan for a callable's signature.

    Inspects *func* once and stashes the data
    :func:`check_function_args` and :func:`build_row_invoker` need on
    every call: the positional / var-positional / var-keyword
    parameter slots, the resolved annotation map, and a coercer
    closure that maps ``(args, kwargs)`` to coerced ``(args, kwargs)``.

    Per-row dispatch goes through the same plan so an N-row
    ``Dataset.apply(func)`` pays the inspection cost once, not N
    times. Resolves to a no-op coercer when :func:`inspect.signature`
    rejects *func* (C builtins, callables without a Python signature)
    so unintrospectable callables still flow through.
    """

    __slots__ = (
        "func",
        "sig",
        "resolved",
        "positional_params",
        "keyword_only_params",
        "var_positional",
        "var_keyword",
        "declared_keywords",
        "coercer",
        "introspectable",
    )

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        positional_params: list[inspect.Parameter] = []
        keyword_only_params: list[inspect.Parameter] = []
        var_positional: Optional[inspect.Parameter] = None
        var_keyword: Optional[inspect.Parameter] = None
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            self.sig = None
            self.resolved = {}
            self.positional_params = positional_params
            self.keyword_only_params = keyword_only_params
            self.var_positional = var_positional
            self.var_keyword = var_keyword
            self.declared_keywords = frozenset()
            self.coercer = _identity_coercer
            self.introspectable = False
            return

        self.sig = sig
        self.resolved = _resolved_annotations(func)
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                var_positional = p
            elif p.kind is inspect.Parameter.VAR_KEYWORD:
                var_keyword = p
            elif p.kind is inspect.Parameter.KEYWORD_ONLY:
                keyword_only_params.append(p)
            else:
                positional_params.append(p)
        self.positional_params = positional_params
        self.keyword_only_params = keyword_only_params
        self.var_positional = var_positional
        self.var_keyword = var_keyword
        # Names callable by keyword (POSITIONAL_ONLY excluded).
        self.declared_keywords = frozenset(
            p.name for p in (*positional_params, *keyword_only_params)
            if p.kind is not inspect.Parameter.POSITIONAL_ONLY
        )
        self.coercer = self._make_coercer()
        self.introspectable = True

    def _make_coercer(self):
        resolved = self.resolved
        positional_params = self.positional_params
        var_positional = self.var_positional
        var_keyword = self.var_keyword
        sig_parameters = self.sig.parameters

        from yggdrasil.data.cast import convert

        def _coerce(name: str, value: Any, fallback: Any) -> Any:
            ann = resolved.get(name, fallback)
            if ann is inspect.Parameter.empty:
                return value
            if isinstance(ann, str):
                LOGGER.debug(
                    "_SignaturePlan: unresolved annotation %r=%r — "
                    "skipping coercion", name, ann,
                )
                return value
            try:
                return convert(value, ann)
            except (TypeError, ValueError) as exc:
                LOGGER.debug(
                    "_SignaturePlan: convert(%s -> %r) failed for %r — "
                    "skipping coercion (%s)",
                    type(value).__name__, ann, name, exc,
                )
                return value

        n_positional = len(positional_params)

        def _coercer(args: tuple, kwargs: Mapping[str, Any]):
            coerced_args: list[Any] = []
            for i, value in enumerate(args):
                if i < n_positional:
                    p = positional_params[i]
                    coerced_args.append(_coerce(p.name, value, p.annotation))
                elif var_positional is not None:
                    coerced_args.append(_coerce(
                        var_positional.name, value, var_positional.annotation,
                    ))
                else:
                    coerced_args.append(value)
            coerced_kwargs: dict[str, Any] = {}
            for name, value in kwargs.items():
                p = sig_parameters.get(name)
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

        return _coercer


def _identity_coercer(args: tuple, kwargs: Mapping[str, Any]) -> tuple[tuple, dict[str, Any]]:
    """Pass-through coercer used when a callable can't be introspected."""
    return tuple(args), dict(kwargs)


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

    Hot-path callers that invoke the same function repeatedly
    should reach for :func:`build_row_invoker` instead, which
    caches the :class:`_SignaturePlan` across calls.
    """
    kwargs_dict: dict[str, Any] = dict(kwargs or {})
    if not args and not kwargs_dict:
        return args, kwargs_dict
    plan = _SignaturePlan(func)
    return plan.coercer(args, kwargs_dict)


# ---------------------------------------------------------------------------
# Per-row dispatch — adapt a callable to the row shape it receives
# ---------------------------------------------------------------------------

def build_row_invoker(func: Callable[..., Any]) -> Callable[[Any], Any]:
    """Return a per-row dispatcher that adapts row shape to *func*'s signature.

    Pre-computes the signature once and returns a callable
    ``invoker(row) -> result`` that:

    * Passes ``row`` directly when *func* has exactly one
      positional parameter and no ``**kwargs`` catch-all
      (the common ``def f(row): ...`` shape).
    * Spreads ``row`` as ``**kwargs`` when *func* has multiple
      named parameters or accepts ``**kwargs`` and ``row`` is a
      :class:`Mapping`. Keys missing from the declared parameter
      list are dropped unless a ``**kwargs`` catch-all is present;
      keys missing from the row stay unset (so defaults apply).
    * Spreads ``row`` as ``*args`` when *func* has only a
      ``*args`` catch-all (no other positional param) and ``row``
      is a :class:`tuple` or :class:`list`.
    * Falls back to ``func(row)`` for anything else.

    Coerces annotated arguments through the pre-built
    :class:`_SignaturePlan` coercer so a function annotated
    ``def f(id: int, name: str)`` called against a dict with
    string keys still gets its ``id`` coerced to ``int``.

    When the dict-spread call raises ``TypeError`` (e.g. the function
    rejected the spread shape), the invoker retries with
    ``func(row)`` once so a row that happens to be a ``dict`` but
    means "an opaque mapping value" still reaches the function.
    Other exceptions propagate.

    Picklable: the returned closure references *func* and the plan
    by reference; both pickle through the standard cloudpickle path
    used elsewhere in :mod:`yggdrasil.pickle`.
    """
    plan = _SignaturePlan(func)
    coercer = plan.coercer

    if not plan.introspectable:
        # No signature to dispatch on — call func(row) and let it
        # raise the natural TypeError if the row shape is wrong.
        def _invoker_uninspectable(row: Any) -> Any:
            return func(row)
        return _invoker_uninspectable

    has_var_kw = plan.var_keyword is not None
    has_var_pos = plan.var_positional is not None
    n_positional = len(plan.positional_params)
    declared_keywords = plan.declared_keywords
    keyword_only = plan.keyword_only_params

    # Shape 1: one positional param, no **kwargs / keyword-only — pass
    # the row through as the sole argument. Covers the
    # ``def f(row): ...`` / ``def f(x: int): ...`` / lambdas case.
    if n_positional == 1 and not has_var_kw and not keyword_only:
        only_param = plan.positional_params[0]
        only_name = only_param.name
        only_ann = plan.resolved.get(only_name, only_param.annotation)
        if only_ann is inspect.Parameter.empty or isinstance(only_ann, str):
            # No coercion to apply — call straight through. Fastest path.
            def _invoker_single(row: Any) -> Any:
                return func(row)
            return _invoker_single

        from yggdrasil.data.cast import convert

        # Fast skip for the common "row already matches annotation" case —
        # avoids the convert() entry, isinstance recheck, and option
        # normalisation for ~50% of typed rows. The ``isinstance`` guard
        # is cheap (C-level type check) and ``type`` annotations are the
        # overwhelmingly common shape; generic aliases / unions fall
        # through to the convert() path the same way ``convert`` does
        # internally.
        ann_is_type = isinstance(only_ann, type)

        def _invoker_single_typed(row: Any) -> Any:
            # Whole row already matches the annotation — straight through.
            if ann_is_type and isinstance(row, only_ann):
                return func(row)
            # Mapping row + arg name matches a key → the caller's intent
            # is "give me ``row[arg_name]``", not the whole mapping. Covers
            # the common ``def f(id: int)`` over typed-mode dict rows /
            # dynamic-mode dict-shaped pickled objects without forcing the
            # caller to switch to ``def f(id: int, name: str)`` just to
            # access one column. ``def f(row: dict)`` still gets the whole
            # row via the ``isinstance(row, only_ann)`` fast-skip above.
            if isinstance(row, Mapping) and only_name in row:
                value = row[only_name]
                if ann_is_type and isinstance(value, only_ann):
                    return func(value)
                try:
                    value = convert(value, only_ann)
                except (TypeError, ValueError):
                    pass
                return func(value)
            try:
                row = convert(row, only_ann)
            except (TypeError, ValueError):
                pass
            return func(row)
        return _invoker_single_typed

    # Shape 2: zero positional + **kwargs catch-all — dict rows
    # spread as kwargs, anything else gets passed through positional.
    if n_positional == 0 and has_var_kw and not keyword_only:
        def _invoker_var_kw(row: Any) -> Any:
            if isinstance(row, Mapping):
                args, kwargs = coercer((), row)
            else:
                args, kwargs = coercer((row,), {})
            return func(*args, **kwargs)
        return _invoker_var_kw

    # Shape 3: zero positional + *args catch-all — tuple/list rows
    # spread as positional, anything else gets passed as a single arg.
    if n_positional == 0 and has_var_pos and not has_var_kw and not keyword_only:
        def _invoker_var_pos(row: Any) -> Any:
            if isinstance(row, (tuple, list)):
                args, kwargs = coercer(tuple(row), {})
            else:
                args, kwargs = coercer((row,), {})
            return func(*args, **kwargs)
        return _invoker_var_pos

    # Shape 4: generic — multi-arg signatures, or mixes that need
    # per-row dispatch. Mapping rows spread as kwargs (filtered to
    # declared names unless **kwargs catches the rest); tuple/list
    # rows spread as *args when there's a var-positional slot;
    # everything else falls through as a single positional arg.
    def _invoker_generic(row: Any) -> Any:
        if isinstance(row, Mapping):
            if has_var_kw:
                kw = dict(row)
            else:
                kw = {k: v for k, v in row.items() if k in declared_keywords}
            args, kwargs = coercer((), kw)
            try:
                return func(*args, **kwargs)
            except TypeError:
                # The dict-as-kwargs spread didn't match the
                # function's actual signature (missing required
                # arg, wrong shape, …). Best-effort fallback: try
                # the single-arg form before giving up.
                args, kwargs = coercer((row,), {})
                return func(*args, **kwargs)
        if isinstance(row, (tuple, list)) and has_var_pos and n_positional == 0:
            args, kwargs = coercer(tuple(row), {})
            return func(*args, **kwargs)
        args, kwargs = coercer((row,), {})
        return func(*args, **kwargs)

    return _invoker_generic


# ---------------------------------------------------------------------------
# Per-batch dispatch — vectorise the column cast when the function takes
# a single column by name + type hint
# ---------------------------------------------------------------------------

def build_batch_invoker(
    func: Callable[..., Any],
) -> Callable[["pa.RecordBatch"], list[Any]]:
    """Return a per-batch dispatcher ``invoker(batch) -> list[Any]``.

    When *func* has a single positional annotated parameter whose name
    matches a column in the incoming :class:`pyarrow.RecordBatch`, the
    whole column is cast to the target Arrow type via
    :func:`pyarrow.compute.cast` (vectorised, one C++ kernel call) and
    then iterated through *func* — skipping the per-row dict
    reconstruction that ``batch.to_pylist()`` would otherwise do.

    Falls back to per-row dispatch via :func:`build_row_invoker` for
    any other shape (multi-arg, ``**kwargs``, no column name match,
    annotation that doesn't map to an Arrow type, …). The fallback
    materialises ``batch.to_pylist()`` once and calls the row invoker
    over the resulting dicts — exactly what the caller would have
    done by hand, kept inside this helper so apply pipelines have a
    single dispatch point.

    The fast path mirrors the dict-key extraction
    :func:`build_row_invoker` already does on dynamic-mode rows,
    moved up one level to operate over a whole Arrow column at a
    time. Per-row :func:`yggdrasil.data.cast.convert` calls collapse
    into one ``pa.compute.cast`` plus an isinstance fast-skip when
    the column already carries the right type.
    """
    plan = _SignaturePlan(func)
    row_invoker = build_row_invoker(func)

    arrow_target = None
    only_name: Optional[str] = None
    only_ann: Any = None
    if (plan.introspectable
            and len(plan.positional_params) == 1
            and not plan.var_keyword
            and not plan.keyword_only_params):
        only_param = plan.positional_params[0]
        only_name = only_param.name
        only_ann = plan.resolved.get(only_name, only_param.annotation)
        if isinstance(only_ann, type):
            # Map ``int`` / ``float`` / ``str`` / ``bool`` / ``bytes`` /
            # ``datetime`` / ``date`` / ``Decimal`` … to the Arrow type
            # the cast registry would target. Generic aliases / unions /
            # PEP-563-unresolved strings stay ``None`` and the path falls
            # back to per-row dispatch.
            try:
                from yggdrasil.data.types.base import DataType
                arrow_target = DataType.from_pytype(only_ann).to_arrow()
            except Exception:
                arrow_target = None

    if only_name is None or arrow_target is None:
        # No vectorisable shape — straight per-row dispatch over the
        # batch's row dicts. Skipping the early-empty check here is
        # fine: ``batch.to_pylist()`` on a zero-row batch yields ``[]``
        # and the comprehension drops out the same way.
        def _batch_row_only(batch: "pa.RecordBatch") -> list[Any]:
            return [row_invoker(r) for r in batch.to_pylist()]
        return _batch_row_only

    ann_is_type = isinstance(only_ann, type)

    def _batch_invoker(batch: "pa.RecordBatch") -> list[Any]:
        # Look up the column by arg name. Missing column → fall back to
        # the row-by-row path (which lets ``row_invoker`` find the
        # value elsewhere or call ``func(row)`` directly).
        try:
            col = batch.column(only_name)
        except (KeyError, ValueError):
            return [row_invoker(r) for r in batch.to_pylist()]

        import pyarrow as _pa
        # Vectorise the cast — one C++ kernel call instead of N Python
        # ``convert(value, ann)`` round trips. ``safe=True`` keeps the
        # existing pickle-vs-convert error tone (overflow / inexact
        # raises rather than silently truncates); the fallback below
        # absorbs the raise so a partly-castable column still reaches
        # the function.
        if col.type != arrow_target:
            try:
                import pyarrow.compute as _pc
                col = _pc.cast(col, arrow_target)
            except (_pa.ArrowInvalid, _pa.ArrowNotImplementedError,
                    _pa.ArrowTypeError):
                # Cast not supported for this column shape — defer to
                # per-row ``convert`` via the row invoker, which has
                # access to the wider yggdrasil cast registry.
                return [row_invoker(r) for r in batch.to_pylist()]

        # Genuine row endpoint: the workload IS "yield Python values
        # to a user function". ``Array.to_pylist`` on a primitive
        # column is the canonical Arrow→Python bridge for this shape
        # — same C-bridge cost as ``[col[i].as_py() for ...]`` with
        # one less Python frame per cell.
        values = col.to_pylist()
        # When the column was already the right Arrow type AND the
        # Python value's runtime type matches the annotation, skip
        # the per-cell isinstance check entirely (the cast either
        # succeeded loud or was unnecessary). For all other cases the
        # isinstance gate keeps the row invoker out of the hot loop.
        if ann_is_type:
            # ``None`` cells skip the isinstance check (None isn't an
            # instance of ``int`` / ``str`` / … but the function may
            # still accept it via Optional). Pass them through unchanged.
            out: list[Any] = []
            for v in values:
                if v is None or isinstance(v, only_ann):
                    out.append(func(v))
                else:
                    out.append(row_invoker(v))
            return out
        return [func(v) for v in values]
    return _batch_invoker


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
