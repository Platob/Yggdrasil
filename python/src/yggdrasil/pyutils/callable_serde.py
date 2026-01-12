"""pyutils.callable_serde module documentation."""

from __future__ import annotations

import base64
import dis
import importlib
import inspect
import json
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypeVar, Union

import dill

__all__ = ["CallableSerde"]

T = TypeVar("T", bound="CallableSerde")


# ---------- internal helpers ----------

_MAGIC = b"CS1"  # CallableSerde framing v1
_FLAG_COMPRESSED = 1


def _resolve_attr_chain(mod: Any, qualname: str) -> Any:
    """
    _resolve_attr_chain documentation.
    
    Args:
        mod: Parameter.
        qualname: Parameter.
    
    Returns:
        The result.
    """

    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _find_pkg_root_from_file(file_path: Path) -> Optional[Path]:
    """
    Walk up parents while __init__.py exists.
    Return the directory that should be on sys.path (parent of top package dir).
    """
    file_path = file_path.resolve()
    d = file_path.parent

    top_pkg_dir = None
    while (d / "__init__.py").is_file():
        top_pkg_dir = d
        d = d.parent

    return top_pkg_dir if top_pkg_dir else None


def _callable_file_line(fn: Callable[..., Any]) -> Tuple[Optional[str], Optional[int]]:
    """
    _callable_file_line documentation.
    
    Args:
        fn: Parameter.
    
    Returns:
        The result.
    """

    file = None
    line = None
    try:
        file = inspect.getsourcefile(fn) or inspect.getfile(fn)
    except Exception:
        file = None
    if file:
        try:
            _, line = inspect.getsourcelines(fn)
        except Exception:
            line = None
    return file, line


def _referenced_global_names(fn: Callable[..., Any]) -> Set[str]:
    """
    Names that the function *actually* resolves from globals/namespaces at runtime.
    Uses bytecode to avoid shipping random junk.
    """
    names: Set[str] = set()
    try:
        for ins in dis.get_instructions(fn):
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str):
                names.add(ins.argval)
    except Exception:
        # fallback: less precise
        try:
            names.update(getattr(fn.__code__, "co_names", ()) or ())
        except Exception:
            pass

    names.discard("__builtins__")
    return names


def _is_importable_reference(fn: Callable[..., Any]) -> bool:
    """
    _is_importable_reference documentation.
    
    Args:
        fn: Parameter.
    
    Returns:
        The result.
    """

    mod_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not mod_name or not qualname:
        return False
    if "<locals>" in qualname:
        return False
    try:
        mod = importlib.import_module(mod_name)
        obj = _resolve_attr_chain(mod, qualname)
        return callable(obj)
    except Exception:
        return False


def _pick_zlib_level(n: int, limit: int) -> int:
    """
    Ramp compression level 1..9 based on how much payload exceeds byte_limit.
    ratio=1 -> level=1
    ratio=4 -> level=9
    clamp beyond.
    """
    ratio = n / max(1, limit)
    x = min(1.0, max(0.0, (ratio - 1.0) / 3.0))
    return max(1, min(9, int(round(1 + 8 * x))))


def _encode_result_blob(raw: bytes, byte_limit: int) -> bytes:
    """
    For small payloads: return raw dill bytes (backwards compat).
    For large payloads: wrap in framed header + zlib-compressed bytes (if beneficial).
    """
    if len(raw) <= byte_limit:
        return raw

    level = _pick_zlib_level(len(raw), byte_limit)
    compressed = zlib.compress(raw, level)

    # If compression doesn't help, keep raw
    if len(compressed) >= len(raw):
        return raw

    # Frame:
    # MAGIC(3) + flags(u8) + orig_len(u32) + level(u8) + data
    header = _MAGIC + struct.pack(">BIB", _FLAG_COMPRESSED, len(raw), level)
    return header + compressed


def _decode_result_blob(blob: bytes) -> bytes:
    """
    If framed, decompress if flagged, return raw dill bytes.
    Else treat as raw dill bytes.
    """
    if not blob.startswith(_MAGIC):
        return blob

    if len(blob) < 3 + 1 + 4 + 1:
        raise ValueError("Framed result too short / corrupted.")

    flags, orig_len, _level = struct.unpack(">BIB", blob[3 : 3 + 1 + 4 + 1])
    data = blob[3 + 1 + 4 + 1 :]

    if flags & _FLAG_COMPRESSED:
        raw = zlib.decompress(data)
        if orig_len and len(raw) != orig_len:
            raise ValueError(f"Decompressed length mismatch: got {len(raw)}, expected {orig_len}")
        return raw

    return data


def _dump_env(
    fn: Callable[..., Any],
    *,
    include_globals: bool,
    include_closure: bool,
    filter_used_globals: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (env, meta).
    env is dill-able and contains:
      - "globals": {name: value}  (filtered to used names if enabled)
      - "closure": {freevar: value} (capture only; injection not generally safe)
    """
    env: Dict[str, Any] = {}
    meta: Dict[str, Any] = {
        "missing_globals": [],
        "skipped_globals": [],
        "skipped_closure": [],
        "filter_used_globals": bool(filter_used_globals),
    }

    if include_globals:
        g = getattr(fn, "__globals__", None) or {}
        names = sorted(_referenced_global_names(fn)) if filter_used_globals else sorted(set(g.keys()))

        env_g: Dict[str, Any] = {}
        for name in names:
            if name not in g:
                meta["missing_globals"].append(name)
                continue
            try:
                dill.dumps(g[name], recurse=True)
                env_g[name] = g[name]
            except Exception:
                meta["skipped_globals"].append(name)

        if env_g:
            env["globals"] = env_g

    if include_closure:
        freevars = getattr(getattr(fn, "__code__", None), "co_freevars", ()) or ()
        closure = getattr(fn, "__closure__", None) or ()

        clo: Dict[str, Any] = {}
        if freevars and closure and len(freevars) == len(closure):
            for name, cell in zip(freevars, closure):
                try:
                    val = cell.cell_contents
                    dill.dumps(val, recurse=True)
                    clo[name] = val
                except Exception:
                    meta["skipped_closure"].append(name)

        if clo:
            env["closure"] = clo

    return env, meta


# ---------- main class ----------

@dataclass
class CallableSerde:
    """
    Core field: `fn`
    Serialized/backing fields used when fn isn't present yet.

    kind:
      - "auto": resolve import if possible else dill
      - "import": module + qualname
      - "dill": dill_b64

    Optional env payload:
      - env_b64: dill(base64) of {"globals": {...}, "closure": {...}}
    """
    fn: Optional[Callable[..., Any]] = None

    _kind: str = "auto"  # "auto" | "import" | "dill"
    _module: Optional[str] = None
    _qualname: Optional[str] = None
    _pkg_root: Optional[str] = None
    _dill_b64: Optional[str] = None

    _env_b64: Optional[str] = None
    _env_meta: Optional[Dict[str, Any]] = None

    # ----- construction -----

    @classmethod
    def from_callable(cls: type[T], x: Union[Callable[..., Any], T]) -> T:
        """
        from_callable documentation.
        
        Args:
            x: Parameter.
        
        Returns:
            The result.
        """

        if isinstance(x, cls):
            return x

        obj = cls(fn=x)  # type: ignore[return-value]

        return obj

    # ----- lazy-ish properties (computed on access) -----

    @property
    def module(self) -> Optional[str]:
        """
        module documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self._module or (getattr(self.fn, "__module__", None) if self.fn else None)

    @property
    def qualname(self) -> Optional[str]:
        """
        qualname documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        return self._qualname or (getattr(self.fn, "__qualname__", None) if self.fn else None)

    @property
    def file(self) -> Optional[str]:
        """
        file documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.fn:
            return None
        f, _ = _callable_file_line(self.fn)
        return f

    @property
    def line(self) -> Optional[int]:
        """
        line documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.fn:
            return None
        _, ln = _callable_file_line(self.fn)
        return ln

    @property
    def pkg_root(self) -> Optional[str]:
        """
        pkg_root documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self._pkg_root:
            return self._pkg_root
        if not self.file:
            return None
        root = _find_pkg_root_from_file(Path(self.file))
        return str(root) if root else None

    @property
    def relpath_from_pkg_root(self) -> Optional[str]:
        """
        relpath_from_pkg_root documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if not self.file or not self.pkg_root:
            return None
        try:
            return str(Path(self.file).resolve().relative_to(Path(self.pkg_root).resolve()))
        except Exception:
            return self.file

    @property
    def importable(self) -> bool:
        """
        importable documentation.
        
        Args:
            None.
        
        Returns:
            The result.
        """

        if self.fn is None:
            return bool(self.module and self.qualname and "<locals>" not in (self.qualname or ""))
        return _is_importable_reference(self.fn)

    # ----- serde API -----

    def dump(
        self,
        *,
        prefer: str = "import",          # "import" | "dill"
        dump_env: str = "none",          # "none" | "globals" | "closure" | "both"
        filter_used_globals: bool = True,
    ) -> Dict[str, Any]:
        """
        dump documentation.
        
        Args:
            prefer: Parameter.
            dump_env: Parameter.
            filter_used_globals: Parameter.
        
        Returns:
            The result.
        """

        kind = prefer
        if kind == "import" and not self.importable:
            kind = "dill"

        out: Dict[str, Any] = {
            "kind": kind,
            "module": self.module,
            "qualname": self.qualname,
            "pkg_root": self.pkg_root,
            "file": self.file,
            "line": self.line,
            "relpath_from_pkg_root": self.relpath_from_pkg_root,
        }

        if kind == "dill":
            if self._dill_b64 is None:
                if self.fn is None:
                    raise ValueError("No callable available to dill-dump.")
                payload = dill.dumps(self.fn, recurse=True)
                self._dill_b64 = base64.b64encode(payload).decode("ascii")
            out["dill_b64"] = self._dill_b64

        if dump_env != "none":
            if self.fn is None:
                raise ValueError("dump_env requested but fn is not present.")
            include_globals = dump_env in ("globals", "both")
            include_closure = dump_env in ("closure", "both")
            env, meta = _dump_env(
                self.fn,
                include_globals=include_globals,
                include_closure=include_closure,
                filter_used_globals=filter_used_globals,
            )
            self._env_meta = meta
            if env:
                self._env_b64 = base64.b64encode(dill.dumps(env, recurse=True)).decode("ascii")
                out["env_b64"] = self._env_b64
                out["env_meta"] = meta

        return out

    @classmethod
    def load(cls: type[T], d: Dict[str, Any], *, add_pkg_root_to_syspath: bool = True) -> T:
        """
        load documentation.
        
        Args:
            d: Parameter.
            add_pkg_root_to_syspath: Parameter.
        
        Returns:
            The result.
        """

        obj = cls(
            fn=None,
            _kind=d.get("kind", "auto"),
            _module=d.get("module"),
            _qualname=d.get("qualname"),
            _pkg_root=d.get("pkg_root"),
            _dill_b64=d.get("dill_b64"),
        )
        obj._env_b64 = d.get("env_b64")
        obj._env_meta = d.get("env_meta")

        if add_pkg_root_to_syspath and obj._pkg_root and obj._pkg_root not in sys.path:
            sys.path.insert(0, obj._pkg_root)

        return obj  # type: ignore[return-value]

    def materialize(self, *, add_pkg_root_to_syspath: bool = True) -> Callable[..., Any]:
        """
        materialize documentation.
        
        Args:
            add_pkg_root_to_syspath: Parameter.
        
        Returns:
            The result.
        """

        if self.fn is not None:
            return self.fn

        if add_pkg_root_to_syspath and self.pkg_root and self.pkg_root not in sys.path:
            sys.path.insert(0, self.pkg_root)

        kind = self._kind
        if kind == "auto":
            kind = "import" if (self.module and self.qualname and "<locals>" not in (self.qualname or "")) else "dill"

        if kind == "import":
            if not self.module or not self.qualname:
                raise ValueError("Missing module/qualname for import load.")
            mod = importlib.import_module(self.module)
            fn = _resolve_attr_chain(mod, self.qualname)
            if not callable(fn):
                raise TypeError("Imported object is not callable.")
            self.fn = fn
            return fn

        if kind == "dill":
            if not self._dill_b64:
                raise ValueError("Missing dill_b64 for dill load.")
            payload = base64.b64decode(self._dill_b64.encode("ascii"))
            fn = dill.loads(payload)
            if not callable(fn):
                raise TypeError("Dill payload did not decode to a callable.")
            self.fn = fn
            return fn

        raise ValueError(f"Unknown kind: {kind}")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        __call__ documentation.
        
        Args:
            *args: Parameter.
            **kwargs: Parameter.
        
        Returns:
            The result.
        """

        fn = self.materialize()
        return fn(*args, **kwargs)

    # ----- command execution bridge -----

    def to_command(
        self,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        result_tag: str = "__CALLABLE_SERDE_RESULT__",
        prefer: str = "dill",
        byte_limit: int = 256_000,
        dump_env: str = "none",  # "none" | "globals" | "closure" | "both"
        filter_used_globals: bool = True,
    ) -> str:
        """
        Returns Python code string to execute in another interpreter.
        Prints one line: "{result_tag}:{base64(blob)}"
        where blob is raw dill bytes or framed+zlib.
        """
        args = args or ()
        kwargs = kwargs or {}

        serde_dict = self.dump(
            prefer=prefer,
            dump_env=dump_env,
            filter_used_globals=filter_used_globals,
        )
        serde_json = json.dumps(serde_dict, ensure_ascii=False)

        call_payload_b64 = base64.b64encode(
            dill.dumps((args, kwargs), recurse=True)
        ).decode("ascii")

        # NOTE: plain string template + replace. No f-string. No brace escaping.
        template = r"""
import base64, json, sys, struct, zlib, importlib, dis
import dill

RESULT_TAG = __RESULT_TAG__
BYTE_LIMIT = __BYTE_LIMIT__

MAGIC = b"CS1"
FLAG_COMPRESSED = 1

def _resolve_attr_chain(mod, qualname: str):
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj

def _pick_level(n: int, limit: int) -> int:
    ratio = n / max(1, limit)
    x = min(1.0, max(0.0, (ratio - 1.0) / 3.0))
    return max(1, min(9, int(round(1 + 8 * x))))

def _encode_result(raw: bytes, byte_limit: int) -> bytes:
    if len(raw) <= byte_limit:
        return raw
    level = _pick_level(len(raw), byte_limit)
    compressed = zlib.compress(raw, level)
    if len(compressed) >= len(raw):
        return raw
    header = MAGIC + struct.pack(">BIB", FLAG_COMPRESSED, len(raw), level)
    return header + compressed

def _needed_globals(fn) -> set[str]:
    names = set()
    try:
        for ins in dis.get_instructions(fn):
            if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME") and isinstance(ins.argval, str):
                names.add(ins.argval)
    except Exception:
        try:
            names.update(getattr(fn.__code__, "co_names", ()) or ())
        except Exception:
            pass
    names.discard("__builtins__")
    return names

def _apply_env(fn, env: dict, filter_used: bool):
    if not env:
        return
    g = getattr(fn, "__globals__", None)
    if not isinstance(g, dict):
        return

    env_g = env.get("globals") or {}
    if env_g:
        if filter_used:
            needed = _needed_globals(fn)
            for name in needed:
                if name in env_g:
                    g.setdefault(name, env_g[name])
        else:
            for name, val in env_g.items():
                g.setdefault(name, val)

serde = json.loads(__SERDE_JSON__)

pkg_root = serde.get("pkg_root")
if pkg_root and pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

kind = serde.get("kind")
if kind == "import":
    mod = importlib.import_module(serde["module"])
    fn = _resolve_attr_chain(mod, serde["qualname"])
elif kind == "dill":
    fn = dill.loads(base64.b64decode(serde["dill_b64"]))
else:
    if serde.get("module") and serde.get("qualname") and "<locals>" not in serde.get("qualname", ""):
        mod = importlib.import_module(serde["module"])
        fn = _resolve_attr_chain(mod, serde["qualname"])
    else:
        fn = dill.loads(base64.b64decode(serde["dill_b64"]))

env_b64 = serde.get("env_b64")
if env_b64:
    env = dill.loads(base64.b64decode(env_b64))
    meta = serde.get("env_meta") or {}
    _apply_env(fn, env, bool(meta.get("filter_used_globals", True)))

args, kwargs = dill.loads(base64.b64decode(__CALL_PAYLOAD_B64__))

res = fn(*args, **kwargs)
raw = dill.dumps(res, recurse=True)
blob = _encode_result(raw, BYTE_LIMIT)

# No f-string. No braces. No drama.
sys.stdout.write(str(RESULT_TAG) + ":" + base64.b64encode(blob).decode("ascii") + "\n")
""".strip()

        code = (
            template
            .replace("__RESULT_TAG__", repr(result_tag))
            .replace("__BYTE_LIMIT__", str(int(byte_limit)))
            .replace("__SERDE_JSON__", repr(serde_json))
            .replace("__CALL_PAYLOAD_B64__", repr(call_payload_b64))
        )

        return code

    @staticmethod
    def parse_command_result(output: str, *, result_tag: str = "__CALLABLE_SERDE_RESULT__") -> Any:
        """
        Parse stdout/stderr combined text, find last "{result_tag}:{b64}" line.
        Supports raw dill or framed+zlib compressed payloads.
        """
        prefix = f"{result_tag}:"
        b64 = None
        for line in reversed(output.splitlines()):
            if line.startswith(prefix):
                b64 = line[len(prefix):].strip()
                break
        if not b64:
            raise ValueError(f"Result tag not found in output: {result_tag!r}")

        blob = base64.b64decode(b64.encode("ascii"))
        raw = _decode_result_blob(blob)
        return dill.loads(raw)
