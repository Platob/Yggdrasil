"""Callable serialization helpers for cross-process execution.

Design goals:
- Prefer import-by-reference when possible (module + qualname), fallback to dill.
- Optional environment payload: selected globals and/or closure values.
- Cross-process bridge: generate a self-contained Python command string that:
    1) materializes the callable
    2) decodes args/kwargs payload
    3) executes
    4) emits a single tagged base64 line with a compressed result blob

Compression/framing:
- CS2 framing only (no CS1 logic).
- Frame header: MAGIC(3) + codec(u8) + orig_len(u32) + param(u8) + data
- Codecs:
    0 raw (rarely used; mostly means "no frame")
    1 zlib
    2 lzma
    3 zstd (optional dependency)
"""

from __future__ import annotations

import base64
import binascii
import dis
import importlib
import inspect
import io
import os
import secrets
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, TypeVar, Union, TYPE_CHECKING

import dill

if TYPE_CHECKING:
    from ..databricks.workspaces import Workspace

__all__ = ["CallableSerde"]

T = TypeVar("T", bound="CallableSerde")

# ---------------------------
# Framing / compression (CS2)
# ---------------------------

_MAGIC = b"CS2"

_CODEC_RAW = 0
_CODEC_ZLIB = 1
_CODEC_LZMA = 2
_CODEC_ZSTD = 3


def _try_import_zstd():
    try:
        import zstandard as zstd  # type: ignore
        return zstd
    except Exception:
        return None


def _try_import_lzma():
    try:
        import lzma  # type: ignore
        return lzma
    except Exception:
        return None


def _pick_zlib_level(n: int, limit: int) -> int:
    """Ramp compression level 1..9 based on how far we exceed the byte_limit."""
    ratio = n / max(1, limit)
    x = min(1.0, max(0.0, (ratio - 1.0) / 3.0))
    return max(1, min(9, int(round(1 + 8 * x))))


def _frame(codec: int, orig_len: int, param: int, payload: bytes) -> bytes:
    return _MAGIC + struct.pack(">BIB", int(codec) & 0xFF, int(orig_len), int(param) & 0xFF) + payload


def _encode_with_candidates(raw: bytes, *, byte_limit: int, allow_zstd: bool) -> bytes:
    """Choose the smallest among available codecs; fall back to raw if not beneficial."""
    if len(raw) <= byte_limit:
        return raw

    candidates: list[bytes] = []

    if allow_zstd:
        zstd = _try_import_zstd()
        if zstd is not None:
            for lvl in (6, 10, 15):
                try:
                    c = zstd.ZstdCompressor(level=lvl).compress(raw)
                    candidates.append(_frame(_CODEC_ZSTD, len(raw), lvl, c))
                except Exception:
                    pass

    lvl = _pick_zlib_level(len(raw), byte_limit)
    try:
        c = zlib.compress(raw, lvl)
        candidates.append(_frame(_CODEC_ZLIB, len(raw), lvl, c))
    except Exception:
        pass

    if not candidates:
        return raw

    best = min(candidates, key=len)
    return best if len(best) < len(raw) else raw


def _encode_result_blob(raw: bytes, byte_limit: int) -> bytes:
    """Result payload: zstd (if available) -> lzma -> zlib."""
    return _encode_with_candidates(raw, byte_limit=byte_limit, allow_zstd=True)


def _encode_wire_blob_stdlib(raw: bytes, byte_limit: int) -> bytes:
    """Wire payload (args/kwargs): stdlib-only (lzma -> zlib)."""
    return _encode_with_candidates(raw, byte_limit=byte_limit, allow_zstd=False)


def _decode_result_blob(blob: bytes) -> bytes:
    """Decode raw or CS2 framed data (no CS1 support)."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 3:
        return blob  # type: ignore[return-value]

    if not blob.startswith(_MAGIC):
        return blob

    if len(blob) < 3 + 6:
        raise ValueError("CS2 framed blob too short / truncated.")

    codec, orig_len, _param = struct.unpack(">BIB", blob[3 : 3 + 6])
    data = blob[3 + 6 :]

    if codec == _CODEC_RAW:
        raw = data
    elif codec == _CODEC_ZLIB:
        raw = zlib.decompress(data)
    # elif codec == _CODEC_LZMA:
    #     raw = lzma.decompress(data)
    elif codec == _CODEC_ZSTD:
        zstd = _try_import_zstd()
        if zstd is None:
            raise RuntimeError("CS2 uses zstd but 'zstandard' is not installed.")
        raw = zstd.ZstdDecompressor().decompress(data, max_output_size=int(orig_len) if orig_len else 0)
    else:
        raise ValueError(f"Unknown CS2 codec: {codec}")

    if orig_len and len(raw) != orig_len:
        raise ValueError(f"Decoded length mismatch: got {len(raw)}, expected {orig_len}")
    return raw


# ---------------------------
# Callable reference helpers
# ---------------------------

def _resolve_attr_chain(mod: Any, qualname: str) -> Any:
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _find_pkg_root_from_file(file_path: Path) -> Optional[Path]:
    file_path = file_path.resolve()
    d = file_path.parent

    top_pkg_dir = None
    while (d / "__init__.py").is_file():
        top_pkg_dir = d
        d = d.parent

    return top_pkg_dir if top_pkg_dir else None


def _callable_file_line(fn: Callable[..., Any]) -> Tuple[Optional[str], Optional[int]]:
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
    names: Set[str] = set()
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


def _is_importable_reference(fn: Callable[..., Any]) -> bool:
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


# ---------------------------
# Environment snapshot
# ---------------------------

def _dump_env(
    fn: Callable[..., Any],
    *,
    include_globals: bool,
    include_closure: bool,
    filter_used_globals: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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


# ----------
# Main class
# ----------

@dataclass
class CallableSerde:
    """
    Core field: `fn`

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
        if isinstance(x, cls):
            return x
        return cls(fn=x)  # type: ignore[return-value]

    # ----- properties -----

    @property
    def module(self) -> Optional[str]:
        return self._module or (getattr(self.fn, "__module__", None) if self.fn else None)

    @property
    def qualname(self) -> Optional[str]:
        return self._qualname or (getattr(self.fn, "__qualname__", None) if self.fn else None)

    @property
    def file(self) -> Optional[str]:
        if not self.fn:
            return None
        f, _ = _callable_file_line(self.fn)
        return f

    @property
    def line(self) -> Optional[int]:
        if not self.fn:
            return None
        _, ln = _callable_file_line(self.fn)
        return ln

    @property
    def pkg_root(self) -> Optional[str]:
        if self._pkg_root:
            return self._pkg_root
        if not self.file:
            return None
        root = _find_pkg_root_from_file(Path(self.file))
        return str(root) if root else None

    @property
    def relpath_from_pkg_root(self) -> Optional[str]:
        if not self.file or not self.pkg_root:
            return None
        try:
            return str(Path(self.file).resolve().relative_to(Path(self.pkg_root).resolve()))
        except Exception:
            return self.file

    @property
    def importable(self) -> bool:
        if self.fn is None:
            return bool(self.module and self.qualname and "<locals>" not in (self.qualname or ""))
        return _is_importable_reference(self.fn)

    # ----- serde API -----

    def dump(
        self,
        *,
        prefer: str = "import",           # "import" | "dill"
        dump_env: str = "none",           # "none" | "globals" | "closure" | "both"
        filter_used_globals: bool = True,
        env_keys: Optional[Iterable[str]] = None,
        env_variables: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
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

        env_variables = env_variables or {}
        if env_keys:
            for env_key in env_keys:
                existing = os.getenv(env_key)
                if existing:
                    env_variables[env_key] = existing

        if env_variables:
            out["osenv"] = env_variables

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
        fn = self.materialize()
        return fn(*args, **kwargs)

    # -------------------------
    # Command execution bridge
    # -------------------------

    def to_command(
        self,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        *,
        result_tag: str = "__CALLABLE_SERDE_RESULT__",
        prefer: str = "dill",
        byte_limit: int = 64 * 1024,
        dump_env: str = "none",           # "none" | "globals" | "closure" | "both"
        filter_used_globals: bool = True,
        env_keys: Optional[Iterable[str]] = None,
        env_variables: Optional[Dict[str, str]] = None,
        file_dump_limit: int = 512 * 1024,
        transaction_id: Optional[str] = None
    ) -> str:
        """
        Returns Python code string to execute in another interpreter.
        Emits exactly one line to stdout:
            "{result_tag}:{base64(blob)}\\n"
        where blob is raw dill bytes or CS2 framed.
        """
        import json

        args = args or ()
        kwargs = kwargs or {}

        serde_dict = self.dump(
            prefer=prefer,
            dump_env=dump_env,
            filter_used_globals=filter_used_globals,
            env_keys=env_keys,
            env_variables=env_variables,
        )
        serde_json = json.dumps(serde_dict, ensure_ascii=False)

        # args/kwargs payload: stdlib-only compression (lzma/zlib)
        call_raw = dill.dumps((args, kwargs), recurse=True)
        call_blob = _encode_wire_blob_stdlib(call_raw, int(byte_limit))
        call_payload_b64 = base64.b64encode(call_blob).decode("ascii")
        transaction_id = transaction_id or secrets.token_urlsafe(16)

        template = r"""
import base64, json, os, sys
import dill
import pandas

from yggdrasil.databricks import Workspace
from yggdrasil.pyutils.callable_serde import (
    CallableSerde,
    _decode_result_blob,
    _encode_result_blob,
)

RESULT_TAG = __RESULT_TAG__
BYTE_LIMIT = __BYTE_LIMIT__
FILE_DUMP_LIMIT = __FILE_DUMP_LIMIT__
TRANSACTION_ID = __TRANSACTION_ID__

def _needed_globals(fn) -> set[str]:
    import dis
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
    if not env_g:
        return

    if filter_used:
        needed = _needed_globals(fn)
        for name in needed:
            if name in env_g:
                g.setdefault(name, env_g[name])
    else:
        for name, val in env_g.items():
            g.setdefault(name, val)

serde = json.loads(__SERDE_JSON__)

cs = CallableSerde.load(serde, add_pkg_root_to_syspath=True)
fn = cs.materialize(add_pkg_root_to_syspath=True)

osenv = serde.get("osenv")
if osenv:
    for k, v in osenv.items():
        os.environ[k] = v

env_b64 = serde.get("env_b64")
if env_b64:
    env = dill.loads(base64.b64decode(env_b64))
    meta = serde.get("env_meta") or {}
    _apply_env(fn, env, bool(meta.get("filter_used_globals", True)))

call_blob = base64.b64decode(__CALL_PAYLOAD_B64__)
call_raw = _decode_result_blob(call_blob)
args, kwargs = dill.loads(call_raw)

res = fn(*args, **kwargs)

if isinstance(res, pandas.DataFrame):
    dump_path = Workspace().shared_cache_path("/cmd/" + TRANSACTION_ID + ".parquet")
    
    with dump_path.open(mode="wb") as f:
        res.to_parquet(f)
        
    blob = "DBXPATH:" + str(dump_path)
else:
    raw = dill.dumps(res)
    blob = _encode_result_blob(raw, BYTE_LIMIT)
    
    if len(blob) > FILE_DUMP_LIMIT:
        dump_path = Workspace().shared_cache_path("/cmd/" + TRANSACTION_ID)
        
        with dump_path.open(mode="wb") as f:
            f.write_all_bytes(data=blob)
            
        blob = "DBXPATH:" + str(dump_path)
    else:
        blob = base64.b64encode(blob).decode('ascii')

sys.stdout.write(f"{RESULT_TAG}:{len(blob)}:{blob}\n")
sys.stdout.flush()
"""

        return (
            template
            .replace("__RESULT_TAG__", repr(result_tag))
            .replace("__BYTE_LIMIT__", str(int(byte_limit)))
            .replace("__SERDE_JSON__", repr(serde_json))
            .replace("__CALL_PAYLOAD_B64__", repr(call_payload_b64))
            .replace("__FILE_DUMP_LIMIT__", str(int(file_dump_limit)))
            .replace("__TRANSACTION_ID__", repr(str(transaction_id)))
        )

    @staticmethod
    def parse_command_result(
        output: str,
        *,
        result_tag: str = "__CALLABLE_SERDE_RESULT__",
        workspace: Optional["Workspace"] = None
    ) -> Any:
        """
        Expect last tagged line:
            "{result_tag}:{blob_nbytes}:{b64}"

        We use blob_nbytes to compute expected base64 char length and detect truncation
        before decoding/decompressing.
        """
        prefix = f"{result_tag}:"
        if prefix not in output:
            raise ValueError(f"Result tag not found in output: {result_tag}")

        # Grab everything after the LAST occurrence of the tag
        _, tail = output.rsplit(prefix, 1)

        # Parse "{nbytes}:{b64}"
        try:
            nbytes_str, string_result = tail.split(":", 1)
        except ValueError as e:
            raise ValueError(
                f"Malformed result line after tag {result_tag}. "
                "Expected '{tag}:{nbytes}:{b64}'."
            ) from e

        try:
            content_length = int(nbytes_str)
        except ValueError as e:
            raise ValueError(f"Malformed byte count '{nbytes_str}' after tag {result_tag}") from e

        if content_length < 0:
            raise ValueError(f"Negative byte count {content_length} after tag {result_tag}")

        string_result = string_result[:content_length]

        if len(string_result) != content_length:
            raise ValueError(
                "Got truncated result content from command, got %s bytes and expected %s bytes" % (
                    len(string_result),
                    content_length
                )
            )

        if string_result.startswith("DBXPATH:"):
            from ..databricks.workspaces import Workspace

            workspace = Workspace() if workspace is None else workspace
            path = workspace.dbfs_path(
                string_result.replace("DBXPATH:", "")
            )

            if path.name.endswith(".parquet"):
                import pandas

                with path.open(mode="rb") as f:
                    buf = io.BytesIO(f.read_all_bytes())

                path.rmfile()
                buf.seek(0)
                return pandas.read_parquet(buf)

            with path.open(mode="rb") as f:
                blob = f.read_all_bytes()

            path.rmfile()
        else:
            # Strict base64 decode (rejects junk chars)
            try:
                blob = base64.b64decode(string_result.encode("ascii"), validate=True)
            except (UnicodeEncodeError, binascii.Error) as e:
                raise ValueError("Invalid base64 payload after result tag (corrupted/contaminated).") from e

        raw = _decode_result_blob(blob)
        try:
            result = dill.loads(raw)
        except Exception as e:
            raise ValueError("Failed to dill.loads decoded payload") from e

        return result
