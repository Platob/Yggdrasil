"""URL with canonical equality.

Three design notes that drive the changes vs. the previous version:

1. **Single port representation.** The internal sentinel ``_NO_PORT = 0``
   was leaking past ``with_scheme(inplace=True)`` and ``with_host`` while
   ``from_str`` / ``from_dict`` were normalizing absent ports to ``None``.
   The dataclass auto-eq then reported ``port=0 != port=None`` for two
   URLs that render to the exact same string. ``__post_init__`` now
   normalizes ``0`` to ``None`` once, centrally — every constructor and
   ``replace`` flows through it.

2. **Windows drive letters in ``from_str``.** ``urlsplit("C:\\Users\\x")``
   returns ``scheme='c', path='\\Users\\x'``. The previous fix-up only
   stripped the bogus single-letter scheme, leaving the path mangled
   and the drive letter dropped. We now reattach the drive and convert
   backslashes to forward slashes inside this branch only — so
   ``URL.from_str("C:\\Users\\x")`` and ``URL.from_(Path("C:\\Users\\x"))``
   produce equal URLs on Windows, and the rest of the URL parser stays
   untouched for non-Windows inputs.

3. **In-process object handles via ``mem://``.** :meth:`URL.from_memory_address`
   encodes ``id(obj)`` as a URL path so a Python object can round-trip
   through code paths that expect a URL string (cache keys, MediaIO
   dispatch, pipeline configs). Same-process, same-interpreter only;
   the URL is a raw id() handle and the caller is responsible for
   keeping a strong reference to the referent. See the method's
   docstring for the full lifetime contract.
"""

from __future__ import annotations

import ctypes
import fnmatch
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Mapping, Sequence
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

import pyarrow as pa

from yggdrasil import rs as _rs
from yggdrasil.io.parameters import anonymize_parameters
from yggdrasil.lazy_imports import (
    bytes_io_class,
    media_type_class,
    mime_type_class,
)

# Hot-path delegation: when the Rust extension is loaded, hand off URL
# parsing and percent-encoding to native kernels. The Python fallback
# below is preserved for environments without `yggrs` (e.g. odd
# packaging environments where the optional native wheel isn't present);
# `ygg` now requires `yggrs`, so in normal installs this branch is
# always taken. Functions are read once at module-import time so the
# hot path stays a direct attribute lookup.
_RS_URL = _rs.io_url

__all__ = [
    "URL",
    "URL_SCHEMA",
    "URL_STRUCT",
    "resolve_memory_address",
]

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

_SAFE_PATH = "/:@-._~!$&'()*+,;="
_SAFE_QUERY = "-._~!$'()*+,;=:@/?&="
_SAFE_FRAGMENT = "-._~!$&'()*+,;=:@/?"

_NO_PORT = 0

# Scheme used for in-process object handles. ``mem://<hex_addr>``.
# Same-process, same-interpreter only — see URL.from_memory_address.
_MEMORY_SCHEME: str = "mem"

# Pointer width in hex digits — 16 on 64-bit, 8 on 32-bit. Used so that
# every emitted memory URL has the same visual length on a given host;
# the parser accepts unpadded hex too, so this is purely cosmetic.
_MEMORY_HEX_WIDTH: int = (sys.maxsize.bit_length() + 1 + 3) // 4


def _lower_if(value: str) -> str:
    return value.lower() if value else ""


def _strip_trailing_dot(host: str) -> str:
    return host[:-1] if host.endswith(".") else host


def _normalize_path(
    path: str,
    os_find: bool
) -> str:
    if not path:
        return "/"

    elif path == "/":
        return path

    elif os_find and not path.startswith("/"):
        # Pre-process the input for os.path.realpath so it behaves on both
        # POSIX and Windows. Two cases matter:
        #
        # 1. POSIX absolute path like "/tmp/x". realpath("/tmp/x") == "/tmp/x",
        #    so feed it through as-is. Do NOT lstrip("/") — that would make
        #    the path cwd-relative and realpath would re-anchor it to $PWD.
        #
        # 2. Windows drive path, which arrives in file-URL form "/C:/Users/x".
        #    Windows' realpath chokes on the leading slash and emits a broken
        #    "/C:Users\x" (the slash AFTER the colon gets dropped as realpath
        #    tries to normalize the odd mixed form). Strip the leading slash
        #    only in this specific shape so realpath sees "C:/Users/x" and
        #    does the right thing.
        #
        # The drive-letter pattern is "/X:" (colon at index 2, where X is a
        # single letter). That's unambiguous: no real POSIX path has a colon
        # at index 2 of a top-level absolute path.
        to_resolve = path
        if len(path) >= 3 and path[0] == "/" and path[2] == ":" and path[1].isalpha():
            to_resolve = path[1:]

        resolved = os.path.realpath(to_resolve).replace("\\", "/")
        return resolved if resolved.startswith("/") else "/" + resolved

    return path


def _remove_default_port(scheme: str, host: str, port: int) -> int:
    if not scheme or not host or port <= 0:
        return _NO_PORT
    return _NO_PORT if _DEFAULT_PORTS.get(scheme) == port else port


def _strip_windows_drive_slash(path: str) -> str:
    """Inverse of the ``from_pathlib`` Windows fix-up.

    Transforms ``/C:/Users/x`` → ``C:/Users/x`` for any leading
    ``/X:`` pattern. Used by both :meth:`URL.to_pathlib` and
    :meth:`URL.__fspath__` so the two agree on the drive-letter
    handling without duplicating the rule.
    """
    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        return path[1:]
    return path


def _format_memory_address(address: int) -> str:
    """Render a non-negative ``id()`` as ``0x``-prefixed lowercase hex,
    zero-padded to the platform pointer width.

    Emitted with a leading ``/`` by :meth:`URL.from_memory_address` so
    the resulting URL path is already rooted and bypasses
    :func:`_normalize_path`'s empty-path → ``"/"`` coercion.
    """
    if address < 0:
        raise ValueError(f"memory address must be non-negative, got {address}")
    return f"0x{address:0{_MEMORY_HEX_WIDTH}x}"


# Cached host component for ``mem://<host>/<addr>`` URLs. Looked up
# once via :func:`socket.gethostname` (or :func:`platform.node` as a
# fallback) and lower-cased so the rendered URL matches the casing of
# every other URL component. Falls back to ``"localhost"`` if the
# system has no usable hostname configured.
_LOCAL_HOSTNAME: "str | None" = None


def _local_hostname() -> str:
    """Return the local machine's hostname, lower-cased and cached.

    Used as the host component for memory URLs so two ``mem://`` URLs
    from different hosts never collide on a shared cache key. The
    lookup runs once per process; subsequent calls return the cached
    value.
    """
    global _LOCAL_HOSTNAME
    if _LOCAL_HOSTNAME is None:
        import socket
        try:
            name = socket.gethostname()
        except OSError:
            name = ""
        if not name:
            import platform
            name = platform.node() or "localhost"
        _LOCAL_HOSTNAME = name.strip().lower() or "localhost"
    return _LOCAL_HOSTNAME


def _parse_memory_address(text: str) -> int:
    """Parse the path portion of a ``mem://`` URL back into an integer.

    Accepts with or without ``0x`` prefix, with or without a leading
    ``/`` (URL paths are rooted). Rejects empty strings, signed values,
    and non-hex input.
    """
    if not text:
        raise ValueError("memory address is empty")
    s = text.lstrip("/").lower()
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        raise ValueError(f"memory address is empty after stripping prefix: {text!r}")
    try:
        value = int(s, 16)
    except ValueError as exc:
        raise ValueError(f"not a valid hex memory address: {text!r}") from exc
    if value < 0:
        raise ValueError(f"memory address must be non-negative, got {value}")
    return value


def resolve_memory_address(address: int) -> object:
    """Dereference an integer address back to the Python object.

    Uses ``ctypes.cast(addr, py_object).value`` — the standard CPython
    trick. O(1), but a *raw* dereference: the caller MUST hold a strong
    reference to the object for the URL's entire lifetime. If the object
    has been garbage-collected, the slot will be reused and this
    function returns a different object (or, if the slot was freed and
    not yet reused, may segfault).

    Module-level so non-URL callers (tests, MediaIO dispatch sites) can
    use it without constructing a URL instance.
    """
    return ctypes.cast(address, ctypes.py_object).value


def _join_segment(seg: Any) -> str:
    """Coerce a :meth:`URL.joinpath` argument to a path string.

    URL-valued arguments contribute only their :attr:`path` component
    — the scheme/host/query of the RHS is dropped. Use :meth:`URL.join`
    instead when full URL-reference joining (per RFC 3986) is needed.
    :class:`os.PathLike` and ``str`` pass through ``os.fspath``. All
    other types raise :class:`TypeError` rather than coerce silently,
    because ``url / 42`` is almost certainly a bug.
    """
    if isinstance(seg, URL):
        return seg.path
    if isinstance(seg, (str, os.PathLike)):
        return os.fspath(seg)
    raise TypeError(
        f"joinpath/truediv segment must be str, URL, or os.PathLike, "
        f"got {type(seg).__name__}"
    )


def _encode_userinfo(userinfo: str) -> str:
    if not userinfo:
        return ""
    if _RS_URL is not None:
        return _RS_URL.encode_userinfo(userinfo)
    return quote(userinfo, safe=":!$&'()*+,;=")


def _encode_path(path: str, safe: str = _SAFE_PATH) -> str:
    # The Rust kernel uses the canonical safe set + the same "no space →
    # also keep '%'" rule as the Python fallback. Custom safe sets are a
    # rare debug knob, so detour into the Python fallback when one is
    # supplied.
    if _RS_URL is not None and safe is _SAFE_PATH:
        return _RS_URL.encode_path(path)
    if safe:
        if " " not in path:
            safe = safe + "%"
    return quote(path, safe=safe or "")


def _encode_query(query: str) -> str:
    if _RS_URL is not None:
        return _RS_URL.encode_query(query)
    return quote(query, safe=_SAFE_QUERY + "%")


def _encode_fragment(fragment: str) -> str:
    if _RS_URL is not None:
        return _RS_URL.encode_fragment(fragment)
    return quote(fragment, safe=_SAFE_FRAGMENT + "%")


def _decode_maybe(value: str, decode: bool) -> str:
    return unquote(value) if decode and value else value


def _s(value: str | None) -> str:
    return value or ""


def _p(value: int | None) -> int:
    return value or _NO_PORT


def _normalize_query(query: str) -> str:
    if _RS_URL is not None:
        return _RS_URL.normalize_query(query)
    query = query.lstrip("?")
    if not query:
        return ""
    items = parse_qsl(query, keep_blank_values=True)
    items.sort(key=lambda item: (item[0], item[1]))
    return urlencode(items, doseq=True)


def _parse_port(value: Any) -> int:
    if value in (None, "", 0):
        return _NO_PORT
    if isinstance(value, int):
        return value if value > 0 else _NO_PORT
    text = str(value)
    return int(text) if text.isdigit() and int(text) > 0 else _NO_PORT


def _parse_netloc(netloc: str, *, decode: bool) -> tuple[str, str, int]:
    if _RS_URL is not None:
        userinfo, host, port = _RS_URL.parse_netloc(netloc, decode=decode)
        return userinfo, host, port
    if not netloc:
        return "", "", _NO_PORT

    userinfo = ""
    hostport = netloc

    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        userinfo = _decode_maybe(userinfo, decode)

    port = _NO_PORT

    if hostport.startswith("["):
        rb = hostport.find("]")
        if rb == -1:
            return userinfo, hostport, _NO_PORT

        host = hostport[1:rb]
        rest = hostport[rb + 1 :]
        if rest.startswith(":") and len(rest) > 1:
            port = _parse_port(rest[1:])
        return userinfo, host, port

    if ":" in hostport:
        host, port_text = hostport.rsplit(":", 1)
        port = _parse_port(port_text)
        return userinfo, host, port

    return userinfo, hostport, _NO_PORT


def _normalize_components(
    *,
    scheme: str,
    userinfo: str,
    host: str,
    port: int,
    path: str,
    query: str,
    fragment: str,
) -> tuple[str, str, int, str, str, str]:
    scheme_n = _lower_if(scheme)
    host_n = _strip_trailing_dot(_lower_if(host))
    port_n = _remove_default_port(scheme_n, host_n, port)
    # Only resolve local filesystem paths for explicit file:// URLs.
    # An empty scheme means a schemaless HTTP-like URL (e.g. "example.com/path"),
    # NOT a local path — calling os.path.realpath() on those would corrupt them.
    path_n = _normalize_path(path, os_find=scheme_n == "file")
    query_n = _normalize_query(query)
    fragment_n = fragment.lstrip("#")

    return scheme_n, host_n, port_n, path_n, query_n, fragment_n


# ---------------------------------------------------------------------------
# Arrow schema — single source of truth for the URL struct shape used by
# request / response / userinfo serializers. Defined here next to the
# :class:`URL` class so every consumer references the same column
# ordering, types, and nullability flags.
#
# Kept as raw pyarrow types (rather than wrapped in
# :class:`yggdrasil.data.schema.Schema`) because ``data.data_field``
# transitively imports ``io.enums.media_type``, which imports back into
# this module — wrapping URL_SCHEMA here would create an import cycle.
# Engines that need the richer wrapper rebuild it on top of these
# columns.
# ---------------------------------------------------------------------------

URL_SCHEMA: pa.Schema = pa.schema([
    pa.field("scheme",   pa.string(), nullable=False),
    pa.field("userinfo", pa.string(), nullable=True),
    pa.field("host",     pa.string(), nullable=False),
    pa.field("port",     pa.int32(),  nullable=True),
    pa.field("path",     pa.string(), nullable=False),
    pa.field("query",    pa.string(), nullable=True),
    pa.field("fragment", pa.string(), nullable=True),
])

URL_STRUCT: pa.StructType = pa.struct(list(URL_SCHEMA))


@dataclass(frozen=True, slots=True)
class URL(os.PathLike):
    scheme: str = ""
    userinfo: str | None = None
    host: str = ""
    port: int | None = None
    path: str = "/"
    query: str | None = None
    fragment: str | None = None

    _str_enc: str | None = field(default=None, init=False, repr=False, compare=False)
    _str_raw: str | None = field(default=None, init=False, repr=False, compare=False)
    _anonymized: bool | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Centrally normalize the port=0 sentinel to None. Without this,
        # two URLs that render identically can compare unequal because
        # one constructor flowed port through `port or None` and another
        # left the int `_NO_PORT` (0) in place. Frozen dataclass — must
        # use object.__setattr__.
        if self.port == 0:
            object.__setattr__(self, "port", None)

    @classmethod
    def is_urlish(cls, value: Any) -> bool:
        """True iff the value is a string or URL-like object."""
        if isinstance(value, cls):
            return True

        if isinstance(value, str):
            if len(value) > 256 * 1024:
                return False

            if "://" in value or "/":
                return True

        return isinstance(value, os.PathLike)

    @property
    def parts(self):
        if not self.path or self.path == "/":
            return []
        return self.path.lstrip("/").split("/")

    @property
    def extensions(self) -> list[str]:
        """Return the path's extensions, lowercased, leading dot stripped.

        Examples::

            URL.from_("/data/file.csv").extensions           == ["csv"]
            URL.from_("/data/file.tar.gz").extensions        == ["tar", "gz"]
            URL.from_("/data/archive.csv.zst").extensions    == ["csv", "zst"]
            URL.from_("/data/README").extensions             == []
            URL.from_("/data/.hidden").extensions            == []   # dotfile
            URL.from_("/data/.env.local").extensions         == ["local"]

        The list ordering is outer-to-inner: for ``archive.csv.zst`` you
        get ``["csv", "zst"]``, matching the codec/media-type refactor
        convention (outer format first, compression codec last). Leading
        dotfile marker isn't treated as an extension.
        """
        if not self.path or self.path == "/":
            return []
        return [s.lstrip(".").lower() for s in PurePosixPath(self.path).suffixes]

    @property
    def name(self) -> str:
        """Last segment of the path, trailing slash stripped.

        Examples::

            URL.from_("/a/b/c").name   == "c"
            URL.from_("/a/b/c/").name  == "c"   # trailing slash ignored
            URL.from_("/a/b/").name    == "b"
            URL.from_("/").name        == ""
            URL.from_("").name         == ""

        ``PurePosixPath`` normalizes the trailing slash, so directory-style
        URLs still return their last non-empty segment.
        """
        if not self.path or self.path == "/":
            return ""
        elif self.path.endswith("/"):
            return self.parts[-2]
        return self.parts[-1]

    @property
    def stem(self) -> str:
        """The :attr:`name` with its final extension removed.

        Matches :attr:`pathlib.PurePosixPath.stem` semantics — only the
        last suffix is stripped, not all of them:

            URL.from_("/a/file.csv").stem          == "file"
            URL.from_("/a/archive.csv.zst").stem   == "archive.csv"
            URL.from_("/a/README").stem            == "README"
            URL.from_("/a/.hidden").stem           == ".hidden"  # dotfile
            URL.from_("/a/.env.local").stem        == ".env"
            URL.from_("/").stem                    == ""
            URL.from_("").stem                     == ""

        Use :attr:`extensions` if you need every suffix, or call
        ``.stem`` repeatedly to peel layers.
        """
        if not self.path or self.path == "/":
            return ""
        return PurePosixPath(self.path).stem

    @property
    def parent(self) -> URL:
        """The URL one path segment up, with query/fragment/authority preserved.

        Matches :attr:`pathlib.PurePosixPath.parent` semantics at the
        path level — trailing slashes are ignored and the root is its
        own parent:

            URL.from_("https://e.com/a/b/c").parent      # https://e.com/a/b
            URL.from_("https://e.com/a/b/c/").parent     # https://e.com/a/b
            URL.from_("https://e.com/a").parent          # https://e.com/
            URL.from_("https://e.com/").parent           # https://e.com/
            URL.from_("/a/b?x=1#frag").parent            # /a?x=1#frag

        Query, fragment, scheme, host, userinfo, and port are carried
        over unchanged. If the inherited query/fragment don't make
        sense for the parent URL, strip them yourself via
        :meth:`with_query` / :meth:`with_fragment`.
        """
        if not self.path or self.path == "/":
            # Root is its own parent (pathlib semantics) and also the
            # sensible answer when path is empty (the dataclass default
            # is "/", so "" shouldn't occur in practice but guard anyway).
            return self._replace(path="/")
        parent_path = str(PurePosixPath(self.path).parent)
        # PurePosixPath("").parent is ".", not "/" — coerce to "/" for
        # URL semantics. Can't reach this branch given the guard above,
        # but belt-and-braces.
        if parent_path == ".":
            parent_path = "/"
        return self._replace(path=parent_path)

    @property
    def parents(self) -> tuple[URL, ...]:
        """The full ancestry chain, closest first.

        A tuple of URLs walking up from :attr:`parent` to the root.
        Empty when the URL has no meaningful path:

            URL.from_("https://e.com/a/b/c").parents
                # (https://e.com/a/b, https://e.com/a, https://e.com/)
            URL.from_("https://e.com/a").parents
                # (https://e.com/,)
            URL.from_("https://e.com/").parents
                # ()

        Like :attr:`parent`, every URL in the chain carries the same
        query, fragment, and authority as ``self``.
        """
        if not self.path or self.path == "/":
            return ()
        return tuple(
            self._replace(path=str(p))
            for p in PurePosixPath(self.path).parents
        )

    @property
    def media_type(self):
        return self.infer_media_type(default=None)

    @property
    def mime_types(self):
        mc = mime_type_class()
        ext = self.extensions

        if not ext:
            return []

        return [mc.from_str(_) for _ in ext]

    def infer_media_type(self, default: "MediaType" = ...):
        return media_type_class().from_url(self, default=default)

    @property
    def mime_type(self):
        """The inner :class:`MimeType` of this URL's :attr:`media_type`.

        For ``/data/archive.csv.zst`` this is ``CSV`` — the codec
        wrapper is reported separately via :attr:`codec`. Returns
        ``None`` when :attr:`media_type` is ``None``.
        """
        mt = self.media_type
        return mt.mime_type if mt is not None else None

    @property
    def codec(self):
        """The :class:`Codec` peeled from :attr:`media_type`, if any.

        ``None`` for uncompressed URLs (``/data/file.csv``) and for
        URLs with no filename. For ``/data/archive.csv.zst`` this is
        the ZSTD codec.
        """
        mt = self.media_type
        return mt.codec if mt is not None else None

    # ------------------------------------------------------------------
    # Memory-address handles
    # ------------------------------------------------------------------

    @property
    def is_memory_address(self) -> bool:
        """True iff this URL is a ``mem://`` in-process handle.

        Cheap predicate for dispatch sites that need to branch between
        real I/O (``file``, ``http``, ``s3``, ...) and direct object
        pickup. Pairs with :attr:`memory_address` and
        :meth:`resolve_memory_address` for the actual retrieval.
        """
        return self.scheme == _MEMORY_SCHEME

    @property
    def memory_address(self) -> int:
        """Integer ``id()`` encoded in this ``mem://`` URL.

        Raises :class:`ValueError` if this is not a memory URL or if the
        path is malformed. Use :attr:`is_memory_address` to guard the
        lookup if the scheme is unknown.
        """
        if not self.is_memory_address:
            raise ValueError(
                f"not a memory-address URL (scheme={self.scheme!r}); "
                f"use is_memory_address to guard access"
            )
        return _parse_memory_address(self.path)

    def resolve_memory_address(self) -> object:
        """Dereference this ``mem://`` URL back to the original Python object.

        Wraps the module-level :func:`resolve_memory_address` for
        ergonomics. The lifetime caveats apply: the caller must have
        kept a strong reference to the object since the URL was built.
        See :meth:`from_memory_address` for the full contract.
        """
        return resolve_memory_address(self.memory_address)

    @classmethod
    def empty(
        cls,
        new_instance: bool = False
    ) -> "URL":
        if new_instance:
            return cls()
        return _EMPTY_URL

    @staticmethod
    def path_encode(path: str, safe: str = _SAFE_PATH) -> str:
        return _encode_path(path, safe=safe)

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        default_scheme: str | None = None,
        decode: bool = False,
        normalize: bool = True,
    ):
        return cls.from_(obj, default_scheme=default_scheme, decode=decode, normalize=normalize)

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        default_scheme: str | None = None,
        decode: bool = False,
        normalize: bool = True,
        default: Any = ...
    ) -> URL:
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, str):
            return cls.from_str(
                str(obj),
                default_scheme=default_scheme,
                decode=decode,
                normalize=normalize
            )

        if isinstance(obj, Path):
            return cls.from_pathlib(
                obj,
                default_scheme=default_scheme,
                decode=decode,
                normalize=normalize,
            )

        if isinstance(obj, Mapping):
            return cls.from_dict(obj, decode=decode, normalize=normalize)

        if obj is None:
            if default is ...:
                raise ValueError("Cannot parse URL from None")
            return default

        bio = bytes_io_class()
        if isinstance(obj, bio):
            return cls.from_(
                obj.url,
                default_scheme=default_scheme, decode=decode, normalize=normalize
            )

        return cls.from_str(
            str(obj),
            default_scheme=default_scheme,
            decode=decode,
            normalize=normalize
        )

    @classmethod
    def from_memory_address(cls, obj: object) -> "URL":
        """Build a ``mem://<hostname>/<hex_addr>`` URL pointing at ``obj``.

        Encodes ``id(obj)`` so the URL can round-trip through code
        paths that expect a string or :class:`URL` (cache keys,
        MediaIO dispatch, pipeline configs). The returned URL is a
        *handle*, not a persistent reference: it is valid only within
        this process and only while the caller holds a strong
        reference to ``obj``. The host component is the local machine's
        hostname (cached process-wide) so two memory URLs minted on
        different hosts never alias even when ``id()`` happens to
        collide across machines.

        Lifetime contract:

        - **Same-process, same-interpreter only.** No pickling, no
          cross-process transport, no persistence across restarts.
        - **Caller owns the reference.** This constructor does NOT
          take a reference to ``obj``. If ``obj`` is garbage-collected
          before :meth:`resolve_memory_address` runs, the address will
          be reused and resolution returns a different object — or
          crashes the interpreter if the slot was freed.

        See :meth:`resolve_memory_address` for round-trip retrieval and
        :attr:`is_memory_address` for dispatch-site predicates.

        Implementation: the rendered path has a leading ``/`` so it is
        a well-formed rooted URL path and bypasses
        :func:`_normalize_path`'s empty-path coercion. Constructed
        directly through ``cls(...)`` rather than :meth:`from_dict`
        because there is nothing to normalize on a hex address — the
        full :func:`_normalize_components` pipeline would be wasted
        work and is in fact unsafe (we don't want anyone deciding to
        ``realpath`` a hex string later).
        """
        address = id(obj)
        return cls(
            scheme=_MEMORY_SCHEME,
            host=_local_hostname(),
            path="/" + _format_memory_address(address),
        )

    @classmethod
    def is_pathish(cls, obj: Any) -> bool:
        """Return True when *obj* is something :meth:`from_` can resolve
        without resorting to the ``str(obj)`` fallback.

        Accepts: :class:`URL` instances, :class:`str`, :class:`pathlib.Path`
        (and any :class:`os.PathLike`), :class:`Mapping` (handled by
        :meth:`from_dict`), and objects that expose a ``.url`` attribute
        (the duck-typed fallback in :meth:`from_`).

        Rejects ``None`` — :meth:`from_` raises on ``None``, so it is
        not pathish. Arbitrary objects without any of the above shapes
        also return False: :meth:`from_` would coerce them via
        ``str(obj)`` and produce nonsense, so calling ``is_pathish``
        first is the right guard.

        This mirrors the ``is_pathish`` protocol used elsewhere in the
        codebase (``path_class().is_pathish`` in ``MimeType.from_``,
        ``MediaType.from_``), so ``URL`` can participate in the same
        dispatch shape.
        """
        if obj is None:
            return False
        if isinstance(obj, (cls, str, Path, os.PathLike, Mapping)):
            return True
        return hasattr(obj, "url")

    @classmethod
    def from_str(
        cls,
        raw: str,
        *,
        default_scheme: str | None = None,
        decode: bool = False,
        normalize: bool = True,
    ) -> URL:
        if _RS_URL is not None:
            # Rust handles split + netloc + Windows drive fix-up + entity
            # decoding + the missing-authority recovery. We keep the
            # final normalize pass in Python because `_normalize_path`'s
            # `os.path.realpath` branch for `file://` URLs is platform-
            # sensitive and lives in the Python helper.
            scheme, userinfo, host, port, path, query, fragment = _RS_URL.parse_url(
                raw,
                default_scheme=default_scheme,
                decode=decode,
                normalize=False,
            )
        else:
            split = urlsplit(raw)
            userinfo, host, port = _parse_netloc(split.netloc, decode=decode)

            scheme = split.scheme

            path = _decode_maybe(split.path, decode)

            # A single-letter "scheme" is a Windows drive letter (e.g. C:/path
            # or C:\path). Reattach the drive to the path, normalize backslash
            # separators inside this branch, and force the file scheme so
            # ``URL.from_str("C:\\foo")`` matches ``URL.from_(Path("C:\\foo"))``.
            if scheme and len(scheme) == 1 and scheme.isalpha():
                drive = scheme.upper()
                rest = path.replace("\\", "/")
                if not rest.startswith("/"):
                    rest = "/" + rest
                path = f"/{drive}:{rest}"
                scheme = "file"

            scheme = default_scheme or scheme

            # URLs sourced from XML/HTML payloads often arrive with ``&`` encoded
            # as the entity ``&amp;`` (e.g. an Atom feed link, an HTML attribute,
            # an XML-RPC body). Without this fix, ``urlsplit`` keeps the literal
            # ``amp;`` and the parameter separator ``&`` is lost — the next pair
            # in the query gets glued onto the previous key (``amp;update_id``)
            # and round-trips through Arrow keep the broken form. Decode the
            # entity here so the query/fragment parse as the caller intended;
            # callers that genuinely need a literal ``&amp;`` must percent-encode
            # the ``&`` themselves.
            query = _decode_maybe(split.query, decode).replace("&amp;", "&")
            fragment = _decode_maybe(split.fragment, decode).replace("&amp;", "&")

            # Fix-up for URLs that lack "//" authority (e.g. "http:example.com/path")
            # but NOT for schemaless strings — those would incorrectly extract the
            # first path segment as the host.
            if scheme and scheme not in ("file",) and not host and path:
                if "/" in path:
                    host, path = path.split("/", 1)
                    path = "/" + path.lstrip("/")
                else:
                    host = path
                    path = "/"

        if normalize:
            scheme, host, port, path, query, fragment = _normalize_components(
                scheme=scheme,
                userinfo=userinfo,
                host=host,
                port=port,
                path=path,
                query=query,
                fragment=fragment,
            )

        return cls(
            scheme=scheme,
            userinfo=userinfo or None,
            host=host,
            port=port or None,
            path=path or "/",
            query=query or None,
            fragment=fragment or None,
        )

    @classmethod
    def from_pathlib(
        cls,
        path: Path | str,
        *,
        default_scheme: str | None = None,
        decode: bool = False,
        normalize: bool = True,
    ) -> URL:
        """Build a URL from a ``pathlib.Path`` (or a string path).

        Expands ``~``, resolves to an absolute posix path, and prepends a
        leading slash to Windows drive-letter paths so ``C:/Users/x``
        becomes ``/C:/Users/x`` — the form file URLs use. Passing a
        string goes through ``Path`` first, so ``from_pathlib("~/data.csv")``
        and ``from_pathlib(Path("~/data.csv"))`` produce equal URLs.

        Use ``from_str`` instead if the input is already a URL string
        (``file:///...``) — this method treats its input as a filesystem
        path, not a URL.
        """
        if not isinstance(path, Path):
            path = Path(path)

        resolved = path.expanduser().resolve(strict=False).as_posix()

        # Windows drive path must be /C:/... in file URLs. `as_posix()`
        # yields "C:/Users/x" on Windows; we need "/C:/Users/x".
        if len(resolved) >= 2 and resolved[1] == ":":
            resolved = "/" + resolved

        scheme = default_scheme or "file"

        return cls.from_dict(
            {"scheme": scheme, "path": resolved},
            decode=decode,
            normalize=normalize,
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        decode: bool = False,
        normalize: bool = True,
    ) -> URL:
        if not data:
            return cls()

        raw = data.get("url") or data.get("raw")
        if raw is not None:
            return cls.from_str(str(raw), decode=decode, normalize=normalize)

        scheme = str(data.get("scheme") or "")
        path = str(data.get("path") or "")
        query = str(data.get("query") or "")
        fragment = str(data.get("fragment") or "")

        netloc = data.get("netloc")
        if netloc is None:
            netloc = data.get("authority")

        userinfo = ""
        host = ""
        port = _NO_PORT

        if netloc is not None:
            userinfo, host, port = _parse_netloc(str(netloc), decode=decode)

        if data.get("userinfo") is not None:
            userinfo = _decode_maybe(str(data["userinfo"]), decode)

        if data.get("host") is not None:
            host = _decode_maybe(str(data["host"]), decode)

        if "port" in data:
            port = _parse_port(data.get("port"))

        path = _decode_maybe(path, decode)
        query = _decode_maybe(query, decode)
        fragment = _decode_maybe(fragment, decode)

        if normalize:
            scheme, host, port, path, query, fragment = _normalize_components(
                scheme=scheme,
                userinfo=userinfo,
                host=host,
                port=port,
                path=path,
                query=query,
                fragment=fragment,
            )
        else:
            query = query.lstrip("?")
            fragment = fragment.lstrip("#")

        return cls(
            scheme=scheme,
            userinfo=userinfo or None,
            host=host,
            port=port or None,
            path=path or "",
            query=query or None,
            fragment=fragment or None,
        )

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self):
        return f"URL({self.to_string()!r})"

    def __fspath__(self) -> str:
        """Return a filesystem-style string so :class:`URL` satisfies
        :class:`os.PathLike`.

        Rule: when the scheme is ``file`` or empty (bare path), return
        the path component with the Windows drive-letter fix-up applied
        (``/C:/Users/x`` → ``C:/Users/x``) so ``open(url)`` works the
        same as ``open(url.to_pathlib())``. For any other scheme return
        the raw :attr:`path`.

        Callers that want the whole URL as a string — including scheme,
        host, query, fragment — should use :func:`str` or
        :meth:`to_string` explicitly. ``os.fspath`` is about filesystem
        paths, and a full ``https://…`` string is not one.

        Unlike :meth:`to_pathlib`, this method does not raise on an
        empty path or on extra URL components; the ``os.fspath`` contract
        is that it should return a string for anything registered as
        ``PathLike``. An empty path yields an empty string.
        """
        if self.scheme in ("", "file"):
            return _strip_windows_drive_slash(self.path)
        return self.path

    def __truediv__(self, other: object) -> URL:
        """``url / seg`` is shorthand for :meth:`joinpath`.

        Semantics changed from earlier versions: previously this
        *replaced* the path (``URL("/a") / "b"`` → ``/b``), which
        contradicted the method name and every other URL/path
        library. It now *joins*, matching :class:`pathlib.PurePath`:
        ``URL("/a") / "b"`` → ``/a/b``.
        """
        return self.joinpath(other)

    def joinpath(self, *segments: Any) -> URL:
        """Append one or more path segments, pathlib-style.

        ``URL("https://e.com/a").joinpath("b", "c")`` → ``https://e.com/a/b/c``.
        Scheme, host, userinfo, port, query, and fragment are carried
        over from ``self`` — only the path changes.

        Semantics match :class:`pathlib.PurePosixPath.joinpath`:

        - A segment starting with ``/`` is treated as absolute and
          resets the accumulated path::

                URL("/a/b").joinpath("/c")  →  URL("/c")

        - ``..`` is NOT resolved implicitly. ``URL("/a/b").joinpath("..")``
          yields ``/a/b/..``. If you want lexical resolution, run the
          result through :meth:`with_path(..., os_find=True)` or handle
          it at the call site — URL stays syntactic by default.

        - URL-valued segments contribute only their :attr:`path`. Use
          :meth:`join` instead for RFC 3986 reference resolution (which
          honours scheme/host on the RHS).

        - Accepts :class:`str`, :class:`URL`, or any :class:`os.PathLike`.
          Anything else raises :class:`TypeError` — ``url / 42`` is
          almost certainly a bug.

        Trailing slashes on ``self.path`` are preserved by
        :class:`PurePosixPath` semantics: ``/a/b/`` joined with ``c``
        gives ``/a/b/c``.
        """
        if not segments:
            return self

        base = PurePosixPath(self.path) if self.path else PurePosixPath("/")
        segments = [_join_segment(s) for s in segments]
        joined = str(base.joinpath(*segments))
        if segments and segments[-1].endswith("/") and not joined.endswith("/"):
            joined = joined + "/"
        return self._replace(path=joined)

    # ------------------------------------------------------------------
    # Pattern matching and relative-path queries
    # ------------------------------------------------------------------

    def match_pattern(self, pattern: str) -> bool:
        """Glob-match against :attr:`name`, then against the full URL.

        Returns ``True`` on either hit. The basename is checked first
        because ``*.csv``-style filters are the common case and the
        full URL rendering can be expensive on remote backends (forces
        :meth:`to_string`). Uses :func:`fnmatch.fnmatch`, so the
        pattern syntax is shell-style: ``*``, ``?``, ``[abc]``,
        ``[!abc]``.

        The full rendering is the encoded URL string (scheme + host +
        path + query + fragment), so patterns like
        ``"https://*.com/data/*"`` work.
        """
        if fnmatch.fnmatch(self.name, pattern):
            return True
        return fnmatch.fnmatch(self.to_string(), pattern)

    def matches_patterns(self, patterns: Iterable[str] | None) -> bool:
        """Glob-match against any pattern in *patterns*.

        ``None`` or an empty iterable returns ``False`` — "no patterns"
        means "no matches", not "everything matches". Callers that want
        the "include everything when no filter" behaviour should guard
        at the call site (``if not patterns or url.matches_patterns(patterns)``).

        Basename is checked against every pattern first before the
        full URL rendering is considered, since :attr:`name` is cheap
        and ``*.ext``-style filters are the common case.
        """
        if patterns is None:
            return False

        # Materialize once — a generator would be exhausted by the
        # first loop and silently skip the full-URL pass.
        pats = patterns if isinstance(patterns, (tuple, list)) else tuple(patterns)
        if not pats:
            return False

        name = self.name
        for pat in pats:
            if fnmatch.fnmatch(name, pat):
                return True

        full = self.to_string()
        for pat in pats:
            if fnmatch.fnmatch(full, pat):
                return True

        return False

    def is_relative_to(self, other: Any) -> bool:
        """True when ``self.path`` is equal to or below ``other.path``.

        Semantics follow :meth:`pathlib.PurePath.is_relative_to` at
        the path level, with an extra URL-aware constraint: when
        ``other`` is a fully-qualified URL (has scheme or host), its
        scheme and host must match ``self``'s. A file under one
        authority is never "relative to" a file under a different
        authority.

        ``other`` is coerced via :meth:`from_` so strings, dicts,
        :class:`Path` instances, and bare URLs all work::

            URL("https://e.com/a/b").is_relative_to("/a")          # True
            URL("https://e.com/a/b").is_relative_to("https://e.com/a")  # True
            URL("https://e.com/a/b").is_relative_to("https://other.com/a")  # False
            URL("https://e.com/a").is_relative_to("https://e.com/a/b")  # False
        """
        try:
            other_url = self.from_(other)
        except (ValueError, TypeError):
            return False

        # Authority check: if `other` carries a scheme or host, they
        # must match `self`'s. A bare path ``/a`` relative check
        # ignores authority.
        if other_url.scheme and other_url.scheme != self.scheme:
            return False
        if other_url.host and other_url.host != self.host:
            return False

        # Lean on PurePosixPath for the path-level check so trailing
        # slashes, roots, and normalization behave correctly.
        self_path = PurePosixPath(self.path or "/")
        other_path = PurePosixPath(other_url.path or "/")
        return self_path.is_relative_to(other_path)

    def relative_to(self, other: Any) -> URL:
        """Return a URL whose path is ``self.path`` re-rooted under ``other``.

        Raises :class:`ValueError` when ``self`` is not under ``other``
        (check with :meth:`is_relative_to` first if the relationship
        is unknown).

        The returned URL keeps ``self``'s scheme/host/userinfo/port/
        query/fragment — only the path changes. The new path is the
        relative tail, with a leading ``/`` so it remains a well-formed
        URL path (a URL path without a leading slash is ambiguous with
        ``urllib.parse``). An exact match ``self == other`` yields a
        path of ``"/"``.
        """
        other_url = self.from_(other)

        if not self.is_relative_to(other_url):
            raise ValueError(
                f"{self.to_string()!r} is not relative to {other_url.to_string()!r}. "
                "Use is_relative_to() to test the relationship first."
            )

        self_path = PurePosixPath(self.path or "/")
        other_path = PurePosixPath(other_url.path or "/")
        tail = self_path.relative_to(other_path)

        # PurePosixPath.relative_to returns a relative path (no leading
        # slash); URL paths should be rooted, so prepend "/". A pure
        # "." (same directory) collapses to "/".
        tail_str = str(tail)
        new_path = "/" if tail_str in (".", "") else "/" + tail_str
        return self._replace(path=new_path)

    def _replace(self, **changes: Any) -> URL:
        return replace(self, **changes)

    @property
    def user(self) -> str | None:
        ui = self.userinfo
        if ui is None:
            return None
        if ":" in ui:
            return unquote(ui.split(":", 1)[0])
        return unquote(ui)

    @property
    def password(self) -> str | None:
        ui = self.userinfo
        if ui is None or ":" not in ui:
            return None
        return unquote(ui.split(":", 1)[1])

    def with_user_password(self, user: str | None, password: str | None = None) -> URL:
        if user is None and password is None:
            if self.userinfo:
                return self.with_userinfo(None)
            return self

        user_encoded = "" if user is None else quote(str(user), safe="")
        if password is None:
            return self.with_userinfo(user_encoded)

        password_encoded = quote(str(password), safe="")
        return self.with_userinfo(f"{user_encoded}:{password_encoded}")

    @property
    def query_dict(self) -> Mapping[str, tuple[str, ...]]:
        if not self.query:
            return {}

        out: dict[str, list[str]] = {}
        for key, value in parse_qsl(self.query, keep_blank_values=True):
            out.setdefault(key, []).append(value)

        return {key: tuple(values) for key, values in out.items()}

    @property
    def is_absolute(self) -> bool:
        return bool(self.scheme) and bool(self.host)

    @property
    def is_http(self):
        return self.scheme in ("http", "https")

    def to_string(self, *, encode: bool = True) -> str:
        cache_name = "_str_enc" if encode else "_str_raw"
        cached = getattr(self, cache_name)
        if cached is not None:
            return cached

        scheme = self.scheme
        host = self.host
        userinfo = _s(self.userinfo)
        path = self.path
        query = _s(self.query)
        fragment = _s(self.fragment)
        port = _p(self.port)

        if encode:
            userinfo = _encode_userinfo(userinfo)
            path = _encode_path(path)
            query = _encode_query(query)
            fragment = _encode_fragment(fragment)

        netloc = ""
        if host:
            host_text = f"[{host}]" if ":" in host and not host.startswith("[") else host
            netloc = host_text

            if port > 0:
                netloc = f"{netloc}:{port}"

            if userinfo:
                netloc = f"{userinfo}@{netloc}"

        rendered = urlunsplit((scheme, netloc, path, query, fragment))
        object.__setattr__(self, cache_name, rendered)
        return rendered

    @property
    def authority(self) -> str:
        if not self.host:
            return ""

        host_text = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        authority = host_text

        if _p(self.port) > 0:
            authority = f"{authority}:{self.port}"

        if self.userinfo:
            authority = f"{self.userinfo}@{authority}"

        return authority

    def to_struct_dict(self) -> dict[str, Any]:
        """Flatten into the dict shape that matches :data:`URL_STRUCT`.

        Used by request / response / userinfo serializers to populate a
        nested URL struct column without each module re-implementing the
        same field-by-field mapping.
        """
        return {
            "scheme":   self.scheme or "",
            "userinfo": self.userinfo,
            "host":     self.host or "",
            "port":     self.port if self.port not in (None, 0) else None,
            "path":     self.path or "",
            "query":    self.query,
            "fragment": self.fragment,
        }

    def to_pathlib(self, *, strict: bool = True) -> Path:
        """Convert this URL to a ``pathlib.Path``.

        The inverse of ``from_pathlib``: strips the leading slash from
        Windows drive-letter paths so ``/C:/Users/x`` → ``C:/Users/x``,
        then hands the result to ``Path``. Does **not** re-resolve or
        expand ``~`` — the round-trip ``URL.from_pathlib(p).to_pathlib()``
        gives an already-absolute ``Path``, and calling ``.resolve()`` on
        an already-resolved path would force unwanted filesystem I/O
        (e.g. following symlinks). Callers who want the resolved form
        can call ``.resolve()`` on the returned ``Path`` themselves.

        By default only the ``file`` scheme is accepted and the URL must
        not carry a query, fragment, host, userinfo, or port — converting
        those to a ``Path`` would silently drop data. Pass ``strict=False``
        to relax both checks: any scheme is accepted and extra URL
        components are ignored. Use that mode only when you've already
        validated the URL shape elsewhere.

        Raises:
            ValueError: if the URL's path is empty, or (when ``strict``)
                if the scheme is not ``file`` or extra components are
                present.

        Platform note: on Linux, ``Path("C:/x")`` yields a ``PosixPath``
        that treats ``C:`` as a directory name — a ``file:///C:/x`` URL
        can only round-trip correctly on Windows. This matches how
        ``from_pathlib`` was specified.
        """
        if strict:
            if self.scheme and self.scheme != "file":
                raise ValueError(
                    f"to_pathlib requires scheme='file' (got {self.scheme!r}); "
                    f"pass strict=False to coerce any scheme"
                )
            extras = [
                name
                for name, value in (
                    ("query", self.query),
                    ("fragment", self.fragment),
                    ("host", self.host or None),
                    ("userinfo", self.userinfo),
                    ("port", self.port),
                )
                if value
            ]
            if extras:
                raise ValueError(
                    f"to_pathlib cannot represent URL components "
                    f"{extras} as a filesystem path; pass strict=False to drop them"
                )

        raw = self.path
        if not raw or raw == "/":
            raise ValueError(
                f"Cannot convert URL with empty path to Path: {self.to_string(encode=False)!r}"
            )

        return Path(_strip_windows_drive_slash(raw))

    def join(self, ref: str | URL) -> URL:
        base = self.to_string(encode=True)
        target = ref.to_string(encode=True) if isinstance(ref, URL) else ref
        return URL.from_str(urljoin(base, target), normalize=True)

    def with_scheme(self, scheme: str | None, inplace: bool = False) -> URL:
        scheme_text = _lower_if(_s(scheme))
        port = _remove_default_port(scheme_text, _strip_trailing_dot(_lower_if(self.host)), _p(self.port))
        # ``port`` is the int sentinel (0 for "no port"); normalize to None
        # so equality with constructor-built URLs holds.
        port_attr = port or None

        if inplace:
            object.__setattr__(self, "scheme", scheme_text)
            object.__setattr__(self, "port", port_attr)
            # Invalidate cached renderings — the scheme just changed.
            object.__setattr__(self, "_str_enc", None)
            object.__setattr__(self, "_str_raw", None)
            return self
        else:
            return self._replace(scheme=scheme_text, port=port_attr)

    def with_userinfo(self, userinfo: str | None) -> URL:
        return self._replace(userinfo=userinfo or None)

    def with_host(self, host: str | None) -> URL:
        host_text = _strip_trailing_dot(_lower_if(_s(host)))
        port = _remove_default_port(self.scheme, host_text, _p(self.port))
        return self._replace(host=host_text, port=port or None)

    def with_path(
        self,
        path: str | None,
        *,
        os_find: bool = False
    ) -> URL:
        return self._replace(path=_normalize_path(_s(path), os_find=os_find))

    def with_query(self, query: str | None) -> URL:
        query_text = _s(query).lstrip("?")
        return self._replace(query=query_text or None)

    def with_fragment(self, fragment: str | None) -> URL:
        fragment_text = _s(fragment).lstrip("#")
        return self._replace(fragment=fragment_text or None)

    def query_items(self, *, keep_blank_values: bool = True) -> tuple[tuple[str, str], ...]:
        if not self.query:
            return ()
        return tuple(parse_qsl(self.query, keep_blank_values=keep_blank_values))

    def query_mapping(self, *, keep_blank_values: bool = True) -> Mapping[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key, value in self.query_items(keep_blank_values=keep_blank_values):
            out.setdefault(key, []).append(value)
        return out

    def add_query_item(
        self,
        key: str,
        value: str | None | Sequence[str | None],
        replace: bool = True,
    ) -> URL:
        """Add or replace one or more values for ``key`` in the query string.

        ``value`` may be:

        - a scalar (``str`` or ``None``) — appended as a single ``(key, v)``
          pair, with ``None`` coerced to ``""``,
        - a ``list`` / ``tuple`` of scalars — appended as one pair per
          element. ``None`` elements coerce to ``""``.

        When ``replace=True`` (the default), every existing pair with the
        same ``key`` is removed first. Combined with an empty sequence,
        this clears the key entirely; combined with a non-empty sequence,
        it sets the key to exactly those values. With ``replace=False``,
        new pairs are appended without touching existing ones.

        Generators / arbitrary iterables are intentionally not accepted:
        a single-pass iterator that gets exhausted during validation
        would be a footgun, and ``str`` would be ambiguously iterable.
        Materialize first if you have one.
        """
        if key is None:
            raise ValueError("key cannot be None")

        key_text = str(key)

        if isinstance(value, (list, tuple)):
            new_values: list[str] = ["" if v is None else str(v) for v in value]
        else:
            new_values = ["" if value is None else str(value)]

        items = list(self.query_items(keep_blank_values=True))
        if replace:
            items = [(k, v) for k, v in items if k != key_text]

        items.extend((key_text, v) for v in new_values)
        items.sort(key=lambda item: (item[0], item[1]))
        return self.with_query(urlencode(items, doseq=True))

    def with_query_items(
        self,
        items: Mapping[str, str | Sequence[str]] | Iterable[tuple[str, str]],
        *,
        sort_keys: bool = True,
    ) -> URL:
        pairs: list[tuple[str, str]] = []

        if isinstance(items, Mapping):
            for key, value in items.items():
                if isinstance(value, (list, tuple)):
                    pairs.extend((str(key), str(v)) for v in value)
                else:
                    pairs.append((str(key), str(value)))
        else:
            pairs.extend((str(key), str(value)) for key, value in items)

        if sort_keys:
            pairs.sort(key=lambda item: (item[0], item[1]))

        return self.with_query(urlencode(pairs, doseq=True))

    def anonymize(
        self,
        mode: Literal["remove", "redact"] = "remove",
        *,
        sort_keys: bool = True,
    ) -> URL:
        if self._anonymized:
            return self

        if self._anonymized is None:
            result = self

            if self.query:
                current = self.query_dict
                anonymized = anonymize_parameters(current, mode=mode)
                if anonymized != current:
                    result = result.with_query_items(anonymized, sort_keys=sort_keys)

            if self.userinfo:
                result = result.with_userinfo("<redacted>" if mode == "redact" else None)

            object.__setattr__(result, "_anonymized", True)
            return result

        return self

    def endswith(self, value: str):
        if value and self.path:
            return self.path.endswith(value)
        return False


_EMPTY_URL = URL()