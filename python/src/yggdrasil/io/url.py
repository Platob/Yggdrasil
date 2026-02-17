from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, Union, Any, Iterable
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)

from yggdrasil.io.parameters import anonymize_parameters

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

# RFC 3986-ish safe sets
_SAFE_PATH = "/:@-._~!$&'()*+,;="
# Query must NOT escape separators '&' and '=' when query is already structured
_SAFE_QUERY = "-._~!$'()*+,;=:@/?&="
_SAFE_FRAGMENT = "-._~!$&'()*+,;=:@/?"

# port == 0 means "absent"
_NO_PORT = 0


def _lower_if(s: str) -> str:
    return s.lower() if s else ""


def _strip_trailing_dot(host: str) -> str:
    if not host:
        return host
    return host[:-1] if host.endswith(".") else host


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    # collapse repeated slashes (except keep single "/")
    if path != "/" and "//" in path:
        while "//" in path:
            path = path.replace("//", "/")
    return path


def _remove_default_port(scheme: str, host: str, port: int) -> int:
    """
    Return 0 when port is absent or is the scheme default.
    """
    if not scheme or not host or port <= 0:
        return _NO_PORT
    default = _DEFAULT_PORTS.get(scheme)
    return _NO_PORT if default == port else port


def _encode_userinfo(userinfo: str) -> str:
    return quote(userinfo, safe=":!$&'()*+,;=") if userinfo else ""


def _encode_path(path: str) -> str:
    # keep existing '%' to avoid double-encoding
    return quote(path, safe=_SAFE_PATH + "%")


def _encode_query(query: str) -> str:
    # Query is structured: keep '&' and '='; keep '%' to avoid double-encoding
    return quote(query, safe=_SAFE_QUERY + "%")


def _encode_fragment(fragment: str) -> str:
    return quote(fragment, safe=_SAFE_FRAGMENT + "%")


def _decode_maybe(s: str, decode: bool) -> str:
    return unquote(s) if decode and s else s


def _parse_netloc(netloc: str, *, decode: bool) -> tuple[str, str, int]:
    """
    Parse netloc into (userinfo, host, port). Port uses 0 as "absent".
    Handles:
      - userinfo@host:port
      - IPv6: [::1]:8080
      - host without port
    """
    if not netloc:
        return "", "", _NO_PORT

    userinfo = ""
    hostport = netloc

    if "@" in netloc:
        userinfo, hostport = netloc.rsplit("@", 1)
        userinfo = _decode_maybe(userinfo, decode)

    host = ""
    port = _NO_PORT

    if hostport.startswith("["):
        rb = hostport.find("]")
        if rb == -1:
            # malformed, treat whole as host
            return userinfo, hostport, _NO_PORT
        host = hostport[1:rb]
        rest = hostport[rb + 1 :]
        if rest.startswith(":") and len(rest) > 1:
            p = rest[1:]
            if p.isdigit():
                port = int(p)
        return userinfo, host, port

    if ":" in hostport:
        h, p = hostport.rsplit(":", 1)
        host = h
        if p.isdigit():
            port = int(p)
    else:
        host = hostport

    return userinfo, host, port


@dataclass(frozen=True, slots=True)
class URL:
    scheme: str = ""
    userinfo: str = ""
    host: str = ""
    port: int = _NO_PORT  # 0 means absent
    path: str = ""
    query: str = ""
    fragment: str = ""

    @classmethod
    def parse_any(
        cls,
        obj: Any,
        *,
        decode: bool = False,
        normalize: bool = True,
    ):
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, dict):
            return cls.parse_dict(
                obj,
                decode=decode,
                normalize=normalize
            )

        return cls.parse(
            raw=str(obj),
            decode=decode,
            normalize=normalize,
        )

    @classmethod
    def parse(
        cls,
        raw: str,
        *,
        decode: bool = False,
        normalize: bool = True,
    ) -> "URL":
        sp = urlsplit(raw)

        scheme = sp.scheme or ""
        userinfo, host, port = _parse_netloc(sp.netloc, decode=decode)

        path = _decode_maybe(sp.path, decode)
        query = _decode_maybe(sp.query, decode)
        fragment = _decode_maybe(sp.fragment, decode)

        if not normalize:
            return URL(
                scheme=scheme,
                userinfo=userinfo,
                host=host,
                port=port,
                path=path,
                query=query,
                fragment=fragment,
            )

        scheme_n = _lower_if(scheme)
        host_n = _strip_trailing_dot(_lower_if(host))
        path_n = _normalize_path(path)
        port_n = _remove_default_port(scheme_n, host_n, port)

        query_n = query
        if query_n:
            items = parse_qsl(query_n, keep_blank_values=True)
            items.sort(key=lambda kv: (kv[0], kv[1]))
            query_n = urlencode(items, doseq=True)

        return cls(
            scheme=scheme_n,
            userinfo=userinfo,
            host=host_n,
            port=port_n,
            path=path_n,
            query=query_n,
            fragment=fragment,
        )

    @classmethod
    def parse_dict(
        cls,
        d: Mapping[str, Any],
        *,
        decode: bool = False,
        normalize: bool = True,
    ) -> "URL":
        """
        Build a URL from a mapping.

        Supported inputs:
          - {"url": "..."} (delegates to parse)
          - {"scheme","netloc","path","query","fragment"} (like urlsplit result)
          - {"scheme","userinfo","host","port","path","query","fragment"}
          - {"scheme","netloc", ...} plus overrides (userinfo/host/port)

        Notes:
          - port == 0 / None => absent
          - if both netloc and (userinfo/host/port) are provided, explicit fields win
        """
        if not d:
            return cls()

        # 1) Common shortcut: {"url": "..."}
        raw = d.get("url") or d.get("raw")
        if raw is not None:
            return cls.parse(str(raw), decode=decode, normalize=normalize)

        # 2) Pull split-ish fields
        scheme = str(d.get("scheme") or "")
        path = str(d.get("path") or "")
        query = str(d.get("query") or "")
        fragment = str(d.get("fragment") or "")

        # Optional full authority in one string
        netloc = d.get("netloc")
        if netloc is None:
            # Some libs use "authority"
            netloc = d.get("authority")
        netloc_s = str(netloc) if netloc is not None else ""

        # 3) Parse netloc if present
        userinfo = ""
        host = ""
        port = _NO_PORT
        if netloc_s:
            userinfo, host, port = _parse_netloc(netloc_s, decode=decode)

        # 4) Explicit overrides win (for dicts that already have fields split out)
        if "userinfo" in d and d["userinfo"] is not None:
            userinfo = str(d["userinfo"])
            userinfo = _decode_maybe(userinfo, decode)

        if "host" in d and d["host"] is not None:
            host = str(d["host"])
            host = _decode_maybe(host, decode)

        if "port" in d:
            p = d.get("port")
            if p is None or p == "" or p == 0:
                port = _NO_PORT
            elif isinstance(p, int):
                port = p if p > 0 else _NO_PORT
            else:
                ps = str(p)
                port = int(ps) if ps.isdigit() and int(ps) > 0 else _NO_PORT

        # 5) Decode other parts (optionally)
        path = _decode_maybe(path, decode)
        query = _decode_maybe(query, decode)
        fragment = _decode_maybe(fragment, decode)

        if not normalize:
            return cls(
                scheme=scheme,
                userinfo=userinfo,
                host=host,
                port=port,
                path=path,
                query=query.lstrip("?"),
                fragment=fragment.lstrip("#"),
            )

        # 6) Apply the same normalization policy as parse()
        scheme_n = _lower_if(scheme)
        host_n = _strip_trailing_dot(_lower_if(host))
        path_n = _normalize_path(path)
        port_n = _remove_default_port(scheme_n, host_n, port)

        query_n = query.lstrip("?")
        if query_n:
            items = parse_qsl(query_n, keep_blank_values=True)
            items.sort(key=lambda kv: (kv[0], kv[1]))
            query_n = urlencode(items, doseq=True)

        fragment_n = fragment.lstrip("#")

        return cls(
            scheme=scheme_n,
            userinfo=userinfo,
            host=host_n,
            port=port_n,
            path=path_n,
            query=query_n,
            fragment=fragment_n,
        )

    def __str__(self) -> str:
        return self.to_string()

    def __truediv__(self, other):
        if isinstance(other, str):
            return self.with_path(path=other)
        raise ValueError("Cannot join %s with %s" % (self, type(other)))

    @property
    def query_dict(self) -> Mapping[str, Tuple[str, ...]]:
        """
        Parsed query as an immutable mapping: key -> tuple(values...)

        - Preserves original order of values per key (matching parse_qsl order)
        - Empty query => {}
        - Keeps blank values (k=) and keys without '=' (k) as ("",)
        """
        if not self.query:
            return {}

        out: dict[str, list[str]] = {}
        for k, v in parse_qsl(self.query, keep_blank_values=True):
            out.setdefault(k, []).append(v)

        return {k: tuple(vs) for k, vs in out.items()}

    @property
    def is_absolute(self):
        return bool(self.scheme) and bool(self.host)

    def to_string(self, *, encode: bool = True) -> str:
        scheme = self.scheme
        host = self.host
        userinfo = self.userinfo
        path = self.path
        query = self.query
        fragment = self.fragment

        if encode:
            if userinfo:
                userinfo = _encode_userinfo(userinfo)
            path = _encode_path(path)
            query = _encode_query(query)
            fragment = _encode_fragment(fragment)

        netloc = ""
        if host:
            out_host = host
            # bracket IPv6
            if ":" in out_host and not out_host.startswith("["):
                out_host = f"[{out_host}]"
            netloc = out_host

            if self.port > 0:
                netloc = f"{netloc}:{self.port}"

            if userinfo:
                netloc = f"{userinfo}@{netloc}"

        return urlunsplit((scheme, netloc, path, query, fragment))

    @property
    def authority(self) -> str:
        if not self.host:
            return ""
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if self.port > 0:
            netloc = f"{netloc}:{self.port}"
        if self.userinfo:
            netloc = f"{self.userinfo}@{netloc}"
        return netloc

    def join(self, ref: Union[str, "URL"]) -> "URL":
        base = self.to_string(encode=True)
        target = ref.to_string(encode=True) if isinstance(ref, URL) else ref
        return URL.parse(urljoin(base, target), normalize=True)

    # Immutable edits
    def with_scheme(self, scheme: str) -> "URL":
        scheme_n = _lower_if(scheme)
        return URL(
            scheme=scheme_n,
            userinfo=self.userinfo,
            host=self.host,
            port=_remove_default_port(scheme_n, self.host, self.port),
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_userinfo(self, userinfo: str) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=userinfo,
            host=self.host,
            port=self.port,
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_host(self, host: str) -> "URL":
        host_n = _strip_trailing_dot(_lower_if(host))
        port_n = _remove_default_port(self.scheme, host_n, self.port)
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=host_n,
            port=port_n,
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_path(self, path: str) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=_normalize_path(path),
            query=self.query,
            fragment=self.fragment,
        )

    def with_query(self, query: str) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=self.path,
            query=query.lstrip("?"),
            fragment=self.fragment,
        )

    def with_fragment(self, fragment: str) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=self.path,
            query=self.query,
            fragment=fragment.lstrip("#"),
        )

    # Query helpers
    def query_items(self, *, keep_blank_values: bool = True) -> Tuple[Tuple[str, str], ...]:
        if not self.query:
            return ()
        return tuple(parse_qsl(self.query, keep_blank_values=keep_blank_values))

    def with_query_items(self, items: Mapping[str, str] | Tuple[Tuple[str, str], ...]) -> "URL":
        q = urlencode(items if isinstance(items, tuple) else list(items.items()), doseq=True)
        return self.with_query(q)

    # ------------------- URL anonymization -------------------

    def anonymize(
        self,
        mode: str = "redact",  # "redact" | "hash"
    ) -> "URL":
        """
        Returns a new URL with sensitive parts redacted.

        - Query: for keys in sensitive_query_keys, replace *values* with query_replacement.
          Keeps the key names so your logs remain groupable.
        - Userinfo: if strip_userinfo, removes user:pass@ completely.
        - Fragment: optionally redact entire fragment (sometimes people shove tokens in there).
        - No mutation; URL is frozen.

        normalize:
          - False (default): preserve existing ordering/format as much as possible, only touching
            the sensitive bits (good for debugging parity).
          - True: canonicalize like parse(normalize=True) after editing.
        """
        result = self

        if self.query:
            params = self.query_dict
            anon = anonymize_parameters(params, mode=mode)

            if params != anon:
                result = self.with_query_items(anon)

        if self.userinfo:
            result = self.with_userinfo(
                userinfo="<redacted>"
            )

        return result

    def xxh3_64(
        self,
        *,
        exclude_userinfo: bool = False,
        exclude_scheme: bool = False,
        exclude_host: bool = False,
        exclude_port: bool = False,
        exclude_path: bool = False,
        exclude_query: bool | list[str] | None = None,
        exclude_fragment: bool = False,
    ) -> int:
        """
        Stable 64-bit xxh3 hash of a canonical URL with selective exclusions.

        exclude_query semantics:
        - None           → include full query (sorted, canonical)
        - True           → exclude entire query
        - list[str]      → exclude only those query parameter names
        """
        from ..xxhash import xxhash

        scheme = "" if exclude_scheme else self.scheme
        host = "" if exclude_host else self.host
        userinfo = "" if (exclude_userinfo or exclude_host) else self.userinfo

        path = "" if exclude_path else self.path
        fragment = "" if exclude_fragment else self.fragment

        query = ""
        if exclude_query is not True:
            query = self.query
            if query:
                items = parse_qsl(query, keep_blank_values=True)
                if isinstance(exclude_query, list):
                    excl = set(exclude_query)
                    items = [(k, v) for (k, v) in items if k not in excl]
                items.sort(key=lambda kv: (kv[0], kv[1]))
                query = urlencode(items, doseq=True)

        port = _NO_PORT if (exclude_port or exclude_host) else self.port

        netloc = ""
        if host:
            out_host = host
            if ":" in out_host and not out_host.startswith("["):
                out_host = f"[{out_host}]"
            netloc = out_host

            if port > 0:
                netloc = f"{netloc}:{port}"

            if userinfo:
                netloc = f"{userinfo}@{netloc}"

        s = urlunsplit((scheme, netloc, path, query, fragment))

        h = xxhash.xxh3_64()
        h.update(s.encode("utf-8"))
        return h
