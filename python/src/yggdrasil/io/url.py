from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple, Union, Any, Literal
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

_SAFE_PATH = "/:@-._~!$&'()*+,;="
_SAFE_QUERY = "-._~!$'()*+,;=:@/?&="
_SAFE_FRAGMENT = "-._~!$&'()*+,;=:@/?"

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
    if path != "/" and "//" in path:
        while "//" in path:
            path = path.replace("//", "/")
    return path


def _remove_default_port(scheme: str, host: str, port: int) -> int:
    if not scheme or not host or port <= 0:
        return _NO_PORT
    default = _DEFAULT_PORTS.get(scheme)
    return _NO_PORT if default == port else port


def _encode_userinfo(userinfo: str) -> str:
    return quote(userinfo, safe=":!$&'()*+,;=") if userinfo else ""


def _encode_path(path: str) -> str:
    return quote(path, safe=_SAFE_PATH + "%")


def _encode_query(query: str) -> str:
    return quote(query, safe=_SAFE_QUERY + "%")


def _encode_fragment(fragment: str) -> str:
    return quote(fragment, safe=_SAFE_FRAGMENT + "%")


def _decode_maybe(s: str, decode: bool) -> str:
    return unquote(s) if decode and s else s


def _parse_netloc(netloc: str, *, decode: bool) -> tuple[str, str, int]:
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


def _s(x: str | None) -> str:
    """None-safe string: None -> ''."""
    return x or ""


def _p(x: int | None) -> int:
    """None-safe port: None -> 0 (absent)."""
    return x or _NO_PORT


@dataclass(frozen=True, slots=True)
class URL:
    scheme: str | None = None
    userinfo: str | None = None
    host: str | None = None
    port: int | None = None  # None/0 means absent
    path: str | None = None
    query: str | None = None
    fragment: str | None = None

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
            return cls.parse_dict(obj, decode=decode, normalize=normalize)

        return cls.parse(raw=str(obj), decode=decode, normalize=normalize)

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
                scheme=scheme or None,
                userinfo=userinfo or None,
                host=host or None,
                port=port or None,
                path=path or None,
                query=query or None,
                fragment=fragment or None,
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
            scheme=scheme_n or None,
            userinfo=userinfo or None,
            host=host_n or None,
            port=port_n or None,
            path=path_n or None,
            query=query_n or None,
            fragment=fragment or None,
        )

    @classmethod
    def parse_dict(
        cls,
        d: Mapping[str, Any],
        *,
        decode: bool = False,
        normalize: bool = True,
    ) -> "URL":
        if not d:
            return cls()

        raw = d.get("url") or d.get("raw")
        if raw is not None:
            return cls.parse(str(raw), decode=decode, normalize=normalize)

        scheme = str(d.get("scheme") or "")
        path = str(d.get("path") or "")
        query = str(d.get("query") or "")
        fragment = str(d.get("fragment") or "")

        netloc = d.get("netloc")
        if netloc is None:
            netloc = d.get("authority")
        netloc_s = str(netloc) if netloc is not None else ""

        userinfo = ""
        host = ""
        port = _NO_PORT
        if netloc_s:
            userinfo, host, port = _parse_netloc(netloc_s, decode=decode)

        if "userinfo" in d and d["userinfo"] is not None:
            userinfo = _decode_maybe(str(d["userinfo"]), decode)

        if "host" in d and d["host"] is not None:
            host = _decode_maybe(str(d["host"]), decode)

        if "port" in d:
            p = d.get("port")
            if p is None or p == "" or p == 0:
                port = _NO_PORT
            elif isinstance(p, int):
                port = p if p > 0 else _NO_PORT
            else:
                ps = str(p)
                port = int(ps) if ps.isdigit() and int(ps) > 0 else _NO_PORT

        path = _decode_maybe(path, decode)
        query = _decode_maybe(query, decode)
        fragment = _decode_maybe(fragment, decode)

        if not normalize:
            return cls(
                scheme=scheme or None,
                userinfo=userinfo or None,
                host=host or None,
                port=port or None,
                path=path or None,
                query=(query.lstrip("?") or None),
                fragment=(fragment.lstrip("#") or None),
            )

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
            scheme=scheme_n or None,
            userinfo=userinfo or None,
            host=host_n or None,
            port=port_n or None,
            path=path_n or None,
            query=query_n or None,
            fragment=fragment_n or None,
        )

    def __str__(self) -> str:
        return self.to_string()

    def __truediv__(self, other):
        if isinstance(other, str):
            return self.with_path(path=other)
        raise ValueError("Cannot join %s with %s" % (self, type(other)))

    @property
    def query_dict(self) -> Mapping[str, Tuple[str, ...]]:
        q = _s(self.query)
        if not q:
            return {}

        out: dict[str, list[str]] = {}
        for k, v in parse_qsl(q, keep_blank_values=True):
            out.setdefault(k, []).append(v)

        return {k: tuple(vs) for k, vs in out.items()}

    @property
    def is_absolute(self):
        return bool(_s(self.scheme)) and bool(_s(self.host))

    def to_string(self, *, encode: bool = True) -> str:
        scheme = _s(self.scheme)
        host = _s(self.host)
        userinfo = _s(self.userinfo)
        path = _s(self.path)
        query = _s(self.query)
        fragment = _s(self.fragment)
        port = _p(self.port)

        if encode:
            if userinfo:
                userinfo = _encode_userinfo(userinfo)
            path = _encode_path(path)
            query = _encode_query(query)
            fragment = _encode_fragment(fragment)

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

        return urlunsplit((scheme, netloc, path, query, fragment))

    @property
    def authority(self) -> str:
        host = _s(self.host)
        if not host:
            return ""
        out_host = host
        if ":" in out_host and not out_host.startswith("["):
            out_host = f"[{out_host}]"
        netloc = out_host

        port = _p(self.port)
        if port > 0:
            netloc = f"{netloc}:{port}"

        userinfo = _s(self.userinfo)
        if userinfo:
            netloc = f"{userinfo}@{netloc}"

        return netloc

    def join(self, ref: Union[str, "URL"]) -> "URL":
        base = self.to_string(encode=True)
        target = ref.to_string(encode=True) if isinstance(ref, URL) else ref
        return URL.parse(urljoin(base, target), normalize=True)

    # Immutable edits (nullable in, nullable out)
    def with_scheme(self, scheme: str | None) -> "URL":
        scheme_n = _lower_if(_s(scheme))
        host = _s(self.host)
        port_n = _remove_default_port(scheme_n, _strip_trailing_dot(_lower_if(host)), _p(self.port))
        return URL(
            scheme=scheme_n or None,
            userinfo=self.userinfo,
            host=self.host,
            port=port_n or None,
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_userinfo(self, userinfo: str | None) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=(userinfo or None),
            host=self.host,
            port=self.port,
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_host(self, host: str | None) -> "URL":
        host_n = _strip_trailing_dot(_lower_if(_s(host)))
        port_n = _remove_default_port(_s(self.scheme), host_n, _p(self.port))
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=host_n or None,
            port=port_n or None,
            path=self.path,
            query=self.query,
            fragment=self.fragment,
        )

    def with_path(self, path: str | None) -> "URL":
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=(_normalize_path(_s(path)) or None),
            query=self.query,
            fragment=self.fragment,
        )

    def with_query(self, query: str | None) -> "URL":
        q = _s(query).lstrip("?")
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=self.path,
            query=q or None,
            fragment=self.fragment,
        )

    def with_fragment(self, fragment: str | None) -> "URL":
        f = _s(fragment).lstrip("#")
        return URL(
            scheme=self.scheme,
            userinfo=self.userinfo,
            host=self.host,
            port=self.port,
            path=self.path,
            query=self.query,
            fragment=f or None,
        )

    # Query helpers
    def query_items(self, *, keep_blank_values: bool = True) -> Tuple[Tuple[str, str], ...]:
        q = _s(self.query)
        if not q:
            return ()
        return tuple(parse_qsl(q, keep_blank_values=keep_blank_values))

    def with_query_items(
        self,
        items: Mapping[str, str] | Tuple[Tuple[str, str], ...],
        *,
        sort_keys: bool = True,
    ) -> "URL":
        seq = items if isinstance(items, tuple) else list(items.items())
        if sort_keys:
            seq = sorted(seq, key=lambda kv: kv[0])
        q = urlencode(seq, doseq=True)
        return self.with_query(q)

    def anonymize(
        self,
        mode: Literal["remove", "redact", "hash"] = "remove",
        *,
        sort_keys: bool = True
    ) -> "URL":
        result = self

        if _s(self.query):
            params = self.query_dict
            anon = anonymize_parameters(params, mode=mode)
            if params != anon:
                result = self.with_query_items(anon, sort_keys=sort_keys)

        if _s(self.userinfo):
            result = result.with_userinfo(userinfo="<redacted>" if mode == "redact" else None)

        return result

    @property
    def is_databricks(self) -> bool:
        return self.scheme == "dbfs"

    def to_databricks_table(self):
        if not self.is_databricks:
            raise ValueError(
                f"Expected Databricks URI with scheme 'dbfs', got scheme={self.scheme!r}"
            )

        if not self.query:
            raise ValueError(
                "Databricks URI is missing query string. "
                "Expected query params: catalog_name, schema_name, table_name"
            )

        from yggdrasil.databricks.sql.table import Table
        from yggdrasil.databricks.workspaces import Workspace

        # query_dict is assumed to be like: {"catalog_name": ["x"], ...}
        items: dict[str, Any] = {
            k: (v[0] if isinstance(v, (list, tuple)) and v else v)
            for k, v in self.query_dict.items()
        }

        required = ("catalog_name", "schema_name", "table_name")
        missing = [key for key in required if not items.get(key)]

        if missing:
            raise ValueError(
                "Missing required Databricks table query params: "
                f"{', '.join(missing)}. "
                "Expected query params: catalog_name, schema_name, table_name"
            )

        catalog_name = items["catalog_name"]
        schema_name = items["schema_name"]
        table_name = items["table_name"]

        if not self.host:
            raise ValueError("Databricks URI is missing host")

        return Table(
            workspace=Workspace(host=self.host),
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
