from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Sequence, Type, TypeVar, Optional
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

__all__ = [
    "URL",
    "URLResource",
    "get_registered_url_resource",
    "register_url_resource",
    "registered_url_schemes",
    "url_resource_class",
]
T = TypeVar("T", bound="URLResource")

_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}

_SAFE_PATH = "/:@-._~!$&'()*+,;="
_SAFE_QUERY = "-._~!$'()*+,;=:@/?&="
_SAFE_FRAGMENT = "-._~!$&'()*+,;=:@/?"

_NO_PORT = 0

_REGISTRY: Dict[str, Type["URLResource"]] = {}


def _lower_if(value: str) -> str:
    return value.lower() if value else ""


def _strip_trailing_dot(host: str) -> str:
    return host[:-1] if host.endswith(".") else host


def _normalize_path(path: str) -> str:
    if not path:
        return "/"

    elif path == "/":
        return path

    elif not path.startswith("/"):
        path = "/" + os.path.realpath(path).replace("\\", "/")

    return path


def _remove_default_port(scheme: str, host: str, port: int) -> int:
    if not scheme or not host or port <= 0:
        return _NO_PORT
    return _NO_PORT if _DEFAULT_PORTS.get(scheme) == port else port


def _encode_userinfo(userinfo: str) -> str:
    return quote(userinfo, safe=":!$&'()*+,;=") if userinfo else ""


def _encode_path(path: str) -> str:
    return quote(path, safe=_SAFE_PATH + "%")


def _encode_query(query: str) -> str:
    return quote(query, safe=_SAFE_QUERY + "%")


def _encode_fragment(fragment: str) -> str:
    return quote(fragment, safe=_SAFE_FRAGMENT + "%")


def _decode_maybe(value: str, decode: bool) -> str:
    return unquote(value) if decode and value else value


def _s(value: str | None) -> str:
    return value or ""


def _p(value: int | None) -> int:
    return value or _NO_PORT


def _normalize_query(query: str) -> str:
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


def _looks_like_local_path(raw: str) -> bool:
    if not raw:
        return False

    stripped = raw.strip()
    split = urlsplit(stripped)

    if split.scheme:
        if len(split.scheme) == 1 and len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in ("\\", "/"):
            return True
        return False

    return stripped.startswith(("/", "~", "./", "../")) or "/" in stripped or "\\" in stripped


def _pathlike_to_file_parts(raw: str | Path) -> tuple[str, str]:
    path = raw if isinstance(raw, Path) else Path(raw)
    path = path.expanduser().resolve(strict=False)
    return "file", path.as_posix()


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
    path_n = _normalize_path(path)
    query_n = _normalize_query(query)
    fragment_n = fragment.lstrip("#")

    return scheme_n, host_n, port_n, path_n, query_n, fragment_n


@dataclass(frozen=True, slots=True)
class URL:
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

    @classmethod
    def empty(cls) -> "URL":
        return _EMPTY_URL

    @staticmethod
    def path_encode(path: str) -> str:
        return _encode_path(path)

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        default_scheme: Optional[str] = None,
        decode: bool = False,
        normalize: bool = True,
    ) -> URL:
        if isinstance(obj, cls):
            return obj

        if isinstance(obj, Mapping):
            return cls.parse_dict(obj, decode=decode, normalize=normalize)

        if isinstance(obj, Path):
            scheme, path = _pathlike_to_file_parts(obj)
            return cls.parse_dict(
                {"scheme": default_scheme or scheme, "path": path},
                decode=decode,
                normalize=normalize,
            )

        if obj is None:
            raise ValueError("Cannot parse URL from None")

        return cls.parse_str(
            str(obj),
            default_scheme=default_scheme,
            decode=decode,
            normalize=normalize
        )

    @classmethod
    def parse_str(
        cls,
        raw: str,
        *,
        default_scheme: Optional[str] = None,
        decode: bool = False,
        normalize: bool = True,
    ) -> URL:
        split = urlsplit(raw)
        userinfo, host, port = _parse_netloc(split.netloc, decode=decode)

        scheme = default_scheme or split.scheme
        path = _decode_maybe(split.path, decode)
        query = _decode_maybe(split.query, decode)
        fragment = _decode_maybe(split.fragment, decode)

        if scheme not in ("file", None) and not host and path:
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
    def parse_dict(
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
            return cls.parse_str(str(raw), decode=decode, normalize=normalize)

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
        return f"URL<{self.to_string()!r}>"

    def __truediv__(self, other: object) -> URL:
        if not isinstance(other, str):
            raise ValueError(f"Cannot join {self} with {type(other)}")
        return self.with_path(other)

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

    def join(self, ref: str | URL) -> URL:
        base = self.to_string(encode=True)
        target = ref.to_string(encode=True) if isinstance(ref, URL) else ref
        return URL.parse_str(urljoin(base, target), normalize=True)

    def with_scheme(self, scheme: str | None) -> URL:
        scheme_text = _lower_if(_s(scheme))
        port = _remove_default_port(scheme_text, _strip_trailing_dot(_lower_if(self.host)), _p(self.port))
        return self._replace(scheme=scheme_text, port=port or None)

    def with_userinfo(self, userinfo: str | None) -> URL:
        return self._replace(userinfo=userinfo or None)

    def with_host(self, host: str | None) -> URL:
        host_text = _strip_trailing_dot(_lower_if(_s(host)))
        port = _remove_default_port(self.scheme, host_text, _p(self.port))
        return self._replace(host=host_text, port=port or None)

    def with_path(self, path: str | None) -> URL:
        return self._replace(path=_normalize_path(_s(path)))

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

    def add_query_item(self, key: str, value: str | None, replace: bool = True) -> URL:
        if key is None:
            raise ValueError("key cannot be None")

        key_text = str(key)
        value_text = "" if value is None else str(value)

        items = list(self.query_items(keep_blank_values=True))
        if replace:
            items = [(k, v) for k, v in items if k != key_text]

        items.append((key_text, value_text))
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


class URLResource(ABC):
    @classmethod
    @abstractmethod
    def url_scheme(cls) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_url(self, scheme: str | None = None) -> URL:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_parsed_url(cls: Type[T], url: URL) -> T:
        raise NotImplementedError

    @classmethod
    def from_url(cls: Type[T], url: URL | str | Any) -> T:
        default_scheme = None if cls is URLResource else cls.url_scheme()
        parsed = URL.parse(url, default_scheme=default_scheme)
        scheme = parsed.scheme.strip().lower()

        if cls is URLResource:
            if not scheme:
                raise ValueError(
                    "Cannot dispatch URLResource.from_url() without a scheme. "
                    f"Got url={parsed.to_string(encode=False)!r}"
                )

            impl = _REGISTRY.get(scheme)
            if impl is None:
                registered = ", ".join(registered_url_schemes()) or "(none)"
                raise ValueError(
                    f"No URLResource registered for scheme {scheme!r}. "
                    f"Registered schemes: {registered}"
                )

            return impl.from_parsed_url(parsed)  # type: ignore[return-value]

        return cls.from_parsed_url(parsed)


def get_registered_url_resource(scheme: str) -> Type["URLResource"] | None:
    return _REGISTRY.get((scheme or "").strip().lower())


def registered_url_schemes() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def url_resource_class(cls: Type[T] | None = None):
    def decorator(resource_cls: Type[T]) -> Type[T]:
        return register_url_resource(resource_cls)

    return decorator if cls is None else decorator(cls)


def register_url_resource(resource_cls: Type[T]) -> Type[T]:
    if not issubclass(resource_cls, URLResource):
        raise TypeError(f"Can only register URLResource subclasses, got {resource_cls!r}")

    scheme = resource_cls.url_scheme().strip().lower()
    if not scheme:
        raise ValueError(f"{resource_cls.__name__}.url_scheme() must return a non-empty scheme")

    _REGISTRY[scheme] = resource_cls
    return resource_cls


@url_resource_class
@dataclass(frozen=True, slots=True)
class FileResource(URLResource):
    path: Path

    @classmethod
    def url_scheme(cls) -> str:
        return "file"

    def to_url(self, scheme: str | None = None) -> URL:
        target_scheme = (scheme or self.url_scheme()).strip().lower()
        if target_scheme != "file":
            raise ValueError(f"{self.__class__.__name__} only supports the 'file' scheme, got {target_scheme!r}")

        normalized = self.path.expanduser().resolve(strict=False).as_posix()

        # Windows drive path must be /C:/... in file URLs
        if len(normalized) >= 2 and normalized[1] == ":":
            normalized = "/" + normalized

        return URL(
            scheme="file",
            path=normalized,
        )

    @classmethod
    def from_parsed_url(cls: Type["FileResource"], url: URL) -> "FileResource":
        raw_path = url.path or ""
        if not raw_path:
            raise ValueError("File URL is missing a path")

        path_str = raw_path

        # file:///C:/x -> Path("C:/x") on Windows-style input
        if len(path_str) >= 3 and path_str[0] == "/" and path_str[2] == ":":
            path_str = path_str[1:]

        return cls(path=Path(path_str).expanduser().resolve(strict=False))

    @classmethod
    def from_path(cls, value: str | Path) -> "FileResource":
        return cls(path=Path(value).expanduser().resolve(strict=False))

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def joinpath(self, *parts: str) -> "FileResource":
        return FileResource(self.path.joinpath(*parts).resolve(strict=False))

    def __truediv__(self, other: str) -> "FileResource":
        if not isinstance(other, str):
            raise ValueError(f"Cannot join {self} with {type(other)}")
        return self.joinpath(other)


_EMPTY_URL = URL()
