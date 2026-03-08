from __future__ import annotations

from pathlib import Path

import pytest

import yggdrasil.io.url as mod
from yggdrasil.io.url import (
    URL,
    URLResource,
    get_registered_url_resource,
    register_url_resource,
    registered_url_schemes,
    url_resource_class,
)


def test_parse_str_normalizes_scheme_host_default_port_and_query_sorting():
    url = URL.parse_str("HTTPS://Example.COM.:443//a//b?z=2&a=3&a=1#frag")

    assert url.scheme == "https"
    assert url.host == "example.com"
    assert url.port is None
    assert url.path == "/a/b"
    assert url.query == "a=1&a=3&z=2"
    assert url.fragment == "frag"


def test_parse_str_without_normalization_preserves_values():
    url = URL.parse_str("HTTPS://Example.COM.:443//a//b?z=2&a=3#frag", normalize=False)

    assert url.scheme == "https"
    assert url.host == "Example.COM."
    assert url.port == 443
    assert url.path == "//a//b"
    assert url.query == "z=2&a=3"
    assert url.fragment == "frag"


def test_parse_dict_empty_returns_default_url():
    url = URL.parse_dict({})

    assert url == URL()


def test_parse_dict_from_raw_shortcut():
    url = URL.parse_dict({"raw": "https://example.com/x?a=2&a=1"})

    assert url.scheme == "https"
    assert url.host == "example.com"
    assert url.path == "/x"
    assert url.query == "a=1&a=2"


def test_parse_dict_authority_and_explicit_overrides():
    url = URL.parse_dict(
        {
            "scheme": "https",
            "authority": "alice:pw@example.com:443",
            "host": "OtherHost.COM.",
            "port": "8443",
            "path": "//x//y",
            "query": "?b=2&a=1",
            "fragment": "#frag",
        }
    )

    assert url.scheme == "https"
    assert url.userinfo == "alice:pw"
    assert url.host == "otherhost.com"
    assert url.port == 8443
    assert url.path == "/x/y"
    assert url.query == "a=1&b=2"
    assert url.fragment == "frag"


def test_parse_path_object_to_file_url(tmp_path: Path):
    file_path = tmp_path / "data.txt"
    url = URL.parse(file_path)

    assert url.scheme == "file"
    assert url.host == ""
    assert url.path.endswith("/data.txt")


def test_parse_local_path_string_coerces_to_file_url(tmp_path: Path):
    file_path = tmp_path / "nested" / "file.txt"
    url = URL.parse(file_path)

    assert url.scheme == "file"
    assert url.path.endswith("/nested/file.txt")

def test_parse_ipv6_netloc():
    url = URL.parse_str("http://user:pw@[2001:db8::1]:8080/a")

    assert url.userinfo == "user:pw"
    assert url.host == "2001:db8::1"
    assert url.port == 8080
    assert url.authority == "user:pw@[2001:db8::1]:8080"


def test_parse_invalid_ipv6_keeps_hostport_literal():
    url = URL.parse_str("http://[2001:db8::1]/a", normalize=False)

    assert url.host == "2001:db8::1"
    assert url.port is None


def test_user_and_password_properties_decode_values():
    url = URL(userinfo="alice%40corp:p%40ss")

    assert url.user == "alice@corp"
    assert url.password == "p@ss"


def test_user_and_password_edge_cases():
    assert URL(userinfo=None).user is None
    assert URL(userinfo=None).password is None
    assert URL(userinfo="alice").user == "alice"
    assert URL(userinfo="alice").password is None
    assert URL(userinfo=":pw").user == ""
    assert URL(userinfo=":pw").password == "pw"
    assert URL(userinfo="alice:").user == "alice"
    assert URL(userinfo="alice:").password == ""


def test_with_user_password_user_only():
    url = URL("https", host="example.com").with_user_password("alice@example.com")

    assert url.userinfo == "alice%40example.com"
    assert url.user == "alice@example.com"
    assert url.password is None


def test_with_user_password_user_and_password():
    url = URL("https", host="example.com").with_user_password("alice@example.com", "p@ss")

    assert url.userinfo == "alice%40example.com:p%40ss"
    assert url.user == "alice@example.com"
    assert url.password == "p@ss"


def test_with_user_password_none_none_removes_userinfo():
    url = URL("https", userinfo="alice:pw", host="example.com").with_user_password(None, None)

    assert url.userinfo is None


def test_query_dict_and_query_mapping():
    url = URL(query="a=1&a=2&b=")

    assert url.query_dict == {"a": ("1", "2"), "b": ("",)}
    assert url.query_mapping() == {"a": ["1", "2"], "b": [""]}


def test_query_items_respects_blank_values():
    url = URL(query="a=&b=2")

    assert url.query_items() == (("a", ""), ("b", "2"))


def test_to_string_encoded_and_raw_cache_paths():
    url = URL(
        scheme="https",
        userinfo="alice:pw",
        host="example.com",
        path="/a b",
        query="x=hello world",
        fragment="frag ment",
    )

    encoded_first = url.to_string(encode=True)
    encoded_second = url.to_string(encode=True)
    raw_first = url.to_string(encode=False)
    raw_second = url.to_string(encode=False)

    assert encoded_first == encoded_second
    assert raw_first == raw_second
    assert "%20" in encoded_first
    assert " " in raw_first


def test_to_string_wraps_ipv6_host():
    url = URL(scheme="http", host="2001:db8::1", port=8080, path="/")

    assert url.to_string() == "http://[2001:db8::1]:8080/"


def test_authority_empty_without_host():
    assert URL().authority == ""


def test_is_absolute_requires_scheme_and_host():
    assert URL(scheme="https", host="example.com").is_absolute is True
    assert URL(scheme="https", host="").is_absolute is False
    assert URL(scheme="", host="example.com").is_absolute is False


def test_join_relative_url():
    base = URL.parse_str("https://example.com/a/b/")
    joined = base.join("../c?q=1")

    assert joined.to_string() == "https://example.com/a/c?q=1"


def test_truediv_replaces_path():
    url = URL("https", host="example.com", path="/a") / "/b//c"

    assert url.path == "/b/c"


def test_truediv_rejects_non_string():
    with pytest.raises(ValueError):
        _ = URL("https", host="example.com") / 123


def test_with_host_normalizes_and_removes_default_port():
    url = URL(scheme="https", host="EXAMPLE.COM", port=443).with_host("Other.COM.")

    assert url.host == "other.com"
    assert url.port is None


def test_with_query_and_fragment_strip_prefixes():
    url = URL().with_query("?a=1").with_fragment("#frag")

    assert url.query == "a=1"
    assert url.fragment == "frag"


def test_add_query_item_replace_true():
    url = URL(query="a=2&a=1&b=3").add_query_item("a", "9", replace=True)

    assert url.query == "a=9&b=3"


def test_add_query_item_replace_false():
    url = URL(query="a=2").add_query_item("a", "1", replace=False)

    assert url.query == "a=1&a=2"


def test_add_query_item_none_key_raises():
    with pytest.raises(ValueError):
        URL().add_query_item(None, "x")  # type: ignore[arg-type]


def test_with_query_items_from_mapping_scalar_and_sequence():
    url = URL().with_query_items({"b": "2", "a": ("3", "1")})

    assert url.query == "a=1&a=3&b=2"


def test_with_query_items_from_iterable_no_sort():
    url = URL().with_query_items((("b", "2"), ("a", "1")), sort_keys=False)

    assert url.query == "b=2&a=1"


def test_anonymize_remove_mode(monkeypatch):
    def fake_anonymize(params, mode):
        assert mode == "remove"
        return {"safe": ("1",)}

    monkeypatch.setattr(mod, "anonymize_parameters", fake_anonymize)

    url = URL(
        scheme="https",
        userinfo="alice:pw",
        host="example.com",
        query="secret=1&safe=1",
    )

    out = url.anonymize(mode="remove")

    assert out.userinfo is None
    assert out.query == "safe=1"


def test_anonymize_redact_mode(monkeypatch):
    def fake_anonymize(params, mode):
        assert mode == "redact"
        return params

    monkeypatch.setattr(mod, "anonymize_parameters", fake_anonymize)

    url = URL(
        scheme="https",
        userinfo="alice:pw",
        host="example.com",
        query="a=1",
    )

    out = url.anonymize(mode="redact")

    assert out.userinfo == "<redacted>"
    assert out.query == "a=1"


class DemoResource(URLResource):
    def __init__(self, name: str):
        self.name = name

    @classmethod
    def url_scheme(cls) -> str:
        return "demo"

    def to_url(self, scheme: str | None = None) -> URL:
        return URL(scheme=scheme or self.url_scheme(), host="example.com", path=f"/{self.name}")

    @classmethod
    def from_parsed_url(cls, url: URL) -> "DemoResource":
        return cls(url.path.lstrip("/"))


class OtherDemoResource(URLResource):
    def __init__(self, value: str):
        self.value = value

    @classmethod
    def url_scheme(cls) -> str:
        return "demo"

    def to_url(self, scheme: str | None = None) -> URL:
        return URL(scheme=scheme or self.url_scheme(), path=f"/{self.value}")

    @classmethod
    def from_parsed_url(cls, url: URL) -> "OtherDemoResource":
        return cls(url.path.lstrip("/"))


def test_register_url_resource_success():
    register_url_resource(DemoResource)

    assert get_registered_url_resource("demo") is DemoResource
    assert registered_url_schemes() == ("demo", "file")


def test_register_url_resource_rejects_non_subclass():
    with pytest.raises(TypeError):
        register_url_resource(str)  # type: ignore[arg-type]


def test_register_url_resource_requires_non_empty_scheme():
    class BadResource(URLResource):
        @classmethod
        def url_scheme(cls) -> str:
            return ""

        def to_url(self, scheme: str | None = None) -> URL:
            return URL()

        @classmethod
        def from_parsed_url(cls, url: URL):
            return cls()

    with pytest.raises(ValueError, match="must return a non-empty scheme"):
        register_url_resource(BadResource)


def test_register_url_resource_overwrite():
    register_url_resource(DemoResource)
    register_url_resource(OtherDemoResource, overwrite=True)

    assert get_registered_url_resource("demo") is OtherDemoResource


def test_url_resource_class_decorator():
    @url_resource_class
    class DecoratedResource(URLResource):
        @classmethod
        def url_scheme(cls) -> str:
            return "decorated"

        def to_url(self, scheme: str | None = None) -> URL:
            return URL(scheme=scheme or "decorated", path="/x")

        @classmethod
        def from_parsed_url(cls, url: URL):
            return cls()

    assert get_registered_url_resource("decorated") is DecoratedResource


def test_url_resource_class_decorator_with_overwrite():
    register_url_resource(DemoResource)

    @url_resource_class(overwrite=True)
    class ReplacedDemoResource(URLResource):
        @classmethod
        def url_scheme(cls) -> str:
            return "demo"

        def to_url(self, scheme: str | None = None) -> URL:
            return URL(scheme=scheme or "demo", path="/y")

        @classmethod
        def from_parsed_url(cls, url: URL):
            return cls()

    assert get_registered_url_resource("demo") is ReplacedDemoResource


def test_urlresource_from_url_dispatches_from_base_class():
    register_url_resource(DemoResource)

    resource = URLResource.from_url("demo://example.com/thing")

    assert isinstance(resource, DemoResource)
    assert resource.name == "thing"


def test_urlresource_subclass_from_url_accepts_no_scheme():
    resource = DemoResource.from_url("/thing")

    assert isinstance(resource, DemoResource)
    assert resource.name == "thing"
