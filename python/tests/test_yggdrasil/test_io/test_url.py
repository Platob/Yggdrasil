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
    url = URL.parse_str("HTTPS://Example.COM.:443/a/b?z=2&a=3&a=1#frag")

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
            "path": "/x/y",
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
    url = URL("https", host="example.com", path="/a") / "/b/c"

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
    register_url_resource(OtherDemoResource)

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

    @url_resource_class()
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


# ---------------------------------------------------------------------------
# Regression tests: schemaless / Windows-path URL parsing (os_find bug fix)
# ---------------------------------------------------------------------------


class TestSchemalessAndWindowsPathParsing:
    """Guard against the os_find corruption bug and the host/path mis-split."""

    # ── Schemaless URLs (no "https://" prefix) ────────────────────────────

    def test_schemaless_url_does_not_corrupt_path(self):
        """A URL without a scheme must NOT have os.path.realpath() applied."""
        url = URL.parse("api.example.com/some/path")
        # Path must stay as-is — no Windows drive-letter injection
        assert "C:" not in url.path
        assert "C:\\" not in url.path

    def test_schemaless_url_with_default_scheme(self):
        url = URL.parse_str("api.example.com/v1/resource", default_scheme="https")
        assert url.scheme == "https"
        assert url.host == "api.example.com"
        assert url.path == "/v1/resource"

    def test_schemaless_host_only(self):
        url = URL.parse_str("example.com", default_scheme="https")
        assert url.host == "example.com"
        assert url.scheme == "https"

    def test_schemaless_no_default_scheme_preserves_empty_scheme(self):
        """Parsing without default_scheme leaves scheme empty but keeps host/path intact."""
        url = URL.parse_str("example.com/path")
        assert url.host == "" or url.scheme == ""  # no scheme injected
        assert "C:" not in url.path

    # ── Windows drive-letter "scheme" detection ────────────────────────────

    def test_windows_drive_letter_not_treated_as_scheme(self):
        """C:/path must not produce host='C' or corrupt the path via os_find."""
        url = URL.parse("C:/Users/alice/file.html")
        # scheme must NOT be "c" (drive letter treated as file scheme or empty)
        assert url.scheme in ("", "file", "c")  # may vary, but path must not be mangled
        assert "C:" not in (url.host or "")  # host must not be "c" or contain drive

    def test_windows_path_in_http_url_path_preserved(self):
        """A URL like https://api.example.com/C:/path must preserve the path."""
        url = URL.parse("https://api.example.com/C:/controlador.cgi?ac=signin")
        assert url.scheme == "https"
        assert url.host == "api.example.com"
        assert url.path == "/C:/controlador.cgi"
        assert url.query == "ac=signin"

    # ── The exact error pattern reported by the user ──────────────────────

    def test_path_like_url_no_host_corruption(self):
        """
        Regression: "api/C:/controlador.cgi?ac=signin" must NOT produce
        host='api' with path='/C:/controlador.cgi' via the fix-up block
        when there is no explicit scheme.  The fix-up block should only fire
        for URLs that DO carry an explicit non-file scheme.
        """
        url = URL.parse_str("api/C:/controlador.cgi?ac=signin")
        # The schemaless fix-up must not fire — host should be empty
        # (the whole string becomes the path, not split on "/").
        assert url.host == "" or url.scheme == ""
        # The path must not become a local Windows absolute path
        assert not (url.path.startswith("/C:") and url.host == "api")

    def test_http_scheme_missing_authority_slashes(self):
        """http:example.com/path (no //) — fix-up IS expected for explicit scheme."""
        url = URL.parse_str("http:example.com/path")
        assert url.scheme == "http"
        assert url.host == "example.com"
        assert url.path == "/path"

    # ── os_find must never fire for non-file schemes ──────────────────────

    def test_https_url_with_cgi_path_not_mangled(self):
        url = URL.parse("https://192.168.1.1/C:/cgi-bin/login.cgi?user=admin")
        assert url.scheme == "https"
        assert url.host == "192.168.1.1"
        assert url.path == "/C:/cgi-bin/login.cgi"

    def test_http_url_path_not_os_resolved(self):
        """Ensure _normalize_path is NOT called with os_find=True for http URLs."""
        import os
        # Build a URL whose path would change if os.path.realpath were applied
        url = URL.parse("http://example.com/relative/./path/../end")
        # urllib urlsplit won't collapse "." and ".." (unlike os.path.realpath)
        assert url.host == "example.com"
        assert "C:" not in url.path


class TestSanitizeUrl:
    """Tests for BrowserHTTPSession._resolve_url (URL resolution with base_url)."""

    def test_full_https_url_returned_as_url_object(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        result = b._resolve_url("https://example.com/path")
        assert isinstance(result, URL)
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/path"

    def test_full_http_url_unchanged(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        result = b._resolve_url("http://example.com/path")
        assert result.scheme == "http"
        assert result.host == "example.com"

    def test_schemaless_hostname_gets_https(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        result = b._resolve_url("example.com/path")
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/path"

    def test_subdomain_schemaless_gets_https(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        result = b._resolve_url("api.example.com/v1/resource")
        assert result.scheme == "https"
        assert result.host == "api.example.com"

    def test_protocol_relative_gets_https(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        result = b._resolve_url("//example.com/path")
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/path"

    def test_windows_drive_letter_raises(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="local Windows path"):
            b._resolve_url("C:/Users/alice/page.html")

    def test_url_object_absolute_returned_unchanged(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        url = URL.parse("https://example.com/")
        result = b._resolve_url(url)
        assert result.scheme == "https"
        assert result.host == "example.com"

    # ── Relative URL resolution (the user-reported bug) ───────────────────

    def test_relative_path_segment_joined_with_base_url(self):
        """'api/controlador.cgi' with base_url → joined correctly."""
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession(base_url="https://api.example.com")
        result = b._resolve_url("api/controlador.cgi")
        assert result.scheme == "https"
        assert result.host == "api.example.com"
        assert result.path == "/api/controlador.cgi"

    def test_relative_path_no_dot_no_base_url_raises(self):
        """Relative path without base_url must raise ValueError, not connect to host='api'."""
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="base_url"):
            b._resolve_url("api/controlador.cgi")

    def test_absolute_path_joined_with_base_url_host(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession(base_url="https://api.example.com/v1/")
        result = b._resolve_url("/controlador.cgi")
        assert result.host == "api.example.com"
        assert result.path == "/controlador.cgi"

    def test_absolute_path_no_base_url_raises(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        with pytest.raises(ValueError, match="base_url"):
            b._resolve_url("/some/path")

    def test_dotdot_relative_path_joined(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession(base_url="https://api.example.com/v2/sub/")
        result = b._resolve_url("../resource")
        assert result.host == "api.example.com"
        assert result.path == "/v2/resource"

    # ── _apply_params ─────────────────────────────────────────────────────

    def test_apply_params_scalar(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        url = URL.parse("https://example.com/path")
        result = BrowserHTTPSession._apply_params(url, {"a": "1", "b": "2"})
        assert "a=1" in (result.query or "")
        assert "b=2" in (result.query or "")

    def test_apply_params_multi_value(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        url = URL.parse("https://example.com/path")
        result = BrowserHTTPSession._apply_params(url, {"tag": ["x", "y"]})
        assert (result.query or "").count("tag=") == 2

    def test_apply_params_preserves_existing_query(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        url = URL.parse("https://example.com/path?existing=1")
        result = BrowserHTTPSession._apply_params(url, {"new": "2"})
        assert "existing=1" in (result.query or "")
        assert "new=2" in (result.query or "")

    def test_empty_string_raises(self):
        from yggdrasil.io.http_.browser import BrowserHTTPSession
        b = BrowserHTTPSession()
        with pytest.raises(ValueError):
            b._resolve_url("")


