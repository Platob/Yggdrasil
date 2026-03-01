# test_url_nullable.py
# Pytest unit tests for the nullable URL dataclass behavior.

from yggdrasil.io.url import URL  # adjust import path if your module name differs


def test_default_ctor_all_none_is_empty_string_roundtrip():
    u = URL()
    assert u.scheme is None
    assert u.host is None
    assert u.userinfo is None
    assert u.port is None
    assert u.path is None
    assert u.query is None
    assert u.fragment is None

    assert u.to_string() == ""
    assert str(u) == ""
    assert u.authority == ""
    assert u.is_absolute is False
    assert u.query_dict == {}


def test_to_string_none_fields_dont_crash_and_render_empty():
    u = URL(
        scheme=None,
        userinfo=None,
        host=None,
        port=None,
        path=None,
        query=None,
        fragment=None,
    )
    assert u.to_string() == ""


def test_parse_minimal_url_sets_nones_for_missing_parts():
    u = URL.parse_str("https://example.com")
    assert u.scheme == "https"
    assert u.host == "example.com"
    assert u.userinfo is None
    assert u.path is None  # urlsplit gives empty path; we store None
    assert u.query is None
    assert u.fragment is None
    assert u.port is None  # default port removed -> absent (None)
    assert u.is_absolute is True
    assert u.to_string() == "https://example.com"


def test_parse_keeps_non_default_port():
    u = URL.parse_str("https://example.com:8443")
    assert u.scheme == "https"
    assert u.host == "example.com"
    assert u.port == 8443
    assert u.to_string() == "https://example.com:8443"


def test_parse_ipv6_bracketing_roundtrip():
    u = URL.parse_str("http://[::1]:8080/a/b")
    assert u.scheme == "http"
    assert u.host == "::1"
    assert u.port == 8080
    assert u.path == "/a/b"
    assert u.to_string() == "http://[::1]:8080/a/b"
    assert u.authority == "[::1]:8080"


def test_parse_userinfo_roundtrip():
    u = URL.parse_str("https://user:pass@example.com/path")
    assert u.userinfo == "user:pass"
    assert u.host == "example.com"
    assert u.to_string() == "https://user:pass@example.com/path"
    assert u.authority == "user:pass@example.com"


def test_parse_normalization_host_lower_and_strip_trailing_dot():
    u = URL.parse_str("HTTPS://EXAMPLE.COM./x")
    assert u.scheme == "https"
    assert u.host == "example.com"
    assert u.to_string() == "https://example.com/x"


def test_parse_normalization_collapse_double_slashes():
    u = URL.parse_str("https://example.com//a///b")
    assert u.path == "/a/b"
    assert u.to_string() == "https://example.com/a/b"


def test_parse_normalization_sorts_query_pairs():
    u = URL.parse_str("https://example.com/?b=2&a=2&a=1")
    # sorted by (k, v) => a=1&a=2&b=2
    assert u.query == "a=1&a=2&b=2"
    assert u.to_string() == "https://example.com/?a=1&a=2&b=2"


def test_query_dict_groups_values_and_keeps_order_per_key():
    u = URL.parse_str("https://example.com/?k=2&k=1&x=&y")
    # parse(normalize=True) sorts query by (k,v), so k becomes 1 then 2
    assert u.query_dict == {"k": ("1", "2"), "x": ("",), "y": ("",)}


def test_parse_dict_empty_mapping_returns_defaults():
    u = URL.parse_dict({})
    assert u.to_string() == ""


def test_parse_dict_url_shortcut():
    u = URL.parse_dict({"url": "https://example.com/a?b=1"})
    assert u.scheme == "https"
    assert u.host == "example.com"
    assert u.path == "/a"
    assert u.query == "b=1"
    assert u.to_string() == "https://example.com/a?b=1"


def test_parse_dict_split_fields_with_netloc():
    u = URL.parse_dict(
        {
            "scheme": "https",
            "netloc": "User:Pass@Example.com.:443",
            "path": "/p",
            "query": "b=2&a=1",
            "fragment": "frag",
        },
        normalize=True,
        decode=False,
    )
    assert u.scheme == "https"
    assert u.userinfo == "User:Pass"
    assert u.host == "example.com"
    assert u.port is None  # 443 removed
    assert u.query == "a=1&b=2"
    assert u.fragment == "frag"
    assert u.to_string() == "https://User:Pass@example.com/p?a=1&b=2#frag"


def test_parse_dict_explicit_overrides_win_over_netloc():
    u = URL.parse_dict(
        {
            "scheme": "https",
            "netloc": "u1@wrong.com:123",
            "userinfo": "u2",
            "host": "Right.com",
            "port": 8443,
            "path": "/x",
        },
        normalize=True,
    )
    assert u.userinfo == "u2"
    assert u.host == "right.com"
    assert u.port == 8443
    assert u.to_string() == "https://u2@right.com:8443/x"


def test_parse_dict_port_none_or_zero_means_absent():
    u1 = URL.parse_dict({"scheme": "https", "host": "example.com", "port": None})
    u2 = URL.parse_dict({"scheme": "https", "host": "example.com", "port": 0})
    assert u1.port is None
    assert u2.port is None
    assert u1.to_string() == "https://example.com"
    assert u2.to_string() == "https://example.com"


def test_with_scheme_accepts_none_and_reapplies_default_port_removal():
    u = URL.parse_str("http://example.com:80/x")
    assert u.port is None
    u2 = u.with_scheme("https")
    assert u2.scheme == "https"
    assert u2.port is None
    assert u2.to_string() == "https://example.com/x"

    u3 = u2.with_scheme(None)
    assert u3.scheme is None
    assert u3.to_string() == "//example.com/x"  # urlunsplit behavior for empty scheme


def test_with_host_normalizes_and_recalculates_default_port():
    u = URL.parse_str("https://example.com:8443/x")
    u2 = u.with_host("EXAMPLE.com.")
    assert u2.host == "example.com"
    assert u2.port == 8443  # still non-default
    assert u2.to_string() == "https://example.com:8443/x"

    u3 = URL.parse_str("https://example.com:443/x").with_host("Example.com.")
    assert u3.port is None  # default removed
    assert u3.to_string() == "https://example.com/x"


def test_with_userinfo_none_clears():
    u = URL.parse_str("https://user@example.com/x")
    assert u.userinfo == "user"
    u2 = u.with_userinfo(None)
    assert u2.userinfo is None
    assert u2.to_string() == "https://example.com/x"


def test_with_path_none_clears():
    u = URL.parse_str("https://example.com/x")
    u2 = u.with_path(None)
    assert u2.path is None
    assert u2.to_string() == "https://example.com"


def test_with_query_and_fragment_strip_prefix_and_allow_none():
    u = URL.parse_str("https://example.com/x")
    u2 = u.with_query("?a=1").with_fragment("#f")
    assert u2.query == "a=1"
    assert u2.fragment == "f"
    assert u2.to_string() == "https://example.com/x?a=1#f"

    u3 = u2.with_query(None).with_fragment(None)
    assert u3.query is None
    assert u3.fragment is None
    assert u3.to_string() == "https://example.com/x"


def test_join_works_with_nullable_parts():
    base = URL.parse_str("https://example.com/base/")
    out = base.join("child")
    assert out.to_string() == "https://example.com/base/child"

    # join should still work if base has path None (acts like empty)
    base2 = URL.parse_str("https://example.com").with_path(None)
    out2 = base2.join("/x")
    assert out2.to_string() == "https://example.com/x"


def test_truediv_calls_with_path_and_normalizes_slashes():
    u = URL.parse_str("https://example.com/base")
    out = u / "//x///y"
    assert out.path == "/x/y"
    assert out.to_string() == "https://example.com/x/y"


def test_anonymize_userinfo_and_query():
    u = URL.parse_str("https://user:pass@example.com/x?token=abc&ok=1")
    a = u.anonymize(mode="redact")
    # userinfo always redacted if present
    assert a.userinfo == "<redacted>"
    # query values for sensitive keys depend on your anonymize_parameters implementation.
    # We at least assert it stays parseable and retains keys.
    assert "token=" in (a.query or "")
    assert "ok=" in (a.query or "")
