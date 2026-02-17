# test_url.py
import pytest

from yggdrasil.io.url import URL


# ----------------------------
# Defaults / basic invariants
# ----------------------------

def test_defaults_are_exact_and_no_port_leaks():
    u = URL()
    assert u.scheme == ""
    assert u.userinfo == ""
    assert u.host == ""
    assert u.port == 0  # 0 means "absent"
    assert u.path == ""
    assert u.query == ""
    assert u.fragment == ""
    assert u.is_absolute is False
    assert u.to_string() == ""  # no scheme/netloc/path => empty


# ----------------------------
# Parse + normalize + to_string
# ----------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://example.com", "https://example.com"),
        ("https://EXAMPLE.com", "https://example.com"),              # host lowercased
        ("HTTP://Example.Com", "http://example.com"),                # scheme+host lowercased
        ("https://example.com:443", "https://example.com"),          # default port removed
        ("http://example.com:80", "http://example.com"),             # default port removed
        ("wss://example.com:443", "wss://example.com"),              # default port removed
        ("ws://example.com:80", "ws://example.com"),                 # default port removed
        ("https://example.com./", "https://example.com/"),           # trailing dot stripped
        ("https://example.com:444/a", "https://example.com:444/a"),  # non-default port kept
    ],
)
def test_parse_normalize_and_to_string(raw, expected):
    u = URL.parse(raw, normalize=True)
    assert u.to_string() == expected


@pytest.mark.parametrize(
    "raw, expected_path",
    [
        ("https://example.com/a//b///c", "/a/b/c"),  # collapse repeated slashes
        ("https://example.com/", "/"),
        ("https://example.com", ""),                # urlsplit empty path
    ],
)
def test_path_normalization(raw, expected_path):
    u = URL.parse(raw, normalize=True)
    assert u.path == expected_path


def test_parse_normalize_sets_port_0_when_absent_or_default():
    u1 = URL.parse("https://example.com", normalize=True)
    assert u1.port == 0

    u2 = URL.parse("https://example.com:443", normalize=True)
    assert u2.port == 0

    u3 = URL.parse("https://example.com:444", normalize=True)
    assert u3.port == 444


def test_is_absolute_logic():
    assert URL.parse("https://example.com/a").is_absolute is True
    assert URL.parse("/relative/path").is_absolute is False
    assert URL.parse("mailto:test@example.com").is_absolute is False  # no host


# ----------------------------
# userinfo / ipv6 / authority
# ----------------------------

def test_parse_userinfo_host_port_and_to_string_encodes_userinfo():
    u = URL.parse("https://user:pa ss@example.com:444/a", decode=True, normalize=True)
    assert u.userinfo == "user:pa ss"
    assert u.host == "example.com"
    assert u.port == 444
    assert u.to_string() == "https://user:pa%20ss@example.com:444/a"


def test_ipv6_parsing_and_output_bracketing():
    u = URL.parse("https://[2001:db8::1]:8443/a", normalize=True)
    assert u.host == "2001:db8::1"
    assert u.port == 8443
    assert u.to_string() == "https://[2001:db8::1]:8443/a"

    u2 = URL.parse("https://[2001:db8::1]/a", normalize=True)
    assert u2.port == 0
    assert u2.to_string() == "https://[2001:db8::1]/a"


def test_authority_property():
    u = URL.parse("https://user:pw@example.com:444/a", decode=True, normalize=True)
    # authority uses raw userinfo (not encoded)
    assert u.authority == "user:pw@example.com:444"

    u2 = URL.parse("https://example.com/a", normalize=True)
    assert u2.authority == "example.com"  # port absent => not shown

    ipv6 = URL.parse("https://[2001:db8::1]:444/a", normalize=True)
    assert ipv6.authority == "[2001:db8::1]:444"


# ----------------------------
# decode flag behavior
# ----------------------------

def test_decode_false_keeps_percent_sequences_and_roundtrips():
    raw = "https://example.com/a%20b?x=a%2Fb&y=1#z%20t"
    u = URL.parse(raw, decode=False, normalize=True)
    assert u.path == "/a%20b"
    assert u.query == "x=a%2Fb&y=1"
    assert u.fragment == "z%20t"
    assert u.to_string() == raw


def test_decode_true_decodes_components_and_reencodes():
    raw = "https://example.com/a%20b?x=a%2Fb&y=1#z%20t"
    u = URL.parse(raw, decode=True, normalize=True)
    assert u.path == "/a b"
    assert u.query == "x=a/b&y=1"
    assert u.fragment == "z t"
    # '/' and '&' and '=' should remain in query; spaces re-encoded
    assert u.to_string() == "https://example.com/a%20b?x=a/b&y=1#z%20t"


# ----------------------------
# join
# ----------------------------

def test_join_with_str_and_with_url():
    base = URL.parse("https://example.com/a/b/c", normalize=True)

    out = base.join("../d?x=1&y=2")
    assert out.to_string() == "https://example.com/a/d?x=1&y=2"

    ref = URL.parse("/zzz", normalize=True)
    out2 = base.join(ref)
    assert out2.to_string() == "https://example.com/zzz"


# ----------------------------
# immutable edits: with_*
# ----------------------------

def test_with_scheme_updates_default_port_rules():
    u = URL.parse("http://example.com:80/a", normalize=True)
    assert u.port == 0
    assert u.to_string() == "http://example.com/a"

    u2 = u.with_scheme("https")
    assert u2.scheme == "https"
    assert u2.port == 0
    assert u2.to_string() == "https://example.com/a"

    u3 = URL.parse("http://example.com:443/a", normalize=True)
    assert u3.port == 443  # not default for http

    u4 = u3.with_scheme("https")
    assert u4.port == 0  # becomes default for https, removed
    assert u4.to_string() == "https://example.com/a"


def test_with_host_normalizes_and_removes_default_port_if_applicable():
    u = URL.parse("https://example.com:443/a", normalize=False)
    assert u.port == 443  # normalize=False keeps it

    u2 = u.with_host("EXAMPLE.com.")  # lower + strip dot; remove default https port
    assert u2.host == "example.com"
    assert u2.port == 0
    assert u2.to_string() == "https://example.com/a"


def test_with_path_normalizes_slashes():
    u = URL.parse("https://example.com", normalize=True)
    u2 = u.with_path("/a//b///c")
    assert u2.path == "/a/b/c"
    assert u2.to_string() == "https://example.com/a/b/c"


def test_with_query_and_fragment_strip_prefixes_and_do_not_escape_separators():
    u = URL.parse("https://example.com/a", normalize=True)
    u2 = u.with_query("?x=1&y=2").with_fragment("#frag")
    assert u2.query == "x=1&y=2"
    assert u2.fragment == "frag"
    assert u2.to_string() == "https://example.com/a?x=1&y=2#frag"


# ----------------------------
# query helpers
# ----------------------------

def test_query_items_parsing_and_keep_blank_values():
    u = URL.parse("https://example.com/a?x=1&y=&z", normalize=True)
    assert u.query_items(keep_blank_values=True) == (("x", "1"), ("y", ""), ("z", ""))
    assert u.query_items(keep_blank_values=False) == (("x", "1"),)


def test_with_query_items_from_mapping_and_tuple_and_string_output():
    u = URL.parse("https://example.com/a", normalize=True)

    u2 = u.with_query_items({"b": "2", "a": "1"})
    assert set(u2.query_items()) == {("a", "1"), ("b", "2")}
    assert u2.to_string() in ("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2")

    u3 = u.with_query_items((("a", "1"), ("b", "2")))
    assert u3.query_items() == (("a", "1"), ("b", "2"))
    assert u3.to_string() == "https://example.com/a?a=1&b=2"


def test_sort_query_params_canonical():
    u = URL.parse("https://example.com/a?b=2&a=1&a=0", normalize=True, sort_query=True)
    assert u.query == "a=0&a=1&b=2"
    assert u.to_string() == "https://example.com/a?a=0&a=1&b=2"


def test_sort_query_params_blank_and_duplicates():
    u = URL.parse("https://example.com/?z&b=&a=2&a=1", normalize=True, sort_query=True)
    assert u.query == "a=1&a=2&b=&z="
    assert u.to_string() == "https://example.com/?a=1&a=2&b=&z="


def test_sort_query_params_no_query_is_noop():
    u = URL.parse("https://example.com/a", normalize=True, sort_query=True)
    assert u.query == ""
    assert u.to_string() == "https://example.com/a"


# ----------------------------
# xxh3 hashing
# ----------------------------

def test_xxh3_digest_is_stable_under_query_order_when_included():
    u1 = URL.parse("https://example.com/a?b=2&a=1", normalize=True, sort_query=False)
    u2 = URL.parse("https://example.com/a?a=1&b=2", normalize=True, sort_query=False)
    # digest canonicalizes+sorts query internally
    assert u1.xxh3_64() == u2.xxh3_64()


def test_xxh3_digest_exclude_query_true_ignores_query():
    u1 = URL.parse("https://example.com/a?a=1", normalize=True)
    u2 = URL.parse("https://example.com/a?a=999", normalize=True)
    assert u1.xxh3_64(exclude_query=True) == u2.xxh3_64(exclude_query=True)


def test_xxh3_digest_exclude_query_list_filters_specific_params():
    u1 = URL.parse("https://example.com/a?a=1&ts=100&b=2", normalize=True)
    u2 = URL.parse("https://example.com/a?a=1&ts=999&b=2", normalize=True)
    assert u1.xxh3_64(exclude_query=["ts"]) == u2.xxh3_64(exclude_query=["ts"])


def test_xxh3_digest_default_excludes_scheme_fragment_and_userinfo():
    # defaults: exclude_userinfo=True, exclude_scheme=True, exclude_fragment=True
    u1 = URL.parse("http://user:pw@example.com/a?b=2#frag", normalize=True)
    u2 = URL.parse("https://example.com/a?b=2#other", normalize=True)
    assert u1.xxh3_64() == u2.xxh3_64()


def test_xxh3_digest_can_include_scheme_when_requested():
    u1 = URL.parse("http://example.com/a", normalize=True)
    u2 = URL.parse("https://example.com/a", normalize=True)
    assert u1.xxh3_64(exclude_scheme=False) != u2.xxh3_64(exclude_scheme=False)


def test_xxh3_digest_exclude_host_collapses_netloc():
    u1 = URL.parse("https://example.com/a", normalize=True)
    u2 = URL.parse("https://other.com/a", normalize=True)
    assert u1.xxh3_64(exclude_host=True) == u2.xxh3_64(exclude_host=True)


def test_xxh3_digest_exclude_port_ignores_port_differences():
    u1 = URL.parse("https://example.com:444/a", normalize=True)
    u2 = URL.parse("https://example.com:555/a", normalize=True)
    assert u1.xxh3_64(exclude_scheme=False, exclude_port=True) == u2.xxh3_64(
        exclude_scheme=False, exclude_port=True
    )
