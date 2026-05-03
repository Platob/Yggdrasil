"""Tests for yggdrasil.io.url.URL."""

from __future__ import annotations

from pathlib import Path as _Path

import pytest

from yggdrasil.io.url import URL, resolve_memory_address


# ---------------------------------------------------------------------------
# Construction / parsing
# ---------------------------------------------------------------------------


class TestFromStr:
    def test_full_http_url(self):
        u = URL.from_str("https://user:pw@example.com:8443/data?x=1#frag")
        assert u.scheme == "https"
        assert u.host == "example.com"
        assert u.port == 8443
        assert u.path == "/data"
        assert u.query == "x=1"
        assert u.fragment == "frag"
        assert u.userinfo == "user:pw"

    def test_roundtrip_str(self):
        url = URL.from_str("https://user:pw@example.com:8443/data?x=1&pass=abc#frag")
        str_url = url.to_string()
        assert URL.from_str(str_url) == url

    def test_default_port_is_normalized_away(self):
        assert URL.from_str("http://example.com:80/").port is None
        assert URL.from_str("https://example.com:443/").port is None

    def test_host_lowercased(self):
        assert URL.from_str("HTTP://EXAMPLE.com/").host == "example.com"

    def test_query_normalized_alphabetically(self):
        u = URL.from_str("https://example.com/?b=2&a=1")
        # query is sorted lexicographically by key, then value
        assert u.query == "a=1&b=2"

    def test_trailing_dot_in_host_stripped(self):
        assert URL.from_str("https://example.com./").host == "example.com"

    def test_empty_path_becomes_root(self):
        assert URL.from_str("https://example.com").path == "/"

    def test_html_entity_ampersand_in_query_decoded(self):
        u = URL.from_str("https://example.com/api?foo=1&amp;update_id=202605032129")
        assert dict(u.query_items()) == {"foo": "1", "update_id": "202605032129"}
        assert "amp;" not in u.to_string()

    def test_html_entity_ampersand_in_fragment_decoded(self):
        u = URL.from_str("https://example.com/p#a=1&amp;b=2")
        assert u.fragment == "a=1&b=2"


class TestFromPathlib:
    def test_pathlib_path_becomes_file_url(self, tmp_path):
        u = URL.from_pathlib(tmp_path)
        assert u.scheme == "file"
        # tmp_path is absolute and POSIX-flavored on linux
        assert u.path.startswith("/")

    def test_string_path_via_from_(self, tmp_path):
        u = URL.from_(_Path(tmp_path) / "file.csv")
        assert u.scheme == "file"
        assert u.path.endswith("/file.csv")


class TestFromDict:
    def test_keyword_components(self):
        u = URL.from_dict(
            {"scheme": "https", "host": "example.com", "path": "/x", "port": 9000}
        )
        assert u.scheme == "https"
        assert u.host == "example.com"
        assert u.port == 9000
        assert u.path == "/x"

    def test_url_key_short_circuits_to_from_str(self):
        u = URL.from_dict({"url": "https://example.com/"})
        assert u.host == "example.com"

    def test_empty_mapping_yields_empty_url(self):
        u = URL.from_dict({})
        assert u == URL.empty()


class TestFromGeneric:
    def test_url_passthrough(self):
        u1 = URL.from_str("https://example.com/")
        assert URL.from_(u1) is u1

    def test_str_input(self):
        assert URL.from_("https://example.com/").host == "example.com"

    def test_none_with_default(self):
        sentinel = object()
        assert URL.from_(None, default=sentinel) is sentinel

    def test_none_without_default_raises(self):
        with pytest.raises(ValueError):
            URL.from_(None)


# ---------------------------------------------------------------------------
# Identity / immutability
# ---------------------------------------------------------------------------


class TestEquality:
    def test_two_urls_with_default_port_equal(self):
        a = URL.from_str("http://example.com:80/")
        b = URL.from_str("http://example.com/")
        assert a == b

    def test_inequality_on_path(self):
        assert URL.from_str("https://e.com/a") != URL.from_str("https://e.com/b")

    def test_hashable(self):
        u = URL.from_str("https://example.com/")
        assert hash(u) == hash(URL.from_str("https://example.com/"))


class TestEmptySingleton:
    def test_singleton_by_default(self):
        assert URL.empty() is URL.empty()

    def test_new_instance_flag_returns_distinct(self):
        assert URL.empty(new_instance=True) is not URL.empty()


# ---------------------------------------------------------------------------
# Path / name / stem / extensions
# ---------------------------------------------------------------------------


class TestPathProperties:
    def test_parts(self):
        assert URL.from_str("https://e.com/a/b/c").parts == ["a", "b", "c"]

    def test_name(self):
        assert URL.from_str("https://e.com/a/b/c").name == "c"

    def test_name_empty_for_root(self):
        assert URL.from_str("https://e.com/").name == ""

    def test_stem_strips_last_suffix_only(self):
        assert URL.from_str("/a/archive.csv.zst").stem == "archive.csv"

    def test_extensions_lowercased(self):
        assert URL.from_str("/a/archive.CSV.Zst").extensions == ["csv", "zst"]

    def test_extensions_empty_for_extensionless(self):
        assert URL.from_str("/a/README").extensions == []


class TestParentChain:
    def test_parent(self):
        u = URL.from_str("https://e.com/a/b/c")
        assert u.parent == URL.from_str("https://e.com/a/b")

    def test_root_is_its_own_parent(self):
        root = URL.from_str("https://e.com/")
        assert root.parent == root

    def test_parents_chain(self):
        u = URL.from_str("https://e.com/a/b/c")
        chain = u.parents
        assert chain == (
            URL.from_str("https://e.com/a/b"),
            URL.from_str("https://e.com/a"),
            URL.from_str("https://e.com/"),
        )


class TestJoinpath:
    def test_simple_join(self):
        u = URL.from_str("https://e.com/a")
        assert u.joinpath("b", "c") == URL.from_str("https://e.com/a/b/c")

    def test_truediv_operator(self):
        u = URL.from_str("https://e.com/a")
        assert (u / "b") == URL.from_str("https://e.com/a/b")

    def test_absolute_segment_resets_path(self):
        u = URL.from_str("https://e.com/a")
        assert u.joinpath("/x").path == "/x"

    def test_no_segments_returns_self(self):
        u = URL.from_str("https://e.com/a")
        assert u.joinpath() is u

    def test_invalid_segment_type_raises(self):
        u = URL.from_str("https://e.com/")
        with pytest.raises(TypeError):
            u.joinpath(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Query / userinfo helpers
# ---------------------------------------------------------------------------


class TestUserPassword:
    def test_user_only(self):
        u = URL.from_str("https://alice@example.com/")
        assert u.user == "alice"
        assert u.password is None

    def test_user_and_password(self):
        u = URL.from_str("https://alice:pw@example.com/")
        assert u.user == "alice"
        assert u.password == "pw"

    def test_with_user_password(self):
        base = URL.from_str("https://example.com/")
        u = base.with_user_password("alice", "pw")
        assert u.user == "alice"
        assert u.password == "pw"

    def test_with_user_password_clears_when_none(self):
        base = URL.from_str("https://alice:pw@example.com/")
        u = base.with_user_password(None, None)
        assert u.userinfo is None


class TestQueryHelpers:
    def test_query_dict(self):
        u = URL.from_str("https://e.com/?a=1&b=2&a=3")
        d = u.query_dict
        assert d == {"a": ("1", "3"), "b": ("2",)}

    def test_query_items_tuple(self):
        u = URL.from_str("https://e.com/?a=1&b=2")
        assert u.query_items() == (("a", "1"), ("b", "2"))

    def test_add_query_item_replaces_by_default(self):
        u = URL.from_str("https://e.com/?a=1")
        new = u.add_query_item("a", "2")
        assert new.query == "a=2"

    def test_add_query_item_can_keep_existing(self):
        u = URL.from_str("https://e.com/?a=1")
        new = u.add_query_item("a", "2", replace=False)
        # parse_qsl + sort puts both pairs in order
        items = sorted(new.query_items())
        assert items == [("a", "1"), ("a", "2")]

    def test_with_query_items_mapping(self):
        u = URL.from_str("https://e.com/")
        new = u.with_query_items({"a": "1", "b": ["2", "3"]})
        items = dict(new.query_items())
        assert items["a"] == "1"
        assert "b" in new.query


class TestWithMethods:
    def test_with_scheme(self):
        u = URL.from_str("https://e.com/")
        assert u.with_scheme("http").scheme == "http"

    def test_with_host(self):
        u = URL.from_str("https://e.com/")
        assert u.with_host("other.com").host == "other.com"

    def test_with_path(self):
        u = URL.from_str("https://e.com/a")
        assert u.with_path("/b").path == "/b"

    def test_with_query(self):
        u = URL.from_str("https://e.com/?a=1")
        assert u.with_query(None).query is None

    def test_with_fragment(self):
        u = URL.from_str("https://e.com/#x")
        assert u.with_fragment(None).fragment is None


# ---------------------------------------------------------------------------
# Pattern / relative-to
# ---------------------------------------------------------------------------


class TestMatchPattern:
    def test_basename_match(self):
        u = URL.from_str("https://e.com/data/file.csv")
        assert u.match_pattern("*.csv")

    def test_full_url_match(self):
        u = URL.from_str("https://e.com/data/x")
        assert u.match_pattern("*e.com*")

    def test_no_match(self):
        assert not URL.from_str("https://e.com/file.csv").match_pattern("*.json")


class TestMatchesPatterns:
    def test_returns_false_when_none(self):
        u = URL.from_str("https://e.com/file.csv")
        assert u.matches_patterns(None) is False

    def test_returns_false_when_empty(self):
        assert not URL.from_str("https://e.com/file.csv").matches_patterns([])

    def test_first_match_short_circuits(self):
        u = URL.from_str("https://e.com/file.csv")
        assert u.matches_patterns(["*.csv", "*.json"])


class TestIsRelativeTo:
    def test_strict_prefix(self):
        u = URL.from_str("https://e.com/a/b/c")
        assert u.is_relative_to(URL.from_str("https://e.com/a"))

    def test_equal_paths(self):
        u = URL.from_str("https://e.com/a")
        assert u.is_relative_to(URL.from_str("https://e.com/a"))

    def test_sibling_not_relative(self):
        u = URL.from_str("https://e.com/a")
        assert not u.is_relative_to(URL.from_str("https://e.com/b"))

    def test_mismatched_host_rejected(self):
        u = URL.from_str("https://e.com/a")
        assert not u.is_relative_to(URL.from_str("https://other.com/a"))


class TestRelativeTo:
    def test_basic(self):
        u = URL.from_str("https://e.com/a/b/c")
        rel = u.relative_to("https://e.com/a")
        assert rel.path == "/b/c"

    def test_exact_match_returns_root(self):
        u = URL.from_str("https://e.com/a")
        assert u.relative_to("https://e.com/a").path == "/"

    def test_raises_when_not_relative(self):
        u = URL.from_str("https://e.com/a")
        with pytest.raises(ValueError):
            u.relative_to("https://e.com/b")


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    def test_is_absolute(self):
        assert URL.from_str("https://e.com/").is_absolute is True
        assert URL.from_str("/just/a/path").is_absolute is False

    def test_is_http(self):
        assert URL.from_str("https://e.com/").is_http is True
        assert URL.from_str("file:///x").is_http is False

    def test_authority(self):
        u = URL.from_str("https://alice@example.com:9000/")
        assert "example.com" in u.authority
        assert "9000" in u.authority


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------


class TestAnonymize:
    def test_removes_userinfo(self):
        u = URL.from_str("https://alice:pw@example.com/")
        assert u.anonymize().userinfo is None

    def test_redacts_userinfo(self):
        u = URL.from_str("https://alice:pw@example.com/")
        assert u.anonymize("redact").userinfo == "<redacted>"

    def test_strips_sensitive_query(self):
        u = URL.from_str("https://e.com/?token=abc&keep=1")
        anonymized = u.anonymize()
        assert "token" not in (anonymized.query or "")
        assert "keep" in (anonymized.query or "")


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


class TestToString:
    def test_round_trip(self):
        s = "https://example.com/data/file.csv"
        assert URL.from_str(s).to_string() == s

    def test_str_is_to_string(self):
        u = URL.from_str("https://e.com/")
        assert str(u) == u.to_string()


class TestToPathlib:
    def test_strict_requires_file_scheme(self):
        u = URL.from_str("https://e.com/data")
        with pytest.raises(ValueError):
            u.to_pathlib()

    def test_strict_false_drops_url_components(self):
        u = URL.from_str("https://e.com/data")
        p = u.to_pathlib(strict=False)
        assert isinstance(p, _Path)

    def test_file_scheme_round_trip(self, tmp_path):
        url = URL.from_pathlib(tmp_path)
        assert url.to_pathlib() == tmp_path


class TestFspathProtocol:
    def test_file_url(self, tmp_path):
        url = URL.from_pathlib(tmp_path / "x.csv")
        # os.fspath should give the bare local path
        import os
        assert os.fspath(url).endswith("/x.csv")

    def test_http_url_returns_path_only(self):
        u = URL.from_str("https://e.com/data/x")
        import os
        assert os.fspath(u) == "/data/x"


# ---------------------------------------------------------------------------
# Memory addressing
# ---------------------------------------------------------------------------


class TestMemoryAddress:
    def test_round_trip(self):
        obj = ["live"]
        url = URL.from_memory_address(obj)
        assert url.is_memory_address
        assert url.memory_address == id(obj)
        assert url.resolve_memory_address() is obj

    def test_resolve_helper(self):
        obj = {"x": 1}
        addr = id(obj)
        assert resolve_memory_address(addr) is obj

    def test_memory_address_on_non_mem_url_raises(self):
        u = URL.from_str("https://e.com/")
        assert u.is_memory_address is False
        with pytest.raises(ValueError):
            _ = u.memory_address
