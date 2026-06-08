"""Tests for the URL class from yggdrasil.url."""
from __future__ import annotations

import os
import pickle
import pathlib

import pytest

from yggdrasil.url import URL, resolve_memory_address


class TestFromStr:

    def test_full_url_with_all_components(self):
        url = URL.from_str("http://alice:s3cret@example.com:8080/api/v1?b=2&a=1#top")
        assert url.scheme == "http"
        assert url.userinfo == "alice:s3cret"
        assert url.host == "example.com"
        assert url.port == 8080
        assert url.path == "/api/v1"
        assert url.query == "a=1&b=2"
        assert url.fragment == "top"

    def test_round_trip_str_url_str(self):
        raw = "https://example.com:9090/path/to/resource?key=val#frag"
        url = URL.from_str(raw)
        assert URL.from_str(str(url)) == url

    def test_default_port_http_normalized(self):
        url = URL.from_str("http://example.com:80/path")
        assert url.port is None

    def test_default_port_https_normalized(self):
        url = URL.from_str("https://example.com:443/path")
        assert url.port is None

    def test_host_lowercased(self):
        url = URL.from_str("https://EXAMPLE.COM/path")
        assert url.host == "example.com"

    def test_query_sorted_alphabetically(self):
        url = URL.from_str("https://example.com/?z=1&a=2&m=3")
        assert url.query == "a=2&m=3&z=1"

    def test_trailing_dot_in_host_stripped(self):
        url = URL.from_str("https://example.com./path")
        assert url.host == "example.com"

    def test_empty_path_becomes_slash(self):
        url = URL.from_str("https://example.com")
        assert url.path == "/"

    def test_html_entity_amp_decoded_in_query(self):
        url = URL.from_str("https://example.com/path?a=1&amp;b=2")
        assert url.query == "a=1&b=2"

    def test_html_entity_amp_decoded_in_fragment(self):
        url = URL.from_str("https://example.com/path#a&amp;b")
        assert url.fragment == "a&b"

    def test_windows_drive_letter_path(self):
        url = URL.from_str("C:\\Users\\x\\file.txt")
        assert url.scheme == "file"
        assert "/C:/Users/x/file.txt" in url.path


class TestTrailingSlashPreserved:

    def test_https_dir_path(self):
        url = URL.from_str("https://example.com/data/")
        assert url.path.endswith("/")

    def test_nested_dir(self):
        url = URL.from_str("https://example.com/a/b/c/")
        assert url.path == "/a/b/c/"

    def test_root(self):
        url = URL.from_str("https://example.com/")
        assert url.path == "/"

    def test_no_trailing_slash_stays_unsuffixed(self):
        url = URL.from_str("https://example.com/data")
        assert not url.path.endswith("/")

    def test_s3_prefix(self):
        url = URL.from_str("s3://bucket/prefix/")
        assert url.path.endswith("/")

    def test_s3_root(self):
        url = URL.from_str("s3://bucket/")
        assert url.path == "/"

    def test_file_dir(self):
        url = URL.from_str("file:///tmp/data/")
        assert url.path.endswith("/")

    def test_dbfs_volume_dir(self):
        url = URL.from_str("dbfs://Volumes/catalog/schema/volume/")
        assert url.path.endswith("/")

    def test_query_fragment_dont_swallow_trailing_slash(self):
        url = URL.from_str("https://example.com/dir/?q=1#f")
        assert url.path.endswith("/")

    def test_round_trip_preserves_trailing_slash(self):
        raw = "https://example.com/data/"
        url = URL.from_str(raw)
        assert URL.from_str(str(url)).path.endswith("/")


class TestFromPathlib:

    def test_pathlib_path_becomes_file_url(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        url = URL.from_pathlib(p)
        assert url.scheme == "file"
        assert url.path.endswith("/data.csv")

    def test_from_generic_with_path(self, tmp_path):
        p = tmp_path / "test.txt"
        p.touch()
        url = URL.from_(p)
        assert url.scheme == "file"
        assert url.path.endswith("/test.txt")


class TestFromDict:

    def test_keyword_components(self):
        url = URL.from_dict({
            "scheme": "https",
            "host": "example.com",
            "port": 9090,
            "path": "/api",
            "query": "a=1",
            "fragment": "top",
        })
        assert url.scheme == "https"
        assert url.host == "example.com"
        assert url.port == 9090
        assert url.path == "/api"

    def test_url_key_shortcuts_to_from_str(self):
        url = URL.from_dict({"url": "https://example.com/path?x=1"})
        assert url.scheme == "https"
        assert url.path == "/path"

    def test_empty_mapping_returns_empty_url(self):
        url = URL.from_dict({})
        assert url == URL.empty()


class TestFrom:

    def test_url_passthrough_returns_same_object(self):
        original = URL.from_str("https://example.com/path")
        assert URL.from_(original) is original

    def test_string_input(self):
        url = URL.from_("https://example.com/path")
        assert url.scheme == "https"
        assert url.path == "/path"

    def test_none_with_default_returns_sentinel(self):
        sentinel = object()
        assert URL.from_(None, default=sentinel) is sentinel

    def test_none_without_default_raises(self):
        with pytest.raises(ValueError, match="Cannot parse URL from None"):
            URL.from_(None)

    def test_mapping_input(self):
        url = URL.from_({"scheme": "https", "host": "example.com", "path": "/x"})
        assert url.host == "example.com"

    def test_pathlike_input(self, tmp_path):
        p = tmp_path / "readme.txt"
        p.touch()
        url = URL.from_(p)
        assert url.scheme == "file"


class TestEquality:

    def test_default_port_urls_equal(self):
        a = URL.from_str("http://example.com:80/path")
        b = URL.from_str("http://example.com/path")
        assert a == b

    def test_inequality_on_different_path(self):
        a = URL.from_str("http://example.com/a")
        b = URL.from_str("http://example.com/b")
        assert a != b

    def test_hashable_and_hashes_equal(self):
        a = URL.from_str("https://example.com/path")
        b = URL.from_str("https://example.com/path")
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_equality_with_str(self):
        url = URL.from_str("https://example.com/path")
        assert url == str(url)

    def test_port_zero_normalized_to_none(self):
        url = URL(scheme="https", host="example.com", port=0, path="/")
        assert url.port is None


class TestEmptySingleton:

    def test_empty_returns_singleton(self):
        assert URL.empty() is URL.empty()

    def test_new_instance_returns_distinct(self):
        a = URL.empty()
        b = URL.empty(new_instance=True)
        assert a == b
        assert a is not b


class TestPathProperties:

    def test_parts(self):
        url = URL.from_str("https://example.com/a/b/c")
        assert url.parts == ["a", "b", "c"]

    def test_name(self):
        assert URL.from_str("https://example.com/a/b/c").name == "c"

    def test_name_for_root(self):
        assert URL.from_str("https://example.com/").name == ""

    def test_stem_strips_last_suffix_only(self):
        assert URL.from_str("https://example.com/archive.csv.zst").stem == "archive.csv"

    def test_extensions_lowercased(self):
        url = URL.from_str("https://example.com/file.CSV.GZ")
        assert url.extensions == ["csv", "gz"]

    def test_extensions_empty_for_extensionless(self):
        url = URL.from_str("https://example.com/README")
        assert url.extensions == []


class TestParentChain:

    def test_parent(self):
        url = URL.from_str("https://example.com/a/b/c")
        assert url.parent.path == "/a/b"

    def test_root_is_its_own_parent(self):
        url = URL.from_str("https://example.com/")
        assert url.parent.path == "/"

    def test_parents_chain(self):
        url = URL.from_str("https://example.com/a/b/c")
        paths = [p.path for p in url.parents]
        assert paths == ["/a/b", "/a", "/"]


class TestJoinpath:

    def test_simple_join(self):
        url = URL.from_str("https://example.com/a")
        result = url.joinpath("b")
        assert result.path == "/a/b"

    def test_truediv_operator(self):
        url = URL.from_str("https://example.com/a")
        result = url / "b"
        assert result.path == "/a/b"

    def test_absolute_segment_resets_path(self):
        url = URL.from_str("https://example.com/a/b")
        result = url.joinpath("/c")
        assert result.path == "/c"

    def test_no_segments_returns_self(self):
        url = URL.from_str("https://example.com/a")
        assert url.joinpath() is url

    def test_invalid_segment_type_raises_typeerror(self):
        url = URL.from_str("https://example.com/a")
        with pytest.raises(TypeError, match="joinpath/truediv"):
            url.joinpath(42)

    def test_plain_string_fast_path_matches_pathlib(self):
        url = URL.from_str("https://example.com/base")
        result = url.joinpath("sub", "file.txt")
        assert result.path == "/base/sub/file.txt"

    def test_trailing_slash_base_no_double_slash(self):
        url = URL.from_str("https://example.com/base/")
        result = url.joinpath("child")
        assert "//" not in result.path
        assert result.path == "/base/child"

    def test_dot_segment_falls_back_to_pathlib(self):
        url = URL.from_str("https://example.com/a/b")
        result = url.joinpath(".")
        assert result.path == "/a/b"


class TestQueryHelpers:

    def test_query_dict_multi_value(self):
        url = URL.from_str("https://example.com/?a=1&a=2&b=3")
        d = url.query_dict
        assert d["a"] == ("1", "2")
        assert d["b"] == ("3",)

    def test_query_items_tuple(self):
        url = URL.from_str("https://example.com/?x=1&y=2")
        items = url.query_items()
        assert isinstance(items, tuple)
        assert ("x", "1") in items

    def test_query_mapping(self):
        url = URL.from_str("https://example.com/?k=v1&k=v2")
        m = url.query_mapping()
        assert m["k"] == ["v1", "v2"]

    def test_add_param_replace(self):
        url = URL.from_str("https://example.com/?a=old")
        result = url.add_param("a", "new", replace=True)
        assert result.query_dict["a"] == ("new",)

    def test_add_param_keep_existing(self):
        url = URL.from_str("https://example.com/?a=old")
        result = url.add_param("a", "new", replace=False)
        assert result.query_dict["a"] == ("new", "old")

    def test_add_params_from_mapping(self):
        url = URL.from_str("https://example.com/")
        result = url.add_params({"x": "1", "y": "2"})
        assert "x" in result.query_dict
        assert "y" in result.query_dict

    def test_with_query_items_mapping(self):
        url = URL.from_str("https://example.com/")
        result = url.with_query_items({"a": "1", "b": ["2", "3"]})
        assert result.query_dict["a"] == ("1",)
        assert result.query_dict["b"] == ("2", "3")

    def test_add_param_with_none_key_raises(self):
        url = URL.from_str("https://example.com/")
        with pytest.raises(ValueError, match="key cannot be None"):
            url.add_param(None, "val")


class TestUserPassword:

    def test_user_only(self):
        url = URL.from_str("https://alice@example.com/")
        assert url.user == "alice"
        assert url.password is None

    def test_user_and_password(self):
        url = URL.from_str("https://alice:s3cret@example.com/")
        assert url.user == "alice"
        assert url.password == "s3cret"

    def test_with_user_password_set(self):
        url = URL.from_str("https://example.com/")
        result = url.with_user_password("bob", "pass123")
        assert result.user == "bob"
        assert result.password == "pass123"

    def test_with_user_password_clear(self):
        url = URL.from_str("https://alice:pass@example.com/")
        result = url.with_user_password(None, None)
        assert result.userinfo is None


class TestWithMethods:

    def test_with_scheme(self):
        url = URL.from_str("http://example.com/")
        assert url.with_scheme("https").scheme == "https"

    def test_with_host(self):
        url = URL.from_str("https://example.com/")
        assert url.with_host("other.com").host == "other.com"

    def test_with_path(self):
        url = URL.from_str("https://example.com/old")
        assert url.with_path("/new").path == "/new"

    def test_with_query(self):
        url = URL.from_str("https://example.com/")
        assert url.with_query("k=v").query == "k=v"

    def test_with_fragment(self):
        url = URL.from_str("https://example.com/")
        assert url.with_fragment("section").fragment == "section"

    def test_with_userinfo(self):
        url = URL.from_str("https://example.com/")
        result = url.with_userinfo("user:pass")
        assert result.userinfo == "user:pass"

    def test_with_scheme_inplace(self):
        url = URL.from_str("http://example.com/path")
        returned = url.with_scheme("https", inplace=True)
        assert returned is url
        assert url.scheme == "https"


class TestMatchPattern:

    def test_basename_match(self):
        url = URL.from_str("https://example.com/data/file.csv")
        assert url.match_pattern("*.csv") is True

    def test_full_url_match(self):
        url = URL.from_str("https://example.com/data/file.csv")
        assert url.match_pattern("https://example.com/data/*") is True

    def test_no_match(self):
        url = URL.from_str("https://example.com/data/file.csv")
        assert url.match_pattern("*.parquet") is False


class TestMatchesPatterns:

    def test_none_returns_false(self):
        url = URL.from_str("https://example.com/data.csv")
        assert url.matches_patterns(None) is False

    def test_empty_list_returns_false(self):
        url = URL.from_str("https://example.com/data.csv")
        assert url.matches_patterns([]) is False

    def test_first_match_short_circuits(self):
        url = URL.from_str("https://example.com/data.csv")
        assert url.matches_patterns(["*.csv", "*.parquet"]) is True


class TestIsRelativeTo:

    def test_strict_prefix(self):
        url = URL.from_str("https://example.com/a/b/c")
        assert url.is_relative_to("https://example.com/a") is True

    def test_equal_paths(self):
        url = URL.from_str("https://example.com/a")
        assert url.is_relative_to("https://example.com/a") is True

    def test_sibling_not_relative(self):
        url = URL.from_str("https://example.com/a")
        assert url.is_relative_to("https://example.com/b") is False

    def test_mismatched_host_rejected(self):
        url = URL.from_str("https://example.com/a/b")
        assert url.is_relative_to("https://other.com/a") is False


class TestRelativeTo:

    def test_basic(self):
        url = URL.from_str("https://example.com/a/b/c")
        result = url.relative_to("https://example.com/a")
        assert result.path == "/b/c"

    def test_exact_match_returns_root(self):
        url = URL.from_str("https://example.com/a")
        result = url.relative_to("https://example.com/a")
        assert result.path == "/"

    def test_raises_when_not_relative(self):
        url = URL.from_str("https://example.com/a")
        with pytest.raises(ValueError, match="is not relative to"):
            url.relative_to("https://example.com/b")


class TestDerivedProperties:

    def test_is_absolute(self):
        assert URL.from_str("https://example.com/path").is_absolute is True
        assert URL(path="/just/a/path").is_absolute is False

    def test_is_http(self):
        assert URL.from_str("https://example.com/").is_http is True
        assert URL.from_str("s3://bucket/key").is_http is False

    def test_authority(self):
        url = URL.from_str("https://user:pass@example.com:8080/path")
        assert url.authority == "user:pass@example.com:8080"

    def test_is_dir_sink(self):
        assert URL.from_str("https://example.com/dir/").is_dir_sink is True
        assert URL.from_str("https://example.com/file.csv").is_dir_sink is False

    def test_is_pathish_url(self):
        assert URL.is_pathish(URL.from_str("https://example.com/")) is True

    def test_is_pathish_str(self):
        assert URL.is_pathish("https://example.com/") is True

    def test_is_pathish_path(self):
        assert URL.is_pathish(pathlib.Path("/tmp")) is True

    def test_is_pathish_none(self):
        assert URL.is_pathish(None) is False

    def test_is_pathish_mapping(self):
        assert URL.is_pathish({"scheme": "https"}) is True

    def test_is_urlish(self):
        assert URL.is_urlish(URL.from_str("https://x.com/")) is True
        assert URL.is_urlish("https://x.com/") is True
        assert URL.is_urlish(42) is False


class TestAnonymize:

    def test_removes_userinfo(self):
        url = URL.from_str("https://alice:secret@example.com/path")
        anon = url.anonymize("remove")
        assert anon.userinfo is None

    def test_redacts_userinfo(self):
        url = URL.from_str("https://alice:secret@example.com/path")
        anon = url.anonymize("redact")
        assert anon.userinfo == "<redacted>"

    def test_strips_sensitive_query_params(self):
        url = URL.from_str("https://example.com/path?token=abc123&name=ok")
        anon = url.anonymize("remove")
        assert "token" not in (anon.query or "")
        assert "name=ok" in (anon.query or "")

    def test_repr_redacts_userinfo(self):
        url = URL.from_str("https://alice:secret@example.com/path")
        r = repr(url)
        assert "secret" not in r
        assert "<redacted>" in r

    def test_repr_redacts_sensitive_query(self):
        url = URL.from_str("https://example.com/path?token=secret123")
        r = repr(url)
        assert "secret123" not in r

    def test_str_keeps_original(self):
        url = URL.from_str("https://alice:secret@example.com/path")
        s = str(url)
        assert "alice:secret" in s

    def test_anonymize_caching(self):
        url = URL.from_str("https://user:pw@example.com/path")
        first = url.anonymize("remove")
        second = url.anonymize("remove")
        assert first is second


class TestToString:

    def test_round_trip(self):
        raw = "https://example.com/path?key=val#frag"
        url = URL.from_str(raw)
        assert str(url) == raw

    def test_str_is_to_string(self):
        url = URL.from_str("https://example.com/path")
        assert str(url) == url.to_string()

    def test_encode_false_skips_encoding(self):
        url = URL.from_str("https://example.com/path with spaces")
        raw = url.to_string(encode=False)
        assert "path with spaces" in raw


class TestToPathlib:

    def test_strict_requires_file_scheme(self):
        url = URL.from_str("https://example.com/path")
        with pytest.raises(ValueError, match="scheme='file'"):
            url.to_pathlib(strict=True)

    def test_strict_false_drops_url_components(self):
        url = URL.from_str("https://example.com/some/path?q=1#frag")
        p = url.to_pathlib(strict=False)
        assert isinstance(p, pathlib.Path)

    def test_file_scheme_round_trip(self, tmp_path):
        original = tmp_path / "data.csv"
        original.touch()
        url = URL.from_pathlib(original)
        restored = url.to_pathlib()
        assert restored == original


class TestFspathProtocol:

    def test_file_url(self, tmp_path):
        p = tmp_path / "file.txt"
        p.touch()
        url = URL.from_pathlib(p)
        assert os.fspath(url) == str(p)

    def test_http_url_returns_path_only(self):
        url = URL.from_str("https://example.com/api/v1")
        assert os.fspath(url) == "/api/v1"


class TestMemoryAddress:

    def test_round_trip(self):
        obj = {"key": "value"}
        url = URL.from_memory_address(obj)
        assert url.is_memory_address is True
        assert url.resolve_memory_address() is obj

    def test_resolve_memory_address_standalone(self):
        obj = [1, 2, 3]
        addr = id(obj)
        assert resolve_memory_address(addr) is obj

    def test_non_mem_url_is_memory_address_false(self):
        url = URL.from_str("https://example.com/path")
        assert url.is_memory_address is False

    def test_non_mem_url_memory_address_raises(self):
        url = URL.from_str("https://example.com/path")
        with pytest.raises(ValueError, match="not a memory-address URL"):
            url.memory_address


class TestJoin:

    def test_absolute_path_ref_on_same_authority(self):
        base = URL.from_str("https://example.com/a/b")
        result = base.join("/c/d")
        assert result.host == "example.com"
        assert result.path == "/c/d"

    def test_relative_ref(self):
        base = URL.from_str("https://example.com/a/b")
        result = base.join("c")
        assert result.path == "/a/c"

    def test_full_url_ref_replaces_everything(self):
        base = URL.from_str("https://example.com/a")
        result = base.join("https://other.com/x")
        assert result.host == "other.com"
        assert result.path == "/x"


class TestToStructDict:

    def test_produces_correct_keys(self):
        url = URL.from_str("https://user@example.com:9090/path?q=1#f")
        d = url.to_struct_dict()
        assert set(d.keys()) == {"scheme", "userinfo", "host", "port", "path", "query", "fragment"}
        assert d["scheme"] == "https"
        assert d["host"] == "example.com"

    def test_port_none_when_absent(self):
        url = URL.from_str("https://example.com/path")
        d = url.to_struct_dict()
        assert d["port"] is None


class TestPickle:

    def test_pickle_round_trip_file_url(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        url = URL.from_pathlib(p)
        restored = pickle.loads(pickle.dumps(url))
        assert restored == url
        assert restored.scheme == "file"

    def test_pickle_round_trip_http_url(self):
        url = URL.from_str("https://example.com:9090/path?q=v#f")
        restored = pickle.loads(pickle.dumps(url))
        assert restored == url
        assert restored.port == 9090


class TestEndswith:

    def test_matches_suffix(self):
        url = URL.from_str("https://example.com/data/file.csv")
        assert url.endswith(".csv") is True

    def test_no_match(self):
        url = URL.from_str("https://example.com/data/file.csv")
        assert url.endswith(".parquet") is False

    def test_empty_path(self):
        url = URL(path="")
        assert url.endswith(".csv") is False
