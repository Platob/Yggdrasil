"""Unit tests for :class:`yggdrasil.io.url.URL`.

Organized by concern. Tests that require the yggdrasil package's MimeType /
MediaType registry are guarded with ``pytest.importorskip`` so the rest of
the suite runs in any environment.

Run with::

    pytest test_url.py -v
"""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import pytest

from yggdrasil.io.enums import MimeTypes
from yggdrasil.io.url import URL


# =============================================================================
# Construction — from_str
# =============================================================================


class TestFromStr:
    def test_full_http_url(self):
        url = URL.from_str("https://user:pass@example.com:8443/a/b?x=1&y=2#frag")
        assert url.scheme == "https"
        assert url.userinfo == "user:pass"
        assert url.host == "example.com"
        assert url.port == 8443
        assert url.path == "/a/b"
        assert url.query == "x=1&y=2"
        assert url.fragment == "frag"

    def test_bare_path_no_scheme(self):
        url = URL.from_str("/tmp/data.csv")
        assert url.scheme == ""
        assert url.host == ""
        assert url.path == "/tmp/data.csv"

    def test_empty_string(self):
        url = URL.from_str("")
        assert url.scheme == ""
        assert url.host == ""
        assert url.path == "/"

    def test_default_port_elided(self):
        # http on 80 and https on 443 should not retain the explicit port.
        assert URL.from_str("http://example.com:80/").port is None
        assert URL.from_str("https://example.com:443/").port is None
        assert URL.from_str("ws://example.com:80/").port is None
        assert URL.from_str("wss://example.com:443/").port is None

    def test_non_default_port_retained(self):
        assert URL.from_str("http://example.com:8080/").port == 8080
        assert URL.from_str("https://example.com:8443/").port == 8443

    def test_scheme_lowercased(self):
        assert URL.from_str("HTTPS://Example.COM/").scheme == "https"

    def test_host_lowercased(self):
        assert URL.from_str("https://Example.COM/").host == "example.com"

    def test_host_trailing_dot_stripped(self):
        assert URL.from_str("https://example.com./").host == "example.com"

    def test_query_sorted_canonically(self):
        # Canonical equality requires deterministic query ordering.
        u1 = URL.from_str("https://e.com/?b=2&a=1")
        u2 = URL.from_str("https://e.com/?a=1&b=2")
        assert u1 == u2
        assert u1.query == "a=1&b=2"

    def test_fragment_hash_stripped(self):
        assert URL.from_str("https://e.com/#frag").fragment == "frag"

    def test_schemaless_host_like(self):
        # "example.com/path" has no scheme — should land as a bare path
        # (the "fixup" for host-in-path only fires when a scheme exists).
        url = URL.from_str("example.com/path")
        assert url.scheme == ""
        # No host extraction without a scheme.
        assert url.host == ""

    def test_scheme_without_authority(self):
        # "http:example.com/path" (no //) — fixup should extract host.
        url = URL.from_str("http:example.com/path")
        assert url.scheme == "http"
        assert url.host == "example.com"
        assert url.path == "/path"

    def test_ipv6_host(self):
        url = URL.from_str("http://[::1]:8080/x")
        assert url.host == "::1"
        assert url.port == 8080

    def test_ipv6_no_port(self):
        url = URL.from_str("http://[::1]/")
        assert url.host == "::1"
        assert url.port is None


class TestFromStrWindowsDrive:
    """Windows drive-letter paths go through a dedicated fix-up path.

    These tests pass ``normalize=False`` because normalization on a
    ``file://`` URL runs :func:`os.path.realpath` — on a non-Windows
    host the Windows drive path gets re-anchored to the current
    directory. We're testing the parser's drive-fix-up layer, not the
    post-parse filesystem resolution.
    """

    def test_forward_slash_drive(self):
        url = URL.from_str("C:/Users/x", normalize=False)
        assert url.scheme == "file"
        assert url.path == "/C:/Users/x"

    def test_backslash_drive(self):
        url = URL.from_str("C:\\Users\\x", normalize=False)
        assert url.scheme == "file"
        assert url.path == "/C:/Users/x"

    def test_drive_uppercased(self):
        # Drive letter should be canonicalized to upper case.
        assert URL.from_str("c:/tmp/x", normalize=False).path == "/C:/tmp/x"


# =============================================================================
# Construction — from_pathlib
# =============================================================================


class TestFromPathlib:
    def test_absolute_path(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        url = URL.from_pathlib(p)
        assert url.scheme == "file"
        assert url.path.endswith("/data.csv")

    def test_string_coerced_to_path(self, tmp_path):
        p = tmp_path / "data.csv"
        p.touch()
        url = URL.from_pathlib(str(p))
        assert url.scheme == "file"
        assert url.path.endswith("/data.csv")

    def test_tilde_expanded(self):
        # ~ expansion must happen — can't know the actual home dir, but
        # the produced path must not start with ~.
        url = URL.from_pathlib("~/does_not_exist_12345.csv")
        assert not url.path.startswith("/~")
        assert "~" not in url.path


# =============================================================================
# Construction — from_dict
# =============================================================================


class TestFromDict:
    def test_raw_key(self):
        url = URL.from_dict({"url": "https://example.com/x"})
        assert url == URL.from_str("https://example.com/x")

    def test_raw_alias(self):
        url = URL.from_dict({"raw": "https://example.com/x"})
        assert url == URL.from_str("https://example.com/x")

    def test_components(self):
        url = URL.from_dict(
            {
                "scheme": "https",
                "host": "example.com",
                "port": 8443,
                "path": "/a/b",
                "query": "x=1",
            }
        )
        assert url.scheme == "https"
        assert url.host == "example.com"
        assert url.port == 8443
        assert url.path == "/a/b"
        assert url.query == "x=1"

    def test_netloc(self):
        url = URL.from_dict({"scheme": "https", "netloc": "u:p@example.com:8443"})
        assert url.userinfo == "u:p"
        assert url.host == "example.com"
        assert url.port == 8443

    def test_authority_alias(self):
        url = URL.from_dict({"scheme": "https", "authority": "example.com:9000"})
        assert url.host == "example.com"
        assert url.port == 9000

    def test_empty_dict(self):
        assert URL.from_dict({}) == URL()


# =============================================================================
# Construction — from_ (dispatcher)
# =============================================================================


class TestFrom:
    def test_url_passthrough(self):
        u = URL.from_str("https://e.com/x")
        assert URL.from_(u) is u  # same instance, no copy

    def test_string(self):
        assert URL.from_("https://e.com/x") == URL.from_str("https://e.com/x")

    def test_path(self, tmp_path):
        p = tmp_path / "x.csv"
        p.touch()
        assert URL.from_(p).scheme == "file"

    def test_dict(self):
        assert URL.from_({"url": "https://e.com/x"}) == URL.from_str("https://e.com/x")

    def test_none_raises(self):
        with pytest.raises(ValueError):
            URL.from_(None)


# =============================================================================
# Canonical equality
# =============================================================================


class TestCanonicalEquality:
    """The whole point of __post_init__'s port normalization and
    from_str/from_dict's `port or None` treatment is that any two
    constructors producing URLs that render to the same string compare
    equal.
    """

    def test_port_zero_equals_port_none(self):
        # The internal sentinel should never leak into the public port field.
        # URL(port=0) and URL(port=None) must be equal.
        a = URL(scheme="http", host="e.com", port=0)
        b = URL(scheme="http", host="e.com", port=None)
        assert a == b
        assert a.port is None
        assert b.port is None

    def test_default_port_elided_equality(self):
        # http://e.com/ and http://e.com:80/ should be equal after normalization.
        a = URL.from_str("http://e.com/")
        b = URL.from_str("http://e.com:80/")
        assert a == b

    def test_query_order_equality(self):
        a = URL.from_str("https://e.com/?b=2&a=1")
        b = URL.from_str("https://e.com/?a=1&b=2")
        assert a == b

    def test_trailing_dot_host_equality(self):
        a = URL.from_str("https://e.com./")
        b = URL.from_str("https://e.com/")
        assert a == b

    def test_scheme_case_equality(self):
        a = URL.from_str("HTTPS://e.com/")
        b = URL.from_str("https://e.com/")
        assert a == b


# =============================================================================
# to_string
# =============================================================================


class TestToString:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://example.com/", "https://example.com/"),
            ("https://u:p@example.com:8443/a?x=1#f", "https://u:p@example.com:8443/a?x=1#f"),
            ("/tmp/x", "/tmp/x"),
            # Empty string → URL with default path "/" → renders as "/".
            ("", "/"),
        ],
    )
    def test_roundtrip(self, raw, expected):
        assert URL.from_str(raw).to_string() == expected

    def test_ipv6_bracketed(self):
        url = URL(scheme="http", host="::1", port=8080, path="/")
        s = url.to_string()
        assert "[::1]" in s
        assert "[::1]:8080" in s

    def test_cached(self):
        url = URL.from_str("https://e.com/x")
        a = url.to_string()
        b = url.to_string()
        # Same object; cache hit. This is a behavioural detail but worth locking in.
        assert a == b
        # Internal cache slot is populated.
        assert url._str_enc == a


# =============================================================================
# parts
# =============================================================================


class TestParts:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/", []),
            ("", []),
            ("/a", ["a"]),
            ("/a/b/c", ["a", "b", "c"]),
            ("/a/b/c/", ["a", "b", "c", ""]),  # trailing slash leaves empty tail
        ],
    )
    def test_parts(self, path, expected):
        assert URL(path=path).parts == expected


# =============================================================================
# extensions
# =============================================================================


class TestExtensions:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/data/file.csv", ["csv"]),
            ("/data/file.tar.gz", ["tar", "gz"]),
            ("/data/archive.csv.zst", ["csv", "zst"]),
            ("/data/FILE.CSV", ["csv"]),  # lowercased
            ("/data/README", []),
            ("/", []),
            ("", []),
            ("/data/", []),
            ("/data/.hidden", []),  # dotfile, not extension
            ("/data/.env.local", ["local"]),
            ("/path/to/weird.name.2024-01-01.parquet", ["name", "2024-01-01", "parquet"]),
        ],
    )
    def test_extensions(self, path, expected):
        assert URL(path=path).extensions == expected


# =============================================================================
# name
# =============================================================================


class TestName:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/a/b/c", "c"),
            ("/a/b/c/", "c"),  # trailing slash stripped
            ("/a/b/", "b"),
            ("/", ""),
            ("", ""),
            ("/x", "x"),
            ("/data/archive.csv.zst", "archive.csv.zst"),
            ("/data/archive.csv.zst/", "archive.csv.zst"),
        ],
    )
    def test_name(self, path, expected):
        assert URL(path=path).name == expected


# =============================================================================
# stem
# =============================================================================


class TestStem:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/a/file.csv", "file"),
            # Only the final suffix is peeled — use `extensions` for all.
            ("/a/archive.csv.zst", "archive.csv"),
            ("/a/README", "README"),
            ("/a/.hidden", ".hidden"),  # dotfile, no extension
            ("/a/.env.local", ".env"),
            ("/", ""),
            ("", ""),
            # Trailing slash: PurePosixPath treats "b" as the name.
            ("/a/b/", "b"),
        ],
    )
    def test_stem(self, path, expected):
        assert URL(path=path).stem == expected

    def test_default_url(self):
        # The dataclass default is "/", so a bare URL() has empty stem.
        assert URL().stem == ""


# =============================================================================
# parent
# =============================================================================


class TestParent:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/a/b/c", "/a/b"),
            ("/a/b/c/", "/a/b"),  # trailing slash ignored
            ("/a", "/"),
            ("/", "/"),  # root is its own parent
            ("", "/"),  # empty path treated as root
        ],
    )
    def test_parent_path(self, path, expected):
        url = URL(path=path) if path else URL(path="")
        assert url.parent.path == expected

    def test_returns_url(self):
        assert isinstance(URL(path="/a/b").parent, URL)

    def test_carries_query_and_fragment(self):
        url = URL.from_str("https://e.com/a/b?x=1&y=2#frag")
        p = url.parent
        assert p.path == "/a"
        assert p.query == "x=1&y=2"
        assert p.fragment == "frag"

    def test_carries_authority(self):
        url = URL.from_str("https://user:pw@example.com:8443/a/b/c")
        p = url.parent
        assert p.scheme == "https"
        assert p.userinfo == "user:pw"
        assert p.host == "example.com"
        assert p.port == 8443

    def test_root_is_own_parent(self):
        # Chain up past root — should stabilize at "/".
        url = URL(path="/a")
        assert url.parent.path == "/"
        assert url.parent.parent.path == "/"
        assert url.parent.parent.parent.path == "/"

    def test_default_url(self):
        # Bare URL() has path="/" which is its own parent.
        assert URL().parent.path == "/"


# =============================================================================
# parents
# =============================================================================


class TestParents:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/a/b/c", ["/a/b", "/a", "/"]),
            ("/a/b", ["/a", "/"]),
            ("/a", ["/"]),
            ("/", []),
            ("", []),
        ],
    )
    def test_parents_paths(self, path, expected):
        url = URL(path=path) if path else URL(path="")
        assert [p.path for p in url.parents] == expected

    def test_returns_tuple(self):
        # Frozen URL plays better with a tuple than a list.
        parents = URL(path="/a/b/c").parents
        assert isinstance(parents, tuple)

    def test_every_element_is_url(self):
        for p in URL(path="/a/b/c").parents:
            assert isinstance(p, URL)

    def test_carries_query_and_fragment(self):
        url = URL.from_str("https://e.com/a/b/c?x=1#frag")
        for p in url.parents:
            assert p.query == "x=1"
            assert p.fragment == "frag"
            assert p.host == "e.com"

    def test_default_url(self):
        assert URL().parents == ()

    def test_first_parent_equals_parent(self):
        url = URL(path="/a/b/c")
        # First element of `parents` must match `.parent`.
        assert url.parents[0] == url.parent


# =============================================================================
# mime_type / codec / media_type
# =============================================================================


@pytest.mark.usefixtures("_media_type_available")
class TestMediaTypeProperties:
    """These properties depend on the MimeType/MediaType registry.
    Guarded by a fixture that skips if the subpackage can't be imported.
    """

    def test_plain_extension(self):
        url = URL.from_str("/data/file.csv")
        assert url.mime_type is not None
        assert url.mime_type.name == "CSV"
        assert url.codec is None

    def test_codec_wrapped(self):
        url = URL.from_str("/data/file.csv.zst")
        assert url.mime_type is not None
        assert url.mime_type.name == "CSV"
        assert url.codec is not None
        assert url.codec.name == "zstd"

    def test_no_extension(self):
        # README has no extension; MediaType resolves to OCTET_STREAM.
        url = URL.from_str("/data/README")
        assert url.mime_type is None

    def test_no_path(self):
        url = URL.from_str("/")
        assert url.mime_type is MimeTypes.FOLDER
        assert url.codec is None

    def test_media_type_matches_mime_and_codec(self):
        url = URL.from_str("/a/file.csv.gz")
        mt = url.media_type
        assert mt is not None
        assert mt.mime_type is url.mime_type
        assert mt.codec is url.codec


@pytest.fixture
def _media_type_available():
    pytest.importorskip("yggdrasil.io.enums.media_type")


# =============================================================================
# is_pathish
# =============================================================================


class TestIsPathish:
    def test_str(self):
        assert URL.is_pathish("hello") is True

    def test_empty_str(self):
        # Consistent with "would from_ accept this" — from_str("") succeeds.
        assert URL.is_pathish("") is True

    def test_path(self):
        assert URL.is_pathish(Path("/tmp")) is True

    def test_mapping(self):
        assert URL.is_pathish({"scheme": "file"}) is True

    def test_url(self):
        assert URL.is_pathish(URL()) is True

    def test_custom_pathlike(self):
        class P:
            def __fspath__(self):
                return "/x"

        assert URL.is_pathish(P()) is True

    def test_object_with_url_attr(self):
        class O:
            url = "https://e.com"

        assert URL.is_pathish(O()) is True

    def test_none(self):
        assert URL.is_pathish(None) is False

    def test_int(self):
        assert URL.is_pathish(42) is False

    def test_bytes(self):
        # bytes aren't handled by from_ today.
        assert URL.is_pathish(b"bytes") is False

    def test_arbitrary_object(self):
        assert URL.is_pathish(object()) is False


# =============================================================================
# __fspath__ / os.PathLike
# =============================================================================


class TestFspath:
    def test_is_pathlike(self):
        assert isinstance(URL(), os.PathLike)

    def test_file_scheme(self):
        url = URL(scheme="file", path="/tmp/x")
        assert os.fspath(url) == "/tmp/x"

    def test_file_scheme_windows_drive(self):
        url = URL(scheme="file", path="/C:/Users/x")
        assert os.fspath(url) == "C:/Users/x"

    def test_empty_scheme(self):
        url = URL(scheme="", path="/tmp/x")
        assert os.fspath(url) == "/tmp/x"

    def test_empty_scheme_windows_drive(self):
        url = URL(scheme="", path="/C:/Users/x")
        assert os.fspath(url) == "C:/Users/x"

    def test_non_file_scheme_returns_raw_path(self):
        # For https etc., return only the path (not the full URL).
        url = URL(scheme="https", host="e.com", path="/a/b")
        assert os.fspath(url) == "/a/b"

    def test_s3_scheme_returns_path(self):
        url = URL(scheme="s3", host="bucket", path="/key")
        assert os.fspath(url) == "/key"

    def test_empty_path_returns_empty_string(self):
        # __fspath__ must not raise on edge cases — just returns "".
        assert os.fspath(URL(scheme="file", path="")) == ""

    def test_root_path_returns_root(self):
        # "/" is not a Windows-drive pattern, passes through.
        assert os.fspath(URL(scheme="file", path="/")) == "/"

    def test_open_integration(self, tmp_path):
        # The whole point of PathLike: open() should accept URL directly.
        p = tmp_path / "test.txt"
        p.write_text("hello")
        url = URL.from_pathlib(p)
        with open(url) as fh:
            assert fh.read() == "hello"


# =============================================================================
# to_pathlib (inverse of from_pathlib)
# =============================================================================


class TestToPathlib:
    def test_basic(self):
        url = URL(scheme="file", path="/tmp/x")
        assert url.to_pathlib() == Path("/tmp/x")

    def test_windows_drive_strip(self):
        url = URL(scheme="file", path="/C:/Users/x")
        # to_pathlib strips the /X: leading slash.
        assert str(url.to_pathlib()).replace("\\", "/") == "C:/Users/x"

    def test_strict_refuses_non_file_scheme(self):
        url = URL(scheme="https", host="e.com", path="/a")
        with pytest.raises(ValueError, match="requires scheme"):
            url.to_pathlib()

    def test_strict_refuses_extras(self):
        url = URL(scheme="file", host="host", path="/x")
        with pytest.raises(ValueError, match="cannot represent"):
            url.to_pathlib()

    def test_non_strict_accepts_any_scheme(self):
        url = URL(scheme="https", host="e.com", path="/a")
        # Does not raise, just drops the extras.
        assert url.to_pathlib(strict=False) == Path("/a")

    def test_empty_path_raises(self):
        url = URL(scheme="file", path="")
        with pytest.raises(ValueError, match="empty path"):
            url.to_pathlib()

    def test_root_path_raises(self):
        url = URL(scheme="file", path="/")
        with pytest.raises(ValueError, match="empty path"):
            url.to_pathlib()

    def test_roundtrip(self, tmp_path):
        p = tmp_path / "roundtrip.csv"
        p.touch()
        url = URL.from_pathlib(p)
        back = url.to_pathlib()
        # Both should refer to the same file (resolve comparison to handle symlinks).
        assert back.resolve() == p.resolve()


# =============================================================================
# user / password
# =============================================================================


class TestUserPassword:
    def test_user_only(self):
        url = URL.from_str("https://alice@e.com/")
        assert url.user == "alice"
        assert url.password is None

    def test_user_and_password(self):
        url = URL.from_str("https://alice:s3cret@e.com/")
        assert url.user == "alice"
        assert url.password == "s3cret"

    def test_percent_decoded(self):
        url = URL.from_str("https://al%40ice:p%3Ass@e.com/")
        assert url.user == "al@ice"
        assert url.password == "p:ss"

    def test_no_userinfo(self):
        url = URL.from_str("https://e.com/")
        assert url.user is None
        assert url.password is None

    def test_with_user_password(self):
        url = URL.from_str("https://e.com/").with_user_password("alice", "s3cret")
        assert url.user == "alice"
        assert url.password == "s3cret"

    def test_with_user_password_clears(self):
        url = URL.from_str("https://alice:s3cret@e.com/").with_user_password(None, None)
        assert url.userinfo is None


# =============================================================================
# Query helpers
# =============================================================================


class TestQuery:
    def test_query_dict(self):
        url = URL.from_str("https://e.com/?a=1&b=2&a=3")
        d = url.query_dict
        assert d == {"a": ("1", "3"), "b": ("2",)}

    def test_add_query_item(self):
        url = URL.from_str("https://e.com/").add_query_item("x", "1")
        assert url.query == "x=1"

    def test_add_query_item_replaces(self):
        url = URL.from_str("https://e.com/?x=1").add_query_item("x", "2")
        assert url.query == "x=2"

    def test_add_query_item_appends(self):
        url = URL.from_str("https://e.com/?x=1").add_query_item("x", "2", replace=False)
        # Canonically sorted.
        assert url.query == "x=1&x=2"

    def test_with_query_items_mapping(self):
        url = URL.from_str("https://e.com/").with_query_items({"a": "1", "b": "2"})
        # Sorted by default.
        assert url.query == "a=1&b=2"

    def test_with_query_items_list_value(self):
        url = URL.from_str("https://e.com/").with_query_items({"a": ["1", "2"]})
        assert url.query == "a=1&a=2"


# =============================================================================
# with_* mutators
# =============================================================================


class TestWithMutators:
    def test_with_scheme(self):
        url = URL.from_str("https://e.com/").with_scheme("http")
        assert url.scheme == "http"

    def test_with_scheme_lowercases(self):
        assert URL.from_str("https://e.com/").with_scheme("HTTP").scheme == "http"

    def test_with_scheme_elides_new_default_port(self):
        # If the scheme change makes the current port the new default, drop it.
        url = URL.from_str("http://e.com:443/").with_scheme("https")
        assert url.port is None

    def test_with_scheme_inplace_sets_scheme(self):
        url = URL.from_str("https://e.com/")
        same = url.with_scheme("http", inplace=True)
        assert same is url
        assert url.scheme == "http"

    def test_with_scheme_inplace_invalidates_cache(self):
        url = URL.from_str("https://e.com/")
        # Prime the cache.
        first = url.to_string()
        url.with_scheme("http", inplace=True)
        second = url.to_string()
        assert first.startswith("https://")
        assert second.startswith("http://")

    def test_with_host(self):
        url = URL.from_str("https://e.com/").with_host("other.com")
        assert url.host == "other.com"

    def test_with_host_lowercases(self):
        assert URL.from_str("https://e.com/").with_host("OTHER.COM").host == "other.com"

    def test_with_path(self):
        url = URL.from_str("https://e.com/").with_path("/a/b")
        assert url.path == "/a/b"

    def test_with_query(self):
        url = URL.from_str("https://e.com/").with_query("x=1")
        assert url.query == "x=1"

    def test_with_query_strips_leading_question(self):
        assert URL.from_str("https://e.com/").with_query("?x=1").query == "x=1"

    def test_with_fragment(self):
        url = URL.from_str("https://e.com/").with_fragment("section")
        assert url.fragment == "section"


# =============================================================================
# join / truediv
# =============================================================================


class TestJoin:
    """URL.join uses RFC 3986 reference resolution (via urljoin)."""

    def test_join_relative(self):
        url = URL.from_str("https://e.com/a/b").join("c")
        assert url.to_string() == "https://e.com/a/c"

    def test_join_absolute_path(self):
        url = URL.from_str("https://e.com/a/b").join("/c")
        assert url.path == "/c"

    def test_join_full_url(self):
        url = URL.from_str("https://e.com/a/b").join("https://other.com/x")
        assert url.host == "other.com"
        assert url.path == "/x"


class TestJoinpath:
    """URL.joinpath and / do pathlib-style segment appending."""

    def test_single_segment(self):
        assert URL.from_str("https://e.com/a").joinpath("b").path == "/a/b"

    def test_multiple_segments(self):
        assert URL.from_str("https://e.com/a").joinpath("b", "c").path == "/a/b/c"

    def test_truediv(self):
        assert (URL.from_str("https://e.com/a") / "b").path == "/a/b"

    def test_truediv_chained(self):
        # / operator should compose: url / "a" / "b"
        assert (URL.from_str("https://e.com/") / "a" / "b").path == "/a/b"

    def test_absolute_segment_resets(self):
        # Matches PurePosixPath semantics.
        assert URL.from_str("https://e.com/a/b").joinpath("/c").path == "/c"

    def test_carries_authority_and_query(self):
        url = URL.from_str("https://user:pw@e.com:8443/a?x=1#frag")
        joined = url.joinpath("b")
        assert joined.scheme == "https"
        assert joined.userinfo == "user:pw"
        assert joined.host == "e.com"
        assert joined.port == 8443
        assert joined.query == "x=1"
        assert joined.fragment == "frag"
        assert joined.path == "/a/b"

    def test_no_segments_returns_self(self):
        url = URL.from_str("https://e.com/a")
        assert url.joinpath() is url

    def test_trailing_slash_handled(self):
        # PurePosixPath normalizes the trailing slash during join.
        assert URL.from_str("https://e.com/a/").joinpath("b").path == "/a/b"

    def test_url_segment_contributes_path_only(self):
        # Joining with a URL should use only its path, not its authority.
        rhs = URL.from_str("https://other.com/x/y")
        result = URL.from_str("https://e.com/a").joinpath(rhs)
        # rhs.path is "/x/y" — absolute, so it resets.
        assert result.host == "e.com"  # authority preserved from LHS
        assert result.path == "/x/y"

    def test_pathlib_segment(self):
        from pathlib import PurePosixPath

        result = URL.from_str("https://e.com/a").joinpath(PurePosixPath("b/c"))
        assert result.path == "/a/b/c"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="str, URL, or os.PathLike"):
            URL.from_str("https://e.com/a") / 42  # type: ignore[operator]

    def test_dotdot_not_resolved(self):
        # URL stays syntactic by default — .. is not collapsed.
        result = URL.from_str("/a/b").joinpath("..")
        assert result.path == "/a/b/.."


# =============================================================================
# Pattern matching
# =============================================================================


class TestMatchPattern:
    def test_basename_match(self):
        assert URL.from_str("https://e.com/data/file.csv").match_pattern("*.csv") is True

    def test_full_url_match(self):
        assert (
            URL.from_str("https://e.com/data/file.csv").match_pattern("https://*.com/*")
            is True
        )

    def test_no_match(self):
        assert URL.from_str("https://e.com/data/file.csv").match_pattern("*.json") is False

    def test_question_mark(self):
        assert URL.from_str("/a/file.c").match_pattern("file.?") is True

    def test_character_class(self):
        assert URL.from_str("/a/file.csv").match_pattern("file.[ct]sv") is True


class TestMatchesPatterns:
    def test_none_returns_false(self):
        assert URL.from_str("/a/file.csv").matches_patterns(None) is False

    def test_empty_returns_false(self):
        assert URL.from_str("/a/file.csv").matches_patterns([]) is False
        assert URL.from_str("/a/file.csv").matches_patterns(()) is False

    def test_first_pattern_matches(self):
        assert URL.from_str("/a/file.csv").matches_patterns(["*.csv", "*.json"]) is True

    def test_second_pattern_matches(self):
        assert URL.from_str("/a/file.json").matches_patterns(["*.csv", "*.json"]) is True

    def test_none_match(self):
        assert URL.from_str("/a/file.parquet").matches_patterns(["*.csv", "*.json"]) is False

    def test_accepts_generator(self):
        # Generators must be materialized so both basename and full-URL
        # passes see the patterns.
        patterns = (p for p in ("*.csv",))
        assert URL.from_str("/a/file.csv").matches_patterns(patterns) is True

    def test_full_url_fallback(self):
        # Basename "file.csv" doesn't match "https://*", but the full URL does.
        url = URL.from_str("https://e.com/data/file.csv")
        assert url.matches_patterns(["https://*"]) is True

    def test_basename_checked_first(self):
        # Regression: ensure basename matches aren't masked by full-URL logic.
        url = URL.from_str("https://e.com/data/file.csv")
        assert url.matches_patterns(["file.csv"]) is True


# =============================================================================
# is_relative_to / relative_to
# =============================================================================


class TestIsRelativeTo:
    def test_strict_prefix(self):
        assert URL.from_str("/a/b/c").is_relative_to("/a") is True
        assert URL.from_str("/a/b/c").is_relative_to("/a/b") is True

    def test_equal_paths(self):
        # Path equal to itself is "relative to" itself.
        assert URL.from_str("/a/b").is_relative_to("/a/b") is True

    def test_not_a_prefix(self):
        assert URL.from_str("/a/b").is_relative_to("/x") is False

    def test_sibling_not_relative(self):
        assert URL.from_str("/a").is_relative_to("/a/b") is False

    def test_trailing_slash_handled(self):
        # Trailing slash on other should not break prefix detection.
        assert URL.from_str("/a/b/c").is_relative_to("/a/") is True

    def test_partial_segment_is_not_prefix(self):
        # "/ab" should not be considered relative to "/a" (naive string prefix
        # would mistakenly say yes; PurePosixPath-based check says no).
        assert URL.from_str("/abc/d").is_relative_to("/ab") is False

    def test_matching_authority(self):
        a = URL.from_str("https://e.com/a/b")
        assert a.is_relative_to("https://e.com/a") is True

    def test_mismatched_host_rejects(self):
        a = URL.from_str("https://e.com/a/b")
        assert a.is_relative_to("https://other.com/a") is False

    def test_mismatched_scheme_rejects(self):
        a = URL.from_str("https://e.com/a/b")
        assert a.is_relative_to("http://e.com/a") is False

    def test_bare_path_ignores_authority(self):
        # When `other` has no scheme/host, only path is checked.
        a = URL.from_str("https://e.com/a/b")
        assert a.is_relative_to("/a") is True

    def test_string_coerced(self):
        assert URL.from_str("/a/b/c").is_relative_to("/a") is True

    def test_pathlib_path_is_resolved(self, tmp_path):
        # Path instances go through from_pathlib which resolves against
        # CWD and the filesystem. A real file under tmp_path should be
        # relative to tmp_path itself.
        f = tmp_path / "data.csv"
        f.touch()
        assert URL.from_pathlib(f).is_relative_to(tmp_path) is True

    def test_invalid_input_returns_false(self):
        # Unparseable `other` — swallow and return False.
        assert URL.from_str("/a/b").is_relative_to(None) is False


class TestRelativeTo:
    def test_basic(self):
        result = URL.from_str("/a/b/c").relative_to("/a")
        assert result.path == "/b/c"

    def test_exact_match_returns_root(self):
        # self == other yields path "/" (a URL path can't be empty).
        result = URL.from_str("/a/b").relative_to("/a/b")
        assert result.path == "/"

    def test_carries_authority_and_query(self):
        url = URL.from_str("https://user:pw@e.com:8443/a/b/c?x=1#frag")
        result = url.relative_to("/a")
        assert result.scheme == "https"
        assert result.userinfo == "user:pw"
        assert result.host == "e.com"
        assert result.port == 8443
        assert result.query == "x=1"
        assert result.fragment == "frag"
        assert result.path == "/b/c"

    def test_raises_when_not_relative(self):
        with pytest.raises(ValueError, match="not relative to"):
            URL.from_str("/a/b").relative_to("/x")

    def test_raises_for_sibling(self):
        with pytest.raises(ValueError):
            URL.from_str("/a").relative_to("/a/b")

    def test_accepts_url_argument(self):
        base = URL.from_str("/a")
        assert URL.from_str("/a/b/c").relative_to(base).path == "/b/c"

    def test_mismatched_authority_raises(self):
        url = URL.from_str("https://e.com/a/b")
        with pytest.raises(ValueError):
            url.relative_to("https://other.com/a")


# =============================================================================
# authority / is_absolute / is_http
# =============================================================================


class TestDerivedProperties:
    def test_authority_full(self):
        url = URL.from_str("https://u:p@e.com:8443/x")
        assert url.authority == "u:p@e.com:8443"

    def test_authority_no_port(self):
        url = URL.from_str("https://e.com/x")
        assert url.authority == "e.com"

    def test_authority_empty_without_host(self):
        url = URL(scheme="", path="/x")
        assert url.authority == ""

    def test_authority_ipv6(self):
        url = URL(scheme="http", host="::1", port=8080, path="/")
        assert url.authority == "[::1]:8080"

    def test_is_absolute_true(self):
        assert URL.from_str("https://e.com/x").is_absolute is True

    def test_is_absolute_false_no_host(self):
        assert URL.from_str("/x").is_absolute is False

    def test_is_absolute_false_no_scheme(self):
        assert URL(host="e.com", path="/x").is_absolute is False

    def test_is_http(self):
        assert URL.from_str("http://e.com/").is_http is True
        assert URL.from_str("https://e.com/").is_http is True
        assert URL.from_str("ftp://e.com/").is_http is False
        assert URL.from_str("/tmp/x").is_http is False


# =============================================================================
# anonymize
# =============================================================================


class TestAnonymize:
    def test_removes_userinfo(self):
        url = URL.from_str("https://alice:pw@e.com/").anonymize("remove")
        assert url.userinfo is None

    def test_redacts_userinfo(self):
        url = URL.from_str("https://alice:pw@e.com/").anonymize("redact")
        assert url.userinfo == "<redacted>"

    def test_noop_when_no_sensitive_data(self):
        original = URL.from_str("https://e.com/a")
        assert original.anonymize() == original

    def test_idempotent(self):
        url = URL.from_str("https://alice:pw@e.com/").anonymize()
        # Second call should be a no-op (cached via _anonymized).
        assert url.anonymize() is url


# =============================================================================
# empty
# =============================================================================


class TestEmpty:
    def test_singleton_by_default(self):
        a = URL.empty()
        b = URL.empty()
        assert a is b  # same cached instance

    def test_new_instance_flag(self):
        a = URL.empty(new_instance=True)
        b = URL.empty(new_instance=True)
        assert a == b
        assert a is not b


# =============================================================================
# Frozen dataclass behaviour
# =============================================================================


class TestFrozen:
    def test_cannot_set_attribute(self):
        url = URL.from_str("https://e.com/")
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            url.scheme = "http"  # type: ignore[misc]

    def test_hashable(self):
        a = URL.from_str("https://e.com/a")
        b = URL.from_str("https://e.com/a")
        assert hash(a) == hash(b)
        assert {a, b} == {a}  # canonical equality → same hash