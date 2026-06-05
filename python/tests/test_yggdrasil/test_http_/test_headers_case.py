"""Case-insensitive header lookups on :class:`HTTPHeaders`.

HTTP field names are case-insensitive (RFC 7230 §3.2). HTTP/2 origins lowercase
every name, so a wire ``location`` / ``retry-after`` must still resolve under a
``"Location"`` / ``"Retry-After"`` query — the redirect and 429-backoff paths
depend on it. The store keeps the original case for serialization.
"""
from __future__ import annotations

from yggdrasil.http_.headers import HTTPHeaders


def test_get_is_case_insensitive():
    h = HTTPHeaders({"location": "/login.html", "content-length": "0"})
    assert h.get("Location") == "/login.html"
    assert h.get("LOCATION") == "/login.html"
    assert h["Content-Length"] == "0"
    assert h.get("missing") is None


def test_contains_is_case_insensitive():
    h = HTTPHeaders({"retry-after": "0"})
    assert "Retry-After" in h
    assert "RETRY-AFTER" in h
    assert "x-absent" not in h


def test_delete_is_case_insensitive():
    h = HTTPHeaders({"X-Token": "abc"})
    del h["x-token"]
    assert "X-Token" not in h
    assert len(h) == 0


def test_original_case_preserved_in_iteration():
    # Lookups normalise case; serialization/iteration keep what the origin sent.
    h = HTTPHeaders({"location": "/x", "X-Custom": "y"})
    assert set(h.keys()) == {"location", "X-Custom"}
    assert h.get("Location") == "/x"


def test_exact_hit_still_works_after_fix():
    h = HTTPHeaders({"Content-Type": "application/json"})
    assert h["Content-Type"] == "application/json"
    assert h.get("content-type") == "application/json"


def test_set_updates_existing_slot_without_duplicating():
    # Writing a different-cased name must update the existing header in place,
    # not add a case-variant sibling (the source of the duplicate
    # content-length seen on HTTP/2 responses).
    h = HTTPHeaders({"content-length": "0"})
    h["Content-Length"] = "5"
    assert len(h) == 1
    assert h["content-length"] == "5"
    assert list(h.keys()) == ["content-length"]  # stored case preserved


def test_setdefault_sees_existing_case_insensitively():
    h = HTTPHeaders({"user-agent": "curl/8"})
    h.setdefault("User-Agent", "browser")  # already present → no-op, no dup
    assert len(h) == 1
    assert h["User-Agent"] == "curl/8"


def test_update_dedupes_case_insensitively():
    h = HTTPHeaders({"location": "/a"})
    h.update({"Location": "/b", "X-New": "1"})
    assert h["location"] == "/b"
    assert len([k for k in h if k.lower() == "location"]) == 1
    assert h["x-new"] == "1"


def test_pop_is_case_insensitive():
    h = HTTPHeaders({"Retry-After": "3"})
    assert h.pop("retry-after") == "3"
    assert "Retry-After" not in h


def test_delete_missing_raises_keyerror():
    h = HTTPHeaders({"X-A": "1"})
    import pytest
    with pytest.raises(KeyError):
        del h["x-absent"]
