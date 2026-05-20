"""Tests for :class:`TimeWindowPolicy` and its wiring on :class:`SchemaSession`.

Two surfaces:

1. :class:`TestTimeWindowPolicy` — pure unit tests on the policy:
   construction, alias coercion, URL rewriting on different
   granularities / key sets, error shapes.
2. :class:`TestSchemaSessionTimeWindow` — end-to-end through
   :meth:`SchemaSession._attach_cache`: assert the request URL is
   actually rewritten before the cache lookup happens, two
   differently-spelled windows collapse to one ``public_url_hash``,
   and a session built without ``time_window`` is a no-op.

No live Databricks calls — :class:`Schema` is mocked, and the
session's network leg is short-circuited by a stub ``_local_send``.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest

from yggdrasil.databricks.schema.schema import Schema
from yggdrasil.databricks.schema.session import SchemaSession, TimeWindowPolicy
from yggdrasil.databricks.table.table import Table
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import Response
from yggdrasil.io.session import Session
from yggdrasil.io.url import URL


# ---------------------------------------------------------------------------
# Singleton-cache hygiene — every test builds a fresh SchemaSession; without
# the clear, an earlier ``base_url`` / ``time_window`` pair re-resolves to
# the same singleton and the new wiring never runs.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_session_singleton_cache():
    Session._INSTANCES.clear()
    yield
    Session._INSTANCES.clear()


# ===========================================================================
# Policy — construction, coercion, application
# ===========================================================================


class TestTimeWindowPolicy:

    # ── construction ──────────────────────────────────────────────────────

    def test_iso_duration_string_accepted(self):
        p = TimeWindowPolicy(granularity="PT1H")
        assert p.granularity == "PT1H"

    def test_timedelta_accepted(self):
        p = TimeWindowPolicy(granularity=dt.timedelta(minutes=15))
        assert p.granularity == dt.timedelta(minutes=15)

    def test_invalid_granularity_raises_at_construction(self):
        with pytest.raises(ValueError, match="not a parseable interval"):
            TimeWindowPolicy(granularity="not-a-duration")

    def test_empty_key_sets_rejected(self):
        with pytest.raises(ValueError, match="at least one of start_keys"):
            TimeWindowPolicy(granularity="PT1H", start_keys=(), end_keys=())

    def test_frozen_dataclass_is_hashable(self):
        a = TimeWindowPolicy(granularity="PT1H")
        b = TimeWindowPolicy(granularity="PT1H")
        # Hashability is what lets the policy participate in the
        # SchemaSession singleton key.
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    # ── from_ coercion ────────────────────────────────────────────────────

    def test_from_passes_through_none(self):
        assert TimeWindowPolicy.from_(None) is None

    def test_from_passes_through_policy_identity(self):
        p = TimeWindowPolicy(granularity="PT1H")
        assert TimeWindowPolicy.from_(p) is p

    def test_from_promotes_iso_string(self):
        out = TimeWindowPolicy.from_("P1D")
        assert isinstance(out, TimeWindowPolicy)
        assert out.granularity == "P1D"

    def test_from_promotes_timedelta(self):
        out = TimeWindowPolicy.from_(dt.timedelta(hours=4))
        assert isinstance(out, TimeWindowPolicy)
        assert out.granularity == dt.timedelta(hours=4)

    def test_from_rejects_other_types(self):
        with pytest.raises(TypeError, match="cannot coerce"):
            TimeWindowPolicy.from_(3600)  # type: ignore[arg-type]

    # ── apply: snap behaviour ─────────────────────────────────────────────

    def test_hourly_snap_with_expand(self):
        p = TimeWindowPolicy(granularity="PT1H")
        url = URL.from_(
            "https://api.example.com/v1/prices"
            "?symbol=AAPL"
            "&start=2026-05-20T10:23:14Z"
            "&end=2026-05-20T10:38:02Z"
        )
        snapped = p.apply(url)

        items = dict(snapped.query_items())
        assert items["start"] == "2026-05-20T10:00:00Z"
        assert items["end"] == "2026-05-20T11:00:00Z"
        # Untouched params survive untouched.
        assert items["symbol"] == "AAPL"

    def test_overlapping_windows_collapse_to_one_url(self):
        """The core cache-key promise: two near-overlap windows on the
        same hour grid produce byte-identical URLs."""
        p = TimeWindowPolicy(granularity="PT1H")
        a = p.apply(URL.from_(
            "https://api.example.com/v1/prices"
            "?start=2026-05-20T10:23:14Z&end=2026-05-20T10:38:02Z"
        ))
        b = p.apply(URL.from_(
            "https://api.example.com/v1/prices"
            "?start=2026-05-20T10:11:05Z&end=2026-05-20T10:55:39Z"
        ))
        assert a.to_string() == b.to_string()

    def test_daily_snap_with_calendar_granularity(self):
        p = TimeWindowPolicy(granularity="P1D")
        snapped = p.apply(URL.from_(
            "https://api.example.com/v1/x"
            "?start=2026-05-20T14:00:00Z&end=2026-05-22T03:00:00Z"
        ))
        items = dict(snapped.query_items())
        assert items["start"] == "2026-05-20T00:00:00Z"
        # Expand=True ceils up — 22T03:00 is not a midnight, snap to 23T00:00.
        assert items["end"] == "2026-05-23T00:00:00Z"

    def test_expand_false_floors_both_ends(self):
        p = TimeWindowPolicy(granularity="PT1H", expand=False)
        snapped = p.apply(URL.from_(
            "https://api.example.com/v1/x"
            "?start=2026-05-20T10:23Z&end=2026-05-20T10:38Z"
        ))
        items = dict(snapped.query_items())
        assert items["start"] == "2026-05-20T10:00:00Z"
        # expand=False: end floors too (tighter cache key, may miss tail).
        assert items["end"] == "2026-05-20T10:00:00Z"

    def test_already_aligned_returns_input_identity(self):
        """Aligned URLs short-circuit — saves a URL rebuild on hot paths."""
        p = TimeWindowPolicy(granularity="PT1H")
        url = URL.from_(
            "https://api.example.com/v1/x"
            "?start=2026-05-20T10:00:00Z&end=2026-05-20T11:00:00Z"
        )
        assert p.apply(url) is url

    def test_url_without_query_is_passthrough(self):
        p = TimeWindowPolicy(granularity="PT1H")
        url = URL.from_("https://api.example.com/v1/x")
        assert p.apply(url) is url

    def test_url_without_matching_keys_is_passthrough(self):
        p = TimeWindowPolicy(granularity="PT1H")
        url = URL.from_("https://api.example.com/v1/x?foo=bar&baz=qux")
        assert p.apply(url) is url

    def test_local_timezone_input_normalised_to_utc(self):
        """``10:23+02:00`` and ``08:23Z`` are the same instant — they
        must produce the same cache key. The ``+`` is form-encoding's
        space character; callers send the offset as ``%2B`` (or use
        ``-`` for west-of-UTC) when they really mean a timezone sign."""
        p = TimeWindowPolicy(granularity="PT1H")
        local = p.apply(URL.from_(
            "https://api.example.com/v1/x?start=2026-05-20T10:23:00%2B02:00"
        ))
        utc = p.apply(URL.from_(
            "https://api.example.com/v1/x?start=2026-05-20T08:23:00Z"
        ))
        assert local.to_string() == utc.to_string()
        assert dict(local.query_items())["start"] == "2026-05-20T08:00:00Z"

    def test_custom_key_sets(self):
        p = TimeWindowPolicy(
            granularity="PT1H",
            start_keys=("dateFrom",),
            end_keys=("dateTo",),
        )
        snapped = p.apply(URL.from_(
            "https://api.example.com/v1/x"
            "?dateFrom=2026-05-20T10:23Z&dateTo=2026-05-20T10:38Z"
            "&start=2026-05-20T10:23Z"  # not in start_keys → untouched
        ))
        items = dict(snapped.query_items())
        assert items["dateFrom"] == "2026-05-20T10:00:00Z"
        assert items["dateTo"] == "2026-05-20T11:00:00Z"
        assert items["start"] == "2026-05-20T10:23Z"

    def test_custom_output_format(self):
        p = TimeWindowPolicy(
            granularity="P1D",
            output_format="%Y-%m-%d",
        )
        snapped = p.apply(URL.from_(
            "https://api.example.com/v1/x?start=2026-05-20T14:00:00Z"
        ))
        assert dict(snapped.query_items())["start"] == "2026-05-20"

    # ── apply: error shapes ───────────────────────────────────────────────

    def test_unparseable_value_raises_with_context(self):
        p = TimeWindowPolicy(granularity="PT1H")
        with pytest.raises(ValueError) as exc:
            p.apply(URL.from_("https://api.example.com/v1/x?start=not-a-date"))
        msg = str(exc.value)
        # Error must answer: what you passed, where, what to try next.
        assert "'start'" in msg
        assert "'not-a-date'" in msg
        assert "ISO 8601" in msg

    def test_empty_value_passes_through(self):
        """``?start=`` (blank) is left alone — there's nothing to snap."""
        p = TimeWindowPolicy(granularity="PT1H")
        snapped = p.apply(URL.from_("https://api.example.com/v1/x?start=&end=2026-05-20T10:38Z"))
        items = dict(snapped.query_items())
        assert items["start"] == ""
        assert items["end"] == "2026-05-20T11:00:00Z"


# ===========================================================================
# SchemaSession integration — _attach_cache rewrites the request URL
# ===========================================================================


class _StubSchemaSession(SchemaSession):
    """SchemaSession with the network leg short-circuited so we can
    poke ``_attach_cache`` end-to-end without touching Databricks or
    a real HTTP server."""

    def __init__(self, *args, **kwargs):
        already = getattr(self, "_initialized", False)
        super().__init__(*args, **kwargs)
        if not already:
            self.sent_urls: list[str] = []

    def _local_send(self, request, config):  # type: ignore[override]
        self.sent_urls.append(request.url.to_string())
        return Response(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json"},
            tags={},
            buffer=b"{}",
            received_at=dt.datetime.fromtimestamp(0, tz=dt.timezone.utc),
        )

    def _load_local_cached_response(self, request, cfg):  # type: ignore[override]
        return None  # force the network leg on every call

    def _load_remote_cached_response(self, request, cfg, spark_session=None):  # type: ignore[override]
        return None

    def _store_local_cached_response(self, response, cfg, root=None):  # type: ignore[override]
        return None

    def _persist_remote(self, *args, **kwargs):  # type: ignore[override]
        return None


def _mock_schema() -> Schema:
    schema = MagicMock(spec=Schema)
    # `_attach_cache` builds a CacheConfig with ``tabular=self.schema.table(name)``;
    # the Table is never executed in these tests, so a bare MagicMock is enough.
    schema.table.side_effect = lambda name: MagicMock(spec=Table)
    schema.full_name.return_value = "main.test"
    return schema


def _request(url: str) -> PreparedRequest:
    return PreparedRequest.prepare("GET", url)


class TestSchemaSessionTimeWindow:

    def test_no_policy_leaves_url_untouched(self):
        session = _StubSchemaSession(
            _mock_schema(),
            base_url="https://api.example.com#nopolicy",
            local_cache=False,
        )
        req = _request("https://api.example.com/v1/x?start=2026-05-20T10:23Z&end=2026-05-20T10:38Z")
        session._attach_cache(req)
        assert dict(req.url.query_items())["start"] == "2026-05-20T10:23Z"

    def test_policy_rewrites_request_url_before_cache_attach(self):
        session = _StubSchemaSession(
            _mock_schema(),
            base_url="https://api.example.com#policy",
            local_cache=False,
            time_window="PT1H",
        )
        req = _request(
            "https://api.example.com/v1/x"
            "?start=2026-05-20T10:23:14Z&end=2026-05-20T10:38:02Z"
        )
        session._attach_cache(req)

        items = dict(req.url.query_items())
        assert items["start"] == "2026-05-20T10:00:00Z"
        assert items["end"] == "2026-05-20T11:00:00Z"

        # The cache config built off this normalised URL must reference
        # the same path-derived table — the policy never touches the path.
        assert req.remote_cache_config is not None

    def test_two_overlapping_windows_share_public_url_hash(self):
        """The point of the whole exercise: snapped → same cache key."""
        session = _StubSchemaSession(
            _mock_schema(),
            base_url="https://api.example.com#sharekey",
            local_cache=False,
            time_window="PT1H",
        )
        a = _request(
            "https://api.example.com/v1/prices"
            "?start=2026-05-20T10:23:14Z&end=2026-05-20T10:38:02Z"
        )
        b = _request(
            "https://api.example.com/v1/prices"
            "?start=2026-05-20T10:11:05Z&end=2026-05-20T10:55:39Z"
        )
        session._attach_cache(a)
        session._attach_cache(b)
        assert a.public_url_hash == b.public_url_hash

    def test_send_pipeline_uses_normalised_url(self):
        """End-to-end: ``send`` walks through ``_attach_cache`` first;
        ``_local_send`` (the network leg) must receive the snapped URL."""
        session = _StubSchemaSession(
            _mock_schema(),
            base_url="https://api.example.com#send",
            local_cache=False,
            time_window=TimeWindowPolicy(granularity="PT1H"),
        )
        session.send(_request(
            "https://api.example.com/v1/x"
            "?start=2026-05-20T10:23Z&end=2026-05-20T10:38Z"
        ))
        assert len(session.sent_urls) == 1
        assert "start=2026-05-20T10%3A00%3A00Z" in session.sent_urls[0]
        assert "end=2026-05-20T11%3A00%3A00Z" in session.sent_urls[0]

    def test_time_window_participates_in_singleton_identity(self):
        """Two sessions with the same base_url but different policies
        are *different* singletons — otherwise the second caller would
        unwittingly reuse the first's URL rewrite."""
        schema = _mock_schema()
        s_hourly = _StubSchemaSession(
            schema,
            base_url="https://api.example.com#identity",
            local_cache=False,
            time_window="PT1H",
        )
        s_daily = _StubSchemaSession(
            schema,
            base_url="https://api.example.com#identity",
            local_cache=False,
            time_window="P1D",
        )
        assert s_hourly is not s_daily

    def test_constructor_accepts_timedelta_shortcut(self):
        session = _StubSchemaSession(
            _mock_schema(),
            base_url="https://api.example.com#td",
            local_cache=False,
            time_window=dt.timedelta(hours=1),
        )
        assert isinstance(session.time_window, TimeWindowPolicy)
        assert session.time_window.granularity == dt.timedelta(hours=1)
