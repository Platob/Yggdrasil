"""Tests for :mod:`yggdrasil.environ.userinfo`.

Focuses on the lazy-identity contract:

- ``get_user_info()`` only resolves ``hostname`` eagerly; every
  other identity field is a memoized property and stays unset
  until first access.
- Each lazy resolver fires at most once, even when the resolver
  is monkeypatched to a counter.
- Wire round-trip via ``to_struct_dict`` / ``from_struct_dict``
  pre-fills caches on the receiving side, so receivers never
  shell out to ``whoami`` etc.
- ``with_email`` rebuilds the name pair from the new email and
  preserves other already-resolved fields.
"""
from __future__ import annotations

import pytest

import yggdrasil.environ.userinfo as userinfo_mod
from yggdrasil.environ.userinfo import (
    USERINFO_SCHEMA,
    USERINFO_STRUCT,
    UserInfo,
    get_user_info,
)


@pytest.fixture(autouse=True)
def _clear_userinfo_cache():
    """Drop the per-class Singleton cache between tests."""
    UserInfo._INSTANCES.clear()
    yield
    UserInfo._INSTANCES.clear()


# ── construction & lazy contract ─────────────────────────────────────────────


class TestLazyResolution:
    def test_get_user_info_only_resolves_hostname(self, monkeypatch):
        # Sentinel resolvers — calling any of them bumps the counter,
        # which lets us assert that ``get_user_info()`` itself does
        # *not* trigger them. Only ``hostname`` should fire.
        calls = {"key": 0, "email": 0, "project": 0, "cwd": 0}

        def boom_key():
            calls["key"] += 1
            return "resolved-key"

        def boom_email():
            calls["email"] += 1
            return "resolved@example.com"

        def boom_project(_cwd: str):
            calls["project"] += 1
            return "ygg-test", "9.9.9"

        def boom_cwd():
            calls["cwd"] += 1
            return "/tmp"

        monkeypatch.setattr(userinfo_mod, "_resolve_key", boom_key)
        monkeypatch.setattr(userinfo_mod, "_resolve_email", boom_email)
        monkeypatch.setattr(userinfo_mod, "_infer_project", boom_project)
        monkeypatch.setattr(userinfo_mod, "_safe_getcwd", boom_cwd)
        monkeypatch.setattr(userinfo_mod.socket, "gethostname", lambda: "host-x")

        info = get_user_info()

        assert info.hostname == "host-x"
        assert calls == {"key": 0, "email": 0, "project": 0, "cwd": 0}
        assert info._key is ...
        assert info._email is ...
        assert info._product is ...
        assert info._product_version is ...
        assert info._cwd_cache is ...

    def test_singleton_is_returned_until_refresh(self, monkeypatch):
        monkeypatch.setattr(userinfo_mod.socket, "gethostname", lambda: "host-x")
        a = get_user_info()
        b = get_user_info()
        assert a is b

        c = get_user_info(refresh=True)
        assert c is not a

    def test_lazy_fields_resolve_on_first_access(self, monkeypatch):
        calls = {"key": 0, "email": 0, "project": 0}
        monkeypatch.setattr(userinfo_mod.socket, "gethostname", lambda: "h")
        monkeypatch.setattr(userinfo_mod, "_resolve_key", lambda: (calls.__setitem__("key", calls["key"] + 1) or "user-1"))
        monkeypatch.setattr(userinfo_mod, "_resolve_email", lambda: (calls.__setitem__("email", calls["email"] + 1) or "first.last@example.com"))
        monkeypatch.setattr(userinfo_mod, "_infer_project", lambda _c: (calls.__setitem__("project", calls["project"] + 1) or ("proj", "1.2.3")))
        monkeypatch.setattr(userinfo_mod, "_safe_getcwd", lambda: "/tmp/x")

        info = get_user_info()

        assert info.key == "user-1"
        assert info.key == "user-1"  # second access — no extra resolve
        assert calls["key"] == 1

        assert info.email == "first.last@example.com"
        assert calls["email"] == 1
        # Names derive from email; resolving them must not call the
        # email resolver a second time.
        assert info.first_name == "First"
        assert info.last_name == "Last"
        assert calls["email"] == 1

        # First access of either project field resolves both slots
        # in a single call to ``_infer_project``.
        assert info.product == "proj"
        assert info.product_version == "1.2.3"
        assert calls["project"] == 1


# ── persistence (wire round-trip) ─────────────────────────────────────────────


class TestStructRoundTrip:
    def test_struct_dict_keys_match_schema(self):
        info = UserInfo(hostname="h", _key="u", _email=None)
        sd = info.to_struct_dict()
        assert set(sd) == {f.name for f in USERINFO_SCHEMA}

    def test_from_struct_dict_prefills_caches(self, monkeypatch):
        # Resolver explosions guarantee the round-trip never falls
        # back to the live machine for an already-known field.
        for name in ("_resolve_key", "_resolve_email"):
            monkeypatch.setattr(userinfo_mod, name, lambda: pytest.fail(f"{name} should not fire"))
        monkeypatch.setattr(
            userinfo_mod,
            "_infer_project",
            lambda _c: pytest.fail("_infer_project should not fire"),
        )

        payload = {
            "hash": 0,
            "key": "user-1",
            "hostname": "host-1",
            "email": "first.last@example.com",
            "first_name": "First",
            "last_name": "Last",
            "product": "ygg",
            "product_version": "1.0.0",
        }
        info = UserInfo.from_struct_dict(payload)

        assert info._key == "user-1"
        assert info._email == "first.last@example.com"
        assert info._first_name == "First"
        assert info._last_name == "Last"
        assert info._product == "ygg"
        assert info._product_version == "1.0.0"

        # Read every property — none of the resolvers should fire.
        assert info.key == "user-1"
        assert info.email == "first.last@example.com"
        assert info.first_name == "First"
        assert info.last_name == "Last"
        assert info.product == "ygg"
        assert info.product_version == "1.0.0"

    def test_round_trip_preserves_all_wire_fields(self):
        original = UserInfo(
            hostname="h",
            _key="u",
            _email="a@b.com",
            _first_name="A",
            _last_name="B",
            _product="ygg",
            _product_version="1.0",
        )
        rebuilt = UserInfo.from_struct_dict(original.to_struct_dict())
        assert rebuilt.to_struct_dict() == original.to_struct_dict()

    def test_struct_dict_force_resolves_lazy_fields(self, monkeypatch):
        monkeypatch.setattr(userinfo_mod.socket, "gethostname", lambda: "h")
        monkeypatch.setattr(userinfo_mod, "_resolve_key", lambda: "u")
        monkeypatch.setattr(userinfo_mod, "_resolve_email", lambda: "a@b.com")
        monkeypatch.setattr(userinfo_mod, "_infer_project", lambda _c: ("ygg", "1.0"))
        monkeypatch.setattr(userinfo_mod, "_safe_getcwd", lambda: "/tmp")

        info = get_user_info()
        sd = info.to_struct_dict()
        # After serialization, every lazy slot is now pinned.
        assert info._key == "u"
        assert info._email == "a@b.com"
        assert info._first_name is None  # local "u" doesn't parse to a name
        assert info._product == "ygg"
        assert sd["key"] == "u"
        assert sd["email"] == "a@b.com"


# ── struct shape ──────────────────────────────────────────────────────────────


class TestSchema:
    def test_struct_matches_schema_fields(self):
        assert [f.name for f in USERINFO_STRUCT] == [
            f.name for f in USERINFO_SCHEMA
        ]


# ── with_email ────────────────────────────────────────────────────────────────


class TestWithEmail:
    def test_replaces_email_and_rebuilds_names(self):
        info = UserInfo(hostname="h", _key="u", _email="old@example.com")
        new = info.with_email("first.last@example.com")
        assert new.email == "first.last@example.com"
        assert new.first_name == "First"
        assert new.last_name == "Last"
        # Original instance is not mutated.
        assert info.email == "old@example.com"

    def test_preserves_other_resolved_fields(self):
        info = UserInfo(
            hostname="h",
            _key="u",
            _email="old@example.com",
            _product="ygg",
            _product_version="1.0",
        )
        new = info.with_email("a.b@x.com")
        assert new._key == "u"
        assert new._product == "ygg"
        assert new._product_version == "1.0"

    def test_with_email_none_clears_names(self):
        info = UserInfo(hostname="h", _email="a@b.com", _first_name="A", _last_name="B")
        new = info.with_email(None)
        assert new.email is None
        assert new.first_name is None
        assert new.last_name is None


# ── singleton wiring ──────────────────────────────────────────────────────────


class TestSingleton:
    """``UserInfo`` is a :class:`Singleton` keyed on the full wire
    identity; in-process pickle collapses to the live instance,
    cross-process pickle rebuilds with transient slots reset."""

    def test_same_identity_returns_same_instance(self):
        a = UserInfo(hostname="h", _key="u", _email="a@b.com")
        b = UserInfo(hostname="h", _key="u", _email="a@b.com")
        assert a is b

    def test_different_identity_returns_different_instance(self):
        a = UserInfo(hostname="h", _key="u", _email="a@b.com")
        b = UserInfo(hostname="h", _key="u", _email="other@b.com")
        assert a is not b

    def test_inprocess_pickle_preserves_singleton(self):
        import pickle

        info = UserInfo(hostname="h", _key="u", _email="a@b.com")
        restored = pickle.loads(pickle.dumps(info))
        assert restored is info

    def test_crossprocess_pickle_resets_transient(self, monkeypatch):
        import pickle

        monkeypatch.setattr(userinfo_mod, "_safe_getcwd", lambda: "/tmp")
        info = UserInfo(hostname="h", _key="u", _email="a@b.com")
        # Force the transient caches to resolve before pickling so we
        # can verify they're cleared on the receiver side.
        _ = info.cwd
        assert info._cwd_cache == "/tmp"

        blob = pickle.dumps(info)
        UserInfo._INSTANCES.clear()
        fresh = pickle.loads(blob)
        assert fresh is not info
        assert fresh.hostname == "h"
        # Per-process derived caches are wiped by ``__setstate__`` so
        # the receiver re-derives them from its own filesystem.
        assert fresh._cwd_cache is ...
        assert fresh._url_cache is ...
        assert fresh._git_url_cache is ...
        # Wire fields survived.
        assert fresh._email == "a@b.com"

    def test_invalidate_singleton_drops_entry(self):
        info = UserInfo(hostname="h", _key="u", _email="a@b.com")
        info.invalidate_singleton()
        again = UserInfo(hostname="h", _key="u", _email="a@b.com")
        assert again is not info


# ── identity grabbers (Databricks notebook / AWS Lambda / AWS Batch) ─────────


class TestIdentityGrabbers:
    """``_resolve_key`` / ``_resolve_email`` / ``_current_compute_url``
    pull identity from the active managed-runtime env before falling
    back to ``whoami``."""

    _ENV_KEYS = (
        "DATABRICKS_RUNTIME_VERSION",
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_LAMBDA_FUNCTION_VERSION",
        "AWS_BATCH_JOB_ID",
        "AWS_BATCH_JQ_NAME",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    )

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for key in self._ENV_KEYS:
            monkeypatch.delenv(key, raising=False)

    def test_databricks_notebook_user_overrides_whoami(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setattr(
            userinfo_mod, "_ctx_tags",
            lambda: {"user": "alice@example.com"},
        )
        # SDK call must NOT fire when the notebook tag is present.
        monkeypatch.setattr(
            userinfo_mod, "_get_dbx_user",
            lambda: pytest.fail("SDK call should be skipped"),
        )
        monkeypatch.setattr(
            userinfo_mod, "_run_quiet",
            lambda *_: pytest.fail("whoami should be skipped"),
        )

        assert userinfo_mod._resolve_key() == "alice@example.com"
        assert userinfo_mod._resolve_email() == "alice@example.com"

    def test_databricks_falls_back_to_sdk_when_tag_missing(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
        monkeypatch.setattr(userinfo_mod, "_ctx_tags", lambda: {})
        monkeypatch.setattr(
            userinfo_mod, "_get_dbx_user",
            lambda: "bob@example.com",
        )
        assert userinfo_mod._resolve_key() == "bob@example.com"

    def test_aws_lambda_key_uses_function_name(self, monkeypatch):
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "process-events")
        # ``whoami`` on Lambda returns sandbox noise — make sure we
        # short-circuit *before* shelling out.
        monkeypatch.setattr(
            userinfo_mod, "_run_quiet",
            lambda *_: pytest.fail("whoami should be skipped"),
        )
        assert userinfo_mod._resolve_key() == "lambda:process-events"

    def test_aws_batch_key_uses_job_id(self, monkeypatch):
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "job-abc-123")
        monkeypatch.setattr(
            userinfo_mod, "_run_quiet",
            lambda *_: pytest.fail("whoami should be skipped"),
        )
        assert userinfo_mod._resolve_key() == "batch:job-abc-123"

    def test_lambda_url_points_at_console(self, monkeypatch):
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "process-events")
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        url = userinfo_mod._aws_current_url()
        assert url is not None
        s = url.to_string()
        assert "console.aws.amazon.com/lambda" in s
        assert "region=us-west-2" in s
        assert "process-events" in s

    def test_batch_url_points_at_console(self, monkeypatch):
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "job-abc-123")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        url = userinfo_mod._aws_current_url()
        assert url is not None
        s = url.to_string()
        assert "console.aws.amazon.com/batch" in s
        assert "region=eu-west-1" in s
        assert "job-abc-123" in s

    def test_aws_url_absent_when_no_managed_runtime(self):
        assert userinfo_mod._aws_current_url() is None

    def test_lambda_identity_dict(self, monkeypatch):
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "f")
        monkeypatch.setenv("AWS_LAMBDA_FUNCTION_VERSION", "5")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        info = userinfo_mod._aws_lambda_identity()
        assert info == {
            "function_name": "f",
            "function_version": "5",
            "region": "us-east-1",
        }

    def test_batch_identity_dict(self, monkeypatch):
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "j")
        monkeypatch.setenv("AWS_BATCH_JQ_NAME", "q")
        info = userinfo_mod._aws_batch_identity()
        assert info == {"job_id": "j", "job_queue": "q"}
