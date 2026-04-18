"""Per-request cache config fields — `local_cache_config`, `remote_cache_config`.

These fields let callers override the session-level cache behavior on a single
request (e.g. force UPSERT for one endpoint while everything else reads from
cache). They're `compare=False` so they don't affect `__eq__`, and they must
survive `copy()` and `anonymize()` because the session consults them *after*
the anonymized copy is created.
"""

from __future__ import annotations

from yggdrasil.io import SaveMode
from yggdrasil.io.send_config import CacheConfig

from .._helpers import make_request


def test_both_fields_default_to_none() -> None:
    req = make_request()

    assert req.local_cache_config is None
    assert req.remote_cache_config is None


def test_fields_can_be_assigned_after_construction() -> None:
    req = make_request()
    cfg = CacheConfig(mode=SaveMode.UPSERT)

    req.local_cache_config = cfg
    req.remote_cache_config = cfg

    assert req.local_cache_config is cfg
    assert req.remote_cache_config is cfg


def test_copy_preserves_both_fields() -> None:
    local_cfg = CacheConfig()
    remote_cfg = CacheConfig(mode=SaveMode.UPSERT)
    req = make_request()
    req.local_cache_config = local_cfg
    req.remote_cache_config = remote_cfg

    copied = req.copy()

    assert copied.local_cache_config is local_cfg
    assert copied.remote_cache_config is remote_cfg


def test_copy_local_cache_config_override_is_applied() -> None:
    original = CacheConfig()
    new = CacheConfig(mode=SaveMode.UPSERT)
    req = make_request()
    req.local_cache_config = original

    copied = req.copy(local_cache_config=new)

    assert copied.local_cache_config is new


def test_copy_can_clear_cache_configs() -> None:
    req = make_request()
    req.local_cache_config = CacheConfig()
    req.remote_cache_config = CacheConfig()

    copied = req.copy(local_cache_config=None, remote_cache_config=None)

    assert copied.local_cache_config is None
    assert copied.remote_cache_config is None


def test_cache_configs_excluded_from_equality() -> None:
    """`compare=False` — two requests with different cache configs still equal."""
    req_a = make_request()
    req_b = make_request()
    req_b.local_cache_config = CacheConfig()

    assert req_a == req_b


def test_anonymize_preserves_cache_configs() -> None:
    """The session pulls effective configs off the *anonymized* request."""
    cfg = CacheConfig()
    req = make_request()
    req.local_cache_config = cfg
    req.remote_cache_config = cfg

    anon = req.anonymize(mode="remove")

    assert anon.local_cache_config is cfg
    assert anon.remote_cache_config is cfg
