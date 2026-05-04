"""Tests for :class:`AWSConfig.refresher` and :meth:`AWSConfig.from_refresher`."""

from __future__ import annotations

import pytest

from yggdrasil.aws.config import AwsCredentials, AWSConfig


def test_from_refresher_seeds_from_callback():
    calls = []

    def refresher() -> AwsCredentials:
        calls.append(1)
        return AwsCredentials(
            access_key_id="AKIA-FRESH",
            secret_access_key="secret-fresh",
            session_token="token-fresh",
            expiration="2099-01-01T00:00:00Z",
        )

    cfg = AWSConfig.from_refresher(refresher, region="eu-central-1")
    assert cfg.access_key_id == "AKIA-FRESH"
    assert cfg.secret_access_key == "secret-fresh"
    assert cfg.session_token == "token-fresh"
    assert cfg.region == "eu-central-1"
    assert cfg.has_refresher()
    assert len(calls) == 1


def test_from_refresher_accepts_mapping():
    def refresher():
        return {
            "access_key": "K",
            "secret_key": "S",
            "token": "T",
            "expiry_time": "2099-01-01T00:00:00Z",
        }

    cfg = AWSConfig.from_refresher(refresher)
    assert cfg.access_key_id == "K"
    assert cfg.secret_access_key == "S"
    assert cfg.session_token == "T"


def test_refresh_metadata_invokes_callback():
    cfg = AWSConfig.from_refresher(
        lambda: AwsCredentials(
            access_key_id="AK",
            secret_access_key="SK",
            session_token="ST",
            expiration="2099-01-01T00:00:00Z",
        ),
    )
    md = cfg.refresh_metadata()
    assert md["access_key"] == "AK"
    assert md["secret_key"] == "SK"
    assert md["token"] == "ST"
    assert md["expiry_time"] == "2099-01-01T00:00:00Z"


def test_refresh_metadata_without_refresher_raises():
    cfg = AWSConfig(
        access_key_id="x", secret_access_key="y",
    )
    with pytest.raises(RuntimeError):
        cfg.refresh_metadata()


def test_from_credentials_attaches_refresher():
    creds = AwsCredentials(access_key_id="A", secret_access_key="B")
    cfg = AWSConfig.from_credentials(
        creds,
        refresher=lambda: creds,
        region="eu-west-1",
    )
    assert cfg.has_refresher()
    assert cfg.region == "eu-west-1"


def test_refresher_excluded_from_equality():
    a = AWSConfig(access_key_id="K", secret_access_key="S", refresher=lambda: None)
    b = AWSConfig(access_key_id="K", secret_access_key="S", refresher=lambda: None)
    assert a == b


def test_refresher_invalid_return_raises():
    cfg = AWSConfig.from_credentials(
        AwsCredentials(access_key_id="A", secret_access_key="B"),
        refresher=lambda: 42,  # not credentials, not mapping
    )
    with pytest.raises(TypeError):
        cfg.refresh_metadata()
