"""AWSRegion enum + region-from-bucket-name inference + S3 wiring."""
from __future__ import annotations

import types

import pytest

from yggdrasil.enums import AWSRegion


class TestEnum:
    def test_is_a_plain_string(self):
        assert AWSRegion.EU_CENTRAL_1 == "eu-central-1"
        assert f"{AWSRegion.EU_CENTRAL_1}" == "eu-central-1"
        assert str(AWSRegion.EU_CENTRAL_1) == "eu-central-1"

    def test_partition(self):
        assert AWSRegion.US_EAST_1.partition == "aws"
        assert AWSRegion.US_GOV_WEST_1.partition == "aws-us-gov"
        assert AWSRegion.CN_NORTH_1.partition == "aws-cn"

    def test_from_(self):
        assert AWSRegion.from_("us-west-2") is AWSRegion.US_WEST_2
        assert AWSRegion.from_(AWSRegion.US_WEST_2) is AWSRegion.US_WEST_2
        assert AWSRegion.from_("US-WEST-2") is AWSRegion.US_WEST_2
        assert AWSRegion.from_("nope", default=None) is None
        assert AWSRegion.from_(None, default=None) is None
        with pytest.raises(ValueError):
            AWSRegion.from_("nope")


class TestFromText:
    @pytest.mark.parametrize("text,expected", [
        ("acme-dls3-eu-central-1-p", AWSRegion.EU_CENTRAL_1),   # the motivating case
        ("us-east-1-logs", AWSRegion.US_EAST_1),
        ("prod.eu-west-1.data", AWSRegion.EU_WEST_1),
        ("x-ap-southeast-1", AWSRegion.AP_SOUTHEAST_1),         # longest match
        ("x-ap-south-1-y", AWSRegion.AP_SOUTH_1),
        ("EU-WEST-2-CAPS", AWSRegion.EU_WEST_2),                # case-insensitive
        ("co-cn-north-1-x", AWSRegion.CN_NORTH_1),
    ])
    def test_finds_region(self, text, expected):
        assert AWSRegion.from_text(text) is expected

    @pytest.mark.parametrize("text", [
        "my-data", "", None, "bad-eu-central-12", "useast1nodash", "x-eu-west-1foo",
    ])
    def test_no_false_positive(self, text):
        assert AWSRegion.from_text(text) is None

    def test_default(self):
        assert AWSRegion.from_text("none-here", default=AWSRegion.US_EAST_1) is AWSRegion.US_EAST_1


class TestS3RegionFromBucketName:
    def _bucket(self, name):
        from tests.test_yggdrasil.test_aws._fake_s3 import reset_s3_singletons
        from yggdrasil.aws.fs.path import S3Bucket

        reset_s3_singletons()
        b = S3Bucket(bucket=name)
        frozen = types.SimpleNamespace(access_key="AK", secret_key="SK", token=None)
        client = types.SimpleNamespace(
            region=None, endpoint_url=None,
            session=types.SimpleNamespace(get_credentials=lambda: types.SimpleNamespace(get_frozen_credentials=lambda: frozen)),
        )
        b._service = types.SimpleNamespace(client=client)
        return b

    def test_sniffs_region_from_bucket(self):
        b = self._bucket("acme-dls3-eu-central-1-p")
        assert b.http.signer.region == "eu-central-1"
        assert "eu-central-1.amazonaws.com" in str(b.http.endpoint)

    def test_defaults_when_unsniffable(self):
        b = self._bucket("plain-bucket")
        assert b.http.signer.region == "us-east-1"

    def teardown_method(self):
        from tests.test_yggdrasil.test_aws._fake_s3 import reset_s3_singletons
        reset_s3_singletons()
