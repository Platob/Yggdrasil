"""AWS Console deep-links: builders + explore_url on client / account / S3."""
from __future__ import annotations

import pytest

from yggdrasil.aws import AWSClient, AWSAccount, AccountService
from yggdrasil.aws.console import account_console_url, s3_bucket_url, s3_object_url, partition_for_region


class TestConsoleUrlBuilders:
    def test_account_home_pins_region(self):
        assert str(account_console_url("us-east-1")) == (
            "https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1"
        )

    def test_account_home_no_region(self):
        assert str(account_console_url(None)) == "https://console.aws.amazon.com/console/home"

    def test_partition_detection(self):
        assert partition_for_region("us-east-1") == "aws"
        assert partition_for_region("us-gov-west-1") == "aws-us-gov"
        assert partition_for_region("cn-north-1") == "aws-cn"

    def test_govcloud_and_china_hosts(self):
        assert "amazonaws-us-gov.com" in str(account_console_url("us-gov-west-1"))
        assert "amazonaws.cn" in str(account_console_url("cn-north-1"))

    def test_s3_bucket_url(self):
        assert str(s3_bucket_url("my-bucket", "eu-west-1")) == (
            "https://s3.console.aws.amazon.com/s3/buckets/my-bucket?region=eu-west-1"
        )

    def test_s3_object_url_encodes_key_prefix(self):
        url = str(s3_object_url("my-bucket", "a/b/c.parquet", "eu-west-1"))
        assert url.startswith("https://s3.console.aws.amazon.com/s3/buckets/my-bucket?")
        assert "prefix=a%2Fb%2Fc.parquet" in url
        assert "region=eu-west-1" in url


class TestClientAccount:
    def test_client_explore_url(self):
        c = AWSClient(region="us-west-2", access_key_id="AK", secret_access_key="SK")
        assert str(c.explore_url) == (
            "https://us-west-2.console.aws.amazon.com/console/home?region=us-west-2"
        )

    def test_client_account_resource(self):
        c = AWSClient(region="eu-central-1", access_key_id="AK", secret_access_key="SK")
        acct = c.account
        assert isinstance(acct, AWSAccount)
        assert acct.region == "eu-central-1"
        assert "eu-central-1.console.aws.amazon.com" in str(acct.explore_url)

    def test_account_service_is_sts_flavored(self):
        assert AccountService.service_name() == "sts"

    def test_account_clickable_repr(self):
        c = AWSClient(region="us-east-1", access_key_id="AK", secret_access_key="SK")
        acct = c.account
        assert repr(acct) == f"AWSAccount({acct.explore_url!r})"
        assert acct._repr_html_().startswith('<a href="https://us-east-1.console.aws.amazon.com')


class TestS3ExploreUrl:
    def test_bucket_and_path_links(self):
        from tests.test_yggdrasil.test_aws._fake_s3 import wire_s3_path, FakeS3, reset_s3_singletons

        reset_s3_singletons()
        p = wire_s3_path(FakeS3(), "s3://my-bucket/dir/obj.parquet", bucket="my-bucket")
        assert "s3/buckets/my-bucket" in str(p.s3_bucket.explore_url)
        assert "prefix=dir%2Fobj.parquet" in str(p.explore_url)
        assert p._repr_html_().startswith('<a href="https://s3.console.aws.amazon.com')
        reset_s3_singletons()
