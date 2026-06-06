"""Unit tests for the specialized AWS Loki behaviors (mocked boto3)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import yggdrasil.aws.loki  # noqa: F401 — registers the fleet
from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend, detect_aws


class TestAWSCapability(unittest.TestCase):
    def test_detect_aws_from_env(self):
        import os

        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AKIA..."}, clear=False):
            self.assertTrue(detect_aws().available)

    def test_detect_aws_offline_false_without_signals(self):
        import os

        clear = {k: "" for k in ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_REGION",
                                 "AWS_DEFAULT_REGION", "AWS_WEB_IDENTITY_TOKEN_FILE")}
        with patch.dict(os.environ, clear, clear=False):
            for k in clear:
                os.environ.pop(k, None)
            with patch("pathlib.Path.exists", return_value=False):
                self.assertFalse(detect_aws().available)


class TestAWSFleet(unittest.TestCase):
    def _loki(self, client):
        loki = Loki()
        loki._backends = [Backend("aws", available=True)]
        p = patch("yggdrasil.aws.AWSClient")
        AC = p.start()
        self.addCleanup(p.stop)
        AC.current.return_value = client
        return loki

    def test_fleet_registered_and_requires_aws(self):
        names = {b.name for b in Loki().behaviors()}
        for n in ("aws-identity", "aws-s3", "aws-ec2", "aws-lambda", "aws-dynamodb",
                  "aws-iam-users", "aws-sqs", "aws-rds", "aws-secrets"):
            self.assertIn(n, names)
        beh = next(b for b in Loki().behaviors() if b.name == "aws-s3")
        self.assertEqual(beh.requires, "aws")

    def test_unavailable_without_session(self):
        loki = Loki()
        loki._backends = [Backend("aws", available=False)]
        with self.assertRaises(RuntimeError):
            loki.run("aws-s3")

    def test_s3_lists_bucket_names(self):
        client = MagicMock()
        client.client.return_value.list_buckets.return_value = {
            "Buckets": [{"Name": "a"}, {"Name": "b"}], "ResponseMetadata": {}}
        out = self._loki(client).run("aws-s3")
        self.assertEqual(out["items"], ["a", "b"])
        client.client.assert_called_with("s3")

    def test_s3_objects_passes_bucket_param(self):
        client = MagicMock()
        client.client.return_value.list_objects_v2.return_value = {
            "Contents": [{"Key": "k1"}, {"Key": "k2"}]}
        self._loki(client).run("aws-s3-objects", Bucket="my-bucket")
        client.client.return_value.list_objects_v2.assert_called_once_with(Bucket="my-bucket")

    def test_ec2_flattens_nested_reservations(self):
        client = MagicMock()
        client.client.return_value.describe_instances.return_value = {
            "Reservations": [{"Instances": [
                {"InstanceId": "i-1", "InstanceType": "t3.micro", "State": {"Name": "running"}}]}]}
        out = self._loki(client).run("aws-ec2")
        self.assertEqual(out["items"], [{"InstanceId": "i-1", "InstanceType": "t3.micro", "Name": "running"}])

    def test_identity_picks_account_arn_userid(self):
        client = MagicMock()
        client.client.return_value.get_caller_identity.return_value = {
            "Account": "123", "Arn": "arn:aws:iam::123:user/me", "UserId": "AIDA",
            "ResponseMetadata": {}}
        out = self._loki(client).run("aws-identity")
        self.assertEqual(out["items"], [{"Account": "123", "Arn": "arn:aws:iam::123:user/me", "UserId": "AIDA"}])

    def test_dynamodb_passthrough_string_list(self):
        client = MagicMock()
        client.client.return_value.list_tables.return_value = {"TableNames": ["t1", "t2"]}
        self.assertEqual(self._loki(client).run("aws-dynamodb")["items"], ["t1", "t2"])


if __name__ == "__main__":
    unittest.main()
