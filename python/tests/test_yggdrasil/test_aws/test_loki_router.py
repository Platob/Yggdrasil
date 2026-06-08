"""Tests for the AWS NL→skill router (autonomous aws-* dispatch)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from yggdrasil.aws.loki.router import route


class TestAwsRouter(unittest.TestCase):
    def test_common_list_requests(self):
        cases = {
            "list my s3 buckets": "aws-s3",
            "show ec2 instances": "aws-ec2",
            "what lambda functions do I have": "aws-lambda",
            "who am i on aws": "aws-identity",
            "list iam roles": "aws-iam-roles",
            "show the sqs queues": "aws-sqs",
            "list dynamodb tables": "aws-dynamodb",
        }
        for text, skill in cases.items():
            hit = route(text)
            self.assertIsNotNone(hit, text)
            self.assertEqual(hit[0], skill, text)

    def test_s3_objects_extracts_bucket(self):
        self.assertEqual(route("objects in bucket my-data"), ("aws-s3-objects", {"bucket": "my-data"}))
        self.assertEqual(route("list objects in my-logs bucket"),
                         ("aws-s3-objects", {"bucket": "my-logs"}))
        # No bucket named → fall back to listing buckets.
        self.assertEqual(route("show s3 objects"), ("aws-s3", {}))

    def test_non_aws_lines_return_none(self):
        for text in ("hello there", "what is a monad", "users", "refactor the parser"):
            self.assertIsNone(route(text), text)

    def test_plan_routes_to_aws_skill_when_available(self):
        from yggdrasil.loki import Loki

        loki = Loki()
        with patch.object(Loki, "has", lambda self, n: n == "aws"):
            p = loki.plan("list my s3 buckets")
        self.assertEqual((p.action, p.skill), ("skill", "aws-s3"))


if __name__ == "__main__":
    unittest.main()
