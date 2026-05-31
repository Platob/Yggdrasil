"""AWS Console deep-link builders.

``explore_url`` on the AWS client / account / S3 resources returns one of
these — a clickable Console URL so you can jump from a Python repl / notebook
straight to the bucket, object, or account home in the browser. Partition-aware
(commercial ``aws``, GovCloud ``aws-us-gov``, China ``aws-cn``) since the
Console lives on a different domain in each.
"""
from __future__ import annotations

from typing import Optional

from yggdrasil.url import URL

__all__ = ["account_console_url", "s3_bucket_url", "s3_object_url", "batch_job_url", "partition_for_region"]


def partition_for_region(region: Optional[str]) -> str:
    if region and region.startswith("cn-"):
        return "aws-cn"
    if region and region.startswith("us-gov-"):
        return "aws-us-gov"
    return "aws"


def _console_host(region: Optional[str]) -> str:
    part = partition_for_region(region)
    if part == "aws-cn":
        return "console.amazonaws.cn"
    if part == "aws-us-gov":
        return "console.amazonaws-us-gov.com"
    # Commercial: the regional subdomain pins the Console to the right region.
    return f"{region}.console.aws.amazon.com" if region else "console.aws.amazon.com"


def _s3_console_host(region: Optional[str]) -> str:
    part = partition_for_region(region)
    if part == "aws-cn":
        return "console.amazonaws.cn"
    if part == "aws-us-gov":
        return "console.amazonaws-us-gov.com"
    return "s3.console.aws.amazon.com"


def account_console_url(region: Optional[str] = None) -> URL:
    """Console home for the account, pinned to *region* when known."""
    url = URL.from_(f"https://{_console_host(region)}/console/home")
    return url.with_query_items({"region": region}) if region else url


def s3_bucket_url(bucket: str, region: Optional[str] = None) -> URL:
    """Console view of an S3 bucket's object list."""
    url = URL.from_(f"https://{_s3_console_host(region)}/s3/buckets/{bucket}")
    return url.with_query_items({"region": region}) if region else url


def batch_job_url(job_id: str, region: Optional[str] = None) -> URL:
    """Console deep-link to an AWS Batch job's detail page."""
    url = URL.from_(f"https://{_console_host(region)}/batch/home")
    if region:
        url = url.with_query_items({"region": region})
    # The Batch console routes the job detail behind a SPA fragment.
    return url.with_fragment(f"jobs/detail/{job_id}")


def s3_object_url(bucket: str, key: str, region: Optional[str] = None) -> URL:
    """Console view focused on an S3 object (or prefix) — the Console uses the
    ``prefix`` query param to scroll to / highlight the key."""
    url = URL.from_(f"https://{_s3_console_host(region)}/s3/buckets/{bucket}")
    items = {"prefix": key} if key else {}
    if region:
        items = {"region": region, **items}
    return url.with_query_items(items) if items else url
