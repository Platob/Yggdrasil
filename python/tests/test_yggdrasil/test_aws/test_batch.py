"""AWS Batch runtime resource + in-AWS detection + S3 proxy skip."""
from __future__ import annotations

from yggdrasil.aws import AWSBatch, BatchService, AWSClient, in_aws_environment


def _batch(env):
    # Bind to a credential-free client so the resource is hermetic.
    client = AWSClient(region="us-east-1", access_key_id="AK", secret_access_key="SK")
    return AWSBatch(service=BatchService(client=client), env=env)


def test_off_batch_is_inert():
    b = _batch({})
    assert b.is_batch is False
    assert b.explore_url is None
    assert b.is_main_node is True
    assert b.to_dict()["is_batch"] is False


def test_array_job_under_fargate():
    b = _batch({
        "AWS_BATCH_JOB_ID": "abc123:0",
        "AWS_BATCH_JOB_ATTEMPT": "2",
        "AWS_BATCH_JQ_NAME": "prod-queue",
        "AWS_BATCH_CE_NAME": "prod-ce",
        "AWS_BATCH_JOB_ARRAY_INDEX": "7",
        "AWS_EXECUTION_ENV": "AWS_ECS_FARGATE",
        "AWS_REGION": "eu-west-1",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/xyz",
    })
    assert b.is_batch and b.job_id == "abc123:0" and b.job_attempt == 2
    assert b.job_queue == "prod-queue" and b.compute_environment == "prod-ce"
    assert b.is_array_job and b.array_index == 7
    assert b.is_fargate and b.has_container_credentials and b.is_aws_environment
    assert b.region == "eu-west-1"
    assert "jobs/detail/abc123:0" in str(b.explore_url)


def test_resource_clickable_repr_via_explore_url():
    b = _batch({"AWS_BATCH_JOB_ID": "j", "AWS_REGION": "us-east-1"})
    assert repr(b) == f"AWSBatch({b.explore_url!r})"
    assert b._repr_html_().startswith('<a href="https://us-east-1.console.aws.amazon.com/batch/home')


def test_multinode_main_vs_worker():
    common = {"AWS_BATCH_JOB_ID": "mn", "AWS_BATCH_JOB_NUM_NODES": "4", "AWS_BATCH_JOB_MAIN_NODE_INDEX": "0"}
    main = _batch({**common, "AWS_BATCH_JOB_NODE_INDEX": "0"})
    worker = _batch({**common, "AWS_BATCH_JOB_NODE_INDEX": "3"})
    assert main.is_multinode and main.is_main_node
    assert worker.is_multinode and not worker.is_main_node


def test_region_falls_back_to_default_region():
    assert _batch({"AWS_BATCH_JOB_ID": "j", "AWS_DEFAULT_REGION": "us-west-2"}).region == "us-west-2"


def test_bad_int_tolerated():
    assert _batch({"AWS_BATCH_JOB_ID": "j", "AWS_BATCH_JOB_ATTEMPT": "nope"}).job_attempt is None


def test_service_follows_current_pattern():
    assert BatchService.service_name() == "batch"
    # Per-class current() singleton (not shared with the base / AccountService).
    assert isinstance(BatchService.current(), BatchService)


def test_client_batch_reads_live_env(monkeypatch):
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "live-job")
    c = AWSClient(region="us-east-1", access_key_id="AK", secret_access_key="SK")
    assert isinstance(c.batch, AWSBatch)
    assert c.batch.is_batch and c.batch.job_id == "live-job"


def test_pickle_round_trip():
    import pickle

    b = _batch({"AWS_BATCH_JOB_ID": "j", "AWS_REGION": "eu-west-1"})
    restored = pickle.loads(pickle.dumps(b))
    assert restored.job_id == "j" and restored.region == "eu-west-1"


class TestInAwsEnvironment:
    def test_signals(self):
        assert in_aws_environment({"AWS_EXECUTION_ENV": "AWS_ECS_FARGATE"})
        assert in_aws_environment({"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/x"})
        assert in_aws_environment({"AWS_BATCH_JOB_ID": "j"})
        assert in_aws_environment({"AWS_LAMBDA_FUNCTION_NAME": "fn"})
        assert in_aws_environment({"ECS_CONTAINER_METADATA_URI_V4": "http://..."})

    def test_not_in_aws(self):
        assert in_aws_environment({}) is False
        assert in_aws_environment({"PATH": "/usr/bin"}) is False


class TestS3ProxySkipInAws:
    def test_skips_proxy_when_in_aws(self, monkeypatch):
        from tests.test_yggdrasil.test_aws._fake_s3 import reset_s3_singletons
        from yggdrasil.aws.fs.path import S3Bucket

        monkeypatch.setenv("AWS_BATCH_JOB_ID", "j")  # → in_aws_environment()
        monkeypatch.setattr(AWSClient, "current", classmethod(
            lambda cls, **kw: cls(region="us-east-1", access_key_id="AK", secret_access_key="SK")
        ))
        reset_s3_singletons()
        http = S3Bucket(bucket="bkt").http
        assert http._no_proxy == "amazonaws.com,amazonaws.com.cn,amazonaws-us-gov.com"
        reset_s3_singletons()

    def test_keeps_proxy_outside_aws(self, monkeypatch):
        from tests.test_yggdrasil.test_aws._fake_s3 import reset_s3_singletons
        from yggdrasil.aws.fs.path import S3Bucket

        for var in ("AWS_BATCH_JOB_ID", "AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME",
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                    "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(AWSClient, "current", classmethod(
            lambda cls, **kw: cls(region="us-east-1", access_key_id="AK", secret_access_key="SK")
        ))
        reset_s3_singletons()
        assert S3Bucket(bucket="bkt").http._no_proxy is None
        reset_s3_singletons()
