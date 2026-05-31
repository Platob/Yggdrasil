"""AWS Batch runtime environment introspection."""
from __future__ import annotations

from yggdrasil.aws import AWSBatchEnvironment, AWSClient


def test_off_batch_is_inert():
    b = AWSBatchEnvironment.from_env({})
    assert b.is_batch is False
    assert b.explore_url is None
    assert b._repr_html_() is None
    assert b.is_main_node is True  # a non-multinode context is its own "main"
    assert b.to_dict()["is_batch"] is False


def test_array_job_under_fargate():
    b = AWSBatchEnvironment.from_env({
        "AWS_BATCH_JOB_ID": "abc123:0",
        "AWS_BATCH_JOB_ATTEMPT": "2",
        "AWS_BATCH_JQ_NAME": "prod-queue",
        "AWS_BATCH_CE_NAME": "prod-ce",
        "AWS_BATCH_JOB_ARRAY_INDEX": "7",
        "AWS_EXECUTION_ENV": "AWS_ECS_FARGATE",
        "AWS_REGION": "eu-west-1",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/credentials/xyz",
    })
    assert b.is_batch
    assert b.job_id == "abc123:0" and b.job_attempt == 2
    assert b.job_queue == "prod-queue" and b.compute_environment == "prod-ce"
    assert b.is_array_job and b.array_index == 7
    assert b.is_fargate and b.has_container_credentials
    assert b.region == "eu-west-1"
    assert "jobs/detail/abc123:0" in str(b.explore_url)
    assert "batch/home" in str(b.explore_url)
    assert b._repr_html_().startswith('<a href="https://eu-west-1.console.aws.amazon.com/batch/home')


def test_region_falls_back_to_default_region():
    b = AWSBatchEnvironment.from_env({"AWS_BATCH_JOB_ID": "j", "AWS_DEFAULT_REGION": "us-west-2"})
    assert b.region == "us-west-2"


def test_multinode_main_vs_worker():
    common = {"AWS_BATCH_JOB_ID": "mn", "AWS_BATCH_JOB_NUM_NODES": "4", "AWS_BATCH_JOB_MAIN_NODE_INDEX": "0"}
    main = AWSBatchEnvironment.from_env({**common, "AWS_BATCH_JOB_NODE_INDEX": "0"})
    worker = AWSBatchEnvironment.from_env({**common, "AWS_BATCH_JOB_NODE_INDEX": "3"})
    assert main.is_multinode and main.is_main_node
    assert worker.is_multinode and not worker.is_main_node
    assert main.num_nodes == 4


def test_raw_captures_only_set_vars():
    b = AWSBatchEnvironment.from_env({"AWS_BATCH_JOB_ID": "j", "UNRELATED": "x"})
    assert b.raw == {"AWS_BATCH_JOB_ID": "j"}


def test_bad_int_is_tolerated():
    b = AWSBatchEnvironment.from_env({"AWS_BATCH_JOB_ID": "j", "AWS_BATCH_JOB_ATTEMPT": "not-a-number"})
    assert b.job_attempt is None


def test_client_batch_accessor_reads_process_env(monkeypatch):
    monkeypatch.setenv("AWS_BATCH_JOB_ID", "live-job")
    monkeypatch.setenv("AWS_BATCH_JQ_NAME", "q")
    c = AWSClient(region="us-east-1", access_key_id="AK", secret_access_key="SK")
    assert c.batch.is_batch
    assert c.batch.job_id == "live-job" and c.batch.job_queue == "q"


def test_picklable():
    import pickle

    b = AWSBatchEnvironment.from_env({"AWS_BATCH_JOB_ID": "j", "AWS_BATCH_JOB_ATTEMPT": "1"})
    assert pickle.loads(pickle.dumps(b)) == b
