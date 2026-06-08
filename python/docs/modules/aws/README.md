# yggdrasil.aws

AWS integration — `AWSClient` singleton, S3 filesystem, and credential management. Mirrors the Databricks-side organization.

**Optional dependency:** `boto3`. Install with `pip install "ygg[aws]"` (or `pip install boto3`).

## One-liner

```python
from yggdrasil.aws import AWSClient
from yggdrasil.aws.fs import S3Path

p = S3Path("s3://my-bucket/data/events.parquet")
print(p.read_bytes()[:50])
```

## AWSClient

`AWSClient` is a singleton-cached client per identity configuration. Passing the same arguments returns the same instance.

```python
from yggdrasil.aws import AWSClient

# Default credential chain (env vars → profile → instance metadata)
client = AWSClient()

# Explicit role (STS assume-role)
client = AWSClient(
    role_arn="arn:aws:iam::123456789012:role/DataReader",
    region="us-east-1",
)

# Named profile
client = AWSClient(profile="prod-readonly")

# IAM Identity Center (SSO) with browser login
client = AWSClient(
    sso_start_url="https://example.awsapps.com/start",
    sso_region="us-east-1",
    sso_account_id="123456789012",
    sso_role_name="DataReader",
)
```

## S3 service and paths

```python
from yggdrasil.aws import AWSClient

client = AWSClient(region="eu-west-1")
s3     = client.s3   # S3Service

# Build a path via the service (credentials already bound)
path = s3.path("s3://my-bucket/prefix/file.parquet")

# Or build directly (uses default credential chain)
from yggdrasil.aws.fs import S3Path
path = S3Path("s3://my-bucket/prefix/file.parquet")
```

## S3Path — file operations

`S3Path` behaves like a `pathlib.Path` over S3:

```python
from yggdrasil.aws.fs import S3Path

path = S3Path("s3://my-bucket/data/events.parquet")

# Read / write bytes
content = path.read_bytes()
path.write_bytes(b"hello world")

# Read / write text
text = path.read_text()
path.write_text("hello world")

# Check existence
print(path.exists())

# List a prefix (directory-like)
prefix = S3Path("s3://my-bucket/data/")
for child in prefix.iterdir():
    print(child)

# Delete
path.unlink(missing_ok=True)

# Copy
path.copy_to(S3Path("s3://my-bucket/data/events_backup.parquet"))
```

## Arrow filesystem integration

Use `S3Path` with pyarrow for Parquet / Feather / CSV I/O:

```python
import pyarrow.parquet as pq
import pyarrow as pa
from yggdrasil.aws import AWSClient

client = AWSClient(region="us-east-1", role_arn="arn:aws:iam::123:role/Reader")
fs     = client.s3.arrow_filesystem()

# Write Parquet
table = pa.table({"id": [1, 2, 3], "score": [9.1, 8.7, 7.4]})
pq.write_table(table, "my-bucket/scores/batch.parquet", filesystem=fs)

# Read Parquet
tbl = pq.read_table("my-bucket/scores/", filesystem=fs)
```

## Credentials

```python
from yggdrasil.aws import AwsCredentials, AWSClient

# STS credentials (from a Databricks storage credential vend, for example)
creds = AwsCredentials(
    access_key_id="AKIA...",
    secret_access_key="...",
    session_token="...",
)
client = AWSClient.from_credentials(creds, region="us-east-1")

# Credentials provider (refreshes automatically)
from yggdrasil.aws import AwsCredentialsProvider
provider = AwsCredentialsProvider(client=client)
```

## Databricks volume → S3 bridge

When a Databricks external volume is backed by S3, the volume vends temporary credentials. Use them to build an `AWSClient`:

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.aws import AWSClient
from yggdrasil.data.enums import Mode

vol   = DatabricksClient().volumes["main.external.s3data"]
creds = vol.temporary_credentials(mode=Mode.READ)

client = AWSClient.from_credentials(creds)
fs     = client.s3.arrow_filesystem()

import pyarrow.parquet as pq
tbl = pq.read_table("s3data/events/", filesystem=fs)
```

## Full S3 ingest pipeline

```python
from yggdrasil.aws import AWSClient
from yggdrasil.aws.fs import S3Path
import pyarrow as pa
import pyarrow.parquet as pq

client = AWSClient(role_arn="arn:aws:iam::123:role/ETL", region="eu-west-1")
fs     = client.s3.arrow_filesystem()

# Read all Parquet files under a prefix
table = pq.read_table("my-bucket/raw/events/2026/05/", filesystem=fs)

# Transform
import polars as pl
df = pl.from_arrow(table).filter(pl.col("status") == "active")

# Write back
pq.write_table(df.to_arrow(), "my-bucket/curated/events_active.parquet", filesystem=fs)
```
