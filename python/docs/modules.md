# Yggdrasil Python module index

This index reflects the current package layout under `python/src/yggdrasil`.

## Core data and schema

- `yggdrasil.data.cast` — converter registry, dispatch, and `CastOptions`.
- `yggdrasil.data.enums` — timezone, currency, and geozone enums.
- `yggdrasil.arrow` — Arrow type inference and Arrow casting helpers.
- `yggdrasil.dataclasses` — dataclass helpers (`dataclass_to_arrow_field`, waiting/expiring utilities).

## Dataframe engines

- `yggdrasil.polars`
- `yggdrasil.pandas`
- `yggdrasil.spark`

## IO and transport

- `yggdrasil.io`
- `yggdrasil.io.http_`
- `yggdrasil.io.buffer`
- `yggdrasil.requests`

## Databricks integrations

- `yggdrasil.databricks`
- `yggdrasil.databricks.account`
- `yggdrasil.databricks.compute`
- `yggdrasil.databricks.fs`
- `yggdrasil.databricks.iam`
- `yggdrasil.databricks.jobs`
- `yggdrasil.databricks.secrets`
- `yggdrasil.databricks.sql`
- `yggdrasil.databricks.workspaces`
- `yggdrasil.databricks.ai`

## Platform and utilities

- `yggdrasil.pyutils`
- `yggdrasil.concurrent`
- `yggdrasil.environ`
- `yggdrasil.fastapi`
- `yggdrasil.pickle`
- `yggdrasil.mongo`
- `yggdrasil.mongoengine`
- `yggdrasil.blake3`
- `yggdrasil.xxhash`
- `yggdrasil.fxrates`

## Focused docs

See `python/docs/modules/README.md` for links to focused module pages and examples.
