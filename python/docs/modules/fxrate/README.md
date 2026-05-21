# yggdrasil.fxrate

FX rate fetching with multi-source fallback, geography enrichment, and a one-call Databricks Job factory for scheduled ingestion.

**Optional dependencies:** `polars` (for DataFrames), `ygg[databricks]` (for `deploy_scheduled_fxrate_job`).

## One-liner

```python
from yggdrasil.fxrate import FxRate

print(FxRate().latest([("EUR", "USD"), ("GBP", "JPY")]))
```

## Latest rates

```python
from yggdrasil.fxrate import FxRate

fx = FxRate()

# One or more currency pairs (str or Currency enum)
df = fx.latest([("EUR", "USD"), ("EUR", "GBP"), ("USD", "JPY")])
# Returns a Polars DataFrame:
#   [source, target, from_timestamp, to_timestamp, sampling, value]
print(df)
```

## Historical rates (date range)

```python
from yggdrasil.fxrate import FxRate
import datetime

fx = FxRate()

df = fx.fetch(
    pairs=[("EUR", "USD"), ("GBP", "USD")],
    start=datetime.date(2026, 1, 1),
    end=datetime.date(2026, 5, 1),
    sampling="1d",   # "1d", "1h", "1m"
)
print(df.head(5))
```

Date arguments accept many formats — ISO strings, `datetime`, epoch seconds, `"now"` / `"utcnow"`:

```python
df = fx.fetch(pairs=[("EUR", "USD")], start="2026-01-01", end="now", sampling="1d")
```

## Geography enrichment

Add country-level geo columns (lat, lon, country_iso) for each currency side:

```python
df = fx.fetch(
    pairs=[("EUR", "USD")],
    start="2026-01-01",
    end="2026-05-01",
    sampling="1d",
    geo=True,          # adds source_lat, source_lon, target_lat, target_lon, ...
)
print(df.columns)
```

## Lazy output (LazyFrame)

```python
df = fx.fetch(pairs=[("EUR", "USD")], start="2026-01-01", end="now", lazy=True)
result = df.filter(pl.col("value") > 1.08).collect()
```

## Convert a value

```python
from yggdrasil.fxrate import FxRate
import datetime

fx = FxRate()

eur_amount = 1000.0
usd_value  = fx.convert(eur_amount, "EUR", "USD", at=datetime.date(2026, 5, 1))
print(f"€{eur_amount} = ${usd_value:.2f}")
```

## Select backends

By default `FxRate` tries `Frankfurter → FawazAhmed → ErApi` in order. Override the chain:

```python
from yggdrasil.fxrate import FxRate, FrankfurterBackend, ErApiBackend

fx = FxRate(backends=[FrankfurterBackend(), ErApiBackend()])
```

Pin to a single backend:

```python
from yggdrasil.fxrate import FxRate, FawazBackend

fx = FxRate(backends=[FawazBackend()])
```

## Single FxQuote

```python
from yggdrasil.fxrate import FxRate, FxQuote

fx   = FxRate()
rows = fx.latest([("EUR", "USD")]).to_struct("_").to_list()

# Or use FxQuote directly (named-tuple row)
quote: FxQuote = rows[0]
print(quote.source, quote.target, quote.value, quote.from_timestamp)
```

## Deploy a scheduled Databricks ingestion Job

```python
from yggdrasil.fxrate import deploy_scheduled_fxrate_job
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()

job = deploy_scheduled_fxrate_job(
    client=client,
    catalog="main",
    schema="iso",
    schedule="0 0 * * *",          # daily at midnight UTC (UNIX cron)
    pairs=[("EUR", "USD"), ("GBP", "USD"), ("JPY", "USD")],
    sampling="1d",
    geo=True,
)
print(job.url)   # Databricks Job UI link
```

## FX columns schema

The DataFrame produced by `FxRate.fetch` / `FxRate.latest` always has these columns:

| Column | Type | Description |
|---|---|---|
| `source` | `string` | ISO 4217 source currency |
| `target` | `string` | ISO 4217 target currency |
| `from_timestamp` | `timestamp[us, UTC]` | Period start |
| `to_timestamp` | `timestamp[us, UTC]` | Period end |
| `sampling` | `string` | Granularity (`"1d"`, `"1h"`, …) |
| `value` | `float64` | Exchange rate (1 source = N target) |

With `geo=True`, additional columns per side: `source_country_iso`, `source_lat`, `source_lon`, `target_country_iso`, `target_lat`, `target_lon`.
