# Skill: trading and commodity market data — exchanges, contracts, codes

## When to use

The user mentions exchanges (CME, ICE, NYMEX, LME, EEX, EPEX, Nord
Pool), instruments (futures, options, swaps, spot, day-ahead /
intra-day power), contract months (M, Q, Y), commodity codes
(`CL`, `NG`, `BZ`, `HH`, `TTF`), trading sessions, settlement
prices, mark-to-market, volatility / IV, OHLCV bars, tick data,
order book, FX crosses, or asks to "ingest market data", "load
prices", "build a forwards curve", "compute returns", "track
positions". Also for energy-system data: ENTSO-E generation /
load / cross-border flows, EIC codes, bidding zones, day-ahead
auction results.

Builds on [`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md)
(the recipe), [`ygg-data-modeling`](ygg-data-modeling.md) (raw_
+ provenance + PK/FK + ISO joins), and
[`ygg-curated-views`](ygg-curated-views.md) (UTC / decimal / ISO
standardisation). This skill adds the **domain conventions** the
generic recipe needs to land trading-grade data correctly.

## Universal codes

| Concept | Standard | Yggdrasil enum / catalog |
| --- | --- | --- |
| Currency | ISO 4217 | `yggdrasil.data.enums.Currency` |
| Country | ISO 3166-1 alpha-2 | `yggdrasil.data.enums.geozone.GeoZoneCatalog` |
| Region / subdivision | ISO 3166-2 | same |
| Exchange (Market Identifier) | ISO 10383 MIC | (no enum yet — store as `mic_iso` string, document) |
| Power bidding zone (ENTSO-E EIC) | EIC 16-char | `yggdrasil.data.enums.geozone.EntsoeBiddingZoneRecord` + `GeoZone.eic` |
| Timezone | IANA | `yggdrasil.data.enums.Timezone` |
| Time unit | ISO 8601 duration | `yggdrasil.data.enums.TimeUnit` |

Column-naming convention (extends [`ygg-curated-views`](ygg-curated-views.md)):

- `mic_iso` — exchange (ISO 10383). `"XCME"`, `"IFEU"`, `"XEEE"`.
- `eic_code` — energy zone (ENTSO-E). `"10YFR-RTE------C"` for France.
- `currency_iso`, `country_iso`, `timezone_iana` — as elsewhere.
- `contract_code`, `contract_month`, `expiration_utc`, `tenor_iso` —
  for forwards / futures (see below).

## `raw_` schema templates

### Spot / OHLCV bars (any tradable instrument)

```python
from yggdrasil.data import Field, DataType, Schema

RAW_OHLCV_SCHEMA = Schema.from_fields([
    # Source-shaped identifiers — keep vendor's exact strings here.
    Field("vendor_symbol", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="Vendor instrument id (e.g. 'CME:CL=F'). Curated layer maps to mic_iso + ticker."),
    Field("vendor_exchange", DataType.string(), nullable=True),
    Field("bucket_start", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True},
          comment="Bar window start, UTC. Bar width carried in `bar_width_iso`."),
    Field("bar_width_iso", DataType.string(), nullable=False,
          comment="ISO-8601 duration ('PT1M', 'PT5M', 'PT1H', 'P1D')."),
    Field("open",   DataType.decimal(28, 10), nullable=True),
    Field("high",   DataType.decimal(28, 10), nullable=True),
    Field("low",    DataType.decimal(28, 10), nullable=True),
    Field("close",  DataType.decimal(28, 10), nullable=True),
    Field("volume", DataType.decimal(28, 4),  nullable=True,
          comment="Traded volume in instrument's native unit (contracts / lots / MWh)."),
    Field("vwap",   DataType.decimal(28, 10), nullable=True),
    Field("currency_iso", DataType.string(), nullable=False,
          comment="Quote currency. ISO 4217."),
    # Provenance — same convention as ygg-data-modeling.
    Field("_ingested_at",   DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True}),
    Field("_source",        DataType.string(),  nullable=False),
    Field("_source_url",    DataType.string(),  nullable=True),
    Field("_payload_hash",  DataType.string(),  nullable=False),
    Field("_batch_id",      DataType.string(),  nullable=False),
])
```

Decimal precision rationale:

- `decimal(28, 10)` for prices — covers crypto-grade precision down
  to 10 places, well within Spark Decimal limits (38 digits).
- `decimal(28, 4)` for volume — exchanges typically ship up to 4
  decimal contracts.
- Use `decimal(36, 18)` only when ingesting on-chain / fractional
  crypto where 18 decimals is the native unit.

### Futures / forwards contract

```python
RAW_CONTRACT_SCHEMA = Schema.from_fields([
    Field("contract_code", DataType.string(), nullable=False,
          tags={"primary_key": True},
          comment="Vendor contract identifier (e.g. 'CLZ26' = CME WTI Dec-2026)."),
    Field("underlying_root", DataType.string(), nullable=False,
          comment="Root symbol ('CL', 'NG', 'BZ', 'TTF', …)."),
    Field("mic_iso", DataType.string(), nullable=False,
          tags={"foreign_key": True},
          metadata={"references": "main.iso.exchange(mic_iso)"},
          comment="Listing exchange. ISO 10383."),
    Field("contract_month", DataType.string(), nullable=False,
          comment="ISO month 'YYYY-MM' (e.g. '2026-12'). Easier joins than the vendor's M/Q/Y code."),
    Field("expiration_utc", DataType.timestamp("UTC"), nullable=False,
          comment="Last trade time, UTC. Source's local timestamp converted at curate time."),
    Field("multiplier", DataType.decimal(18, 6), nullable=False,
          comment="Contract size multiplier (e.g. 1000 bbl for WTI)."),
    Field("tick_size", DataType.decimal(18, 10), nullable=False),
    Field("currency_iso", DataType.string(), nullable=False),
    Field("settle_currency_iso", DataType.string(), nullable=True,
          comment="When different from currency_iso (cross-listed contracts)."),
    # …+ provenance.
])
```

### Energy day-ahead auction (ENTSO-E)

```python
RAW_DAYAHEAD_SCHEMA = Schema.from_fields([
    Field("eic_code", DataType.string(), nullable=False,
          tags={"primary_key": True, "cluster_by": True},
          comment="ENTSO-E bidding zone EIC (e.g. '10YFR-RTE------C')."),
    Field("delivery_start_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True},
          comment="Delivery hour start, UTC. Source local time converted at ingest."),
    Field("delivery_end_utc", DataType.timestamp("UTC"), nullable=False),
    Field("price",  DataType.decimal(18, 6), nullable=False,
          comment="Clearing price. EUR / MWh by default; see currency_iso + unit."),
    Field("currency_iso", DataType.string(), nullable=False),
    Field("unit", DataType.string(), nullable=False,
          comment="'MWh' for energy auctions. Document any deviation."),
    Field("source_resolution_iso", DataType.string(), nullable=False,
          comment="ISO 8601 duration of the auction grid ('PT60M' day-ahead, 'PT15M' intra-day)."),
    # …+ provenance.
])
```

## Curated joins on shared dims

Cross-source queries route through `main.iso.*` rather than vendor-
specific code columns:

```sql
-- All FR day-ahead prices joined with EUR/USD spot for the same hour.
SELECT
    d.delivery_start_utc,
    d.price                 AS price_eur,
    d.price * fx.rate       AS price_usd
FROM main.entsoe.dayahead d
JOIN main.fx.spot fx
       ON fx.pair_iso = 'EURUSD'
      AND fx.observed_utc = date_trunc('HOUR', d.delivery_start_utc)
WHERE d.eic_code = '10YFR-RTE------C'
```

`pair_iso` is the curated ISO 4217 pair string (`'EURUSD'`); the
join goes through the shared FX schema rather than vendor A's
`USD_EUR` vs vendor B's `EURUSD` mismatch.

## Time-series hygiene

1. **Always `_utc`-suffixed timestamps.** Source local time → UTC at
   ingest, never at query time. Document the source timezone in the
   field comment (`comment="Vendor ships America/Chicago — converted to UTC."`).
2. **Bar boundary convention.** `bucket_start` is the *start* of the
   window, not the end. Make this explicit in the column comment.
3. **DST gaps are real.** Spring-forward / fall-back can give an
   energy zone 23 or 25 hours in a day. Don't reject the row; store
   it. Curated layer can fan out by `source_resolution_iso`.
4. **No phantom intervals.** When the source skips a bar (illiquid
   instrument), don't synthesise a NULL row in `raw_`. The curated
   layer can do that with a `GENERATE_SERIES` if downstream needs it.
5. **One-second resolution at most for tick data** unless you have a
   genuine sub-second use case — Delta is bad at billions of rows of
   1µs ticks. Bucket into `raw_<entity>_1s` / `raw_<entity>_1m`
   layers and keep the rawest layer narrow.

## Idempotency for trading data

**Spot / OHLCV ingestion.** Compose the PK from
`(vendor_symbol, bucket_start, _ingested_at)` so a rerun lands a
new row (audit) but `MERGE` on `(vendor_symbol, bucket_start)`
upserts the latest observation. The curated layer dedups to the
latest `_ingested_at` per key.

**Settlement prices.** When the vendor publishes a final settle
after the close (often hours later), use `Mode.UPSERT` on the
`SchemaSession` — the next fetch replaces the row. Or store both
indicative and settle as separate fields: `last_price`,
`settle_price`, `settle_published_at_utc`.

**Corrections / restatements.** Some vendors silently rewrite
history (ENTSO-E, ICE corrections feeds). Always keep `_ingested_at`
in the PK so a re-fetch lands the corrected row as new history,
*and* schedule a daily reconciliation job that diffs the last 7d.

## FX rates

```python
RAW_FX_SCHEMA = Schema.from_fields([
    Field("pair_iso", DataType.string(), nullable=False,
          tags={"primary_key": True, "cluster_by": True},
          comment="'EURUSD' — base + quote, both ISO 4217."),
    Field("observed_utc", DataType.timestamp("UTC"), nullable=False,
          tags={"primary_key": True, "partition_by": True}),
    Field("rate", DataType.decimal(20, 10), nullable=False),
    Field("source", DataType.string(), nullable=False,
          comment="Provider tag ('ecb', 'oanda', 'bloomberg', ...)."),
    # …+ provenance.
])
```

`ygg.fxrates` (`yggdrasil.fxrates`) already ships an FX provider
surface — use that when the source is one of the supported feeds
rather than rolling a custom one.

## Don'ts

- Don't store prices in `float64`. The cast registry will
  strict-mode refuse precision-losing inputs; `decimal(28, 10)` is
  the default for OHLCV.
- Don't join vendor A's `exchange_code` to vendor B's `exchange`.
  Both resolve to `mic_iso` in the curated layer; join on that.
- Don't tag exchange-local timestamps as `timestamp("UTC")`.
  Convert at ingest, keep the original tz in a comment.
- Don't ingest 1s tick data into the same table as 1m bars. Layer
  them: `raw_<entity>_ticks`, `raw_<entity>_1m`, curated rollups
  on top.
- Don't ship a vendor-specific "contract code" string as the join
  key. `contract_month` (`'2026-12'`) + `mic_iso` + `underlying_root`
  is the curated triple every consumer can build against.
- Don't compute returns / volatility / spreads on the raw layer;
  that belongs in the curated views (and the cast registry is the
  wrong place for it). See [`ygg-curated-views`](ygg-curated-views.md).
- Don't poll a market-data API at < 1 s without rate-limit
  awareness. Wire `ErrorNotifyingHTTPSession` so a 429 flood pings
  ops instead of silently dropping rows.
