# Benchmark Results

Generated 2026-06-10T04:29:55Z — branch claude/friendly-bell-gop885
Environment: Python 3.12; polars, pyarrow>=20; scikit-learn + xgboost installed.

All yggdrasil.node bot benchmarks and the fxrate orchestration benchmark
run green. Highlights: analysis pushdown ~5x over eager full read; tabular
inspect ~78x via parquet footer; fs ls ~1.5x via os.scandir.

## bench_v2_endpoints
```

  endpoint     n     p50us    p99us    avgus    req/s    status
  ----------------------------------------------------------------------
  ping          500      262      460      278     3597    200
  stats         500      446      776      397     2517    200
  backend       500      269      506      286     3495    200
  back/sum      500      265      510      286     3492    200
  health        500      263      511      281     3554    200
  audit         500      314      589      419     2388    200
  pyfunc/list   500      265      538      285     3511    200
  pyenv/list    500      272      515      291     3439    200
```

## bench_bot_transport (--repeat 3)
```

==========================================================================================
  yggdrasil.node transport benchmark  (repeat=3, inner=50)
==========================================================================================

--- pickle transport round-trip ---
scenario                                       best µs   median µs      wire B
------------------------------------------------------------------------------
int                                               10.1        10.8          17
dict (100 values)                               1023.2      1074.8       1,447
list[dict] (100 items)                          6741.5      6752.2      11,510
bytes (64 B)                                      12.4        12.6          90
bytes (32 KiB)                                    15.3        15.6      32,794

--- Arrow IPC stream round-trip ---
scenario                                       best µs   median µs      wire B  throughput
------------------------------------------------------------------------------------------
Arrow Table (100 rows × 4 cols)                   26.5        27.0       3,200     120.6 MB/s
Arrow Table (10K rows × 4 cols)                   86.7        96.2     280,744    3237.6 MB/s
Arrow Table (100K rows × 4 cols)                 708.6       714.9   2,901,992    4095.4 MB/s

--- chunked vs single-shot Arrow stream (10K rows) ---
mode                                           best µs   median µs
------------------------------------------------------------------
single-shot                                       58.3        59.4
chunked (8192 rows)                               69.9        71.8

--- serialize_result dispatch (format selection) ---
scenario                                       best µs   median µs      wire B
------------------------------------------------------------------------------
scalar dict → x-python-pickle                    564.7       564.9       1,447
Arrow Table (100 rows) → vnd.apache.arrow.str        11.1        11.1       3,200
Arrow Table (10K rows) → vnd.apache.arrow.str        55.6        57.4     280,744

--- Polars DataFrame transport ---
scenario                                       best µs   median µs      wire B  throughput
------------------------------------------------------------------------------------------
Polars DF (10K × 2) → Arrow stream               282.6       289.7     219,296     776.0 MB/s
Arrow stream → pa.Table (10K × 2)                  9.5         9.5     219,296   23155.2 MB/s

--- /api/call endpoint overhead ---
scenario                                       best ms   median ms
------------------------------------------------------------------
scalar return (add)                               1.54        1.60
tabular return (100 rows)                         3.35        3.64
tabular return (10K rows)                         7.73        8.03

==========================================================================================

```

## bench_bot_messenger (--repeat 3)
```

==================================================================================
  yggdrasil.node messenger benchmark  (repeat=3)
==================================================================================

--- messenger service direct (no HTTP) ---
scenario                                          best us   median us       ops/sec
----------------------------------------------------------------------------------
send_message (general)                               18.0        18.2        55,695
send_message (custom channel)                        17.4        17.8        57,616
list_channels                                        12.3        12.4        81,185
get_messages (50 limit)                              11.4        11.4        87,706
get_channel                                          10.6        10.8        94,001

--- messenger endpoint (full HTTP stack) ---
scenario                                          best ms   median ms
--------------------------------------------------------------------
POST /messenger (send)                               1.68        1.73
GET /messenger/channels                              1.51        1.52
GET /messenger/.../messages (50)                     1.88        1.91

--- throughput (burst send) ---
    100 messages in      0.8 ms  (   124,264 msg/s)
   1000 messages in      6.5 ms  (   153,160 msg/s)
   5000 messages in    109.2 ms  (    45,781 msg/s)

==================================================================================

```

## bench_node_functions
```
Function CRUD (n=100):
  create: 0.0009s (109305 ops/s)
  read:   0.0000s (5388802 ops/s)
  delete: 0.0000s (2330350 ops/s)

Monitor snapshots (n=1000):
  total: 0.0584s (17122 ops/s)
```

## bench_fs_ls
```

  /fs/ls listing  (200  1000  5000 entries)

   entries   old iterdir ms   scandir ls ms   speedup
       220             2.72            1.95     1.40x  (220 entries)
      1100            15.24            9.60     1.59x  (1100 entries)
      5500            89.61           60.58     1.48x  (5500 entries)
```

## bench_audit_writes
```

  audit append:   50000 entries    312.3 ms
                   160096 entries/sec  (6.2 us each)
```

## bench_analysis
```

  wide.parquet: 1,000,000 rows x 30 cols (2 MB), aggregate touches 2 cols

  lazy scan + projection pushdown (2 cols):      18.7 ms   (5 groups)
  eager full read (30 cols) + aggregate:        108.4 ms
  ==>   5.8x  (pushdown skips 28 unused columns)

  downsample 1,000,000 -> 800 pts:      11.2 ms
  ohlc 1,000,000 -> 121 bars:          15.6 ms

  pivot sector x region (3 cols):     22.7 ms   (5 rows x 5 cols)
  eager full read (30) + pivot:      139.2 ms
  ==>   6.1x

  ts.parquet: 50,000 rows, 4 groups — forecast value~ts

  forecast ridge    (ridge   ):   1968.9 ms  4 series x 48h  rmse≈0.001158
  forecast gbr      (gbr     ):   6526.0 ms  4 series x 48h  rmse≈50.148918
  forecast xgboost  (xgboost ):    271.6 ms  4 series x 48h  rmse≈52.344779

```

## bench_tabular_inspect
```

  parquet 43 MB, 500000 rows x 20 cols

  read 1000+1 rows to size:      49.58 ms
  footer-metadata inspect:        0.60 ms    83.3x faster
  (now reports exact row_count=500000, editable=True)

```

## fxrate/bench_fxrate (--repeat 3)
```

--- Input coercion ---
coerce_currency('EUR')                                        best=  352.79 ns  median=  361.56 ns  mean=  360.57 ns
coerce_currency('$') [alias]                                  best=  353.69 ns  median=  356.60 ns  mean=  355.78 ns
coerce_currency(Currency.EUR) [identity]                      best=   83.26 ns  median=   85.84 ns  mean=   85.09 ns
coerce_pair(('EUR','USD'))                                    best=  784.52 ns  median=  794.87 ns  mean=  799.81 ns
coerce_datetime('2024-01-01')                                 best=    1.28 us  median=    1.31 us  mean=    1.31 us
coerce_datetime(ISO w/ tz)                                    best=  461.12 ns  median=  462.07 ns  mean=  466.79 ns
coerce_datetime(epoch int)                                    best=  711.01 ns  median=  719.82 ns  mean=  734.29 ns
coerce_datetime(datetime instance)                            best=  646.29 ns  median=  646.98 ns  mean=  650.48 ns
group_pairs_by_source (10 pairs, 5 sources)                   best=    1.07 us  median=    1.08 us  mean=    1.08 us

--- End-to-end fetch (stubbed transport) ---
fetch 1 pair, 1 day (eager frame)                             best=  191.50 us  median=  195.43 us  mean=  194.35 us
fetch 1 pair, 30 days (eager frame)                           best=  294.57 us  median=  305.41 us  mean=  301.81 us
fetch 3 pairs, 30 days (90 rows)                              best=  501.41 us  median=  515.94 us  mean=  516.97 us
fetch 3 pairs, 30 days (lazy)                                 best=  496.56 us  median=  496.81 us  mean=  497.12 us
latest 1 pair                                                 best=  183.84 us  median=  184.80 us  mean=  185.07 us

--- Fallback walk ---
fetch with 1 fail+1 success backend                           best=  200.41 us  median=  200.70 us  mean=  235.89 us
fetch with 2 fail+1 success backends                          best=  202.77 us  median=  203.16 us  mean=  203.95 us

--- Geography enrichment ---
fetch 3 pairs, 30 days (no geo)                               best=  497.65 us  median=  504.53 us  mean=  504.71 us
fetch 3 pairs, 30 days (geo=True)                             best= 2202.12 us  median= 2259.58 us  mean= 2253.78 us
```
