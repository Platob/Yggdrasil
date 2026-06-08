"""Benchmark the ENTSO-E energy-data *frame* path — finance-style time series.

What this measures (and what it does NOT)
-----------------------------------------
This benchmark stubs the wire — no real ENTSO-E call. It measures everything
the energy-data path does *around* the network: parsing the publication XML
into rows (:func:`yggdrasil.loki.entsoe.parse_timeseries_xml`), assembling the
polars frame (:func:`~yggdrasil.loki.entsoe.to_frame`), and the analytics a
power-market / finance workload runs on the result — multi-zone concat,
price × load join, hourly→daily resample, and a rolling volatility window.

Those are the per-call costs you pay regardless of which upstream you hit, and
the things worth optimising when a day of 15-minute data across a dozen bidding
zones turns into a frame. It does NOT measure HTTP round trips or the
Transparency Platform's own latency.

Usage::

    PYTHONPATH=src python benchmarks/entsoe/bench_entsoe_frames.py
    PYTHONPATH=src python benchmarks/entsoe/bench_entsoe_frames.py --repeat 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import statistics
import time
from typing import Callable

import polars as pl

from yggdrasil.loki import entsoe

# ENTSO-E publication namespace (day-ahead prices); the parser is ns-agnostic
# but real documents carry it, so the fixture does too.
_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"


def _prices_xml(zone_eic: str, *, days: int, step_min: int = 60) -> str:
    """A realistic A44 day-ahead-prices document: ``days`` periods of
    ``1440/step_min`` points each, so the parser walks real volume."""
    per_period = 1440 // step_min
    resolution = f"PT{step_min}M"
    periods = []
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for d in range(days):
        start = (base + dt.timedelta(days=d)).strftime("%Y-%m-%dT%H:%MZ")
        end = (base + dt.timedelta(days=d + 1)).strftime("%Y-%m-%dT%H:%MZ")
        points = "".join(
            f"<Point><position>{i + 1}</position>"
            f"<price.amount>{40 + (i % 24) * 1.5:.2f}</price.amount></Point>"
            for i in range(per_period)
        )
        periods.append(
            f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
            f"<resolution>{resolution}</resolution>{points}</Period>"
        )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Publication_MarketDocument xmlns="{_NS}"><mRID>bench</mRID>'
        f'<TimeSeries><mRID>1</mRID>'
        f'<in_Domain.mRID codingScheme="A01">{zone_eic}</in_Domain.mRID>'
        f'<out_Domain.mRID codingScheme="A01">{zone_eic}</out_Domain.mRID>'
        f'<currency_Unit.name>EUR</currency_Unit.name>'
        f'<price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
        f'{"".join(periods)}</TimeSeries></Publication_MarketDocument>'
    )


def _load_xml(zone_eic: str, *, days: int, step_min: int = 60) -> str:
    """An A65 actual-load document — same shape, ``quantity`` instead of price."""
    per_period = 1440 // step_min
    resolution = f"PT{step_min}M"
    periods = []
    base = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    for d in range(days):
        start = (base + dt.timedelta(days=d)).strftime("%Y-%m-%dT%H:%MZ")
        end = (base + dt.timedelta(days=d + 1)).strftime("%Y-%m-%dT%H:%MZ")
        points = "".join(
            f"<Point><position>{i + 1}</position>"
            f"<quantity>{50000 + (i % 24) * 800}</quantity></Point>"
            for i in range(per_period)
        )
        periods.append(
            f"<Period><timeInterval><start>{start}</start><end>{end}</end></timeInterval>"
            f"<resolution>{resolution}</resolution>{points}</Period>"
        )
    return (
        f'<Publication_MarketDocument xmlns="{_NS}"><mRID>bench</mRID>'
        f'<TimeSeries><mRID>1</mRID>'
        f'<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>'
        f'{"".join(periods)}</TimeSeries></Publication_MarketDocument>'
    )


# --- timing harness — mirrors benchmarks/fxrate/bench_fxrate.py ------------


def _time_one(label: str, fn: Callable[[], object], *, repeat: int, inner: int) -> dict:
    for _ in range(min(inner, 10)):
        fn()
    samples: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return {"label": label, "best": min(samples),
            "median": statistics.median(samples), "mean": statistics.fmean(samples)}


def _fmt(r: dict) -> str:
    scale, unit = 1e6, "us"
    if r["best"] >= 1e-3:
        scale, unit = 1e3, "ms"
    return (f"{r['label']:<52s}  best={r['best']*scale:8.2f} {unit}  "
            f"median={r['median']*scale:8.2f} {unit}  mean={r['mean']*scale:8.2f} {unit}")


# --- scenarios ------------------------------------------------------------


def _parse_scenarios(repeat: int) -> list[dict]:
    eic = entsoe.ZONES["DE_LU"]
    out: list[dict] = []
    for days, step in ((1, 60), (7, 60), (30, 60), (7, 15)):
        xml = _prices_xml(eic, days=days, step_min=step)
        n = (1440 // step) * days
        out.append(_time_one(
            f"parse_timeseries_xml  {days}d @ {step}m ({n} pts)",
            lambda xml=xml: entsoe.parse_timeseries_xml(xml),
            repeat=repeat, inner=200 if n <= 200 else 50,
        ))
        out.append(_time_one(
            f"to_frame              {days}d @ {step}m ({n} pts)",
            lambda xml=xml: entsoe.to_frame(xml, zone="DE_LU", series="day_ahead_prices"),
            repeat=repeat, inner=200 if n <= 200 else 50,
        ))
    return out


def _analytics_scenarios(repeat: int) -> list[dict]:
    """The finance-style frame work: multi-zone concat, price×load join,
    daily resample, rolling volatility — over a month of hourly data."""
    zones = ["DE_LU", "FR", "NL", "BE", "AT", "ES", "PL", "CZ", "IT_NORD", "CH", "DK_1", "SE_3"]
    price_frames = [
        entsoe.to_frame(_prices_xml(entsoe.ZONES[z], days=30), zone=z, series="day_ahead_prices")
        for z in zones
    ]
    load_de = entsoe.to_frame(_load_xml(entsoe.ZONES["DE_LU"], days=30), zone="DE_LU", series="load")

    def concat_zones() -> pl.DataFrame:
        return pl.concat(price_frames)

    panel = concat_zones()

    def price_load_join() -> pl.DataFrame:
        de = panel.filter(pl.col("zone") == "DE_LU").select("timestamp", "value")
        return de.join(load_de.select("timestamp", pl.col("value").alias("load_mw")),
                       on="timestamp", how="inner")

    def daily_resample() -> pl.DataFrame:
        return (panel.group_by_dynamic("timestamp", every="1d", group_by="zone")
                .agg(pl.col("value").mean().alias("avg_price"),
                     pl.col("value").max().alias("peak_price")))

    def rolling_volatility() -> pl.DataFrame:
        de = panel.filter(pl.col("zone") == "DE_LU").sort("timestamp")
        return de.with_columns(
            pl.col("value").rolling_std(window_size=24).alias("vol_24h"))

    def cross_zone_spread() -> pl.DataFrame:
        wide = panel.pivot(values="value", index="timestamp", on="zone")
        return wide.with_columns((pl.col("DE_LU") - pl.col("FR")).alias("DE_FR_spread"))

    return [
        _time_one(f"concat {len(zones)} zones × 30d hourly  ({panel.height} rows)",
                  concat_zones, repeat=repeat, inner=200),
        _time_one("price × load join (inner, on timestamp)", price_load_join,
                  repeat=repeat, inner=200),
        _time_one("daily resample (group_by_dynamic, per zone)", daily_resample,
                  repeat=repeat, inner=100),
        _time_one("rolling 24h volatility (rolling_std)", rolling_volatility,
                  repeat=repeat, inner=200),
        _time_one("cross-zone spread (pivot wide + diff)", cross_zone_spread,
                  repeat=repeat, inner=100),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    sections = [
        ("XML → frame (stubbed wire)", _parse_scenarios(args.repeat)),
        ("Power-market frame analytics", _analytics_scenarios(args.repeat)),
    ]
    for title, results in sections:
        print()
        print(f"--- {title} ---")
        for r in results:
            print(_fmt(r))


if __name__ == "__main__":
    main()
