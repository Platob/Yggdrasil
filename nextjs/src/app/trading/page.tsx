"use client";

import { useState, useCallback } from "react";
import {
  analysisRisk,
  analysisIndicators,
  analysisOhlc,
  analysisForecast,
  type RiskResult,
  type IndicatorsResult,
  type OhlcResult,
  type ForecastResult,
  type FilterSpec,
} from "@/lib/api";
import Chart from "@/components/Chart";

// ── helpers ────────────────────────────────────────────────────────────────

function pct(v: number | null, decimals = 2): string {
  if (v == null) return "—";
  return (v * 100).toFixed(decimals) + "%";
}
function num(v: number | null, decimals = 3): string {
  if (v == null) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
}
function colorClass(v: number | null, positiveGood = true): string {
  if (v == null) return "text-muted";
  const good = positiveGood ? v > 0 : v < 0;
  return good ? "text-emerald" : "text-rose";
}

// ── KPI card ──────────────────────────────────────────────────────────────

function KpiCard({ label, value, color, hint }: { label: string; value: string; color?: string; hint?: string }) {
  return (
    <div className="runic-card p-4 flex flex-col gap-1" title={hint}>
      <span className="text-[10px] text-muted uppercase tracking-widest">{label}</span>
      <span className={`text-xl font-mono font-bold ${color ?? "text-foreground"}`}>{value}</span>
    </div>
  );
}

// ── Risk panel ────────────────────────────────────────────────────────────

function RiskPanel({ risk }: { risk: RiskResult }) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <KpiCard label="Ann. Return" value={pct(risk.ann_return)} color={colorClass(risk.ann_return)} hint="Compound annualised return" />
        <KpiCard label="Ann. Volatility" value={pct(risk.ann_volatility, 2)} color="text-amber" hint="Annualised standard deviation of returns" />
        <KpiCard label="Sharpe" value={num(risk.sharpe_ratio)} color={colorClass(risk.sharpe_ratio)} hint="Risk-adjusted return (mean / vol × √periods)" />
        <KpiCard label="Sortino" value={num(risk.sortino_ratio)} color={colorClass(risk.sortino_ratio)} hint="Sharpe using downside deviation only" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <KpiCard label="Max Drawdown" value={pct(risk.max_drawdown)} color={colorClass(risk.max_drawdown, false)} hint="Worst peak-to-trough loss" />
        <KpiCard label="Calmar" value={num(risk.calmar_ratio)} color={colorClass(risk.calmar_ratio)} hint="Ann. return / |Max drawdown|" />
        <KpiCard label="VaR 95%" value={pct(risk.var_95)} color="text-rose" hint="5th percentile 1-period loss" />
        <KpiCard label="CVaR 95%" value={pct(risk.cvar_95)} color="text-rose" hint="Expected shortfall beyond VaR" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <KpiCard label="Win Rate" value={pct(risk.win_rate)} color={colorClass(risk.win_rate)} hint="Fraction of positive return periods" />
        <KpiCard label="Profit Factor" value={num(risk.profit_factor)} color={colorClass(risk.profit_factor)} hint="Sum of gains / sum of losses" />
        <KpiCard label="Skewness" value={num(risk.skewness)} hint="Return distribution skew (negative = fat left tail)" />
        <KpiCard label="Kurtosis" value={num(risk.kurtosis)} color={risk.kurtosis != null && risk.kurtosis > 1 ? "text-amber" : undefined} hint="Excess kurtosis (fat tails > 0)" />
      </div>
      <div className="text-[10px] text-muted font-mono">
        {risk.n.toLocaleString()} return observations · {risk.periods_per_year} periods/year
        {risk.max_drawdown_peak_i != null && ` · drawdown peak @${risk.max_drawdown_peak_i} trough @${risk.max_drawdown_trough_i}`}
      </div>
    </div>
  );
}

// ── Indicator chart ───────────────────────────────────────────────────────

function IndicatorChart({
  ind,
  priceLabel,
}: {
  ind: IndicatorsResult;
  priceLabel: string;
}) {
  const [view, setView] = useState<"price" | "rsi" | "macd" | "stoch">("price");
  const labels = ind.x;

  const priceKeys = Object.keys(ind.indicators).filter((k) => k.startsWith("sma_") || k.startsWith("ema_") || k.startsWith("bb_"));
  const rsiKey = Object.keys(ind.indicators).find((k) => k.startsWith("rsi_"));
  const macdKey = Object.keys(ind.indicators).includes("macd") ? "macd" : null;
  const stochKey = Object.keys(ind.indicators).find((k) => k.startsWith("stoch_k_"));

  const tabs = [
    { id: "price", label: "Price + MAs" },
    ...(rsiKey ? [{ id: "rsi", label: "RSI" }] : []),
    ...(macdKey ? [{ id: "macd", label: "MACD" }] : []),
    ...(stochKey ? [{ id: "stoch", label: "Stochastic" }] : []),
  ] as { id: "price" | "rsi" | "macd" | "stoch"; label: string }[];

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-[11px] font-mono flex-wrap">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setView(t.id)}
            className={`px-2.5 py-1 rounded ${view === t.id ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>
            {t.label}
          </button>
        ))}
        <span className="ml-auto text-muted/60">{ind.n.toLocaleString()} pts</span>
      </div>

      {view === "price" && (
        <div className="space-y-1">
          <Chart type="line" labels={labels} values={ind.price} color="var(--emerald)" yLabel={priceLabel} height={280} />
          {priceKeys.filter((k) => !k.startsWith("bb_")).map((k) => (
            <div key={k} className="flex items-center gap-2 text-[10px] font-mono text-muted">
              <span className={k.startsWith("sma_") ? "text-amber/80" : "text-frost/80"}>{k}</span>
            </div>
          ))}
          {ind.indicators.bb_upper && (
            <div className="text-[10px] text-muted font-mono">
              Bollinger: mid={num(ind.indicators.bb_mid?.[ind.indicators.bb_mid.length - 1])} ±2σ
            </div>
          )}
        </div>
      )}

      {view === "rsi" && rsiKey && (
        <div className="space-y-1">
          <Chart type="line" labels={labels} values={ind.indicators[rsiKey]} color="var(--amber)" yLabel={rsiKey.toUpperCase()} height={180} />
          <div className="text-[10px] text-muted font-mono">Overbought ≥70 · Oversold ≤30 · Current: {num(ind.indicators[rsiKey][ind.indicators[rsiKey].length - 1])}</div>
        </div>
      )}

      {view === "macd" && macdKey && (
        <div className="space-y-2">
          <Chart type="line" labels={labels} values={ind.indicators.macd} color="var(--frost)" yLabel="MACD" height={140}
            band={ind.indicators.macd_signal ? { min: ind.indicators.macd_signal, max: ind.indicators.macd_signal } : undefined} />
          <Chart type="bar" labels={labels} values={ind.indicators.macd_hist} color="var(--emerald)" yLabel="Histogram" height={80} />
        </div>
      )}

      {view === "stoch" && stochKey && (
        <div className="space-y-1">
          <Chart type="line" labels={labels} values={ind.indicators[stochKey]} color="var(--frost)" yLabel="%K" height={180} />
          <div className="text-[10px] text-muted font-mono">Overbought ≥80 · Oversold ≤20</div>
        </div>
      )}
    </div>
  );
}

// ── Main trading page ─────────────────────────────────────────────────────

const INDICATOR_COLORS: Record<string, string> = {
  "var(--emerald)": "emerald", "var(--frost)": "frost", "var(--amber)": "amber", "var(--rose)": "rose",
};
void INDICATOR_COLORS;

type Tab = "risk" | "indicators" | "candles" | "forecast";

export default function TradingPage() {
  const [path, setPath] = useState("");
  const [priceCol, setPriceCol] = useState("close");
  const [xCol, setXCol] = useState("date");
  const [highCol, setHighCol] = useState("high");
  const [lowCol, setLowCol] = useState("low");
  const [volCol, setVolCol] = useState("volume");
  const [orderBy, setOrderBy] = useState("date");
  const [ppy, setPpy] = useState(252);
  const [tab, setTab] = useState<Tab>("risk");
  const [buckets, setBuckets] = useState(200);
  const [fcHorizon, setFcHorizon] = useState(30);
  const [fcModel, setFcModel] = useState("auto");
  const [risk, setRisk] = useState<RiskResult | null>(null);
  const [ind, setInd] = useState<IndicatorsResult | null>(null);
  const [ohlc, setOhlc] = useState<OhlcResult | null>(null);
  const [forecast, setForecast] = useState<ForecastResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [maWindow, setMaWindow] = useState(20);

  const run = useCallback(async () => {
    if (!path || !priceCol) return;
    setLoading(true); setErr(null);
    try {
      if (tab === "risk") {
        setRisk(await analysisRisk(path, priceCol, {
          order_by: orderBy || undefined, periods_per_year: ppy,
        }));
      } else if (tab === "indicators") {
        setInd(await analysisIndicators(path, priceCol, {
          x: xCol || undefined,
          high: highCol || undefined, low: lowCol || undefined,
          volume: volCol || undefined,
          sma: [20, 50], ema: [12, 26],
          rsi: 14, macd: true, bollinger: 20,
          atr: highCol && lowCol ? 14 : null,
          stoch: highCol && lowCol ? 14 : null,
          obv: !!volCol,
        }));
      } else if (tab === "candles") {
        setOhlc(await analysisOhlc(path, priceCol, {
          x: xCol || undefined, volume: volCol || undefined, buckets,
        }));
      } else {
        setForecast(await analysisForecast(path, priceCol, {
          x: xCol || undefined, horizon: fcHorizon, model: fcModel,
        }));
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : "request failed");
    } finally {
      setLoading(false);
    }
  }, [path, priceCol, xCol, highCol, lowCol, volCol, orderBy, ppy, tab, buckets, fcHorizon, fcModel]);

  const maLine = ohlc ? ohlc.close.map((_, i) => {
    if (i < maWindow - 1) return null;
    let s = 0, c = 0;
    for (let k = i - maWindow + 1; k <= i; k++) { const v = ohlc.close[k]; if (v != null) { s += v; c++; } }
    return c ? s / c : null;
  }) : [];

  const inp = "bg-white/[0.04] border border-white/10 rounded px-2 py-1 outline-none text-[11px] font-mono text-foreground/90 focus:border-frost/40";
  const tabs: { id: Tab; label: string }[] = [
    { id: "risk", label: "Risk Metrics" },
    { id: "indicators", label: "Indicators" },
    { id: "candles", label: "OHLC" },
    { id: "forecast", label: "Forecast" },
  ];

  return (
    <div className="min-h-screen p-6 space-y-6">
      <div className="flex items-start gap-4">
        <div>
          <h1 className="text-2xl font-bold gradient-frost">Trading Analytics</h1>
          <p className="text-muted text-sm mt-1">Risk metrics · Technical indicators · OHLC charts · Forecasting</p>
        </div>
      </div>

      {/* Config strip */}
      <div className="runic-card p-4 space-y-3">
        <div className="flex items-center gap-3 flex-wrap text-[11px] font-mono">
          <label className="text-muted">file path</label>
          <input
            value={path} onChange={(e) => setPath(e.target.value)}
            placeholder="e.g. data/prices.parquet" className={`${inp} w-64`}
          />
          <label className="text-muted">price col</label>
          <input value={priceCol} onChange={(e) => setPriceCol(e.target.value)} className={`${inp} w-20`} />
          <label className="text-muted">x / date</label>
          <input value={xCol} onChange={(e) => setXCol(e.target.value)} className={`${inp} w-20`} />
          <label className="text-muted">high</label>
          <input value={highCol} onChange={(e) => setHighCol(e.target.value)} className={`${inp} w-16`} />
          <label className="text-muted">low</label>
          <input value={lowCol} onChange={(e) => setLowCol(e.target.value)} className={`${inp} w-16`} />
          <label className="text-muted">volume</label>
          <input value={volCol} onChange={(e) => setVolCol(e.target.value)} className={`${inp} w-16`} />
        </div>
        <div className="flex items-center gap-3 flex-wrap text-[11px] font-mono">
          <label className="text-muted">order by</label>
          <input value={orderBy} onChange={(e) => setOrderBy(e.target.value)} className={`${inp} w-20`} />
          <label className="text-muted">periods/yr</label>
          <select value={ppy} onChange={(e) => setPpy(Number(e.target.value))} className={inp}>
            <option value={252}>252 (daily)</option>
            <option value={52}>52 (weekly)</option>
            <option value={12}>12 (monthly)</option>
            <option value={365}>365 (calendar)</option>
          </select>
          {tab === "candles" && <>
            <label className="text-muted">bars</label>
            <input type="number" min={10} max={2000} value={buckets} onChange={(e) => setBuckets(Number(e.target.value) || 200)} className={`${inp} w-16`} />
            <label className="text-muted">MA</label>
            <input type="number" min={1} value={maWindow} onChange={(e) => setMaWindow(Math.max(1, Number(e.target.value) || 1))} className={`${inp} w-14`} />
          </>}
          {tab === "forecast" && <>
            <label className="text-muted">horizon</label>
            <input type="number" min={1} max={500} value={fcHorizon} onChange={(e) => setFcHorizon(Number(e.target.value) || 30)} className={`${inp} w-16`} />
            <select value={fcModel} onChange={(e) => setFcModel(e.target.value)} className={inp}>
              {["auto", "xgboost", "gbr", "ridge"].map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </>}
          <button
            onClick={run} disabled={loading || !path || !priceCol}
            className="px-3 py-1.5 rounded bg-emerald/15 text-emerald border border-emerald/30 disabled:opacity-40 ml-auto"
          >
            {loading ? "computing…" : "Analyze"}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 text-[12px] font-mono border-b border-white/[0.07] pb-0.5">
        {tabs.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`px-4 py-2 rounded-t transition-colors ${tab === t.id ? "bg-frost/10 text-frost border-b-2 border-frost" : "text-muted hover:text-foreground"}`}>
            {t.label}
          </button>
        ))}
      </div>

      {err && <div className="text-rose text-[11px] font-mono p-3 rounded border border-rose/20 bg-rose/5">{err}</div>}

      {/* Risk tab */}
      {tab === "risk" && (
        <div>
          {risk ? (
            <RiskPanel risk={risk} />
          ) : !loading && (
            <div className="runic-card p-10 text-center text-muted text-sm">
              Enter a file path + price column and click <span className="text-emerald">Analyze</span> to compute risk metrics.
            </div>
          )}
        </div>
      )}

      {/* Indicators tab */}
      {tab === "indicators" && (
        <div className="runic-card p-4">
          {ind ? (
            <IndicatorChart ind={ind} priceLabel={priceCol} />
          ) : !loading && (
            <div className="py-10 text-center text-muted text-sm">
              Configure columns above and click <span className="text-emerald">Analyze</span> to compute indicators.
              <div className="mt-2 text-[11px] text-muted/60">
                SMA 20/50 · EMA 12/26 · RSI 14 · MACD 12/26/9 · Bollinger 20 · ATR 14 · Stochastic 14
              </div>
            </div>
          )}
        </div>
      )}

      {/* OHLC candles tab */}
      {tab === "candles" && (
        <div className="runic-card p-4 space-y-2">
          {ohlc ? (
            <>
              <div className="text-[10px] text-muted font-mono">
                {ohlc.bars} bars from {ohlc.source_rows.toLocaleString()} rows
                · last close: {num(ohlc.close[ohlc.close.length - 1], 4)}
              </div>
              <Chart type="candle" labels={ohlc.x}
                ohlc={{ open: ohlc.open, high: ohlc.high, low: ohlc.low, close: ohlc.close }}
                overlay={maLine} volume={ohlc.volume ?? undefined} yLabel={priceCol}
                height={ohlc.volume ? 400 : 340} />
            </>
          ) : !loading && (
            <div className="py-10 text-center text-muted text-sm">
              Set price + x (date) columns and click <span className="text-emerald">Analyze</span>.
            </div>
          )}
        </div>
      )}

      {/* Forecast tab */}
      {tab === "forecast" && (
        <div className="runic-card p-4 space-y-3">
          {forecast ? (
            <>
              <div className="text-[10px] text-muted font-mono">
                model <span className="text-emerald/90">{forecast.model_used}</span> · horizon {forecast.horizon}
                · {forecast.source_rows.toLocaleString()} source rows
              </div>
              {forecast.series.map((s) => {
                const labels = [...s.history_x, ...s.forecast_x];
                const values = [...s.history_y, ...s.forecast_y];
                const bandMin = [...s.history_y, ...s.lower];
                const bandMax = [...s.history_y, ...s.upper];
                return (
                  <div key={s.key || "all"} className="space-y-1">
                    {s.key && <div className="text-[11px] text-frost font-mono">{s.key}</div>}
                    {s.rmse != null && (
                      <div className="text-[10px] text-muted font-mono">
                        RMSE {s.rmse.toFixed(4)} · {s.history_x.length} history → {s.forecast_x.length} forecast
                      </div>
                    )}
                    <Chart type="line" labels={labels} values={values}
                      band={{ min: bandMin, max: bandMax }}
                      color="var(--emerald)" yLabel={priceCol} height={300} />
                  </div>
                );
              })}
            </>
          ) : !loading && (
            <div className="py-10 text-center text-muted text-sm">
              Configure model + horizon and click <span className="text-emerald">Analyze</span>.
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="text-center py-8 text-muted text-sm font-mono">computing…</div>
      )}
    </div>
  );
}
