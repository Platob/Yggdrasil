"use client";

import { useState, useCallback } from "react";
import { useMutation } from "@tanstack/react-query";
import { series, ohlc, aggregate, forecast } from "@/lib/api";
import type {
  SeriesResult,
  OhlcResult,
  AggregateResult,
  ForecastResult,
} from "@/lib/types";
import LineChart from "@/components/LineChart";
import OhlcChart from "@/components/OhlcChart";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

type Tab = "series" | "ohlc" | "aggregate" | "forecast";

const tabs: { id: Tab; label: string }[] = [
  { id: "series", label: "Series" },
  { id: "ohlc", label: "OHLC" },
  { id: "aggregate", label: "Aggregate" },
  { id: "forecast", label: "Forecast" },
];

const aggTypes = ["sum", "mean", "count", "min", "max"] as const;
const forecastModels = ["linear", "arima", "prophet", "naive"];

function inputCls(extra = "") {
  return `px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#1e1e2e] text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-[#3b82f6] transition-colors w-full ${extra}`;
}

function labelCls() {
  return "text-[10px] font-mono uppercase tracking-widest text-gray-500 mb-1 block";
}

export default function AnalysisPage() {
  const [tab, setTab] = useState<Tab>("series");
  const [path, setPath] = useState("");

  // Series
  const [seriesCol, setSeriesCol] = useState("value");
  const [seriesPoints, setSeriesPoints] = useState(200);
  const [seriesResult, setSeriesResult] = useState<SeriesResult | null>(null);

  // OHLC
  const [ohlcCol, setOhlcCol] = useState("price");
  const [ohlcBuckets, setOhlcBuckets] = useState(50);
  const [ohlcResult, setOhlcResult] = useState<OhlcResult | null>(null);

  // Aggregate
  const [aggGroupBy, setAggGroupBy] = useState("");
  const [aggMeasure, setAggMeasure] = useState("");
  const [aggType, setAggType] = useState<"sum" | "mean" | "count" | "min" | "max">("sum");
  const [aggResult, setAggResult] = useState<AggregateResult | null>(null);

  // Forecast
  const [fcastCol, setFcastCol] = useState("value");
  const [fcastXCol, setFcastXCol] = useState("date");
  const [fcastHorizon, setFcastHorizon] = useState(30);
  const [fcastModel, setFcastModel] = useState("linear");
  const [fcastResult, setFcastResult] = useState<ForecastResult | null>(null);

  const [error, setError] = useState<string | null>(null);

  const seriesMut = useMutation({
    mutationFn: () => series({ path, column: seriesCol, points: seriesPoints }),
    onSuccess: (data) => { setSeriesResult(data); setError(null); },
    onError: (e) => setError(String(e)),
  });

  const ohlcMut = useMutation({
    mutationFn: () => ohlc({ path, column: ohlcCol, buckets: ohlcBuckets }),
    onSuccess: (data) => { setOhlcResult(data); setError(null); },
    onError: (e) => setError(String(e)),
  });

  const aggMut = useMutation({
    mutationFn: () =>
      aggregate({ path, group_by: aggGroupBy, measure: aggMeasure, agg: aggType }),
    onSuccess: (data) => { setAggResult(data); setError(null); },
    onError: (e) => setError(String(e)),
  });

  const fcastMut = useMutation({
    mutationFn: () =>
      forecast({
        path,
        column: fcastCol,
        x_column: fcastXCol,
        horizon: fcastHorizon,
        model: fcastModel,
      }),
    onSuccess: (data) => { setFcastResult(data); setError(null); },
    onError: (e) => setError(String(e)),
  });

  const handleRun = useCallback(() => {
    setError(null);
    if (tab === "series") seriesMut.mutate();
    else if (tab === "ohlc") ohlcMut.mutate();
    else if (tab === "aggregate") aggMut.mutate();
    else fcastMut.mutate();
  }, [tab, seriesMut, ohlcMut, aggMut, fcastMut]);

  const isPending =
    seriesMut.isPending || ohlcMut.isPending || aggMut.isPending || fcastMut.isPending;

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-5xl mx-auto flex flex-col gap-6">
        <div>
          <h1 className="text-xl font-bold font-mono text-white mb-1">Analysis</h1>
          <p className="text-xs font-mono text-gray-500">Run data analysis against a file path</p>
        </div>

        {/* File path */}
        <div>
          <label className={labelCls()}>File Path</label>
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/data/trades.parquet"
            className={inputCls()}
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-1 border-b border-[#1e1e2e]">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 text-sm font-mono transition-colors border-b-2 -mb-px ${
                tab === t.id
                  ? "border-[#3b82f6] text-[#60a5fa]"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/30 px-4 py-2 text-red-400 text-xs font-mono">
            {error}
          </div>
        )}

        {/* Tab panels */}
        {tab === "series" && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelCls()}>Column</label>
                <input
                  type="text"
                  value={seriesCol}
                  onChange={(e) => setSeriesCol(e.target.value)}
                  placeholder="value"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>Points: {seriesPoints}</label>
                <input
                  type="range"
                  min={10}
                  max={1000}
                  step={10}
                  value={seriesPoints}
                  onChange={(e) => setSeriesPoints(Number(e.target.value))}
                  className="w-full accent-[#3b82f6] mt-2"
                />
              </div>
            </div>
            <RunButton onClick={handleRun} loading={isPending} />
            {seriesResult && (
              <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4">
                <LineChart data={seriesResult.series} column={seriesResult.column} />
              </div>
            )}
          </div>
        )}

        {tab === "ohlc" && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelCls()}>Price Column</label>
                <input
                  type="text"
                  value={ohlcCol}
                  onChange={(e) => setOhlcCol(e.target.value)}
                  placeholder="price"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>Buckets: {ohlcBuckets}</label>
                <input
                  type="range"
                  min={5}
                  max={200}
                  step={5}
                  value={ohlcBuckets}
                  onChange={(e) => setOhlcBuckets(Number(e.target.value))}
                  className="w-full accent-[#3b82f6] mt-2"
                />
              </div>
            </div>
            <RunButton onClick={handleRun} loading={isPending} />
            {ohlcResult && (
              <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4">
                <OhlcChart candles={ohlcResult.candles} />
              </div>
            )}
          </div>
        )}

        {tab === "aggregate" && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className={labelCls()}>Group By Column</label>
                <input
                  type="text"
                  value={aggGroupBy}
                  onChange={(e) => setAggGroupBy(e.target.value)}
                  placeholder="category"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>Measure Column</label>
                <input
                  type="text"
                  value={aggMeasure}
                  onChange={(e) => setAggMeasure(e.target.value)}
                  placeholder="value"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>Aggregation</label>
                <select
                  value={aggType}
                  onChange={(e) =>
                    setAggType(e.target.value as typeof aggType)
                  }
                  className={inputCls()}
                >
                  {aggTypes.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <RunButton onClick={handleRun} loading={isPending} />
            {aggResult && (
              <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#1e1e2e]">
                      {aggResult.columns.map((col) => (
                        <th
                          key={col}
                          className="px-4 py-2 text-left text-gray-500 uppercase tracking-wider font-normal"
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {aggResult.rows.map((row, i) => (
                      <tr
                        key={i}
                        className="border-b border-[#1e1e2e]/50 hover:bg-[#1a1a24] transition-colors"
                      >
                        {row.map((cell, j) => (
                          <td key={j} className="px-4 py-2 text-gray-200">
                            {cell == null ? "null" : String(cell)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {tab === "forecast" && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <div>
                <label className={labelCls()}>Value Column</label>
                <input
                  type="text"
                  value={fcastCol}
                  onChange={(e) => setFcastCol(e.target.value)}
                  placeholder="value"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>X Column</label>
                <input
                  type="text"
                  value={fcastXCol}
                  onChange={(e) => setFcastXCol(e.target.value)}
                  placeholder="date"
                  className={inputCls()}
                />
              </div>
              <div>
                <label className={labelCls()}>Horizon: {fcastHorizon}</label>
                <input
                  type="range"
                  min={5}
                  max={365}
                  step={5}
                  value={fcastHorizon}
                  onChange={(e) => setFcastHorizon(Number(e.target.value))}
                  className="w-full accent-[#3b82f6] mt-2"
                />
              </div>
              <div>
                <label className={labelCls()}>Model</label>
                <select
                  value={fcastModel}
                  onChange={(e) => setFcastModel(e.target.value)}
                  className={inputCls()}
                >
                  {forecastModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <RunButton onClick={handleRun} loading={isPending} />
            {fcastResult && (
              <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4">
                <ResponsiveContainer width="100%" height={320}>
                  <ReLineChart
                    data={fcastResult.points}
                    margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
                    <XAxis
                      dataKey="x"
                      tick={{ fill: "#6b7280", fontSize: 11, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={{ stroke: "#1e1e2e" }}
                    />
                    <YAxis
                      tick={{ fill: "#6b7280", fontSize: 11, fontFamily: "monospace" }}
                      tickLine={false}
                      axisLine={false}
                      width={60}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#13131a",
                        border: "1px solid #1e1e2e",
                        borderRadius: "8px",
                        fontFamily: "monospace",
                        fontSize: "12px",
                        color: "#e5e7eb",
                      }}
                    />
                    <Legend
                      wrapperStyle={{
                        fontFamily: "monospace",
                        fontSize: "11px",
                        color: "#6b7280",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name={`Forecast (${fcastResult.model})`}
                      stroke="#f59e0b"
                      strokeWidth={2}
                      strokeDasharray="6 3"
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </ReLineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function RunButton({ onClick, loading }: { onClick: () => void; loading: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="self-start px-6 py-2.5 rounded-xl bg-[#3b82f6] text-white text-sm font-mono font-medium hover:bg-[#2563eb] transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
    >
      {loading ? "Running…" : "Run Analysis"}
    </button>
  );
}
