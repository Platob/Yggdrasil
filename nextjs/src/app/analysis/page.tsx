"use client";

import { useMemo, useState } from "react";
import { FlaskConical } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { runAggregate } from "@/lib/api";
import type { AggregateResponse } from "@/lib/types";
import { fmtNum } from "@/lib/format";
import { ErrorBanner, Panel, Spinner } from "@/components/ui";

const AGGS = ["sum", "mean", "min", "max", "count", "median", "std"];
const MAX_SERIES_POINTS = 500;

export default function AnalysisPage() {
  const [path, setPath] = useState("");
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [column, setColumn] = useState("");
  const [agg, setAgg] = useState("sum");
  const [groupBy, setGroupBy] = useState("");
  const [result, setResult] = useState<AggregateResponse | null>(null);
  const [seriesCol, setSeriesCol] = useState("");
  const [seriesData, setSeriesData] = useState<{ i: number; value: number }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // "Load" probes the file with a cheap count aggregate to discover its columns.
  const onLoad = async () => {
    if (!path.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runAggregate({ path: path.trim(), column: "*", agg: "count" });
      setColumns(res.columns);
      setLoadedPath(path.trim());
      if (res.columns.length) {
        setColumn(res.columns[0]);
        setSeriesCol(res.columns[0]);
      }
      setResult(null);
      setSeriesData([]);
    } catch (err) {
      setError((err as Error).message);
      setColumns([]);
      setLoadedPath(null);
    } finally {
      setLoading(false);
    }
  };

  const onRun = async () => {
    if (!loadedPath || !column) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runAggregate({
        path: loadedPath,
        column,
        agg,
        group_by: groupBy || null,
      });
      setResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const onPlotSeries = async () => {
    if (!loadedPath || !seriesCol) return;
    setLoading(true);
    setError(null);
    try {
      // Reuse the aggregate endpoint grouped by row to pull the raw series, then
      // downsample client-side to keep the chart responsive.
      const res = await runAggregate({
        path: loadedPath,
        column: seriesCol,
        agg: "series",
      });
      const values = res.rows.map((r) => Number(r.value));
      const step = Math.max(1, Math.ceil(values.length / MAX_SERIES_POINTS));
      const sampled: { i: number; value: number }[] = [];
      for (let i = 0; i < values.length; i += step) {
        sampled.push({ i, value: values[i] });
      }
      setSeriesData(sampled);
    } catch (err) {
      setError((err as Error).message);
      setSeriesData([]);
    } finally {
      setLoading(false);
    }
  };

  const chartData = useMemo(
    () =>
      result?.rows.map((r) => ({
        group: String(r.group),
        value: Number(r.value),
      })) ?? [],
    [result],
  );

  return (
    <div className="p-4 md:p-6 pt-20 md:pt-6 space-y-6">
      <header className="flex items-center gap-2">
        <FlaskConical className="text-green-400" size={22} />
        <h1 className="text-xl font-semibold text-gray-100">Analysis</h1>
      </header>

      {error && <ErrorBanner message={error} />}

      <Panel title="Data Source">
        <div className="p-4 flex flex-wrap items-end gap-3">
          <div className="flex-1 min-w-[240px]">
            <label className="block text-xs text-gray-500 mb-1">File path</label>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onLoad()}
              placeholder="/path/to/data.parquet"
              className="w-full bg-gray-950 border border-gray-800 rounded-md px-3 py-1.5 text-sm font-mono text-gray-200 focus:outline-none focus:border-green-500"
            />
          </div>
          <button
            onClick={onLoad}
            disabled={loading || !path.trim()}
            className="px-4 py-1.5 rounded-md text-sm bg-green-500/15 text-green-400 hover:bg-green-500/25 disabled:opacity-50"
          >
            Load
          </button>
        </div>
        {loadedPath && (
          <div className="px-4 pb-3 text-xs text-gray-500 font-mono">
            {columns.length} columns · {loadedPath}
          </div>
        )}
      </Panel>

      {columns.length > 0 && (
        <>
          <Panel title="Aggregate">
            <div className="p-4 flex flex-wrap items-end gap-3">
              <Field label="Column">
                <Select value={column} onChange={setColumn} options={columns} />
              </Field>
              <Field label="Aggregation">
                <Select value={agg} onChange={setAgg} options={AGGS} />
              </Field>
              <Field label="Group by">
                <Select
                  value={groupBy}
                  onChange={setGroupBy}
                  options={["", ...columns]}
                  labels={{ "": "(none)" }}
                />
              </Field>
              <button
                onClick={onRun}
                disabled={loading || !column}
                className="px-4 py-1.5 rounded-md text-sm bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 disabled:opacity-50"
              >
                Run Aggregate
              </button>
            </div>

            {loading && !result ? (
              <Spinner />
            ) : result ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4 pt-0">
                <div style={{ height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="#1f2937" vertical={false} />
                      <XAxis
                        dataKey="group"
                        tick={{ fill: "#6b7280", fontSize: 10 }}
                        axisLine={{ stroke: "#1f2937" }}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fill: "#6b7280", fontSize: 10 }}
                        axisLine={{ stroke: "#1f2937" }}
                        tickLine={false}
                        tickFormatter={(v) => fmtNum(v, 0)}
                        width={64}
                      />
                      <Tooltip
                        cursor={{ fill: "#1f2937", opacity: 0.4 }}
                        contentStyle={{
                          background: "#111827",
                          border: "1px solid #374151",
                          borderRadius: 6,
                          fontSize: 12,
                        }}
                      />
                      <Bar dataKey="value" fill="#3b82f6" isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="overflow-auto max-h-[300px]">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-gray-500 border-b border-gray-800 sticky top-0 bg-gray-900">
                        <th className="px-4 py-2 font-medium">
                          {result.group_by ?? "Group"}
                        </th>
                        <th className="px-4 py-2 font-medium text-right">
                          {result.agg}({result.column})
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.rows.map((r, i) => (
                        <tr
                          key={i}
                          className="border-b border-gray-800/60 hover:bg-gray-800/30"
                        >
                          <td className="px-4 py-2 font-mono text-gray-300">
                            {String(r.group)}
                          </td>
                          <td className="px-4 py-2 text-right font-mono text-gray-100">
                            {fmtNum(Number(r.value))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </Panel>

          <Panel title="Series Plot">
            <div className="p-4 flex flex-wrap items-end gap-3">
              <Field label="Column">
                <Select value={seriesCol} onChange={setSeriesCol} options={columns} />
              </Field>
              <button
                onClick={onPlotSeries}
                disabled={loading || !seriesCol}
                className="px-4 py-1.5 rounded-md text-sm bg-blue-500/15 text-blue-400 hover:bg-blue-500/25 disabled:opacity-50"
              >
                Plot
              </button>
              {seriesData.length > 0 && (
                <span className="text-xs text-gray-500 font-mono">
                  {seriesData.length} points (downsampled)
                </span>
              )}
            </div>
            {seriesData.length > 0 && (
              <div className="p-4 pt-0" style={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={seriesData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#1f2937" vertical={false} />
                    <XAxis
                      dataKey="i"
                      tick={{ fill: "#6b7280", fontSize: 10 }}
                      axisLine={{ stroke: "#1f2937" }}
                      tickLine={false}
                      minTickGap={40}
                    />
                    <YAxis
                      tick={{ fill: "#6b7280", fontSize: 10 }}
                      axisLine={{ stroke: "#1f2937" }}
                      tickLine={false}
                      tickFormatter={(v) => fmtNum(v, 0)}
                      width={64}
                    />
                    <Tooltip
                      cursor={{ stroke: "#374151" }}
                      contentStyle={{
                        background: "#111827",
                        border: "1px solid #374151",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </Panel>
        </>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-xs text-gray-500 mb-1">{label}</label>
      {children}
    </div>
  );
}

function Select({
  value,
  onChange,
  options,
  labels,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  labels?: Record<string, string>;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-gray-950 border border-gray-800 rounded-md px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-green-500"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {labels?.[o] ?? o}
        </option>
      ))}
    </select>
  );
}
