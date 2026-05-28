"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { getMetrics, getAudit } from "@/lib/api";
import type { MetricsResponse, AuditEntry } from "@/lib/types";

// ── Relative time ────────────────────────────────────────────
function timeAgo(ts: string | null): string {
  if (!ts) return "--";
  try {
    const now = Date.now();
    const then = new Date(ts).getTime();
    const diffMs = now - then;
    if (diffMs < 0) return "just now";
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
  } catch {
    return "--";
  }
}

function formatDuration(sec: number | null): string {
  if (sec == null) return "--";
  if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

const STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  completed: { bg: "rgba(52,211,153,0.1)", text: "var(--emerald)" },
  success:   { bg: "rgba(52,211,153,0.1)", text: "var(--emerald)" },
  failed:    { bg: "rgba(244,63,94,0.1)",  text: "var(--rose)" },
  error:     { bg: "rgba(244,63,94,0.1)",  text: "var(--rose)" },
  running:   { bg: "rgba(251,191,36,0.1)", text: "var(--amber)" },
  pending:   { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
  queued:    { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
};

function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLES[status] || STATUS_STYLES.pending;
  return (
    <span
      className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded inline-flex items-center gap-1"
      style={{ background: s.bg, color: s.text }}
    >
      {status}
    </span>
  );
}

// ── Per-op terminal colour ───────────────────────────────────
function opColor(op: string): string {
  if (op === "create") return "var(--emerald)";
  if (op === "update") return "var(--frost)";
  if (op === "delete") return "var(--rose)";
  return "var(--foreground-dim)";
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const consoleRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  const fetchMetrics = useCallback(async () => {
    try {
      const m = await getMetrics();
      setMetrics(m);
      setError(false);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAudit = useCallback(async () => {
    try {
      const a = await getAudit(20);
      // Newest first from the API — render oldest -> newest for terminal feel.
      setAudit([...a.entries].reverse());
    } catch {
      // Silently ignore audit errors
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    const id = setInterval(fetchMetrics, 5000);
    return () => clearInterval(id);
  }, [fetchMetrics]);

  useEffect(() => {
    fetchAudit();
    const id = setInterval(fetchAudit, 2000);
    return () => clearInterval(id);
  }, [fetchAudit]);

  // Auto-scroll console to bottom on new entries when enabled.
  useEffect(() => {
    if (autoScroll && consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [audit, autoScroll]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in space-y-4">
          <div className="w-12 h-12 rounded-full bg-rose/10 flex items-center justify-center mx-auto">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--rose)" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          </div>
          <p className="text-sm text-muted">Backend unreachable</p>
        </div>
      </div>
    );
  }

  const maxRuns = Math.max(1, ...metrics.top_by_runs.map((r) => r.runs));
  const maxAvgMs = Math.max(1, ...metrics.top_by_duration.map((r) => r.avg_ms));

  return (
    <div className="relative p-6 space-y-6 overflow-y-auto h-screen animate-in">
      <div className="aurora-bg" />

      {/* Header */}
      <div className="relative flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground glow-frost">Metrics</h1>
          <p className="text-sm text-muted mt-1">
            <span className="font-mono">{metrics.node_id}</span> performance analytics
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          <span className="text-[11px] font-mono text-muted">Live - 5s</span>
        </div>
      </div>

      <div className="relative grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top by runs */}
        <div className="runic-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
            Top 5 Most-Run Functions
          </h2>
          {metrics.top_by_runs.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No runs yet</p>
          ) : (
            <div className="space-y-2">
              {metrics.top_by_runs.slice(0, 5).map((row) => {
                const pct = (row.runs / maxRuns) * 100;
                return (
                  <div key={row.id} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-foreground-dim truncate">{row.name}</span>
                      <span className="text-xs font-mono text-frost gradient-frost font-semibold ml-2 shrink-0">{row.runs}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${pct}%`,
                          background: "linear-gradient(90deg, var(--frost), var(--emerald))",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Top by duration */}
        <div className="runic-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            Top 5 Slowest Functions
          </h2>
          {metrics.top_by_duration.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No runs yet</p>
          ) : (
            <div className="space-y-2">
              {metrics.top_by_duration.slice(0, 5).map((row) => {
                const pct = (row.avg_ms / maxAvgMs) * 100;
                return (
                  <div key={row.id} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-foreground-dim truncate">{row.name}</span>
                      <span className="text-xs font-mono text-amber glow-amber font-semibold ml-2 shrink-0">{formatMs(row.avg_ms)}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${pct}%`,
                          background: "linear-gradient(90deg, var(--amber), var(--rose))",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Success rate */}
        <div className="runic-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            Success Rate Ranking
          </h2>
          {metrics.success_rate.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No runs yet</p>
          ) : (
            <div className="space-y-2">
              {metrics.success_rate.slice(0, 5).map((row) => {
                const pct = row.rate * 100;
                const color = pct >= 90
                  ? "var(--emerald)"
                  : pct >= 60
                  ? "var(--amber)"
                  : "var(--rose)";
                return (
                  <div key={row.id} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-foreground-dim truncate">{row.name}</span>
                      <span
                        className="text-xs font-mono font-semibold ml-2 shrink-0"
                        style={{ color }}
                      >
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, background: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Recent runs */}
        <div className="runic-card p-5 space-y-3 lg:col-span-1">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="2" x2="12" y2="6" />
              <line x1="12" y1="18" x2="12" y2="22" />
              <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
              <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
              <line x1="2" y1="12" x2="6" y2="12" />
              <line x1="18" y1="12" x2="22" y2="12" />
            </svg>
            Recent Runs Timeline
          </h2>
          {metrics.recent_runs.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No recent runs</p>
          ) : (
            <div className="space-y-1">
              {metrics.recent_runs.slice(0, 10).map((run) => {
                const slow = run.duration != null && run.duration > 5;
                return (
                  <div
                    key={run.id}
                    className="flex items-center gap-3 px-2 py-1.5 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
                  >
                    <StatusBadge status={run.status} />
                    <span className="text-[10px] font-mono text-muted shrink-0">#{run.id}</span>
                    <span className="text-[10px] font-mono text-foreground-dim shrink-0">fn#{run.func_id}</span>
                    <span
                      className="text-[10px] font-mono shrink-0 ml-auto"
                      style={{ color: slow ? "var(--amber)" : "var(--muted)" }}
                    >
                      {formatDuration(run.duration)}
                    </span>
                    <span className="text-[10px] font-mono text-muted/60 shrink-0 w-16 text-right">
                      {timeAgo(run.started_at)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Live Console — terminal-style audit stream ───────── */}
      <div className="relative runic-card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="4 17 10 11 4 5" />
              <line x1="12" y1="19" x2="20" y2="19" />
            </svg>
            Live Console
            <span className="relative flex h-2 w-2 ml-1">
              <span
                className="absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ background: "var(--emerald)", animation: "pulse-frost 1.5s ease-in-out infinite" }}
              />
              <span className="relative inline-flex rounded-full h-2 w-2" style={{ background: "var(--emerald)" }} />
            </span>
          </h2>
          <label className="flex items-center gap-1.5 text-[10px] text-muted font-mono cursor-pointer">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
              className="accent-frost"
            />
            auto-scroll
          </label>
        </div>
        <div
          ref={consoleRef}
          className="font-mono text-[11px] leading-relaxed rounded-lg bg-[#04040b] border border-white/[0.06] p-3 h-64 overflow-y-auto"
        >
          {audit.length === 0 ? (
            <p className="text-muted/40 italic">No events yet — waiting for activity...</p>
          ) : (
            audit.map((entry, i) => (
              <div
                key={`${entry.asset_id}-${entry.timestamp}-${i}`}
                className="whitespace-nowrap"
              >
                <span className="text-muted/60">[{entry.timestamp}]</span>{" "}
                <span style={{ color: opColor(entry.operation) }}>{entry.operation}</span>{" "}
                <span className="text-foreground-dim">{entry.asset_type}</span>{" "}
                <span className="text-muted">#{entry.asset_id}</span>
                {entry.detail && (
                  <>
                    {" "}
                    <span className="text-muted/70">({entry.detail})</span>
                  </>
                )}
              </div>
            ))
          )}
          {/* Blinking caret at the bottom */}
          <span
            className="inline-block w-2 h-3 align-middle"
            style={{
              background: "var(--frost)",
              animation: "pulse-frost 1s steps(1) infinite",
            }}
          />
        </div>
      </div>
    </div>
  );
}
