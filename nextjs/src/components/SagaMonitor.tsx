"use client";

import { useCallback, useEffect, useState } from "react";
import {
  getSagaOverview, getMounts, createMount, deleteMount, listMount,
  type SagaOverview, type MountEntry, type MountNode, type TopAsset, type OpLogEntry,
} from "@/lib/api";
import { Sparkline } from "@/components/Sparkline";

// Per mount-kind glyph + colour so the family reads at a glance — mirrors the
// catalog tree's object-type vocabulary.
const KIND: Record<string, { glyph: string; color: string }> = {
  database: { glyph: "⛁", color: "text-violet-400" },
  databricks: { glyph: "◆", color: "text-amber/80" },
  s3: { glyph: "☁", color: "text-frost/80" },
  node: { glyph: "⬡", color: "text-emerald/80" },
  http: { glyph: "🌐", color: "text-frost/70" },
  local: { glyph: "▢", color: "text-muted" },
};
const kindOf = (k: string) => KIND[k] ?? KIND.local;

function fmtBytes(b: number | null | undefined): string {
  if (!b) return "--";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
function fmtNum(n: number | null | undefined): string {
  if (n == null) return "--";
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(1)}K`;
  return `${(n / 1e6).toFixed(2)}M`;
}
function ago(ts: string | null | undefined): string {
  if (!ts) return "--";
  const d = (Date.now() - new Date(ts).getTime()) / 1000;
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

// Op → tint, matching the activity language elsewhere in the app.
const opColor: Record<string, string> = {
  register: "text-emerald/80", update: "text-frost/80", drop: "text-rose/80",
  delete: "text-rose/80", query: "text-amber/80", replicate: "text-violet-400",
};

function Kpi({ label, value, sub, color = "text-foreground" }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="glass-card p-3 flex flex-col gap-0.5">
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className={`font-mono text-xl font-semibold ${color}`}>{value}</span>
      {sub && <span className="text-[10px] text-muted">{sub}</span>}
    </div>
  );
}

interface Props {
  node?: string;
  /** Drop a reference into the SQL editor (alias/sub or full_name). */
  onQuery?: (ref: string) => void;
}

export default function SagaMonitor({ node, onQuery }: Props) {
  const [ov, setOv] = useState<SagaOverview | null>(null);
  const [mounts, setMounts] = useState<MountEntry[]>([]);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [mName, setMName] = useState("");
  const [mTarget, setMTarget] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [listing, setListing] = useState<Record<string, MountNode[]>>({});

  const load = useCallback(async () => {
    setErr("");
    try {
      const [o, m] = await Promise.all([getSagaOverview(node, true), getMounts(node, true)]);
      setOv(o); setMounts(m.mounts);
    } catch (e) { setErr(String(e)); }
  }, [node]);

  useEffect(() => { load(); }, [load]);
  // Live-ish refresh so the monitor reflects ongoing catalog activity.
  useEffect(() => {
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);

  const onAdd = async () => {
    if (!mName.trim() || !mTarget.trim()) return;
    setBusy(true); setErr("");
    try {
      await createMount({ name: mName.trim(), target: mTarget.trim() });
      setMName(""); setMTarget(""); setShowAdd(false);
      await load();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const onDelete = async (name: string) => {
    if (!window.confirm(`Drop mount '${name}'? (the underlying data is untouched)`)) return;
    setBusy(true);
    try { await deleteMount(name); await load(); }
    catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const toggleMount = async (m: MountEntry) => {
    if (expanded === m.name) { setExpanded(null); return; }
    setExpanded(m.name);
    if (!listing[m.name]) {
      try {
        const r = await listMount(m.name, "", node);
        setListing((l) => ({ ...l, [m.name]: r.entries }));
      } catch (e) { setListing((l) => ({ ...l, [m.name]: [] })); setErr(String(e)); }
    }
  };

  if (err && !ov) return <div className="glass-card p-4 text-rose/80 font-mono text-xs break-words">{err}</div>;
  if (!ov) return <div className="glass-card p-4 text-muted text-xs">Loading monitor…</div>;

  const assetTotal = ov.table_count + ov.view_count + ov.forecast_count + ov.other_count;

  return (
    <div className="flex flex-col gap-3 overflow-auto min-h-0 pr-1 animate-in">
      {err && <div className="text-[11px] text-rose/80 font-mono break-words">{err}</div>}

      {/* KPI strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-2">
        <Kpi label="Assets" value={String(assetTotal)} sub={`${ov.table_count} tbl · ${ov.view_count} view · ${ov.forecast_count} fc`} color="text-emerald" />
        <Kpi label="Mounts" value={String(ov.mount_count)} sub={Object.entries(ov.mount_kinds).map(([k, v]) => `${v} ${k}`).join(" · ") || "none"} color="text-violet-400" />
        <Kpi label="Catalogs" value={String(ov.catalog_count)} sub={`${ov.schema_count} schema${ov.schema_count === 1 ? "" : "s"}`} color="text-frost" />
        <Kpi label="Total rows" value={fmtNum(ov.total_rows)} sub={fmtBytes(ov.total_bytes)} />
        <Kpi label="Operations" value={fmtNum(ov.total_ops)} sub={Object.entries(ov.op_counts).map(([k, v]) => `${v} ${k}`).join(" · ")} color="text-amber" />
        <div className="glass-card p-3 flex flex-col gap-0.5">
          <span className="text-[10px] uppercase tracking-wider text-muted">Activity (14d)</span>
          <Sparkline data={ov.daily.length ? ov.daily : [0]} color="var(--frost)" height={34} />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-3">
        {/* ── Mounts management ── */}
        <div className="glass-card p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] uppercase tracking-wide text-muted">Mounts</span>
            <button onClick={() => setShowAdd((s) => !s)} className="text-[11px] text-emerald/90 hover:text-emerald">+ mount</button>
          </div>
          {showAdd && (
            <div className="mb-3 p-2 rounded-lg bg-white/[0.03] border border-white/[0.06] space-y-2">
              <input value={mName} onChange={(e) => setMName(e.target.value)} placeholder="alias (e.g. prod_vol)"
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs outline-none focus:border-frost/30" />
              <input value={mTarget} onChange={(e) => setMTarget(e.target.value)}
                placeholder="target: /Volumes/… · s3://… · npfs://… · postgres://… · sqlite://…"
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono outline-none focus:border-frost/30" />
              <div className="flex justify-end gap-2">
                <button onClick={() => setShowAdd(false)} className="px-2 py-1 text-[11px] text-muted hover:text-foreground">cancel</button>
                <button onClick={onAdd} disabled={busy || !mName.trim() || !mTarget.trim()}
                  className="px-2.5 py-1 rounded text-[11px] font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">create</button>
              </div>
            </div>
          )}
          {mounts.length === 0 && <div className="text-[11px] text-muted py-2">No mounts. Add one to query a volume, S3 prefix, node path, or live database.</div>}
          <div className="space-y-0.5">
            {mounts.map((m) => {
              const k = kindOf(m.kind);
              const open = expanded === m.name;
              return (
                <div key={m.id}>
                  <div className="group flex items-center gap-1.5 py-1 px-1 rounded hover:bg-white/[0.03]">
                    <button onClick={() => toggleMount(m)} className="flex items-center gap-1.5 flex-1 min-w-0 cursor-pointer text-left">
                      <span className="text-[9px] text-muted w-2">{open ? "▾" : "▸"}</span>
                      <span className={k.color}>{k.glyph}</span>
                      <span className="text-xs font-mono truncate">{m.name}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.04] text-muted">{m.kind}</span>
                    </button>
                    <span className="text-[10px] text-muted/70 truncate max-w-[40%] hidden md:inline" title={m.target}>{m.target}</span>
                    <button onClick={() => onDelete(m.name)} title="drop mount"
                      className="opacity-0 group-hover:opacity-100 text-[11px] text-rose/70 hover:text-rose px-1">✕</button>
                  </div>
                  {open && (
                    <div className="ml-5 border-l border-white/[0.06] pl-2 py-0.5 space-y-0.5">
                      {(listing[m.name] ?? []).length === 0 && <div className="text-[10px] text-muted py-1">empty / loading…</div>}
                      {(listing[m.name] ?? []).map((e) => (
                        <div key={e.path} className="group/leaf flex items-center gap-1.5 py-0.5 px-1 rounded hover:bg-white/[0.03]">
                          <span className={e.is_dir ? "text-frost/60" : e.is_tabular ? "text-emerald/70" : "text-muted"}>
                            {e.is_dir ? "▸" : e.is_tabular ? "▦" : "·"}
                          </span>
                          <span className="text-[11px] font-mono truncate flex-1">{e.name}</span>
                          {e.is_tabular && onQuery && (
                            <button onClick={() => onQuery(`${m.name}/${e.path}`)} title="query this"
                              className="opacity-0 group-hover/leaf:opacity-100 text-[10px] text-frost/80 hover:text-frost px-1">query →</button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Recent activity feed ── */}
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">Recent activity</span>
          <div className="mt-2 space-y-0.5 max-h-[340px] overflow-auto">
            {ov.recent.length === 0 && <div className="text-[11px] text-muted py-2">No operations logged yet.</div>}
            {ov.recent.map((r: OpLogEntry, i) => (
              <div key={i} className="flex items-center gap-2 py-1 text-[11px] border-b border-white/[0.03] last:border-0">
                <span className={`font-mono w-16 ${opColor[r.op] ?? "text-muted"}`}>{r.op}</span>
                <span className="flex-1 truncate text-foreground-dim" title={r.detail || r.statement}>{r.detail || r.statement || "—"}</span>
                <span className="text-muted/70 w-16 text-right">{ago(r.ts)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Leaderboards ── */}
      <div className="grid grid-cols-1 xl:grid-cols-[1fr_1fr] gap-3">
        <Leaderboard title="Largest tables" rows={ov.largest} metric={(a) => fmtBytes(a.size_bytes)} sub={(a) => `${fmtNum(a.rows)} rows`} onQuery={onQuery} />
        <Leaderboard title="Busiest assets" rows={ov.busiest} metric={(a) => `${a.ops} ops`} sub={(a) => ago(a.last_op_at)} onQuery={onQuery} />
      </div>
    </div>
  );
}

function Leaderboard({ title, rows, metric, sub, onQuery }: {
  title: string; rows: TopAsset[]; metric: (a: TopAsset) => string; sub: (a: TopAsset) => string;
  onQuery?: (ref: string) => void;
}) {
  return (
    <div className="glass-card p-3">
      <span className="text-[11px] uppercase tracking-wide text-muted">{title}</span>
      <div className="mt-2 space-y-0.5">
        {rows.length === 0 && <div className="text-[11px] text-muted py-2">No data yet.</div>}
        {rows.map((a) => (
          <div key={a.full_name} className="group flex items-center gap-2 py-1 text-[11px] border-b border-white/[0.03] last:border-0">
            <span className="flex-1 truncate font-mono" title={a.full_name}>{a.full_name}</span>
            <span className="text-foreground-dim w-20 text-right">{metric(a)}</span>
            <span className="text-muted/70 w-16 text-right hidden md:inline">{sub(a)}</span>
            {onQuery && (
              <button onClick={() => onQuery(a.full_name)} title="query this"
                className="opacity-0 group-hover:opacity-100 text-[10px] text-frost/80 hover:text-frost px-1">→</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
