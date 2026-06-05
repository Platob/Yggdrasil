"use client";

import { useEffect, useMemo, useState } from "react";
import type { PlanGraph, PlanOp, PlanEdit } from "@/lib/api";

const OP_STYLE: Record<string, { color: string; glyph: string }> = {
  scan: { color: "#67e8f9", glyph: "▦" },        // frost
  join: { color: "#a78bfa", glyph: "⋈" },        // violet
  union: { color: "#a78bfa", glyph: "⊎" },
  filter: { color: "#fbbf24", glyph: "▽" },      // amber
  having: { color: "#fbbf24", glyph: "▽" },
  aggregate: { color: "#34d399", glyph: "Σ" },   // emerald
  distinct: { color: "#f43f5e", glyph: "≠" },    // rose
  project: { color: "#60a5fa", glyph: "⊓" },     // blue
  sort: { color: "#22d3ee", glyph: "↕" },        // cyan
  limit: { color: "#fb923c", glyph: "✂" },       // orange
};

function fmtRows(n: number | null): string {
  if (n == null) return "";
  if (n < 1000) return `${n}`;
  if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1_000_000).toFixed(1)}M`;
}

interface Props {
  graph: PlanGraph;
  busy?: boolean;
  onAnalyze: () => void;
  onApply: (edits: PlanEdit[]) => void;  // re-emit SQL from edits, then run
}

export default function PlanGraphView({ graph, busy, onAnalyze, onApply }: Props) {
  const ops = graph.ops;
  const [step, setStep] = useState(ops.length);
  const [playing, setPlaying] = useState(false);
  const [limitDraft, setLimitDraft] = useState<string>("");

  useEffect(() => { setStep(ops.length); setPlaying(false); }, [graph]);

  useEffect(() => {
    if (!playing) return;
    if (step >= ops.length) { setPlaying(false); return; }
    const t = setTimeout(() => setStep((s) => Math.min(ops.length, s + 1)), 750);
    return () => clearTimeout(t);
  }, [playing, step, ops.length]);

  const maxMs = useMemo(
    () => Math.max(1, ...ops.map((o) => o.elapsed_ms ?? 0)),
    [ops],
  );

  // Leading scan ops feed a join/union; render them on one branched row.
  const scans = ops.filter((o) => o.op === "scan");
  const pipeline = ops.filter((o) => o.op !== "scan");
  const branched = scans.length > 1;

  const play = () => {
    if (step >= ops.length) setStep(0);
    setPlaying(true);
  };

  const dropEditFor = (op: string): PlanEdit | null => {
    if (op === "filter" || op === "having") return { op: "drop_filter" };
    if (op === "aggregate") return { op: "drop_group" };
    if (op === "sort") return { op: "drop_order" };
    if (op === "distinct") return { op: "drop_distinct" };
    if (op === "limit") return { op: "drop_limit" };
    return null;
  };

  const Card = ({ o, idx }: { o: PlanOp; idx: number }) => {
    const st = OP_STYLE[o.op] ?? { color: "#9a9894", glyph: "•" };
    const active = idx < step;
    const drop = dropEditFor(o.op);
    return (
      <div
        className="relative rounded-lg border px-3 py-2 transition-all"
        style={{
          borderColor: active ? `${st.color}55` : "rgba(255,255,255,0.06)",
          background: active ? `${st.color}10` : "rgba(255,255,255,0.02)",
          opacity: active ? 1 : 0.35,
          width: branched && o.op === "scan" ? 180 : 280,
        }}
      >
        <div className="flex items-center gap-2">
          <span className="text-base leading-none" style={{ color: st.color }}>{st.glyph}</span>
          <span className="text-xs font-semibold" style={{ color: st.color }}>{o.title}</span>
          <div className="flex-1" />
          {o.rows != null && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/[0.05] text-foreground-dim">
              {fmtRows(o.rows)} rows
            </span>
          )}
          {drop && (
            <button onClick={() => onApply([drop])} title={`remove ${o.title}`}
              className="text-[11px] text-rose/60 hover:text-rose leading-none">✕</button>
          )}
        </div>
        {o.detail && <div className="text-[11px] font-mono text-foreground-dim truncate mt-0.5" title={o.detail}>{o.detail}</div>}

        {/* limit is live-editable */}
        {o.op === "limit" && (
          <div className="flex items-center gap-1 mt-1">
            <input
              type="number" min={1} placeholder="limit"
              value={limitDraft}
              onChange={(e) => setLimitDraft(e.target.value)}
              className="w-20 bg-white/[0.05] border border-white/[0.08] rounded px-1.5 py-0.5 text-[11px] font-mono outline-none focus:border-frost/40"
            />
            <button
              onClick={() => limitDraft && onApply([{ op: "set_limit", value: Number(limitDraft) }])}
              className="text-[11px] text-frost/80 hover:text-frost">apply</button>
          </div>
        )}

        {/* elapsed bar */}
        {o.elapsed_ms != null && (
          <div className="mt-1.5">
            <div className="h-1.5 rounded-full bg-white/[0.05] overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${Math.max(3, (o.elapsed_ms / maxMs) * 100)}%`, background: st.color }} />
            </div>
            <div className="text-[10px] font-mono text-muted mt-0.5">{o.elapsed_ms} ms</div>
          </div>
        )}
      </div>
    );
  };

  const connector = (
    <div className="flex justify-center py-1" style={{ width: 280 }}>
      <svg width="12" height="16" viewBox="0 0 12 16"><path d="M6 0v11M2 8l4 4 4-4" stroke="rgba(255,255,255,0.25)" strokeWidth="1.5" fill="none" /></svg>
    </div>
  );

  return (
    <div className="p-3">
      {/* controls */}
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        <button onClick={onAnalyze} disabled={busy}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40">
          {busy ? "Analyzing…" : graph.analyzed ? "Re-analyze ↻" : "Analyze ▸"}
        </button>
        <div className="flex items-center gap-1.5">
          <button onClick={() => (playing ? setPlaying(false) : play())}
            className="px-2.5 py-1.5 rounded-lg text-xs bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08]">
            {playing ? "❚❚" : "▶"} <span className="text-muted">step</span>
          </button>
          <input type="range" min={0} max={ops.length} value={step}
            onChange={(e) => { setPlaying(false); setStep(Number(e.target.value)); }}
            className="w-32 accent-frost" />
          <span className="text-[11px] font-mono text-muted">{step}/{ops.length}</span>
        </div>
        {graph.analyzed && graph.total_ms != null && (
          <span className="text-[11px] font-mono text-foreground-dim ml-auto">
            total {graph.total_ms} ms{graph.sampled ? " · sampled" : ""}
          </span>
        )}
      </div>

      {/* graph */}
      <div className="flex flex-col items-center">
        {branched ? (
          <>
            <div className="flex gap-3 flex-wrap justify-center">
              {scans.map((o) => <Card key={o.id} o={o} idx={ops.indexOf(o)} />)}
            </div>
            {connector}
          </>
        ) : (
          scans.map((o, i) => (
            <div key={o.id} className="flex flex-col items-center">
              <Card o={o} idx={ops.indexOf(o)} />
              {(i < scans.length - 1 || pipeline.length > 0) && connector}
            </div>
          ))
        )}
        {pipeline.map((o, i) => (
          <div key={o.id} className="flex flex-col items-center">
            <Card o={o} idx={ops.indexOf(o)} />
            {i < pipeline.length - 1 && connector}
          </div>
        ))}
      </div>

      {/* emitted SQL */}
      <div className="mt-4">
        <div className="text-[11px] uppercase tracking-wide text-muted mb-1">Emitted SQL · {graph.dialect}</div>
        <pre className="text-[12px] font-mono text-frost/90 whitespace-pre-wrap break-words bg-[#06060f] border border-white/[0.06] rounded p-2">{graph.plan_sql || "—"}</pre>
      </div>
    </div>
  );
}
