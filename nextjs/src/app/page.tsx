"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { BrainHero } from "@/components/BrainHero";
import { YggLogoIcon } from "@/components/YggLogo";
import { getNodeCard, getStats, getTopology } from "@/lib/api";
import type { NodeCard, ClusterStats, TopologyResponse } from "@/lib/types";

export default function WelcomePage() {
  const [card, setCard] = useState<NodeCard | null>(null);
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const [topology, setTopology] = useState<TopologyResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getNodeCard()
      .then(setCard)
      .catch(() => setError(true));
  }, []);

  // Live ticker: poll /api/v2/stats every 5s
  useEffect(() => {
    let active = true;
    const tick = () => {
      getStats()
        .then((s) => {
          if (active) setStats(s);
        })
        .catch(() => {
          // Silently ignore stats fetch errors
        });
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  // Topology poll powers the neuron/synapse count pill below the title
  useEffect(() => {
    let active = true;
    const tick = () => {
      getTopology()
        .then((t) => {
          if (active) setTopology(t);
        })
        .catch(() => {
          // Silently ignore; pill just won't render
        });
    };
    tick();
    const id = setInterval(tick, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const nodeId = card?.node_id ?? "---";
  const role = card?.role ?? "hybrid";
  const version = card?.version ?? "";

  // Real graph counts: nodes in the mesh and live peer→self links (synapses).
  const neuronCount = topology?.nodes.length ?? 0;
  const linkCount = Math.max(0, neuronCount - 1);

  const kpis: [string, string | number, string][] = stats
    ? [
        ["nodes", neuronCount, "text-frost"],
        ["links", linkCount, "text-emerald"],
        ["funcs", stats.func_count, "text-foreground"],
        ["envs", stats.env_count, "text-foreground"],
        ["dags", stats.dag_count, "text-foreground"],
        ["active", stats.active_runs, "text-amber"],
        ["runs", stats.total_runs, "text-foreground"],
      ]
    : [];

  return (
    <div className="relative w-full h-screen overflow-hidden brain-wave-bg">
      {/* Aurora ambient layer behind the brain */}
      <div className="aurora-bg z-0" />

      {/* Full-bleed brain hero — the main feature, untouched */}
      <BrainHero className="absolute inset-0 w-full h-full" />

      {/* Subtle edge vignettes so corner chrome stays readable over the brain */}
      <div className="absolute inset-x-0 top-0 h-32 pointer-events-none z-10"
        style={{ background: "linear-gradient(to bottom, rgba(5,5,16,0.7) 0%, transparent 100%)" }} />
      <div className="absolute inset-x-0 bottom-0 h-40 pointer-events-none z-10"
        style={{ background: "linear-gradient(to top, rgba(5,5,16,0.8) 0%, transparent 100%)" }} />

      {/* Twinkling stars */}
      <div className="absolute inset-0 pointer-events-none z-10">
        <span className="twinkle absolute top-[12%] left-[40%] w-1 h-1 rounded-full bg-frost" style={{ animationDelay: "0s" }} />
        <span className="twinkle absolute top-[24%] right-[32%] w-[3px] h-[3px] rounded-full bg-white" style={{ animationDelay: "0.8s" }} />
        <span className="twinkle absolute bottom-[34%] left-[46%] w-[2px] h-[2px] rounded-full bg-frost" style={{ animationDelay: "1.5s" }} />
        <span className="twinkle absolute top-[40%] right-[24%] w-1 h-1 rounded-full bg-emerald" style={{ animationDelay: "2.2s" }} />
      </div>

      {/* Top-left: identity (discrete) */}
      <div className="absolute top-5 left-6 z-20 select-none">
        <div className="flex items-center gap-2.5">
          <YggLogoIcon size={32} />
          <div className="leading-tight">
            <h1 className="text-xl font-bold tracking-[0.28em] uppercase text-foreground"
              style={{ textShadow: "0 0 24px rgba(103,232,249,0.35)" }}>
              Yggdrasil
            </h1>
            <p className="text-[9px] tracking-[0.25em] uppercase text-muted/80">Living Brain · Distributed Computing</p>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-2 text-[11px] font-mono">
          <span className={`w-1.5 h-1.5 rounded-full ${error ? "bg-rose" : "status-online neural-pulse"}`} />
          <span className={error ? "text-rose/80" : "text-frost/80"}>{error ? "Backend unreachable" : nodeId}</span>
          {!error && role && (<><span className="text-border">|</span><span className="text-foreground-dim capitalize">{role}</span></>)}
          {!error && version && (<><span className="text-border">|</span><span className="text-foreground-dim">v{version}</span></>)}
        </div>
      </div>

      {/* Top-right: CTA (discrete) */}
      <Link href="/nodes" className="absolute top-5 right-6 z-20 runic-card group px-4 py-2 text-xs font-semibold text-frost"
        style={{ boxShadow: "0 0 20px rgba(103,232,249,0.08)" }}>
        <span className="group-hover:gradient-frost transition-all">View Nodes →</span>
      </Link>

      {/* Bottom-left: live KPI card (discrete glass) */}
      {kpis.length > 0 && (
        <div className="absolute bottom-6 left-6 z-20 glass-card px-3 py-2.5">
          <div className="flex items-stretch divide-x divide-white/[0.06]">
            {kpis.map(([label, value, color]) => (
              <div key={label} className="px-3 first:pl-1 text-center">
                <div className={`text-lg font-mono font-semibold ${color}`}>{value}</div>
                <div className="text-[9px] uppercase tracking-widest text-muted/70">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottom hint */}
      <p className="absolute bottom-7 right-6 text-[10px] text-white/20 pointer-events-none z-20 whitespace-nowrap hidden md:block">
        Drag to rotate · scroll to zoom · busy nodes sit nearer the core
      </p>
    </div>
  );
}
