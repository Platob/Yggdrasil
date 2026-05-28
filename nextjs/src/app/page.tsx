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

  // Synapse estimate — every neuron is ~6 connected synapses (peer→self
  // + ~3 interneurons + cross-links). Matches the visual density of BrainHero.
  const neuronCount = topology?.nodes.length ?? 0;
  const synapseEstimate = neuronCount > 0 ? neuronCount * 6 + 60 : 60;

  return (
    <div className="relative w-full h-screen overflow-hidden brain-wave-bg">
      {/* Aurora ambient layer behind the brain */}
      <div className="aurora-bg z-0" />

      {/* Full-bleed brain hero */}
      <BrainHero className="absolute inset-0 w-full h-full" />

      {/* Top gradient overlay for readability */}
      <div
        className="absolute inset-x-0 top-0 h-40 pointer-events-none z-10"
        style={{
          background:
            "linear-gradient(to bottom, rgba(5,5,16,0.85) 0%, transparent 100%)",
        }}
      />

      {/* Bottom gradient overlay */}
      <div
        className="absolute inset-x-0 bottom-0 h-60 pointer-events-none z-10"
        style={{
          background:
            "linear-gradient(to top, rgba(5,5,16,0.9) 0%, transparent 100%)",
        }}
      />

      {/* Twinkling stars scattered across the viewport */}
      <div className="absolute inset-0 pointer-events-none z-10">
        <span className="twinkle absolute top-[12%] left-[18%] w-1 h-1 rounded-full bg-frost" style={{ animationDelay: "0s" }} />
        <span className="twinkle absolute top-[24%] right-[22%] w-[3px] h-[3px] rounded-full bg-white" style={{ animationDelay: "0.8s" }} />
        <span className="twinkle absolute bottom-[28%] left-[28%] w-[2px] h-[2px] rounded-full bg-frost" style={{ animationDelay: "1.5s" }} />
        <span className="twinkle absolute top-[38%] right-[14%] w-1 h-1 rounded-full bg-emerald" style={{ animationDelay: "2.2s" }} />
        <span className="twinkle absolute bottom-[18%] right-[32%] w-[2px] h-[2px] rounded-full bg-white" style={{ animationDelay: "1.1s" }} />
      </div>

      {/* Runic constellation near the title */}
      <svg
        className="absolute left-1/2 top-[15%] -translate-x-1/2 pointer-events-none z-10"
        width="220"
        height="120"
        viewBox="0 0 220 120"
        fill="none"
      >
        <line className="constellation-line" x1="20" y1="40" x2="70" y2="20" stroke="rgba(103,232,249,0.3)" strokeWidth="1" />
        <line className="constellation-line" x1="70" y1="20" x2="130" y2="55" stroke="rgba(103,232,249,0.3)" strokeWidth="1" style={{ animationDelay: "0.4s" }} />
        <line className="constellation-line" x1="130" y1="55" x2="190" y2="35" stroke="rgba(103,232,249,0.3)" strokeWidth="1" style={{ animationDelay: "0.8s" }} />
        <line className="constellation-line" x1="130" y1="55" x2="110" y2="100" stroke="rgba(103,232,249,0.3)" strokeWidth="1" style={{ animationDelay: "1.2s" }} />
        <circle className="twinkle" cx="20" cy="40" r="2" fill="#67e8f9" />
        <circle className="twinkle" cx="70" cy="20" r="2.5" fill="#ffffff" style={{ animationDelay: "0.6s" }} />
        <circle className="twinkle" cx="130" cy="55" r="3" fill="#67e8f9" style={{ animationDelay: "1.2s" }} />
        <circle className="twinkle" cx="190" cy="35" r="2" fill="#34d399" style={{ animationDelay: "1.8s" }} />
        <circle className="twinkle" cx="110" cy="100" r="2" fill="#ffffff" style={{ animationDelay: "2.4s" }} />
      </svg>

      {/* Center content overlay */}
      <div className="absolute inset-0 z-20 flex flex-col items-center justify-center pointer-events-none">
        {/* Title */}
        <div className="text-center space-y-4 float">
          <div className="flex justify-center mb-2">
            <YggLogoIcon size={72} />
          </div>
          <h1
            className="text-6xl md:text-7xl font-bold tracking-[0.3em] uppercase"
            style={{
              color: "#e4e2df",
              textShadow:
                "0 0 40px rgba(103,232,249,0.3), 0 0 80px rgba(103,232,249,0.1)",
            }}
          >
            YGGDRASIL
          </h1>

          <p className="text-sm md:text-base text-foreground-dim tracking-widest uppercase">
            A Living Brain of Distributed Computing
          </p>

          {/* Node status line */}
          <div className="flex items-center justify-center gap-3 mt-6">
            <span className="w-2 h-2 rounded-full status-online neural-pulse" />
            <span className="text-xs font-mono text-frost/80">
              {error ? "Backend unreachable" : nodeId}
            </span>
            {role && !error && (
              <>
                <span className="text-border">|</span>
                <span className="text-xs font-mono text-foreground-dim capitalize">
                  {role}
                </span>
              </>
            )}
            {version && !error && (
              <>
                <span className="text-border">|</span>
                <span className="text-xs font-mono text-foreground-dim">
                  v{version}
                </span>
              </>
            )}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-center gap-3 mt-8 pointer-events-auto">
            <Link
              href="/nodes"
              className="
                runic-card group
                px-6 py-2.5 text-sm font-semibold
                text-frost
                transition-all duration-200
              "
              style={{
                boxShadow: "0 0 20px rgba(103,232,249,0.1)",
              }}
            >
              <span className="group-hover:gradient-frost transition-all duration-200">
                View Nodes
              </span>
            </Link>
            {topology && (
              <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-mono text-muted bg-white/[0.03] border border-white/[0.06]">
                {/* Neuron/synapse glyph — small circuit-style icon */}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="2.5" />
                  <circle cx="4" cy="6" r="1.5" />
                  <circle cx="20" cy="6" r="1.5" />
                  <circle cx="4" cy="18" r="1.5" />
                  <circle cx="20" cy="18" r="1.5" />
                  <path d="M5.5 7l4.5 4M18.5 7L14 11M5.5 17L10 13M18.5 17L14 13" />
                </svg>
                <span className="text-frost">{neuronCount}</span>
                <span>neuron{neuronCount !== 1 ? "s" : ""}</span>
                <span className="text-border">&middot;</span>
                <span className="text-emerald">{synapseEstimate}</span>
                <span>synapses</span>
              </div>
            )}
          </div>

          {/* Live stats ticker (auto-refresh every 5s) */}
          {stats && (
            <div className="flex items-center justify-center gap-2 mt-6 text-[11px] font-mono text-foreground-dim">
              <span className="text-frost">{stats.func_count}</span>
              <span>function{stats.func_count !== 1 ? "s" : ""}</span>
              <span className="text-border">&bull;</span>
              <span className="text-emerald">{stats.env_count}</span>
              <span>environment{stats.env_count !== 1 ? "s" : ""}</span>
              <span className="text-border">&bull;</span>
              <span className="text-frost">{stats.dag_count}</span>
              <span>DAG{stats.dag_count !== 1 ? "s" : ""}</span>
              <span className="text-border">&bull;</span>
              <span className="text-amber">{stats.scheduled_dags}</span>
              <span>scheduled</span>
              <span className="text-border">&bull;</span>
              <span className="text-foreground">{stats.total_runs}</span>
              <span>run{stats.total_runs !== 1 ? "s" : ""}</span>
            </div>
          )}
        </div>
      </div>

      {/* Hint text at very bottom */}
      <p className="absolute bottom-6 left-1/2 -translate-x-1/2 text-[10px] text-white/15 pointer-events-none z-20 whitespace-nowrap hidden md:block">
        Drag to rotate &middot; Scroll to zoom
      </p>
    </div>
  );
}
