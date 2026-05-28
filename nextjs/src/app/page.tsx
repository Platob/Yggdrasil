"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Globe } from "@/components/Globe";
import { YggLogoIcon } from "@/components/YggLogo";
import { getNodeCard, getStats } from "@/lib/api";
import type { NodeCard, ClusterStats } from "@/lib/types";

export default function WelcomePage() {
  const [card, setCard] = useState<NodeCard | null>(null);
  const [stats, setStats] = useState<ClusterStats | null>(null);
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

  const nodeId = card?.node_id ?? "---";
  const lat = card?.lat ?? null;
  const lon = card?.lon ?? null;
  const role = card?.role ?? "hybrid";
  const version = card?.version ?? "";

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Full-bleed globe background */}
      <Globe
        lat={lat}
        lon={lon}
        className="absolute inset-0 w-full h-full"
      />

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
            The World Tree of Distributed Computing
          </p>

          {/* Node status line */}
          <div className="flex items-center justify-center gap-3 mt-6">
            <span className="w-2 h-2 rounded-full status-online" />
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
                px-6 py-2.5 rounded-lg text-sm font-semibold
                bg-frost/10 text-frost border border-frost/20
                hover:bg-frost/20 hover:border-frost/40
                transition-all duration-200
                backdrop-blur-sm
              "
              style={{
                boxShadow: "0 0 20px rgba(103,232,249,0.1)",
              }}
            >
              View Nodes
            </Link>
            {card && (
              <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-xs font-mono text-muted bg-white/[0.03] border border-white/[0.06]">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z" />
                  <line x1="2" y1="12" x2="22" y2="12" />
                </svg>
                {lat != null && lon != null
                  ? `${lat.toFixed(2)}, ${lon.toFixed(2)}`
                  : "Locating..."}
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
