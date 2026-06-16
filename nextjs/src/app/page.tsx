import StatCard from "@/components/StatCard";
import QuickNavCard from "@/components/QuickNavCard";

async function getStats() {
  try {
    const base = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";
    const [health, backend] = await Promise.allSettled([
      fetch(`${base}/api/v2/health`, { cache: "no-store" }).then(r => r.json()),
      fetch(`${base}/api/v2/backend`, { cache: "no-store" }).then(r => r.json()),
    ]);
    return {
      health: health.status === "fulfilled" ? health.value as { status: string; uptime: number } : null,
      backend: backend.status === "fulfilled" ? backend.value as { node_id: string; version: string; engines: string[] } : null,
    };
  } catch {
    return { health: null, backend: null };
  }
}

export default async function Dashboard() {
  const { health, backend } = await getStats();
  const uptimeMins = health ? Math.floor(health.uptime / 60) : null;

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto" }}>
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, letterSpacing: "-0.03em", margin: 0 }}>Dashboard</h1>
        <p style={{ color: "var(--muted)", marginTop: 6, fontSize: 13 }}>
          Yggdrasil trading node — real-time FX, analysis, and AI
        </p>
      </div>

      <div style={{ display: "flex", gap: 12, marginBottom: 32, flexWrap: "wrap" }}>
        <StatCard
          label="Node Status"
          value={health?.status === "ok" ? "Online" : "Offline"}
          sub={health ? `${uptimeMins}m uptime` : "Check API_URL"}
          trend={health?.status === "ok" ? "up" : "down"}
        />
        <StatCard
          label="Node ID"
          value={backend?.node_id ?? "—"}
          sub={backend?.version ? `v${backend.version}` : undefined}
        />
        <StatCard
          label="AI Engines"
          value={backend?.engines?.length ?? 0}
          sub={backend?.engines?.slice(0, 2).join(", ")}
        />
        <StatCard
          label="Environment"
          value="Python 3.12"
          sub="FastAPI · Polars · Arrow"
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 16 }}>
        <QuickNavCard href="/fx"       title="FX Rates"  desc="Live currency rates with multi-backend fallback"   icon="₿" color="#f59e0b" />
        <QuickNavCard href="/analysis" title="Analysis"  desc="OHLC charts, aggregation, polars lazy scanning"    icon="⧖" color="#8b5cf6" />
        <QuickNavCard href="/chat"     title="Messenger" desc="In-node chat channels for team coordination"       icon="◎" color="#3b82f6" />
        <QuickNavCard href="/system"   title="System"    desc="Monitor resources, audit log, Python environments" icon="⊙" color="#10b981" />
      </div>
    </div>
  );
}
