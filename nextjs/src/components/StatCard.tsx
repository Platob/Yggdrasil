interface Props {
  label: string;
  value: string | number;
  sub?: string;
  trend?: "up" | "down" | "flat";
}

export default function StatCard({ label, value, sub, trend }: Props) {
  const trendColor = trend === "up" ? "var(--green)" : trend === "down" ? "var(--red)" : "var(--muted)";
  const trendIcon = trend === "up" ? "▲" : trend === "down" ? "▼" : "─";
  return (
    <div className="card" style={{ minWidth: 160 }}>
      <div style={{ color: "var(--muted)", fontSize: 11, fontWeight: 600, letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, fontVariantNumeric: "tabular-nums", letterSpacing: "-0.02em" }}>
        {value}
      </div>
      {sub && (
        <div style={{ marginTop: 6, fontSize: 12, color: trendColor, display: "flex", alignItems: "center", gap: 4 }}>
          {trend && <span style={{ fontSize: 10 }}>{trendIcon}</span>}
          {sub}
        </div>
      )}
    </div>
  );
}
