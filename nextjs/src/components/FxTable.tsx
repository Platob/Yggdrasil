"use client";
import type { FxQuote } from "@/lib/types";

const PAIRS = [
  ["EUR", "USD"],
  ["EUR", "GBP"],
  ["EUR", "JPY"],
  ["USD", "JPY"],
  ["GBP", "USD"],
  ["USD", "CHF"],
  ["AUD", "USD"],
  ["USD", "CAD"],
];

interface Props {
  quotes: FxQuote[];
  loading?: boolean;
}

export default function FxTable({ quotes, loading }: Props) {
  const byPair = new Map(quotes.map(q => [`${q.source}/${q.target}`, q]));

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["Pair", "Rate", "Sampling", "From", ""].map(h => (
              <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: "var(--muted)", fontWeight: 600, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PAIRS.map(([src, tgt]) => {
            const key = `${src}/${tgt}`;
            const q = byPair.get(key);
            const rate = q?.value;
            return (
              <tr key={key} style={{ borderBottom: "1px solid var(--border)", transition: "background 0.1s" }}
                onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
                onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
              >
                <td style={{ padding: "10px 12px", fontWeight: 600 }}>{key}</td>
                <td style={{ padding: "10px 12px", fontVariantNumeric: "tabular-nums" }}>
                  {loading ? <Skeleton /> : rate != null ? rate.toFixed(5) : <span style={{ color: "var(--muted)" }}>—</span>}
                </td>
                <td style={{ padding: "10px 12px", color: "var(--muted)" }}>{q?.sampling ?? "—"}</td>
                <td style={{ padding: "10px 12px", color: "var(--muted)", fontSize: 11 }}>
                  {q ? new Date(q.from_timestamp).toLocaleDateString() : "—"}
                </td>
                <td style={{ padding: "10px 12px" }}>
                  {rate != null && (
                    <span className={`badge ${rate >= 1 ? "badge-green" : "badge-blue"}`}>
                      {rate.toFixed(3)}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Skeleton() {
  return (
    <span style={{
      display: "inline-block",
      width: 70,
      height: 14,
      background: "var(--border)",
      borderRadius: 4,
      animation: "pulse 1.4s ease-in-out infinite",
    }} />
  );
}
