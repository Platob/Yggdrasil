"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard", icon: "⬡" },
  { href: "/market", label: "Market", icon: "📈" },
  { href: "/chat", label: "Loki AI", icon: "✦" },
  { href: "/data", label: "Data", icon: "⛁" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav
      style={{
        width: 200,
        minHeight: "100vh",
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "20px 0",
        flexShrink: 0,
      }}
    >
      <div style={{ padding: "0 20px 24px", borderBottom: "1px solid var(--border)" }}>
        <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent)", letterSpacing: "-0.5px" }}>
          ⬡ Yggdrasil
        </div>
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
          Trading Dashboard
        </div>
      </div>
      <div style={{ marginTop: 16, flex: 1 }}>
        {links.map((l) => {
          const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "10px 20px",
                color: active ? "var(--accent)" : "var(--text-muted)",
                background: active ? "rgba(59,130,246,0.08)" : "transparent",
                borderLeft: active ? "2px solid var(--accent)" : "2px solid transparent",
                textDecoration: "none",
                fontSize: 13,
                fontWeight: active ? 500 : 400,
                transition: "all 0.15s",
              }}
            >
              <span style={{ fontSize: 14 }}>{l.icon}</span>
              {l.label}
            </Link>
          );
        })}
      </div>
      <div style={{ padding: "16px 20px", borderTop: "1px solid var(--border)", color: "var(--text-muted)", fontSize: 11 }}>
        v0.9 · Ygg Node
      </div>
    </nav>
  );
}
