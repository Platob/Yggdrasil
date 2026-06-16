"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/",         label: "Dashboard",  icon: "◈" },
  { href: "/fx",       label: "FX Rates",   icon: "₿" },
  { href: "/analysis", label: "Analysis",   icon: "⧖" },
  { href: "/chat",     label: "Chat",       icon: "◎" },
  { href: "/system",   label: "System",     icon: "⊙" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <nav style={{
      width: 200,
      background: "var(--surface)",
      borderRight: "1px solid var(--border)",
      padding: "24px 12px",
      display: "flex",
      flexDirection: "column",
      gap: 4,
      flexShrink: 0,
    }}>
      <div style={{ padding: "0 8px 20px", fontSize: 18, fontWeight: 700, letterSpacing: "-0.02em" }}>
        <span style={{ color: "var(--accent)" }}>YGG</span> Trading
      </div>
      {links.map(({ href, label, icon }) => {
        const active = path === href;
        return (
          <Link key={href} href={href} style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "8px 12px",
            borderRadius: 8,
            color: active ? "var(--text)" : "var(--muted)",
            background: active ? "rgba(59,130,246,0.12)" : "transparent",
            textDecoration: "none",
            fontWeight: active ? 600 : 400,
            fontSize: 13,
            transition: "all 0.15s",
          }}>
            <span style={{ fontSize: 16, opacity: active ? 1 : 0.7 }}>{icon}</span>
            {label}
          </Link>
        );
      })}
      <div style={{ marginTop: "auto", padding: "12px 8px 0", borderTop: "1px solid var(--border)" }}>
        <StatusDot />
      </div>
    </nav>
  );
}

function StatusDot() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--muted)" }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%",
        background: "var(--green)",
        boxShadow: "0 0 6px var(--green)",
        display: "inline-block",
      }} />
      Node online
    </div>
  );
}
