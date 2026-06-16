"use client";
import Link from "next/link";

interface Props {
  href: string;
  title: string;
  desc: string;
  icon: string;
  color: string;
}

export default function QuickNavCard({ href, title, desc, icon, color }: Props) {
  return (
    <Link href={href} style={{ textDecoration: "none" }}>
      <div
        className="card"
        style={{ cursor: "pointer", transition: "border-color 0.15s, transform 0.1s", height: "100%" }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.borderColor = color;
          (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.borderColor = "var(--border)";
          (e.currentTarget as HTMLElement).style.transform = "translateY(0)";
        }}
      >
        <div style={{ fontSize: 28, marginBottom: 12, color }}>{icon}</div>
        <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 15 }}>{title}</div>
        <div style={{ color: "var(--muted)", fontSize: 12, lineHeight: 1.5 }}>{desc}</div>
      </div>
    </Link>
  );
}
