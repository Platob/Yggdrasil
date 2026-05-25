"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { YggdrasilBrand } from "./logo";

const NAV = [
  {
    href: "/msg",
    label: "Channels",
    desc: "All channels",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
];

export function MsgSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-60 flex flex-col z-50" style={{ background: "#0a0a0a", borderRight: "1px solid rgba(255,255,255,0.07)" }}>
      {/* Logo */}
      <div className="p-5" style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
        <Link href="/">
          <YggdrasilBrand size={26} />
        </Link>
        <p className="text-[10px] uppercase tracking-widest mt-2" style={{ color: "rgba(255,255,255,0.3)" }}>
          Messaging
        </p>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-1">
        {NAV.map((item) => {
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className="group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all"
              style={{
                background: active ? "rgba(242,107,58,0.08)" : "transparent",
                border: `1px solid ${active ? "rgba(242,107,58,0.2)" : "transparent"}`,
                color: active ? "#f26b3a" : "rgba(255,255,255,0.45)",
              }}
            >
              <span style={{ color: active ? "#f26b3a" : "rgba(255,255,255,0.3)" }}>
                {item.icon}
              </span>
              <div>
                <div className="font-medium">{item.label}</div>
                <div className="text-[10px]" style={{ color: active ? "rgba(242,107,58,0.6)" : "rgba(255,255,255,0.25)" }}>
                  {item.desc}
                </div>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Cross-links */}
      <div className="p-3" style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>
        <Link
          href="/bot"
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors"
          style={{ color: "rgba(255,255,255,0.3)" }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
          Bot Control
        </Link>
        <Link
          href="/"
          className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs transition-colors"
          style={{ color: "rgba(255,255,255,0.2)" }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
          </svg>
          Home
        </Link>
      </div>

      {/* Status */}
      <div className="p-4" style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: "#22c55e", boxShadow: "0 0 6px #22c55e" }} />
          <span className="text-[11px]" style={{ color: "rgba(255,255,255,0.3)" }}>Live</span>
        </div>
      </div>
    </aside>
  );
}
