"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, useCallback, useRef } from "react";
import { YggdrasilLogo } from "./logo";

// ── Types ──────────────────────────────────────────────────
type NavItem = {
  href: string;
  label: string;
  icon: React.ReactNode;
  exact?: boolean;
};

type ServiceSection = {
  id: string;
  label: string;
  prefix: string;
  icon: React.ReactNode;
  items: NavItem[];
  active?: boolean;
};

// ── Icons ──────────────────────────────────────────────────
const Icon = {
  grid: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  ),
  terminal: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="4 17 10 11 4 5" /><line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  ),
  globe: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  ),
  chat: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  bolt: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
  chart: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" /><line x1="2" y1="20" x2="22" y2="20" />
    </svg>
  ),
  candleUp: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" /><polyline points="16 7 22 7 22 13" />
    </svg>
  ),
  home: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" /><polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  ),
  sun: (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5" />
      <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
      <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
    </svg>
  ),
  moon: (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ),
  chevronLeft: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  ),
  chevronRight: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  ),
};

// ── Service definitions ────────────────────────────────────
const SERVICES: ServiceSection[] = [
  {
    id: "bot",
    label: "Bot Control",
    prefix: "/bot",
    icon: Icon.bolt,
    items: [
      { href: "/bot", label: "Dashboard", icon: Icon.grid, exact: true },
      { href: "/bot/network", label: "Network", icon: Icon.globe },
      { href: "/bot/execute", label: "Execute", icon: Icon.terminal },
    ],
  },
  {
    id: "msg",
    label: "Messaging",
    prefix: "/msg",
    icon: Icon.chat,
    items: [
      { href: "/msg", label: "Channels", icon: Icon.chat, exact: true },
    ],
  },
  {
    id: "trading",
    label: "Trading",
    prefix: "/trading",
    icon: Icon.candleUp,
    items: [
      { href: "/trading", label: "Markets", icon: Icon.chart, exact: true },
    ],
  },
];

// ── Tooltip wrapper ────────────────────────────────────────
function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="relative group/tip">
      {children}
      <div
        className="pointer-events-none absolute left-full top-1/2 -translate-y-1/2 ml-3 z-[200]
          px-2.5 py-1.5 rounded-md text-xs font-medium whitespace-nowrap
          opacity-0 group-hover/tip:opacity-100 transition-opacity duration-150"
        style={{ background: "var(--card-elevated)", border: "1px solid var(--border-accent)", color: "var(--foreground)" }}
      >
        {label}
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────
export function GlobalSidebar({ onCollapse }: { onCollapse?: (collapsed: boolean) => void }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [dark, setDark] = useState(true);
  const initialized = useRef(false);

  // Restore persisted state
  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    const savedCollapse = localStorage.getItem("ygg-sidebar-collapsed");
    const savedTheme = localStorage.getItem("ygg-theme");
    if (savedCollapse === "true") {
      setCollapsed(true);
      onCollapse?.(true);
    }
    const isDark = savedTheme !== "light";
    setDark(isDark);
    document.documentElement.classList.toggle("light", !isDark);
  }, [onCollapse]);

  const toggleCollapse = useCallback(() => {
    setCollapsed((v) => {
      const next = !v;
      localStorage.setItem("ygg-sidebar-collapsed", String(next));
      onCollapse?.(next);
      return next;
    });
  }, [onCollapse]);

  const toggleTheme = useCallback(() => {
    setDark((v) => {
      const next = !v;
      localStorage.setItem("ygg-theme", next ? "dark" : "light");
      document.documentElement.classList.toggle("light", !next);
      return next;
    });
  }, []);

  // Determine active service
  const activeService = SERVICES.find((s) => pathname.startsWith(s.prefix));

  // Check if a nav item is active
  const isActive = (item: NavItem) =>
    item.exact ? pathname === item.href : pathname.startsWith(item.href);

  const w = collapsed ? "56px" : "224px";

  return (
    <aside
      style={{
        width: w,
        minWidth: w,
        maxWidth: w,
        background: "var(--sidebar-bg)",
        borderRight: "1px solid var(--sidebar-border)",
        transition: "width 200ms ease, min-width 200ms ease, max-width 200ms ease",
      }}
      className="fixed left-0 top-0 h-full flex flex-col z-50 overflow-visible"
    >
      {/* ── Collapse tab — center-right edge of sidebar ── */}
      <button
        onClick={toggleCollapse}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        style={{
          position: "absolute",
          right: "-14px",
          top: "50%",
          transform: "translateY(-50%)",
          width: "14px",
          height: "48px",
          background: "var(--sidebar-bg)",
          border: "1px solid var(--sidebar-border)",
          borderLeft: "none",
          borderRadius: "0 6px 6px 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          color: "var(--muted-foreground)",
          transition: "background 150ms, color 150ms",
          zIndex: 51,
        }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLButtonElement).style.background = "var(--card-hover)";
          (e.currentTarget as HTMLButtonElement).style.color = "var(--foreground)";
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLButtonElement).style.background = "var(--sidebar-bg)";
          (e.currentTarget as HTMLButtonElement).style.color = "var(--muted-foreground)";
        }}
      >
        <svg
          width="8"
          height="12"
          viewBox="0 0 8 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ transition: "transform 200ms", transform: collapsed ? "scaleX(-1)" : "scaleX(1)" }}
        >
          <polyline points="5 1 1 6 5 11" />
        </svg>
      </button>

      {/* ── Logo ── */}
      <div
        className="flex items-center h-14 shrink-0 px-3"
        style={{ borderBottom: "1px solid var(--sidebar-border)" }}
      >
        {!collapsed && (
          <Link href="/" className="flex items-center gap-2.5 flex-1 min-w-0">
            <YggdrasilLogo size={22} className="text-primary shrink-0" />
            <span className="font-bold text-sm tracking-widest uppercase truncate" style={{ color: "var(--foreground)" }}>
              Yggdrasil
            </span>
          </Link>
        )}
        {collapsed && (
          <div className="flex-1 flex justify-center">
            <Link href="/">
              <YggdrasilLogo size={22} className="text-primary" />
            </Link>
          </div>
        )}
      </div>

      {/* ── Nav: services + items ── */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden py-3 space-y-5">
        {SERVICES.map((service) => {
          const isServiceActive = pathname.startsWith(service.prefix);
          const hasItems = service.items.length > 0;

          return (
            <div key={service.id}>
              {/* Section label / service header */}
              {!collapsed ? (
                <div
                  className="flex items-center gap-2 px-3 mb-1"
                >
                  <span style={{ color: isServiceActive ? "var(--primary)" : "var(--muted)", transition: "color 150ms" }}>
                    {service.icon}
                  </span>
                  <span
                    className="text-[10px] font-semibold uppercase tracking-widest truncate"
                    style={{ color: isServiceActive ? "var(--primary)" : "var(--muted)" }}
                  >
                    {service.label}
                  </span>
                  {!hasItems && (
                    <span
                      className="ml-auto text-[9px] font-medium px-1.5 py-0.5 rounded"
                      style={{ background: "var(--border)", color: "var(--muted)" }}
                    >
                      Soon
                    </span>
                  )}
                </div>
              ) : (
                <Tooltip label={service.label}>
                  <div
                    className="flex justify-center mb-1 mx-1 py-1 rounded-md"
                    style={{
                      background: isServiceActive ? "var(--primary-glow)" : "transparent",
                      color: isServiceActive ? "var(--primary)" : "var(--muted)",
                    }}
                  >
                    {service.icon}
                  </div>
                </Tooltip>
              )}

              {/* Items */}
              {hasItems && service.items.map((item) => {
                const active = isActive(item);
                return collapsed ? (
                  <Tooltip key={item.href} label={item.label}>
                    <Link
                      href={item.href}
                      className="flex justify-center items-center py-2 mx-1 rounded-md transition-all duration-150"
                      style={{
                        background: active ? "rgba(242,107,58,0.12)" : "transparent",
                        color: active ? "var(--primary)" : "var(--muted)",
                      }}
                    >
                      {item.icon}
                    </Link>
                  </Tooltip>
                ) : (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex items-center gap-2.5 mx-2 px-2 py-1.5 rounded-md text-sm transition-all duration-150"
                    style={{
                      background: active ? "rgba(242,107,58,0.1)" : "transparent",
                      color: active ? "var(--primary)" : "var(--foreground-dim)",
                      borderLeft: active ? "2px solid var(--primary)" : "2px solid transparent",
                    }}
                  >
                    <span style={{ color: active ? "var(--primary)" : "var(--muted)" }}>
                      {item.icon}
                    </span>
                    <span className="font-medium text-[13px] truncate">{item.label}</span>
                  </Link>
                );
              })}
            </div>
          );
        })}
      </nav>

      {/* ── Bottom: home + dark mode + status ── */}
      <div style={{ borderTop: "1px solid var(--sidebar-border)" }} className="shrink-0 py-2 space-y-0.5">

        {/* Home link */}
        {collapsed ? (
          <Tooltip label="Home">
            <Link
              href="/"
              className="flex justify-center items-center py-2 mx-1 rounded-md transition-colors"
              style={{ color: "var(--muted)" }}
            >
              {Icon.home}
            </Link>
          </Tooltip>
        ) : (
          <Link
            href="/"
            className="flex items-center gap-2.5 mx-2 px-2 py-1.5 rounded-md text-[13px] transition-colors"
            style={{ color: "var(--muted)" }}
          >
            {Icon.home}
            <span className="font-medium">Home</span>
          </Link>
        )}

        {/* Dark mode toggle */}
        {collapsed ? (
          <Tooltip label={dark ? "Light mode" : "Dark mode"}>
            <button
              onClick={toggleTheme}
              className="flex justify-center items-center w-full py-2 mx-1 rounded-md transition-colors"
              style={{ color: "var(--muted)", width: "calc(100% - 8px)" }}
            >
              {dark ? Icon.sun : Icon.moon}
            </button>
          </Tooltip>
        ) : (
          <button
            onClick={toggleTheme}
            className="flex items-center gap-2.5 w-full mx-2 px-2 py-1.5 rounded-md text-[13px] transition-colors"
            style={{ color: "var(--muted)", width: "calc(100% - 16px)" }}
          >
            {dark ? Icon.sun : Icon.moon}
            <span className="font-medium">{dark ? "Light mode" : "Dark mode"}</span>
          </button>
        )}

        {/* Status */}
        <div
          className="flex items-center gap-2 px-3 pt-1.5 pb-1"
        >
          <span
            className="shrink-0 w-1.5 h-1.5 rounded-full"
            style={{ background: "var(--success)", boxShadow: "0 0 6px var(--success)" }}
          />
          {!collapsed && (
            <span className="text-[11px] truncate" style={{ color: "var(--muted)" }}>
              Connected · v0.1.0
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}
