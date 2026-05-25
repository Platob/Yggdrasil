"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Dashboard", icon: "◆" },
  { href: "/execute", label: "Execute", icon: "▶" },
  { href: "/chat", label: "Chat", icon: "◈" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-56 bg-card border-r border-border flex flex-col z-50">
      <div className="p-5 border-b border-border">
        <Link href="/" className="block">
          <pre className="text-[10px] leading-tight text-gold font-mono select-none">
{` __   __ ___ ___
 \\ \\ / // __/ __|
  \\ V /| (_ | (_ |
   |_|  \\___|\\___| `}
          </pre>
        </Link>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                active
                  ? "bg-gold/10 text-gold border border-gold/20"
                  : "text-muted hover:text-foreground hover:bg-card-hover"
              }`}
            >
              <span className={`text-xs ${active ? "text-gold" : ""}`}>{item.icon}</span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-border">
        <p className="text-[10px] text-muted font-mono">YGGDRASIL v0.1.0</p>
      </div>
    </aside>
  );
}
