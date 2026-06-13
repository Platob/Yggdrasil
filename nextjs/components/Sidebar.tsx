"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/dashboard", label: "Dashboard",  icon: "⬛" },
  { href: "/analysis",  label: "Analysis",   icon: "📈" },
  { href: "/files",     label: "Files",      icon: "📁" },
  { href: "/chat",      label: "Chat",       icon: "💬" },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-56 shrink-0 flex flex-col bg-zinc-900 border-r border-zinc-800 h-full">
      <div className="px-5 py-4 border-b border-zinc-800">
        <span className="text-emerald-400 font-bold text-lg tracking-tight">YGG</span>
        <span className="text-zinc-400 text-sm ml-1">node</span>
      </div>
      <nav className="flex-1 p-2 space-y-0.5">
        {NAV.map(({ href, label, icon }) => {
          const active = path === href || path.startsWith(href + "/");
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2 rounded text-sm transition-colors
                ${active
                  ? "bg-zinc-800 text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
                }`}
            >
              <span>{icon}</span>
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-4 py-3 border-t border-zinc-800 text-zinc-600 text-xs">
        yggdrasil.node
      </div>
    </aside>
  );
}
