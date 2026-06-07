"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  CandlestickChart,
  Wallet,
  FlaskConical,
  Menu,
  X,
} from "lucide-react";
import { ping } from "@/lib/api";

const NAV = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/market", label: "Market", icon: CandlestickChart },
  { href: "/portfolio", label: "Portfolio", icon: Wallet },
  { href: "/analysis", label: "Analysis", icon: FlaskConical },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [alive, setAlive] = useState<boolean | null>(null);
  const [nodeId, setNodeId] = useState<string>("");

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const res = await ping();
        if (active) {
          setAlive(res.ok);
          setNodeId(res.node_id);
        }
      } catch {
        if (active) setAlive(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  return (
    <>
      {/* Mobile top bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-30 flex items-center justify-between h-14 px-4 bg-gray-900 border-b border-gray-800">
        <span className="text-xl font-bold text-green-400 tracking-tight">YGG</span>
        <button
          onClick={() => setOpen((v) => !v)}
          className="text-gray-300 hover:text-white"
          aria-label="Toggle navigation"
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {/* Backdrop on mobile */}
      {open && (
        <div
          className="md:hidden fixed inset-0 z-30 bg-black/60"
          onClick={() => setOpen(false)}
        />
      )}

      <aside
        className={`fixed z-40 top-0 left-0 h-screen w-[220px] bg-gray-900 border-r border-gray-800 flex flex-col transition-transform duration-200
          ${open ? "translate-x-0" : "-translate-x-full"} md:translate-x-0`}
      >
        <div className="h-16 flex items-center px-5 border-b border-gray-800">
          <span className="text-2xl font-bold text-green-400 tracking-tight">YGG</span>
          <span className="ml-2 text-xs text-gray-500 uppercase tracking-widest">
            Trading
          </span>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={`flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors
                  ${
                    active
                      ? "bg-gray-800 text-green-400"
                      : "text-gray-400 hover:bg-gray-800/60 hover:text-gray-100"
                  }`}
              >
                <Icon size={18} />
                {label}
              </Link>
            );
          })}
        </nav>

        <div className="px-5 py-4 border-t border-gray-800">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                alive === null
                  ? "bg-gray-500 animate-pulse"
                  : alive
                    ? "bg-green-400"
                    : "bg-red-500"
              }`}
            />
            <span className="text-gray-400">
              {alive === null
                ? "Connecting…"
                : alive
                  ? "Connected"
                  : "Backend offline"}
            </span>
          </div>
          {alive && nodeId && (
            <div className="mt-1 text-[10px] text-gray-600 font-mono truncate">
              {nodeId}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
