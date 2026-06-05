"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { KpiStrip } from "@/components/KpiStrip";

// The Excel task-pane is loaded standalone inside the Excel process via
// the add-in manifest — it must render chrome-free (no sidebar / KPI
// strip / left margin). Every other route gets the full app shell.
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const bare = pathname?.startsWith("/excel/taskpane");

  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    try { setCollapsed(localStorage.getItem("ygg-sidebar") === "collapsed"); } catch { /* ignore */ }
  }, []);
  const toggle = () => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem("ygg-sidebar", next ? "collapsed" : "open"); } catch { /* ignore */ }
      return next;
    });
  };

  if (bare) {
    return <main className="min-h-screen">{children}</main>;
  }

  return (
    <>
      <Sidebar collapsed={collapsed} onToggle={toggle} />
      <main className={`${collapsed ? "ml-[60px]" : "ml-[200px]"} min-h-screen flex flex-col transition-[margin] duration-200`}>
        <KpiStrip />
        <div className="flex-1 min-h-0">{children}</div>
      </main>
    </>
  );
}
