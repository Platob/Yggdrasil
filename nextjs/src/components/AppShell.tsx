"use client";

import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/Sidebar";
import { KpiStrip } from "@/components/KpiStrip";

// The Excel task-pane is loaded standalone inside the Excel process via
// the add-in manifest — it must render chrome-free (no sidebar / KPI
// strip / left margin). Every other route gets the full app shell.
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const bare = pathname?.startsWith("/excel/taskpane");

  if (bare) {
    return <main className="min-h-screen">{children}</main>;
  }

  return (
    <>
      <Sidebar />
      <main className="ml-[200px] min-h-screen flex flex-col">
        <KpiStrip />
        <div className="flex-1 min-h-0">{children}</div>
      </main>
    </>
  );
}
