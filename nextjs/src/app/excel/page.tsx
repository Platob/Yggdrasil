"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getExcelInfo } from "@/lib/api";
import type { ExcelInfo } from "@/lib/types";

export default function ExcelPage() {
  const [info, setInfo] = useState<ExcelInfo | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getExcelInfo().then(setInfo).catch(() => setError(true));
  }, []);

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold">Excel &amp; Power Query</h1>
        <p className="text-sm text-muted">
          Connect Excel to this node — run Python, read/write remote files, and walk the
          filesystem, straight into a sheet.
        </p>
      </header>

      {/* Service status */}
      <div className="glass-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">Service</h2>
        {error && <p className="text-sm text-[var(--rose)]">Excel service unreachable.</p>}
        {info ? (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm font-mono">
            <span className="text-[var(--emerald)]">● online</span>
            <span>node <span className="text-foreground-dim">{info.node_id}</span></span>
            <span>v{info.version}</span>
            <span>formats: {info.table_formats.join(", ")}</span>
            <span>caps: {info.capabilities.join(", ")}</span>
          </div>
        ) : (
          !error && <p className="text-sm text-muted">Checking…</p>
        )}
      </div>

      {/* Capabilities */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { t: "Run Python", d: "Execute a snippet in a PyEnv; the named dataframe lands in a sheet as a typed table." },
          { t: "Read / write files", d: "Pull a parquet/csv/json/arrow file in as a table, or push a sheet back to the node." },
          { t: "Walk the filesystem", d: "Browse the node's files as a drill-down navigation table." },
        ].map((c) => (
          <div key={c.t} className="glass-card p-4 space-y-1">
            <h3 className="text-sm font-semibold text-foreground">{c.t}</h3>
            <p className="text-xs text-muted">{c.d}</p>
          </div>
        ))}
      </div>

      {/* Office.js add-in */}
      <div className="glass-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">Office add-in (web plug-in)</h2>
        <p className="text-sm text-muted">
          The task-pane add-in runs inside Excel and talks to this node directly.
        </p>
        <ol className="text-sm text-foreground-dim list-decimal ml-5 space-y-1">
          <li>Excel → <span className="font-mono">Insert → Add-ins → My Add-ins → Upload My Add-in</span></li>
          <li>Pick <a className="text-[var(--frost)] underline" href="/excel-addin/manifest.xml">manifest.xml</a> (served here)</li>
          <li>Click <span className="font-mono">Open Yggdrasil</span> on the Home tab</li>
        </ol>
        <div className="flex gap-3 pt-1">
          <Link href="/excel/taskpane" className="text-sm px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08]">
            Preview task-pane →
          </Link>
          <a href="/excel-addin/manifest.xml" download className="text-sm px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08]">
            Download manifest
          </a>
        </div>
      </div>

      {/* Power Query */}
      <div className="glass-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">Power Query connector</h2>
        <p className="text-sm text-muted">
          Paste the M module into Excel’s Advanced Editor (no install), or build the
          <span className="font-mono"> .mez</span> for Power BI. Results decode natively via
          <span className="font-mono"> Parquet.Document</span>.
        </p>
        <pre className="text-xs font-mono bg-black/30 rounded-lg p-3 overflow-x-auto text-foreground-dim">{`// after pasting powerquery/YggdrasilExcel.pq:
in Yggdrasil[Execute](
    "import pandas as pd#(lf)df = pd.DataFrame({'x':[1,2,3]})",
    [Packages = {"pandas"}]
)`}</pre>
        <p className="text-xs text-muted">Sources live in <span className="font-mono">powerquery/</span> (YggdrasilExcel.pq, Yggdrasil.pq).</p>
      </div>
    </div>
  );
}
