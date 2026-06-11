"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function SagaPage() {
  const [sql, setSql] = useState("SELECT 1 AS hello");
  const [result, setResult] = useState<{ columns: string[]; rows: unknown[][] } | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);

  const run = async () => {
    setRunning(true); setError(""); setResult(null);
    try {
      const r = await api.sql(sql);
      setResult(r);
    } catch(e: any) { setError(e.message); }
    finally { setRunning(false); }
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Saga SQL</h1>
      <textarea value={sql} onChange={e => setSql(e.target.value)}
        className="w-full bg-gray-900 border border-gray-700 rounded-xl p-4 text-sm font-mono h-32 resize-y"
        placeholder="SELECT * FROM catalog.schema.table" />
      <button onClick={run} disabled={running}
        className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-6 py-2 rounded-lg text-sm font-medium">
        {running ? "Running…" : "Run ▶"}
      </button>
      {error && <div className="text-red-400 text-sm bg-red-900/20 px-4 py-2 rounded-lg">{error}</div>}
      {result && (
        <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                {result.columns.map(c => <th key={c} className="px-4 py-2 text-left text-gray-400 font-medium">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800">
                  {(row as unknown[]).map((cell, j) => <td key={j} className="px-4 py-2">{String(cell)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
