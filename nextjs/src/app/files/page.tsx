"use client";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import type { FsEntry } from "@/lib/types";

export default function FilesPage() {
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [path, setPath] = useState("");
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  const ls = async (p = path) => {
    setLoading(true);
    try {
      const r = await api.ls(p);
      setEntries(r.entries); setTotal(r.total);
    } finally { setLoading(false); }
  };

  useEffect(() => { ls(""); }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Files</h1>
      <div className="flex gap-2">
        <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path..."
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1" />
        <button onClick={() => ls()} className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg text-sm">Browse</button>
      </div>
      <div className="text-xs text-gray-500">{total} entries</div>
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        {loading && <div className="p-4 text-gray-400 text-sm">Loading…</div>}
        {entries.map(e => (
          <div key={e.name} onClick={() => e.is_dir && ls(path ? `${path}/${e.name}` : e.name)}
            className="flex items-center gap-3 px-4 py-2 border-b border-gray-800 hover:bg-gray-800 cursor-pointer text-sm">
            <span className="text-gray-400">{e.is_dir ? "📁" : "📄"}</span>
            <span className="flex-1">{e.name}</span>
            <span className="text-gray-500 text-xs">{e.is_dir ? "" : `${(e.size / 1024).toFixed(1)} KB`}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
