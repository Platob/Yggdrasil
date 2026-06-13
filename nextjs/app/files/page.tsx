"use client";
import { useEffect, useState } from "react";
import FileTree from "@/components/FileTree";
import TabularPreview from "@/components/TabularPreview";
import { lsDir, inspectTabular, readFile } from "@/lib/api";
import type { FsEntry, FsListResult, TabularInspect, FsReadResult } from "@/lib/types";

const TABULAR_EXTS = new Set(["parquet", "csv", "json", "ndjson", "jsonl", "arrow", "feather", "xlsx"]);

function isTabular(entry: FsEntry) {
  const ext = entry.name.split(".").pop()?.toLowerCase() ?? "";
  return TABULAR_EXTS.has(ext);
}

export default function FilesPage() {
  const [root, setRoot] = useState<FsListResult | null>(null);
  const [selected, setSelected] = useState<FsEntry | null>(null);
  const [inspect, setInspect] = useState<TabularInspect | null>(null);
  const [textPreview, setTextPreview] = useState<FsReadResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    lsDir("").then(setRoot).catch(() => setRoot({ entries: [], total: 0 }));
  }, []);

  async function handleSelect(entry: FsEntry) {
    setSelected(entry);
    setInspect(null);
    setTextPreview(null);
    setLoading(true);
    try {
      if (isTabular(entry)) {
        const info = await inspectTabular(entry.path);
        setInspect(info);
      } else {
        const txt = await readFile(entry.path);
        setTextPreview(txt);
      }
    } catch {
      // silently fail — entry may be unreadable
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex gap-5 h-full max-w-6xl mx-auto">
      {/* tree panel */}
      <div className="w-64 shrink-0 bg-zinc-900 rounded-lg border border-zinc-800 p-3 overflow-y-auto">
        <p className="text-zinc-500 text-xs font-medium px-1 mb-2">Node home</p>
        {root === null ? (
          <p className="text-zinc-600 text-xs px-1">Loading…</p>
        ) : root.entries.length === 0 ? (
          <p className="text-zinc-600 text-xs px-1">Empty</p>
        ) : (
          <FileTree
            entries={root.entries}
            onSelectFile={handleSelect}
            selectedPath={selected?.path}
          />
        )}
      </div>

      {/* detail panel */}
      <div className="flex-1 bg-zinc-900 rounded-lg border border-zinc-800 p-5 overflow-auto">
        {!selected && (
          <div className="flex items-center justify-center h-full text-zinc-600 text-sm">
            Select a file to inspect it
          </div>
        )}
        {selected && loading && (
          <div className="text-zinc-500 text-sm">Loading…</div>
        )}
        {selected && !loading && inspect && (
          <TabularPreview
            schema={inspect.schema}
            rowCount={inspect.row_count}
            colCount={inspect.col_count}
            format={inspect.format}
            path={inspect.path}
          />
        )}
        {selected && !loading && textPreview && (
          <div className="space-y-2">
            <div className="flex items-center gap-3 text-xs text-zinc-500">
              <span>{textPreview.size.toLocaleString()} bytes</span>
              {textPreview.truncated && (
                <span className="text-amber-500">truncated at 4 MB</span>
              )}
            </div>
            <pre className="text-xs text-zinc-300 font-mono whitespace-pre-wrap break-all overflow-auto max-h-[60vh] bg-zinc-950 rounded p-3">
              {textPreview.content}
            </pre>
          </div>
        )}
        {selected && !loading && !inspect && !textPreview && (
          <div className="text-zinc-500 text-sm">
            <p className="font-medium text-zinc-300 mb-1">{selected.name}</p>
            <p>Could not preview this file.</p>
          </div>
        )}
      </div>
    </div>
  );
}
