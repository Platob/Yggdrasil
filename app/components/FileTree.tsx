"use client";

import type { FsEntry } from "@/lib/types";
import { Folder, File, ChevronRight } from "lucide-react";

interface FileTreeProps {
  entries: FsEntry[];
  onNavigate: (path: string) => void;
  loading?: boolean;
  error?: string | null;
}

function formatSize(bytes?: number): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(entry: FsEntry): string {
  const raw = entry.modified ?? entry.mtime;
  if (!raw) return "—";
  try {
    return new Date(raw).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  } catch {
    return String(raw);
  }
}

function isDir(entry: FsEntry): boolean {
  const t = (entry.type ?? "").toLowerCase();
  return t === "dir" || t === "directory";
}

export default function FileTree({ entries, onNavigate, loading, error }: FileTreeProps) {
  return (
    <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] overflow-hidden">
      <div className="grid grid-cols-[1fr_80px_120px] border-b border-[#1e1e2e] px-4 py-2">
        <span className="text-[10px] font-mono uppercase tracking-widest text-gray-600">Name</span>
        <span className="text-[10px] font-mono uppercase tracking-widest text-gray-600 text-right">Size</span>
        <span className="text-[10px] font-mono uppercase tracking-widest text-gray-600 text-right">Modified</span>
      </div>
      {error && (
        <div className="text-red-400 text-xs font-mono px-4 py-3">{error}</div>
      )}
      {loading && (
        <div className="flex flex-col gap-2 p-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-6 rounded bg-[#1e1e2e] animate-pulse" />
          ))}
        </div>
      )}
      {!loading && !error && entries.length === 0 && (
        <div className="text-gray-600 text-xs font-mono px-4 py-4">Empty directory.</div>
      )}
      {!loading &&
        !error &&
        entries.map((entry, i) => {
          const dir = isDir(entry);
          return (
            <div
              key={i}
              className={`grid grid-cols-[1fr_80px_120px] items-center px-4 py-2.5 border-b border-[#1e1e2e]/50 last:border-0 transition-colors ${
                dir ? "hover:bg-[#1a1a24] cursor-pointer" : "hover:bg-[#16161f]"
              }`}
              onClick={() => dir && onNavigate(entry.path)}
            >
              <div className="flex items-center gap-2 min-w-0">
                {dir ? (
                  <Folder size={15} className="text-[#60a5fa] shrink-0" />
                ) : (
                  <File size={15} className="text-gray-500 shrink-0" />
                )}
                <span
                  className={`text-sm font-mono truncate ${
                    dir ? "text-[#60a5fa] font-medium" : "text-gray-300"
                  }`}
                >
                  {entry.name}
                </span>
                {dir && (
                  <ChevronRight size={13} className="text-gray-600 shrink-0 ml-auto" />
                )}
              </div>
              <span className="text-xs font-mono text-gray-600 text-right">
                {dir ? "—" : formatSize(entry.size)}
              </span>
              <span className="text-xs font-mono text-gray-600 text-right">{formatDate(entry)}</span>
            </div>
          );
        })}
    </div>
  );
}
