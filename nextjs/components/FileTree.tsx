"use client";
import { useState } from "react";
import type { FsEntry } from "@/lib/types";

interface Props {
  entries: FsEntry[];
  onSelectFile: (entry: FsEntry) => void;
  selectedPath?: string;
}

function sizeLabel(n: number) {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}K`;
  return `${(n / 1024 / 1024).toFixed(1)}M`;
}

export default function FileTree({ entries, onSelectFile, selectedPath }: Props) {
  const dirs = entries.filter((e) => e.is_dir);
  const files = entries.filter((e) => !e.is_dir);

  return (
    <div className="font-mono text-sm">
      {dirs.map((e) => (
        <DirRow key={e.path} entry={e} onSelectFile={onSelectFile} selectedPath={selectedPath} />
      ))}
      {files.map((e) => (
        <button
          key={e.path}
          onClick={() => onSelectFile(e)}
          className={`w-full flex items-center gap-2 px-2 py-1 rounded text-left hover:bg-zinc-800/60 transition-colors
            ${selectedPath === e.path ? "bg-zinc-800 text-zinc-100" : "text-zinc-400"}`}
        >
          <span className="text-zinc-600">📄</span>
          <span className="flex-1 truncate">{e.name}</span>
          <span className="text-zinc-600 text-xs shrink-0">{sizeLabel(e.size)}</span>
        </button>
      ))}
    </div>
  );
}

function DirRow({
  entry, onSelectFile, selectedPath,
}: { entry: FsEntry; onSelectFile: (e: FsEntry) => void; selectedPath?: string }) {
  const [open, setOpen] = useState(false);
  const [children, setChildren] = useState<FsEntry[] | null>(null);

  async function toggle() {
    if (!open && children === null) {
      try {
        const { lsDir } = await import("@/lib/api");
        const res = await lsDir(entry.path);
        setChildren(res.entries);
      } catch {
        setChildren([]);
      }
    }
    setOpen((v) => !v);
  }

  return (
    <div>
      <button
        onClick={toggle}
        className="w-full flex items-center gap-2 px-2 py-1 rounded text-left text-zinc-300 hover:bg-zinc-800/60 transition-colors"
      >
        <span>{open ? "📂" : "📁"}</span>
        <span className="flex-1 truncate">{entry.name}</span>
        <span className="text-zinc-600 text-xs">{open ? "▾" : "▸"}</span>
      </button>
      {open && children !== null && (
        <div className="pl-4">
          {children.length === 0 ? (
            <p className="text-zinc-600 text-xs px-2 py-1">empty</p>
          ) : (
            <FileTree entries={children} onSelectFile={onSelectFile} selectedPath={selectedPath} />
          )}
        </div>
      )}
    </div>
  );
}
