"use client";

import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { lsDir } from "@/lib/api";
import FileTree from "@/components/FileTree";
import { ChevronRight, Home } from "lucide-react";

export default function FilesPage() {
  const [path, setPath] = useState("");
  const [breadcrumbs, setBreadcrumbs] = useState<string[]>([]);

  const lsQ = useQuery({
    queryKey: ["ls", path],
    queryFn: () => lsDir(path),
    retry: false,
  });

  const navigateTo = useCallback(
    (newPath: string) => {
      setPath(newPath);
      // Build breadcrumbs from path segments
      if (!newPath) {
        setBreadcrumbs([]);
        return;
      }
      const parts = newPath.replace(/^\//, "").split("/").filter(Boolean);
      setBreadcrumbs(parts);
    },
    []
  );

  const navigateBreadcrumb = useCallback(
    (index: number) => {
      if (index < 0) {
        setPath("");
        setBreadcrumbs([]);
        return;
      }
      const parts = breadcrumbs.slice(0, index + 1);
      const newPath = "/" + parts.join("/");
      setPath(newPath);
      setBreadcrumbs(parts);
    },
    [breadcrumbs]
  );

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-4xl mx-auto flex flex-col gap-6">
        <div>
          <h1 className="text-xl font-bold font-mono text-white mb-1">File Browser</h1>
          <p className="text-xs font-mono text-gray-500">Navigate the node filesystem</p>
        </div>

        {/* Breadcrumb */}
        <div className="flex items-center gap-1 flex-wrap">
          <button
            onClick={() => navigateBreadcrumb(-1)}
            className="flex items-center gap-1 text-xs font-mono text-[#60a5fa] hover:text-white transition-colors"
          >
            <Home size={13} />
            root
          </button>
          {breadcrumbs.map((crumb, i) => (
            <span key={i} className="flex items-center gap-1">
              <ChevronRight size={12} className="text-gray-600" />
              <button
                onClick={() => navigateBreadcrumb(i)}
                className={`text-xs font-mono transition-colors ${
                  i === breadcrumbs.length - 1
                    ? "text-white"
                    : "text-[#60a5fa] hover:text-white"
                }`}
              >
                {crumb}
              </button>
            </span>
          ))}
        </div>

        {/* Manual path input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") navigateTo(path);
            }}
            placeholder="Enter path (e.g. /data)"
            className="flex-1 px-3 py-2 rounded-lg bg-[#1a1a24] border border-[#1e1e2e] text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-[#3b82f6] transition-colors"
          />
          <button
            onClick={() => navigateTo(path)}
            className="px-4 py-2 rounded-lg bg-[#3b82f6] text-white text-sm font-mono hover:bg-[#2563eb] transition-colors"
          >
            Go
          </button>
        </div>

        {/* Error banner */}
        {lsQ.isError && (
          <div className="rounded-lg border border-red-800 bg-red-950/30 px-4 py-2 text-red-400 text-xs font-mono">
            Failed to list directory — backend may be unreachable or path does not exist
          </div>
        )}

        {/* File tree */}
        <FileTree
          entries={lsQ.data?.entries ?? []}
          onNavigate={navigateTo}
          loading={lsQ.isLoading || lsQ.isFetching}
          error={null}
        />
      </div>
    </div>
  );
}
