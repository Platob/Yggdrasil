"use client";

import type { Channel } from "@/lib/types";
import { Hash } from "lucide-react";

interface ChannelListProps {
  channels: Channel[];
  active: string;
  onSelect: (name: string) => void;
  loading?: boolean;
}

export default function ChannelList({ channels, active, onSelect, loading }: ChannelListProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-[#1e1e2e]">
        <span className="text-xs font-mono uppercase tracking-widest text-gray-500">Channels</span>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {loading && (
          <div className="flex flex-col gap-2 p-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-8 rounded bg-[#1e1e2e] animate-pulse" />
            ))}
          </div>
        )}
        {!loading && channels.length === 0 && (
          <div className="text-gray-600 text-xs font-mono p-2">No channels.</div>
        )}
        {channels.map((ch) => {
          const name = ch.name ?? String(ch.id ?? "unknown");
          const isActive = name === active;
          return (
            <button
              key={name}
              onClick={() => onSelect(name)}
              className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm font-mono transition-colors ${
                isActive
                  ? "bg-[#1e2a3a] text-[#60a5fa]"
                  : "text-gray-400 hover:bg-[#1a1a24] hover:text-gray-200"
              }`}
            >
              <Hash size={14} />
              {name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
