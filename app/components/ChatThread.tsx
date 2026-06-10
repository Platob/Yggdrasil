"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/lib/types";

interface ChatThreadProps {
  messages: Message[];
  loading?: boolean;
  error?: string | null;
}

function formatTs(msg: Message): string {
  const raw = msg.timestamp ?? msg.ts;
  if (!raw) return "";
  try {
    return new Date(raw).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return String(raw);
  }
}

export default function ChatThread({ messages, loading, error }: ChatThreadProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/30 px-4 py-2 text-red-400 text-xs font-mono">
          {error}
        </div>
      )}
      {loading && messages.length === 0 && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex gap-3">
              <div className="h-7 w-20 rounded bg-[#1e1e2e] animate-pulse" />
              <div className="h-7 flex-1 rounded bg-[#1e1e2e] animate-pulse" />
            </div>
          ))}
        </div>
      )}
      {!loading && !error && messages.length === 0 && (
        <div className="text-gray-600 text-sm font-mono text-center mt-8">
          No messages yet. Say something!
        </div>
      )}
      {messages.map((msg, i) => (
        <div key={i} className="flex gap-3 items-start group">
          <div className="shrink-0 flex flex-col items-end gap-0.5">
            <span className="text-xs font-mono font-semibold text-[#60a5fa]">
              {msg.sender ?? "unknown"}
            </span>
            <span className="text-[10px] font-mono text-gray-600">{formatTs(msg)}</span>
          </div>
          <div className="flex-1 bg-[#1a1a24] rounded-lg px-3 py-2 text-sm text-gray-200 font-mono break-words">
            {msg.text}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
