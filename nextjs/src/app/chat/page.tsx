"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import {
  getChannels,
  getMessages,
  sendMessage,
  createMessageStream,
} from "@/lib/api";
import type { ChannelInfo, Message } from "@/lib/types";

// ── Deterministic color from user key ───────────────────────
const USER_COLORS = [
  "var(--frost)",
  "var(--emerald)",
  "var(--amber)",
  "#a78bfa",  // violet
  "#f472b6",  // pink
  "#60a5fa",  // blue
  "#fb923c",  // orange
];

function userColor(key: string): string {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
  }
  return USER_COLORS[Math.abs(hash) % USER_COLORS.length];
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function formatDate(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return "";
  }
}

export default function ChatPage() {
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [activeChannel, setActiveChannel] = useState("general");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Auto-scroll to bottom on new messages
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // ── Fetch channels ───────────────────────────────────────
  useEffect(() => {
    getChannels()
      .then((res) => {
        setChannels(res.channels);
        // If general doesn't exist but other channels do, pick the first
        if (res.channels.length > 0 && !res.channels.find((c) => c.name === "general")) {
          setActiveChannel(res.channels[0].name);
        }
      })
      .catch(() => setError(true));
  }, []);

  // ── Fetch messages for active channel ────────────────────
  useEffect(() => {
    setLoading(true);
    setMessages([]);
    getMessages(activeChannel)
      .then((res) => {
        setMessages(res.messages);
      })
      .catch(() => {
        // Channel might not exist yet — that's OK, start empty
        setMessages([]);
      })
      .finally(() => setLoading(false));
  }, [activeChannel]);

  // ── SSE stream for live messages ─────────────────────────
  useEffect(() => {
    // Close previous stream if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = createMessageStream(activeChannel);
    eventSourceRef.current = es;

    es.onmessage = (event) => {
      try {
        const msg: Message = JSON.parse(event.data);
        setMessages((prev) => {
          // Avoid duplicates by id
          if (prev.some((m) => m.id === msg.id)) return prev;
          return [...prev, msg];
        });
      } catch {
        // Ignore parse errors
      }
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, [activeChannel]);

  // ── Send message ─────────────────────────────────────────
  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    setSending(true);
    setInput("");
    try {
      const msg = await sendMessage(activeChannel, text);
      setMessages((prev) => {
        if (prev.some((m) => m.id === msg.id)) return prev;
        return [...prev, msg];
      });
    } catch {
      // Restore input if send failed
      setInput(text);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen animate-in">
      {/* ── Channel sidebar ──────────────────────────────────── */}
      <div className="w-56 border-r border-border shrink-0 flex flex-col bg-background-elevated/50">
        <div className="p-4 border-b border-border">
          <h2 className="text-xs font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
            Channels
          </h2>
        </div>
        <div className="flex-1 overflow-y-auto py-2 px-2 space-y-0.5">
          {channels.length === 0 && !error ? (
            <div className="px-3 py-2">
              <p className="text-xs text-muted/60 italic">No channels yet</p>
            </div>
          ) : (
            channels.map((ch) => (
              <button
                key={ch.name}
                onClick={() => setActiveChannel(ch.name)}
                className={`
                  w-full text-left px-3 py-2 rounded-lg text-sm transition-all duration-150
                  ${
                    activeChannel === ch.name
                      ? "bg-frost/10 text-frost"
                      : "text-foreground-dim hover:text-foreground hover:bg-white/[0.03]"
                  }
                `}
              >
                <div className="flex items-center gap-2">
                  <span className="text-muted">#</span>
                  <span className="font-medium truncate">{ch.name}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5 text-[10px] text-muted">
                  <span>{ch.message_count} msgs</span>
                  {ch.members.length > 0 && (
                    <span>{ch.members.length} members</span>
                  )}
                </div>
              </button>
            ))
          )}
          {/* Always show the default channel button if not in the list */}
          {channels.length === 0 && (
            <button
              onClick={() => setActiveChannel("general")}
              className={`
                w-full text-left px-3 py-2 rounded-lg text-sm transition-all duration-150
                ${
                  activeChannel === "general"
                    ? "bg-frost/10 text-frost"
                    : "text-foreground-dim hover:text-foreground hover:bg-white/[0.03]"
                }
              `}
            >
              <div className="flex items-center gap-2">
                <span className="text-muted">#</span>
                <span className="font-medium">general</span>
              </div>
            </button>
          )}
        </div>
      </div>

      {/* ── Main chat area ───────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Channel header */}
        <div className="h-14 flex items-center gap-3 px-6 border-b border-border shrink-0">
          <span className="text-frost text-lg font-medium">#</span>
          <h1 className="text-sm font-semibold text-foreground">{activeChannel}</h1>
          {channels.find((c) => c.name === activeChannel)?.members && (
            <span className="text-[11px] text-muted font-mono ml-auto">
              {channels.find((c) => c.name === activeChannel)?.members.length || 0} members
            </span>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-1">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="w-6 h-6 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-3" />
                <p className="text-xs text-muted font-mono">Loading messages...</p>
              </div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-2">
                <p className="text-sm text-muted">Backend unreachable</p>
                <p className="text-xs text-muted/60">Check that the node is running on port 8100</p>
              </div>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-2">
                <div className="w-12 h-12 rounded-full bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.5">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
                  </svg>
                </div>
                <p className="text-sm text-muted">No messages yet</p>
                <p className="text-xs text-muted/60">Be the first to say something in #{activeChannel}</p>
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => {
                const color = userColor(msg.user_key);
                // Show date separator if the day changed
                const prevMsg = i > 0 ? messages[i - 1] : null;
                const showDate = !prevMsg || formatDate(msg.timestamp) !== formatDate(prevMsg.timestamp);
                // Group consecutive messages from the same user
                const showHeader = !prevMsg || prevMsg.user_key !== msg.user_key || showDate;

                return (
                  <div key={msg.id}>
                    {showDate && (
                      <div className="flex items-center gap-3 py-3">
                        <div className="flex-1 h-px bg-white/[0.06]" />
                        <span className="text-[10px] text-muted font-mono">{formatDate(msg.timestamp)}</span>
                        <div className="flex-1 h-px bg-white/[0.06]" />
                      </div>
                    )}
                    <div className={`group px-3 py-1 rounded-lg hover:bg-white/[0.02] transition-colors ${showHeader ? "mt-3" : "mt-0"}`}>
                      {showHeader && (
                        <div className="flex items-baseline gap-2 mb-0.5">
                          <span className="text-sm font-semibold" style={{ color }}>
                            {msg.user_key}
                          </span>
                          <span className="text-[10px] text-muted font-mono">
                            {formatTime(msg.timestamp)}
                          </span>
                        </div>
                      )}
                      <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap break-words">
                        {msg.content}
                      </p>
                    </div>
                  </div>
                );
              })}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input area */}
        <div className="px-6 py-4 border-t border-border shrink-0">
          <div className="glass-card flex items-end gap-2 p-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Message #${activeChannel}...`}
              rows={1}
              className="
                flex-1 bg-transparent text-sm text-foreground placeholder-muted/50
                resize-none outline-none px-3 py-2 max-h-32
              "
              style={{ minHeight: "36px" }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              className="
                shrink-0 px-4 py-2 rounded-lg text-xs font-semibold
                bg-frost/10 text-frost border border-frost/20
                hover:bg-frost/20 hover:border-frost/40
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-all duration-150
              "
            >
              {sending ? (
                <div className="w-4 h-4 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              )}
            </button>
          </div>
          <p className="text-[10px] text-muted/40 mt-1.5 px-1">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
