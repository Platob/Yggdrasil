"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getChannels, getMessages, sendMessage, pollMessages, createChannel,
  type Message, type ChannelInfo,
} from "@/lib/api";

const SENDER_COLORS = [
  "#f26b3a", "#dc2626", "#fb923c",
  "#6b7280", "#9ca3af", "#e5e7eb",
  "#ef4444", "#f97316",
];

function senderColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return SENDER_COLORS[Math.abs(h) % SENDER_COLORS.length];
}

function formatTime(ts: string): string {
  try { return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
  catch { return ts.slice(11, 16); }
}

export default function MsgPage() {
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [channel, setChannel] = useState("general");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sender, setSender] = useState("web-user");
  const [newCh, setNewCh] = useState("");
  const [showNewCh, setShowNewCh] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastIdRef = useRef<string>("");
  const pollRef = useRef(true);

  const loadChannels = useCallback(async () => {
    try { setChannels(await getChannels()); } catch {}
  }, []);

  const loadMessages = useCallback(async () => {
    try {
      const msgs = await getMessages(channel, 200);
      setMessages(msgs);
      if (msgs.length > 0) lastIdRef.current = msgs[msgs.length - 1].id;
    } catch {}
  }, [channel]);

  useEffect(() => {
    loadChannels();
    loadMessages();
    pollRef.current = true;

    const poll = async () => {
      while (pollRef.current) {
        try {
          const newMsgs = await pollMessages(channel, lastIdRef.current, 25);
          if (newMsgs.length > 0) {
            setMessages((prev) => [...prev, ...newMsgs]);
            lastIdRef.current = newMsgs[newMsgs.length - 1].id;
          }
        } catch {
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    };
    poll();
    return () => { pollRef.current = false; };
  }, [channel, loadChannels, loadMessages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim()) return;
    const text = input;
    setInput("");
    try {
      const msg = await sendMessage(text, sender, channel);
      setMessages((prev) => [...prev, msg]);
      lastIdRef.current = msg.id;
    } catch {}
  }

  async function handleCreateChannel() {
    if (!newCh.trim()) return;
    try {
      await createChannel(newCh.trim());
      setShowNewCh(false);
      setNewCh("");
      loadChannels();
      setChannel(newCh.trim());
    } catch {}
  }

  return (
    <div
      className="flex gap-4"
      style={{ height: "calc(100vh - 3rem)" }}
    >
      {/* Channel list */}
      <div
        className="w-52 flex flex-col rounded-xl"
        style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}
      >
        <div
          className="flex items-center justify-between px-4 py-3"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}
        >
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
            Channels
          </span>
          <button
            onClick={() => setShowNewCh(!showNewCh)}
            className="w-6 h-6 flex items-center justify-center rounded-md transition-colors"
            style={{ color: "#f26b3a", background: "rgba(242,107,58,0.1)" }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </button>
        </div>

        {showNewCh && (
          <div className="px-3 py-2" style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
            <div className="flex gap-2">
              <input
                value={newCh}
                onChange={(e) => setNewCh(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateChannel()}
                placeholder="new-channel"
                className="flex-1 px-2 py-1 text-xs rounded-md outline-none"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: "#fff",
                }}
              />
              <button
                onClick={handleCreateChannel}
                className="px-2 py-1 rounded-md text-xs"
                style={{ background: "rgba(242,107,58,0.15)", color: "#f26b3a" }}
              >
                +
              </button>
            </div>
          </div>
        )}

        <div className="flex-1 overflow-auto p-2 space-y-0.5">
          {channels.map((ch) => (
            <button
              key={ch.name}
              onClick={() => setChannel(ch.name)}
              className="w-full text-left px-3 py-2 rounded-lg text-sm transition-all"
              style={{
                background: channel === ch.name ? "rgba(242,107,58,0.08)" : "transparent",
                border: `1px solid ${channel === ch.name ? "rgba(242,107,58,0.2)" : "transparent"}`,
                color: channel === ch.name ? "#f26b3a" : "rgba(255,255,255,0.45)",
              }}
            >
              <div className="flex items-center gap-1.5">
                <span className="opacity-50 text-xs">#</span>
                <span className="font-medium">{ch.name}</span>
              </div>
              <div className="text-[10px] mt-0.5" style={{ color: "rgba(255,255,255,0.25)" }}>
                {ch.message_count} msgs
              </div>
            </button>
          ))}
        </div>

        {/* Sender name */}
        <div className="p-3" style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}>
          <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "rgba(255,255,255,0.25)" }}>
            Your name
          </p>
          <input
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            className="w-full px-2 py-1.5 text-xs rounded-md outline-none"
            style={{
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#fff",
            }}
          />
        </div>
      </div>

      {/* Chat area */}
      <div
        className="flex-1 flex flex-col rounded-xl"
        style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)" }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.07)" }}
        >
          <div className="flex items-center gap-3">
            <span
              className="text-lg font-mono"
              style={{ color: "#f26b3a" }}
            >
              #
            </span>
            <div>
              <h2 className="font-semibold text-white">{channel}</h2>
              <p className="text-xs" style={{ color: "rgba(255,255,255,0.3)" }}>{messages.length} messages</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full" style={{ background: "#22c55e", boxShadow: "0 0 6px #22c55e" }} />
            <span className="text-xs" style={{ color: "rgba(255,255,255,0.3)" }}>Live</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-auto p-5 space-y-1">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="mb-3" style={{ color: "rgba(255,255,255,0.1)" }}>
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              <p className="text-sm" style={{ color: "rgba(255,255,255,0.25)" }}>No messages yet</p>
            </div>
          ) : (
            messages.map((msg, i) => {
              const showHeader = i === 0 || messages[i - 1].sender !== msg.sender;
              return (
                <div key={msg.id} className={showHeader ? "mt-4 first:mt-0" : ""}>
                  {showHeader && (
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-sm font-semibold" style={{ color: senderColor(msg.sender) }}>
                        {msg.sender}
                      </span>
                      <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.2)" }}>
                        {formatTime(msg.timestamp)}
                      </span>
                    </div>
                  )}
                  <p className="text-sm leading-relaxed" style={{ color: "rgba(255,255,255,0.8)" }}>
                    {msg.text}
                  </p>
                </div>
              );
            })
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <form
          onSubmit={handleSend}
          className="p-4"
          style={{ borderTop: "1px solid rgba(255,255,255,0.07)" }}
        >
          <div className="flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Message #${channel}...`}
              autoFocus
              className="flex-1 px-4 py-2.5 rounded-xl text-sm outline-none"
              style={{
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#fff",
              }}
            />
            <button
              type="submit"
              disabled={!input.trim()}
              className="px-5 py-2.5 rounded-xl text-sm font-semibold transition-all disabled:opacity-30"
              style={{
                background: "linear-gradient(135deg, #f26b3a, #dc2626)",
                color: "#fff",
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
