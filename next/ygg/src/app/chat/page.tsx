"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getChannels, getMessages, sendMessage, pollMessages, createChannel,
  type Message, type ChannelInfo,
} from "@/lib/api";

const SENDER_COLORS = [
  "text-red-400", 
  "text-emerald-400", 
  "text-amber-400", 
  "text-sky-400", 
  "text-violet-400", 
  "text-pink-400",
  "text-orange-400",
  "text-teal-400",
];

function senderColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
  return SENDER_COLORS[Math.abs(h) % SENDER_COLORS.length];
}

function formatTime(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return ts.slice(11, 16);
  }
}

export default function ChatPage() {
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
    <div className="flex h-[calc(100vh-3rem)] gap-4 animate-in">
      {/* Channel Sidebar */}
      <div className="w-56 nordic-card flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Channels
          </h3>
          <button 
            onClick={() => setShowNewCh(!showNewCh)} 
            className="w-6 h-6 rounded-md bg-primary/10 text-primary hover:bg-primary/20 flex items-center justify-center transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          </button>
        </div>

        {/* New Channel Input */}
        {showNewCh && (
          <div className="p-3 border-b border-border">
            <div className="flex gap-2">
              <input
                value={newCh}
                onChange={(e) => setNewCh(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateChannel()}
                placeholder="channel-name"
                className="input-nordic flex-1 text-xs py-2"
              />
              <button 
                onClick={handleCreateChannel}
                className="px-2 py-2 bg-primary/10 text-primary rounded-lg hover:bg-primary/20 transition-colors"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </button>
            </div>
          </div>
        )}

        {/* Channel List */}
        <div className="flex-1 overflow-auto p-2 space-y-1">
          {channels.map((ch) => (
            <button
              key={ch.name}
              onClick={() => setChannel(ch.name)}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all ${
                channel === ch.name
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted hover:text-foreground hover:bg-card-hover border border-transparent"
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xs opacity-60">#</span>
                <span className="font-medium">{ch.name}</span>
              </div>
              <div className="flex items-center gap-2 mt-1 text-[10px] text-muted">
                <span>{ch.message_count} msgs</span>
                <span className="opacity-50">|</span>
                <span>{ch.members.length} members</span>
              </div>
            </button>
          ))}
        </div>

        {/* User Identity */}
        <div className="p-3 border-t border-border">
          <label className="text-[10px] uppercase tracking-wider text-muted mb-2 block">Your Name</label>
          <input
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            className="input-nordic w-full text-xs py-2"
            placeholder="Enter your name"
          />
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 nordic-card flex flex-col">
        {/* Chat Header */}
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center">
              <span className="text-primary font-mono text-sm">#</span>
            </div>
            <div>
              <h2 className="font-semibold text-foreground">{channel}</h2>
              <p className="text-xs text-muted">{messages.length} messages</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="status-dot online" />
            <span className="text-xs text-muted">Live</span>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-auto p-4 space-y-1">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-12 h-12 rounded-full bg-card-elevated flex items-center justify-center mb-3">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <p className="text-sm text-muted">No messages yet</p>
              <p className="text-xs text-muted mt-1">Start the conversation</p>
            </div>
          ) : (
            messages.map((msg, i) => {
              const showHeader = i === 0 || messages[i - 1].sender !== msg.sender;
              return (
                <div key={msg.id} className={`${showHeader ? "mt-4 first:mt-0" : ""} animate-in`}>
                  {showHeader && (
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-sm font-semibold ${senderColor(msg.sender)}`}>
                        {msg.sender}
                      </span>
                      <span className="text-[10px] text-muted">{formatTime(msg.timestamp)}</span>
                    </div>
                  )}
                  <p className="text-sm text-foreground leading-relaxed">{msg.text}</p>
                </div>
              );
            })
          )}
          <div ref={bottomRef} />
        </div>

        {/* Message Input */}
        <form onSubmit={handleSend} className="p-4 border-t border-border">
          <div className="flex gap-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={`Message #${channel}...`}
              className="input-nordic flex-1"
              autoFocus
            />
            <button
              type="submit"
              disabled={!input.trim()}
              className="btn-primary px-5"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
