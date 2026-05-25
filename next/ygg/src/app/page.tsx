"use client";

import Link from "next/link";
import { AnimatedYggdrasilTree, YggdrasilLogo } from "@/components/logo";
import { useEffect, useState } from "react";

// Service definitions as constellations
const SERVICES = [
  {
    id: "bot",
    name: "Bot Control",
    description: "Dashboard, Execute, Chat",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 9h.01M15 9h.01M9 15h6" />
      </svg>
    ),
    href: "/bot",
    position: { x: 20, y: 30 },
  },
  {
    id: "trading",
    name: "Trading",
    description: "Market data & execution",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
        <polyline points="16 7 22 7 22 13" />
      </svg>
    ),
    href: "/trading",
    position: { x: 80, y: 25 },
    comingSoon: true,
  },
  {
    id: "data",
    name: "Data Streams",
    description: "Real-time feeds & analytics",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
    href: "/data",
    position: { x: 15, y: 70 },
    comingSoon: true,
  },
  {
    id: "agents",
    name: "AI Agents",
    description: "Autonomous workflows",
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
      </svg>
    ),
    href: "/agents",
    position: { x: 85, y: 75 },
    comingSoon: true,
  },
];

// Constellation connections between services
const CONNECTIONS = [
  { from: "bot", to: "trading" },
  { from: "bot", to: "data" },
  { from: "trading", to: "agents" },
  { from: "data", to: "agents" },
  { from: "bot", to: "agents" },
];

function StarField() {
  const [stars, setStars] = useState<Array<{ id: number; x: number; y: number; size: number; delay: number }>>([]);

  useEffect(() => {
    const newStars = Array.from({ length: 100 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 2 + 1,
      delay: Math.random() * 5,
    }));
    setStars(newStars);
  }, []);

  return (
    <div className="star-field">
      {stars.map((star) => (
        <div
          key={star.id}
          className="star twinkle"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            animationDelay: `${star.delay}s`,
          }}
        />
      ))}
    </div>
  );
}

function ConstellationSVG() {
  const getServicePosition = (id: string) => {
    const service = SERVICES.find((s) => s.id === id);
    return service ? service.position : { x: 50, y: 50 };
  };

  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#f26b3a" stopOpacity="0.3" />
          <stop offset="50%" stopColor="#fb923c" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#f26b3a" stopOpacity="0.3" />
        </linearGradient>
      </defs>
      {CONNECTIONS.map((conn, i) => {
        const from = getServicePosition(conn.from);
        const to = getServicePosition(conn.to);
        return (
          <line
            key={i}
            x1={from.x}
            y1={from.y}
            x2={to.x}
            y2={to.y}
            stroke="url(#lineGradient)"
            strokeWidth="0.15"
            className="constellation-line"
            style={{ animationDelay: `${i * 0.3}s` }}
          />
        );
      })}
    </svg>
  );
}

function ServiceCard({ service, index }: { service: (typeof SERVICES)[0]; index: number }) {
  return (
    <Link
      href={service.comingSoon ? "#" : service.href}
      className={`constellation-card p-6 group ${service.comingSoon ? "cursor-not-allowed opacity-60" : ""}`}
      style={{ animationDelay: `${index * 0.15}s` }}
      onClick={(e) => service.comingSoon && e.preventDefault()}
    >
      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div className={`p-3 rounded-xl ${service.comingSoon ? "bg-muted/10" : "bg-primary/10"} group-hover:bg-primary/20 transition-colors`}>
            <span className={service.comingSoon ? "text-muted" : "text-primary"}>{service.icon}</span>
          </div>
          {service.comingSoon && (
            <span className="text-[10px] uppercase tracking-widest text-muted bg-muted/10 px-2 py-1 rounded">Soon</span>
          )}
        </div>
        <h3 className="text-lg font-semibold text-foreground mb-1 group-hover:text-primary transition-colors">
          {service.name}
        </h3>
        <p className="text-sm text-muted-foreground">{service.description}</p>
      </div>
      
      {/* Decorative corner rune */}
      <div className="absolute bottom-3 right-3 opacity-10 group-hover:opacity-30 transition-opacity">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <path d="M12 2l2 7h7l-5.5 4 2 7-5.5-4-5.5 4 2-7L3 9h7l2-7z" />
        </svg>
      </div>
    </Link>
  );
}

export default function WelcomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <StarField />
      
      {/* Gradient orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl pulse-glow" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-secondary/10 rounded-full blur-3xl pulse-glow" style={{ animationDelay: "2s" }} />
      
      {/* Navigation */}
      <nav className="relative z-50 flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-3">
          <YggdrasilLogo size={36} />
          <span className="font-bold text-xl tracking-tight">YGGDRASIL</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/bot" className="btn-ghost text-sm">
            Dashboard
          </Link>
          <a href="https://github.com/Platob/Yggdrasil" target="_blank" rel="noopener noreferrer" className="btn-ghost text-sm">
            GitHub
          </a>
          <Link href="/bot" className="btn-primary text-sm">
            Launch App
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 flex flex-col items-center justify-center px-8 pt-8 pb-16">
        {/* Animated Tree */}
        <div className={`relative mb-8 ${mounted ? "opacity-100" : "opacity-0"} transition-opacity duration-1000`}>
          <div className="absolute inset-0 bg-primary/20 rounded-full blur-3xl scale-75 glow-intense" />
          <AnimatedYggdrasilTree size={320} className="relative z-10 float" />
        </div>

        {/* Title */}
        <div className="text-center max-w-3xl animate-fade-in-up" style={{ animationDelay: "0.5s" }}>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
            <span className="gradient-text">The World Tree</span>
            <br />
            <span className="text-foreground">of Distributed Systems</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
            Connect your services like branches of the ancient tree.
            <br />
            Execute, monitor, and orchestrate across realms.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link href="/bot" className="btn-primary px-8 py-3 text-base">
              Enter Dashboard
            </Link>
            <Link href="#services" className="btn-secondary px-8 py-3 text-base">
              Explore Services
            </Link>
          </div>
        </div>
      </section>

      {/* Services Constellation */}
      <section id="services" className="relative z-10 px-8 py-24">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in-up">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <span className="text-muted-foreground">Services</span>{" "}
              <span className="text-foreground">Constellation</span>
            </h2>
            <p className="text-muted-foreground max-w-lg mx-auto">
              Each branch of Yggdrasil connects different realms of functionality
            </p>
          </div>

          <div className="relative">
            {/* Constellation lines */}
            <ConstellationSVG />

            {/* Service cards grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
              {SERVICES.map((service, i) => (
                <ServiceCard key={service.id} service={service} index={i} />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="relative z-10 px-8 py-16 border-t border-border">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { label: "Distributed Nodes", value: "Unlimited" },
              { label: "Latency", value: "<10ms" },
              { label: "Uptime", value: "99.9%" },
              { label: "Protocol", value: "WebSocket" },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className="text-2xl md:text-3xl font-bold text-primary mb-1">{stat.value}</div>
                <div className="text-sm text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-8 py-8 border-t border-border">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <YggdrasilLogo size={24} />
            <span className="text-sm text-muted-foreground">Yggdrasil Distributed Bot Framework</span>
          </div>
          <div className="flex items-center gap-6 text-sm text-muted-foreground">
            <a href="https://github.com/Platob/Yggdrasil" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">
              GitHub
            </a>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
