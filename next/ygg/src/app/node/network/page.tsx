"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useRef, useState, useMemo } from "react";
import * as THREE from "three";

// ── Types ────────────────────────────────────────────────────
interface BotNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  status: "online" | "offline" | "pending";
  version: string;
  uptime: number;
}

// ── Mock bot locations ───────────────────────────────────────
const MOCK_BOTS: BotNode[] = [
  { id: "node-001", name: "Primary", lat: 48.8566, lng: 2.3522, status: "online", version: "0.1.0", uptime: 86400 },
  { id: "node-002", name: "US-East", lat: 40.7128, lng: -74.006, status: "online", version: "0.1.0", uptime: 72000 },
  { id: "node-003", name: "US-West", lat: 37.7749, lng: -122.4194, status: "online", version: "0.1.0", uptime: 43200 },
  { id: "node-004", name: "Tokyo", lat: 35.6762, lng: 139.6503, status: "online", version: "0.1.0", uptime: 36000 },
  { id: "node-005", name: "Sydney", lat: -33.8688, lng: 151.2093, status: "pending", version: "0.1.0", uptime: 18000 },
  { id: "node-006", name: "London", lat: 51.5074, lng: -0.1278, status: "online", version: "0.1.0", uptime: 54000 },
  { id: "node-007", name: "Singapore", lat: 1.3521, lng: 103.8198, status: "online", version: "0.1.0", uptime: 28800 },
  { id: "node-008", name: "Sao Paulo", lat: -23.5505, lng: -46.6333, status: "offline", version: "0.0.9", uptime: 0 },
];

// ── Convert lat/lng to 3D position on sphere ─────────────────
function latLngToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

// ── Earth Globe Component (Procedural wireframe) ─────────────
function Earth() {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireRef = useRef<THREE.LineSegments>(null);
  
  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.05;
    }
    if (wireRef.current) {
      wireRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <group>
      {/* Solid dark sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1.98, 64, 64]} />
        <meshStandardMaterial 
          color="#0a0a10"
          metalness={0.2} 
          roughness={0.9}
        />
      </mesh>
      {/* Wireframe grid */}
      <lineSegments ref={wireRef}>
        <wireframeGeometry args={[new THREE.SphereGeometry(2, 24, 18)]} />
        <lineBasicMaterial color="#1e1e2a" transparent opacity={0.6} />
      </lineSegments>
      {/* Glow atmosphere */}
      <mesh>
        <sphereGeometry args={[2.08, 64, 64]} />
        <meshBasicMaterial 
          color="#f26b3a"
          transparent 
          opacity={0.03}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}

// ── Bot Marker Component ─────────────────────────────────────
function BotMarker({ bot, onClick, isSelected }: { bot: BotNode; onClick: () => void; isSelected: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const position = useMemo(() => latLngToVector3(bot.lat, bot.lng, 2.05), [bot.lat, bot.lng]);
  
  const color = bot.status === "online" ? "#4ade80" : bot.status === "pending" ? "#fbbf24" : "#ef4444";
  
  useFrame((state) => {
    if (meshRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 2 + bot.lat) * 0.15;
      meshRef.current.scale.setScalar(isSelected ? scale * 1.5 : scale);
    }
  });

  return (
    <group position={position}>
      <mesh ref={meshRef} onClick={onClick}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color} 
          emissiveIntensity={isSelected ? 1.5 : 0.8} 
        />
      </mesh>
      {/* Glow ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.05, 0.08, 32]} />
        <meshBasicMaterial color={color} transparent opacity={isSelected ? 0.6 : 0.3} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

// ── Connection Lines Between Bots ────────────────────────────
function ConnectionLines({ bots }: { bots: BotNode[] }) {
  const onlineBots = bots.filter(b => b.status === "online");
  
  const lines = useMemo(() => {
    const result: THREE.Vector3[][] = [];
    // Connect to primary node (first one)
    for (let i = 1; i < onlineBots.length; i++) {
      const start = latLngToVector3(onlineBots[0].lat, onlineBots[0].lng, 2.03);
      const end = latLngToVector3(onlineBots[i].lat, onlineBots[i].lng, 2.03);
      result.push([start, end]);
    }
    return result;
  }, [onlineBots]);

  return (
    <>
      {lines.map((points, i) => {
        const positions = new Float32Array([...points[0].toArray(), ...points[1].toArray()]);
        return (
          <line key={i}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[positions, 3]}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#f26b3a" transparent opacity={0.3} />
          </line>
        );
      })}
    </>
  );
}

// ── Scene Component ──────────────────────────────────────────
function Scene({ bots, selectedBot, onSelectBot }: { 
  bots: BotNode[]; 
  selectedBot: string | null; 
  onSelectBot: (id: string | null) => void 
}) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#5b9bd5" />
      
      <Earth />
      
      <ConnectionLines bots={bots} />
      
      {bots.map((bot) => (
        <BotMarker 
          key={bot.id} 
          bot={bot} 
          onClick={() => onSelectBot(selectedBot === bot.id ? null : bot.id)}
          isSelected={selectedBot === bot.id}
        />
      ))}
      
      <OrbitControls 
        enablePan={false} 
        minDistance={3} 
        maxDistance={8}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
      />
    </>
  );
}

// ── Info Panel Component ─────────────────────────────────────
function InfoPanel({ bot }: { bot: BotNode | null }) {
  if (!bot) return null;
  
  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  return (
    <div 
      className="absolute top-4 right-4 w-72 rounded-xl p-4 backdrop-blur-md"
      style={{ 
        background: "rgba(12, 12, 15, 0.9)", 
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center gap-3 mb-3">
        <div 
          className="w-3 h-3 rounded-full"
          style={{ 
            background: bot.status === "online" ? "#4ade80" : bot.status === "pending" ? "#fbbf24" : "#ef4444",
            boxShadow: `0 0 8px ${bot.status === "online" ? "#4ade80" : bot.status === "pending" ? "#fbbf24" : "#ef4444"}`
          }}
        />
        <h3 className="font-bold text-lg" style={{ color: "var(--foreground)" }}>{bot.name}</h3>
      </div>
      
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Node ID</span>
          <span className="font-mono" style={{ color: "var(--foreground-dim)" }}>{bot.id}</span>
        </div>
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Status</span>
          <span className="capitalize" style={{ color: bot.status === "online" ? "#4ade80" : bot.status === "pending" ? "#fbbf24" : "#ef4444" }}>
            {bot.status}
          </span>
        </div>
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Version</span>
          <span style={{ color: "var(--foreground-dim)" }}>{bot.version}</span>
        </div>
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Uptime</span>
          <span style={{ color: "var(--foreground-dim)" }}>{formatUptime(bot.uptime)}</span>
        </div>
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Latitude</span>
          <span className="font-mono" style={{ color: "var(--primary)" }}>{bot.lat.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span style={{ color: "var(--muted)" }}>Longitude</span>
          <span className="font-mono" style={{ color: "var(--primary)" }}>{bot.lng.toFixed(4)}</span>
        </div>
      </div>
    </div>
  );
}

// ── Stats Bar Component ──────────────────────────────────────
function StatsBar({ bots }: { bots: BotNode[] }) {
  const online = bots.filter(b => b.status === "online").length;
  const pending = bots.filter(b => b.status === "pending").length;
  const offline = bots.filter(b => b.status === "offline").length;

  return (
    <div 
      className="absolute bottom-4 left-4 right-4 flex items-center justify-center gap-6 py-3 px-6 rounded-xl backdrop-blur-md"
      style={{ 
        background: "rgba(12, 12, 15, 0.9)", 
        border: "1px solid var(--border)",
      }}
    >
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ background: "#4ade80" }} />
        <span className="text-sm" style={{ color: "var(--muted)" }}>Online</span>
        <span className="font-bold" style={{ color: "#4ade80" }}>{online}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ background: "#fbbf24" }} />
        <span className="text-sm" style={{ color: "var(--muted)" }}>Pending</span>
        <span className="font-bold" style={{ color: "#fbbf24" }}>{pending}</span>
      </div>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full" style={{ background: "#ef4444" }} />
        <span className="text-sm" style={{ color: "var(--muted)" }}>Offline</span>
        <span className="font-bold" style={{ color: "#ef4444" }}>{offline}</span>
      </div>
      <div className="ml-4 pl-4" style={{ borderLeft: "1px solid var(--border)" }}>
        <span className="text-sm" style={{ color: "var(--muted)" }}>Total Nodes</span>
        <span className="font-bold ml-2" style={{ color: "var(--foreground)" }}>{bots.length}</span>
      </div>
    </div>
  );
}

// ── Main Page Component ──────────────────────────────────────
export default function NetworkPage() {
  const [selectedBot, setSelectedBot] = useState<string | null>(null);
  const [bots] = useState<BotNode[]>(MOCK_BOTS);

  const selectedBotData = bots.find(b => b.id === selectedBot) || null;

  return (
    <div className="relative w-full h-[calc(100vh-1rem)]">
      {/* Header */}
      <div 
        className="absolute top-4 left-4 z-10 flex items-center gap-3 py-2 px-4 rounded-xl backdrop-blur-md"
        style={{ 
          background: "rgba(12, 12, 15, 0.9)", 
          border: "1px solid var(--border)",
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: "var(--primary)" }}>
          <circle cx="12" cy="12" r="10" />
          <line x1="2" y1="12" x2="22" y2="12" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
        <h1 className="font-bold" style={{ color: "var(--foreground)" }}>Network Map</h1>
        <span className="text-xs px-2 py-0.5 rounded" style={{ background: "var(--primary-glow)", color: "var(--primary)" }}>
          Live
        </span>
      </div>

      {/* Instructions */}
      <div 
        className="absolute top-4 left-1/2 -translate-x-1/2 z-10 py-1.5 px-3 rounded-lg text-xs"
        style={{ 
          background: "rgba(12, 12, 15, 0.7)", 
          color: "var(--muted)",
        }}
      >
        Drag to rotate · Scroll to zoom · Click node for details
      </div>

      {/* 3D Canvas */}
      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }}
        style={{ background: "radial-gradient(ellipse at center, #0f0f14 0%, #050507 100%)" }}
      >
        <Scene bots={bots} selectedBot={selectedBot} onSelectBot={setSelectedBot} />
      </Canvas>

      {/* Info Panel */}
      <InfoPanel bot={selectedBotData} />

      {/* Stats Bar */}
      <StatsBar bots={bots} />
    </div>
  );
}
