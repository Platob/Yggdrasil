"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import Link from "next/link";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ─── Globe Bot Nodes Data ────────────────────────────────────────────────────

const BOT_NODES = [
  { id: "paris-01", lat: 48.8566, lng: 2.3522, status: "online", label: "Paris" },
  { id: "nyc-01", lat: 40.7128, lng: -74.006, status: "online", label: "New York" },
  { id: "sf-01", lat: 37.7749, lng: -122.4194, status: "online", label: "San Francisco" },
  { id: "tokyo-01", lat: 35.6762, lng: 139.6503, status: "online", label: "Tokyo" },
  { id: "sydney-01", lat: -33.8688, lng: 151.2093, status: "online", label: "Sydney" },
  { id: "london-01", lat: 51.5074, lng: -0.1278, status: "online", label: "London" },
  { id: "singapore-01", lat: 1.3521, lng: 103.8198, status: "online", label: "Singapore" },
  { id: "brazil-01", lat: -23.5505, lng: -46.6333, status: "online", label: "São Paulo" },
  { id: "dubai-01", lat: 25.2048, lng: 55.2708, status: "online", label: "Dubai" },
  { id: "mumbai-01", lat: 19.076, lng: 72.8777, status: "online", label: "Mumbai" },
  { id: "berlin-01", lat: 52.52, lng: 13.405, status: "online", label: "Berlin" },
  { id: "toronto-01", lat: 43.6532, lng: -79.3832, status: "online", label: "Toronto" },
];

// Arc connections between nodes
const ARCS = [
  ["paris-01", "nyc-01"],
  ["nyc-01", "sf-01"],
  ["sf-01", "tokyo-01"],
  ["tokyo-01", "singapore-01"],
  ["singapore-01", "sydney-01"],
  ["london-01", "dubai-01"],
  ["dubai-01", "mumbai-01"],
  ["paris-01", "london-01"],
  ["berlin-01", "paris-01"],
  ["toronto-01", "nyc-01"],
  ["brazil-01", "nyc-01"],
  ["tokyo-01", "sydney-01"],
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function latLngToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

// ─── Globe Component ─────────────────────────────────────────────────────────

function Globe() {
  const globeRef = useRef<THREE.Group>(null);
  const atmosphereRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (globeRef.current) {
      globeRef.current.rotation.y += delta * 0.08;
    }
  });

  // Create graticule (lat/lng grid lines)
  const graticule = useMemo(() => {
    const points: THREE.Vector3[] = [];
    const radius = 2.01;

    // Latitude lines
    for (let lat = -80; lat <= 80; lat += 20) {
      for (let lng = -180; lng <= 180; lng += 4) {
        points.push(latLngToVector3(lat, lng, radius));
        points.push(latLngToVector3(lat, lng + 4, radius));
      }
    }

    // Longitude lines
    for (let lng = -180; lng < 180; lng += 20) {
      for (let lat = -90; lat <= 90; lat += 4) {
        points.push(latLngToVector3(lat, lng, radius));
        points.push(latLngToVector3(lat + 4, lng, radius));
      }
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return geometry;
  }, []);

  return (
    <group ref={globeRef}>
      {/* Core sphere - dark */}
      <mesh>
        <sphereGeometry args={[1.98, 64, 64]} />
        <meshStandardMaterial
          color="#080810"
          metalness={0.3}
          roughness={0.8}
        />
      </mesh>

      {/* Graticule grid */}
      <lineSegments geometry={graticule}>
        <lineBasicMaterial color="#1a1a2e" transparent opacity={0.4} />
      </lineSegments>

      {/* Inner glow */}
      <mesh ref={atmosphereRef}>
        <sphereGeometry args={[2.15, 64, 64]} />
        <meshBasicMaterial
          color="#f26b3a"
          transparent
          opacity={0.04}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Outer atmosphere */}
      <mesh>
        <sphereGeometry args={[2.35, 64, 64]} />
        <meshBasicMaterial
          color="#f26b3a"
          transparent
          opacity={0.015}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}

// ─── Bot Node Points ─────────────────────────────────────────────────────────

function BotNodes() {
  const groupRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    setTime((t) => t + delta);
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.08;
    }
  });

  return (
    <group ref={groupRef}>
      {BOT_NODES.map((node, i) => {
        const pos = latLngToVector3(node.lat, node.lng, 2.02);
        const pulse = 1 + 0.3 * Math.sin(time * 2 + i);

        return (
          <group key={node.id} position={pos}>
            {/* Glow ring */}
            <mesh>
              <ringGeometry args={[0.025 * pulse, 0.045 * pulse, 32]} />
              <meshBasicMaterial
                color="#f26b3a"
                transparent
                opacity={0.3}
                side={THREE.DoubleSide}
              />
            </mesh>
            {/* Core point */}
            <mesh>
              <sphereGeometry args={[0.02, 16, 16]} />
              <meshBasicMaterial color="#f26b3a" />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// ─── Arc Connections ─────────────────────────────────────────────────────────

function ArcConnections() {
  const groupRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    setTime((t) => t + delta);
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.08;
    }
  });

  const arcs = useMemo(() => {
    return ARCS.map(([fromId, toId]) => {
      const from = BOT_NODES.find((n) => n.id === fromId);
      const to = BOT_NODES.find((n) => n.id === toId);
      if (!from || !to) return null;

      const start = latLngToVector3(from.lat, from.lng, 2.02);
      const end = latLngToVector3(to.lat, to.lng, 2.02);

      // Create curved arc
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const dist = start.distanceTo(end);
      mid.normalize().multiplyScalar(2.02 + dist * 0.15);

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const points = curve.getPoints(50);

      return { points, fromId, toId };
    }).filter(Boolean);
  }, []);

  return (
    <group ref={groupRef}>
      {arcs.map((arc, i) => {
        if (!arc) return null;
        const geometry = new THREE.BufferGeometry().setFromPoints(arc.points);

        // Animated dash effect
        const dashOffset = (time * 0.5 + i * 0.3) % 1;

        return (
          <line key={`${arc.fromId}-${arc.toId}`} geometry={geometry}>
            <lineDashedMaterial
              color="#f26b3a"
              transparent
              opacity={0.4}
              dashSize={0.08}
              gapSize={0.04}
            />
          </line>
        );
      })}
    </group>
  );
}

// ─── Stars Background ────────────────────────────────────────────────────────

function Stars() {
  const starsRef = useRef<THREE.Points>(null);

  const starGeometry = useMemo(() => {
    const positions = new Float32Array(2000 * 3);
    for (let i = 0; i < 2000; i++) {
      const radius = 15 + Math.random() * 25;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geometry;
  }, []);

  useFrame((_, delta) => {
    if (starsRef.current) {
      starsRef.current.rotation.y += delta * 0.01;
    }
  });

  return (
    <points ref={starsRef} geometry={starGeometry}>
      <pointsMaterial color="#ffffff" size={0.03} transparent opacity={0.6} />
    </points>
  );
}

// ─── Scene ───────────────────────────────────────────────────────────────────

function GlobeScene() {
  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 3, 5]} intensity={0.4} color="#ffffff" />
      <directionalLight position={[-5, -3, -5]} intensity={0.15} color="#f26b3a" />

      <Stars />
      <Globe />
      <BotNodes />
      <ArcConnections />

      <OrbitControls
        enableZoom={true}
        enablePan={false}
        minDistance={3}
        maxDistance={8}
        autoRotate={false}
        enableDamping
        dampingFactor={0.05}
      />
    </>
  );
}

// ─── Services ────────────────────────────────────────────────────────────────

const SERVICES = [
  { id: "bot", name: "Bot Control", tag: "Active", href: "/bot" },
  { id: "msg", name: "Messaging", tag: "Active", href: "/msg" },
  { id: "network", name: "Network", tag: "Active", href: "/bot/network" },
  { id: "trading", name: "Trading", tag: "Soon", href: "#", comingSoon: true },
  { id: "agents", name: "AI Agents", tag: "Soon", href: "#", comingSoon: true },
];

// ─── Page ────────────────────────────────────────────────────────────────────

export default function WelcomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#030306] overflow-hidden font-sans">
      {/* Globe Canvas - Full Screen Background */}
      <div className="absolute inset-0 z-0">
        {mounted && (
          <Canvas
            camera={{ position: [0, 0, 5], fov: 45 }}
            gl={{ antialias: true, alpha: true }}
            style={{ background: "transparent" }}
          >
            <GlobeScene />
          </Canvas>
        )}
      </div>

      {/* Gradient overlays */}
      <div className="absolute inset-0 z-[1] pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#030306]" />
        <div className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-[#030306] to-transparent" />
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Nav */}
        <nav className="flex items-center justify-between px-8 py-5">
          <div className="flex items-center gap-2.5">
            <svg width="28" height="28" viewBox="0 0 150 150" fill="#f26b3a" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M57.2,36.22c-2.41-6.61-4.38-13.26-5.36-20.21-.29-1.94-.25-2,1.62-2.65,3.77-1.31,7.62-2.34,11.57-2.95,1.83-.28,1.92-.22,2.17,1.65,1.04,7.89,3.26,15.44,6.62,22.65,.38,.82,.59,2.15,1.46,2.19,1.11,.05,1.26-1.36,1.66-2.21,3.35-7.13,5.55-14.6,6.56-22.42,.27-2.11,.33-2.16,2.46-1.85,3.96,.59,7.78,1.71,11.56,2.99,1.72,.59,1.69,.66,1.42,2.53-.9,6.25-2.53,12.31-4.67,18.24-.24,.67-.98,1.46-.26,2.09,.57,.5,1.29-.15,1.9-.4,6.45-2.67,12.07-6.64,16.97-11.6q1.74-1.72,3.68-.1c2.89,2.41,5.66,4.94,8.09,7.82,1.22,1.46,1.21,1.48-.05,2.81-9.64,10.14-21.57,16.63-35.2,19.67-1.4,.42-1.5,.73-.43,1.82,3.55,3.63,7.48,6.81,11.7,9.62,1.32,.88,1.37,.84,2.9-.6,6.5-6.08,13.83-10.9,21.92-14.58,2.37-1.08,4.81-2,7.26-2.9,1.66-.61,1.93-.47,2.61,1.18,1.53,3.69,2.84,7.45,3.67,11.36,.38,1.76,.35,1.84-1.44,2.48-3.91,1.38-7.68,3.09-11.27,5.17-2.32,1.35-4.56,2.83-6.71,4.43-.6,.44-1.55,.8-1.45,1.63,.12,1.02,1.28,.95,2.02,1.19,5.5,1.74,11.13,2.83,16.89,3.28,4.33,.33,3.49,.2,3.15,3.87-.32,3.44-.99,6.83-1.96,10.14-.52,1.77-.57,1.84-2.4,1.7-22.06-1.63-42.83-11.44-58.39-27.1-2.21-2.2-1.8-2.45-4.18-.06-11.99,12.09-26.24,20.29-42.71,24.59-4.98,1.3-10.04,2.16-15.17,2.55-2.37,.18-2.33,.12-3-2.09-1.07-3.54-1.6-7.18-2.01-10.84-.28-2.44-.23-2.52,2.35-2.7,5.61-.39,11.11-1.32,16.49-2.93,.95-.29,1.9-.6,2.84-.93,.81-.29,.9-.82,.27-1.36-5.7-4.56-12.18-8.03-19.05-10.49-2.03-.71-2.05-.74-1.57-2.77,.88-3.74,2.01-7.4,3.57-10.91,.87-1.96,.9-2.01,2.89-1.32,10.82,3.72,20.74,9.8,29.16,17.53,1.47,1.33,1.46,1.3,3.09,.22,4.2-2.83,8.13-6.04,11.6-9.73,1.03-1.24-.9-1.55-1.76-1.74-13.5-3.1-25.07-9.78-34.6-19.81-.57-.58-.68-1.16-.1-1.81,3.01-3.39,6.23-6.55,9.8-9.36,.74-.58,1.24-.24,1.77,.31,5.17,5.46,11.41,9.9,18.42,12.66,.41,.16,.83,.42,1.61,.04" />
            </svg>
            <span className="font-bold text-white tracking-widest text-sm uppercase">Yggdrasil</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs text-white/40 font-mono">{BOT_NODES.length} nodes online</span>
            <Link
              href="/bot"
              className="text-sm font-medium px-4 py-1.5 rounded-full border border-white/20 text-white/70 hover:text-white hover:border-white/40 transition-all"
            >
              Dashboard
            </Link>
          </div>
        </nav>

        {/* Hero */}
        <main className="flex-1 flex flex-col items-center justify-end pb-20 px-8 text-center">
          <div className="space-y-4 mb-10">
            <h1 className="text-5xl md:text-6xl font-bold text-white tracking-tight">
              The World Tree
            </h1>
            <p className="text-base text-white/35 tracking-widest uppercase">
              Distributed Systems Framework
            </p>
            <div className="pt-3">
              <Link
                href="/bot"
                className="inline-flex items-center gap-2 px-8 py-3 rounded-full text-sm font-semibold transition-all hover:scale-105"
                style={{
                  background: "linear-gradient(135deg, #f26b3a, #dc2626)",
                  color: "#fff",
                  boxShadow: "0 0 40px rgba(242,107,58,0.4)",
                }}
              >
                Enter Network
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          </div>

          {/* Service Links */}
          <div className="flex flex-wrap justify-center gap-2 max-w-xl">
            {SERVICES.map((s) => (
              <Link
                key={s.id}
                href={s.href}
                onClick={(e) => s.comingSoon && e.preventDefault()}
                className="px-4 py-2 rounded-full text-xs font-medium transition-all"
                style={{
                  background: s.comingSoon ? "rgba(255,255,255,0.03)" : "rgba(242,107,58,0.1)",
                  borderWidth: 1,
                  borderColor: s.comingSoon ? "rgba(255,255,255,0.06)" : "rgba(242,107,58,0.3)",
                  color: s.comingSoon ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.8)",
                  cursor: s.comingSoon ? "default" : "pointer",
                }}
              >
                {s.name}
                {s.comingSoon && <span className="ml-1.5 text-[10px] opacity-50">Soon</span>}
              </Link>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
