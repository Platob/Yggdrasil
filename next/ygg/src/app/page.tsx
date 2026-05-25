"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import Link from "next/link";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ─── Bot Nodes Data ──────────────────────────────────────────────────────────

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

// Arc connections (PR-style: from -> to)
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

// ─── Constants ───────────────────────────────────────────────────────────────

const GLOBE_RADIUS = 2;
const DOT_DENSITY = 0.03;
const DEG2RAD = Math.PI / 180;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function latLngToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = (lng + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

// Get initial rotation based on user timezone (like GitHub)
function getTimezoneRotation(): number {
  const date = new Date();
  const offset = date.getTimezoneOffset() || 0;
  const maxOffset = 60 * 12;
  return Math.PI * (offset / maxOffset);
}

// ─── Earth Dots Component (GitHub-style) ─────────────────────────────────────

function EarthDots({ rotation }: { rotation: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { camera } = useThree();

  // Generate dots covering the globe
  const { matrices, count } = useMemo(() => {
    const matrices: THREE.Matrix4[] = [];
    const rows = Math.floor(180 / (DOT_DENSITY * 100));

    for (let lat = -90; lat <= 90; lat += 180 / rows) {
      const radius = Math.cos(Math.abs(lat) * DEG2RAD) * GLOBE_RADIUS;
      const circumference = radius * Math.PI * 2;
      const dotsForLat = Math.max(1, Math.floor(circumference / (DOT_DENSITY * 3)));

      for (let x = 0; x < dotsForLat; x++) {
        const lng = -180 + (x * 360) / dotsForLat;

        // Simple land mask approximation (continents rough shape)
        if (!isLand(lat, lng)) continue;

        const pos = latLngToVector3(lat, lng, GLOBE_RADIUS);
        const matrix = new THREE.Matrix4();

        // Orient dot to face outward
        const normal = pos.clone().normalize();
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);

        matrix.compose(pos, quaternion, new THREE.Vector3(1, 1, 1));
        matrices.push(matrix);
      }
    }

    return { matrices, count: matrices.length };
  }, []);

  // Apply matrices to instanced mesh
  useEffect(() => {
    if (!meshRef.current) return;
    matrices.forEach((m, i) => {
      meshRef.current!.setMatrixAt(i, m);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [matrices]);

  // Update dot opacity based on distance from camera (GitHub technique)
  useFrame(() => {
    if (!meshRef.current) return;
    meshRef.current.rotation.y = rotation;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <circleGeometry args={[0.018, 6]} />
      <meshBasicMaterial color="#3a3a4a" transparent opacity={0.8} side={THREE.DoubleSide} />
    </instancedMesh>
  );
}

// Simple land detection (approximation without texture)
function isLand(lat: number, lng: number): boolean {
  // Rough continent boundaries for visualization
  // North America
  if (lat > 25 && lat < 70 && lng > -170 && lng < -50) {
    if (lat > 50 && lng < -140) return true; // Alaska
    if (lat > 25 && lat < 50 && lng > -130 && lng < -65) return true; // USA
    if (lat > 50 && lng > -140 && lng < -55) return true; // Canada
    return false;
  }
  // South America
  if (lat > -55 && lat < 15 && lng > -80 && lng < -35) return true;
  // Europe
  if (lat > 35 && lat < 70 && lng > -10 && lng < 40) return true;
  // Africa
  if (lat > -35 && lat < 35 && lng > -20 && lng < 50) return true;
  // Asia
  if (lat > 5 && lat < 75 && lng > 40 && lng < 180) return true;
  if (lat > 5 && lat < 45 && lng > 65 && lng < 145) return true;
  // Australia
  if (lat > -45 && lat < -10 && lng > 110 && lng < 155) return true;
  // Random ocean dots for visual interest
  return Math.random() < 0.02;
}

// ─── Halo Component (GitHub technique) ───────────────────────────────────────

function Halo() {
  const meshRef = useRef<THREE.Mesh>(null);

  // Custom shader for halo gradient
  const haloMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        color: { value: new THREE.Color("#f26b3a") },
      },
      vertexShader: `
        varying vec3 vNormal;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 color;
        varying vec3 vNormal;
        void main() {
          float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
          gl_FragColor = vec4(color, intensity * 0.4);
        }
      `,
      side: THREE.BackSide,
      transparent: true,
      blending: THREE.AdditiveBlending,
    });
  }, []);

  return (
    <mesh ref={meshRef} material={haloMaterial}>
      <sphereGeometry args={[GLOBE_RADIUS * 1.15, 64, 64]} />
    </mesh>
  );
}

// ─── Globe Core ──────────────────────────────────────────────────────────────

function GlobeCore({ rotation }: { rotation: number }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y = rotation;
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[GLOBE_RADIUS * 0.995, 64, 64]} />
      <meshStandardMaterial
        color="#080810"
        metalness={0.1}
        roughness={0.9}
      />
    </mesh>
  );
}

// ─── Bot Node Spikes (GitHub-style spires) ───────────────────────────────────

function NodeSpikes({ rotation }: { rotation: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    setTime((t) => t + delta);
    if (groupRef.current) {
      groupRef.current.rotation.y = rotation;
    }
  });

  return (
    <group ref={groupRef}>
      {BOT_NODES.map((node, i) => {
        const pos = latLngToVector3(node.lat, node.lng, GLOBE_RADIUS);
        const normal = pos.clone().normalize();
        const spikeHeight = 0.15 + 0.05 * Math.sin(time * 2 + i);

        // Create spike endpoint
        const endPos = pos.clone().add(normal.clone().multiplyScalar(spikeHeight));

        return (
          <group key={node.id}>
            {/* Spike line */}
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([...pos.toArray(), ...endPos.toArray()])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#4ade80" transparent opacity={0.8} />
            </line>
            {/* Spike tip glow */}
            <mesh position={endPos}>
              <sphereGeometry args={[0.025, 8, 8]} />
              <meshBasicMaterial color="#4ade80" />
            </mesh>
            {/* Pulse ring */}
            <mesh position={endPos} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.02, 0.04 + 0.02 * Math.sin(time * 3 + i), 16]} />
              <meshBasicMaterial
                color="#4ade80"
                transparent
                opacity={0.3 + 0.2 * Math.sin(time * 3 + i)}
                side={THREE.DoubleSide}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// ─── Arc Connections (GitHub-style bezier curves) ────────────────────────────

function ArcConnections({ rotation }: { rotation: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const arcsRef = useRef<THREE.Group>(null);
  const [time, setTime] = useState(0);

  useFrame((_, delta) => {
    setTime((t) => t + delta);
    if (groupRef.current) {
      groupRef.current.rotation.y = rotation;
    }
  });

  // Animated arcs
  const arcMeshes = useMemo(() => {
    return ARCS.map(([fromId, toId], index) => {
      const from = BOT_NODES.find((n) => n.id === fromId);
      const to = BOT_NODES.find((n) => n.id === toId);
      if (!from || !to) return null;

      const start = latLngToVector3(from.lat, from.lng, GLOBE_RADIUS);
      const end = latLngToVector3(to.lat, to.lng, GLOBE_RADIUS);

      // Calculate arc height based on distance (GitHub technique)
      const dist = start.distanceTo(end);
      const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const arcHeight = GLOBE_RADIUS + dist * 0.2;
      midPoint.normalize().multiplyScalar(arcHeight);

      // Create bezier curve
      const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);

      // Create tube geometry for the arc
      const tubeGeometry = new THREE.TubeGeometry(curve, 44, 0.008, 8, false);

      return { geometry: tubeGeometry, index, from, to };
    }).filter(Boolean);
  }, []);

  return (
    <group ref={groupRef}>
      <group ref={arcsRef}>
        {arcMeshes.map((arc) => {
          if (!arc) return null;
          // Animate opacity to create "traveling" effect
          const phase = (time * 0.5 + arc.index * 0.3) % 3;
          const opacity = phase < 1 ? phase : phase < 2 ? 1 : 3 - phase;

          return (
            <mesh key={arc.index} geometry={arc.geometry}>
              <meshBasicMaterial
                color="#f26b3a"
                transparent
                opacity={opacity * 0.6}
              />
            </mesh>
          );
        })}
      </group>
    </group>
  );
}

// ─── Landing Effects (GitHub-style) ──────────────────────────────────────────

function LandingEffects({ rotation }: { rotation: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const [effects, setEffects] = useState<{ pos: THREE.Vector3; scale: number; opacity: number; id: number }[]>([]);
  const nextId = useRef(0);

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = rotation;
    }

    // Randomly spawn landing effects
    if (Math.random() < 0.02) {
      const node = BOT_NODES[Math.floor(Math.random() * BOT_NODES.length)];
      const pos = latLngToVector3(node.lat, node.lng, GLOBE_RADIUS + 0.01);
      setEffects((prev) => [...prev, { pos, scale: 0, opacity: 1, id: nextId.current++ }]);
    }

    // Animate effects
    setEffects((prev) =>
      prev
        .map((e) => ({
          ...e,
          scale: e.scale + (1 - e.scale) * 0.06,
          opacity: 1 - e.scale,
        }))
        .filter((e) => e.opacity > 0.01)
    );
  });

  return (
    <group ref={groupRef}>
      {effects.map((e) => (
        <mesh key={e.id} position={e.pos}>
          <ringGeometry args={[0.03 * e.scale, 0.05 * e.scale, 16]} />
          <meshBasicMaterial
            color="#f26b3a"
            transparent
            opacity={e.opacity * 0.5}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// ─── Stars Background ────────────────────────────────────────────────────────

function Stars() {
  const starsRef = useRef<THREE.Points>(null);

  const starGeometry = useMemo(() => {
    const positions = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) {
      const radius = 20 + Math.random() * 30;
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
      starsRef.current.rotation.y += delta * 0.005;
    }
  });

  return (
    <points ref={starsRef} geometry={starGeometry}>
      <pointsMaterial color="#ffffff" size={0.02} transparent opacity={0.5} sizeAttenuation />
    </points>
  );
}

// ─── Main Globe Scene ────────────────────────────────────────────────────────

function GlobeScene() {
  const [rotation, setRotation] = useState(() => getTimezoneRotation());

  useFrame((_, delta) => {
    setRotation((r) => r + delta * 0.06);
  });

  return (
    <>
      {/* Lights - 4 point lights like GitHub */}
      <ambientLight intensity={0.1} />
      <directionalLight position={[5, 3, 5]} intensity={0.3} color="#ffffff" />
      <directionalLight position={[-5, 3, 5]} intensity={0.2} color="#f26b3a" />
      <directionalLight position={[0, -5, 0]} intensity={0.1} color="#4a88c4" />
      <pointLight position={[0, 0, 5]} intensity={0.2} color="#ffffff" />

      <Stars />
      <Halo />
      <GlobeCore rotation={rotation} />
      <EarthDots rotation={rotation} />
      <NodeSpikes rotation={rotation} />
      <ArcConnections rotation={rotation} />
      <LandingEffects rotation={rotation} />

      <OrbitControls
        enableZoom
        enablePan={false}
        minDistance={3.5}
        maxDistance={8}
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.5}
      />
    </>
  );
}

// ─── Services Data ───────────────────────────────────────────────────────────

const SERVICES = [
  { id: "bot", name: "Bot Control", href: "/bot", active: true },
  { id: "msg", name: "Messaging", href: "/msg", active: true },
  { id: "network", name: "Network", href: "/bot/network", active: true },
  { id: "trading", name: "Trading", href: "#", comingSoon: true },
  { id: "agents", name: "AI Agents", href: "#", comingSoon: true },
];

// ─── Page Component ──────────────────────────────────────────────────────────

export default function WelcomePage() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div className="relative min-h-screen bg-[#030306] overflow-hidden font-sans">
      {/* Globe Canvas */}
      <div className="absolute inset-0 z-0">
        {mounted && (
          <Canvas
            camera={{ position: [0, 0, 5.5], fov: 45 }}
            gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
            dpr={[1, 1.5]}
            style={{ background: "transparent" }}
          >
            <GlobeScene />
          </Canvas>
        )}
      </div>

      {/* Gradient overlays for text readability */}
      <div className="absolute inset-0 z-[1] pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-b from-[#030306]/60 via-transparent to-[#030306]" />
        <div className="absolute top-0 left-0 right-0 h-24 bg-gradient-to-b from-[#030306] to-transparent" />
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Nav */}
        <nav className="flex items-center justify-between px-6 md:px-8 py-4">
          <div className="flex items-center gap-2.5">
            <svg width="24" height="24" viewBox="0 0 150 150" fill="#f26b3a" xmlns="http://www.w3.org/2000/svg">
              <path fillRule="evenodd" d="M57.2,36.22c-2.41-6.61-4.38-13.26-5.36-20.21-.29-1.94-.25-2,1.62-2.65,3.77-1.31,7.62-2.34,11.57-2.95,1.83-.28,1.92-.22,2.17,1.65,1.04,7.89,3.26,15.44,6.62,22.65,.38,.82,.59,2.15,1.46,2.19,1.11,.05,1.26-1.36,1.66-2.21,3.35-7.13,5.55-14.6,6.56-22.42,.27-2.11,.33-2.16,2.46-1.85,3.96,.59,7.78,1.71,11.56,2.99,1.72,.59,1.69,.66,1.42,2.53-.9,6.25-2.53,12.31-4.67,18.24-.24,.67-.98,1.46-.26,2.09,.57,.5,1.29-.15,1.9-.4,6.45-2.67,12.07-6.64,16.97-11.6q1.74-1.72,3.68-.1c2.89,2.41,5.66,4.94,8.09,7.82,1.22,1.46,1.21,1.48-.05,2.81-9.64,10.14-21.57,16.63-35.2,19.67-1.4,.42-1.5,.73-.43,1.82,3.55,3.63,7.48,6.81,11.7,9.62,1.32,.88,1.37,.84,2.9-.6,6.5-6.08,13.83-10.9,21.92-14.58,2.37-1.08,4.81-2,7.26-2.9,1.66-.61,1.93-.47,2.61,1.18,1.53,3.69,2.84,7.45,3.67,11.36,.38,1.76,.35,1.84-1.44,2.48-3.91,1.38-7.68,3.09-11.27,5.17-2.32,1.35-4.56,2.83-6.71,4.43-.6,.44-1.55,.8-1.45,1.63,.12,1.02,1.28,.95,2.02,1.19,5.5,1.74,11.13,2.83,16.89,3.28,4.33,.33,3.49,.2,3.15,3.87-.32,3.44-.99,6.83-1.96,10.14-.52,1.77-.57,1.84-2.4,1.7-22.06-1.63-42.83-11.44-58.39-27.1-2.21-2.2-1.8-2.45-4.18-.06-11.99,12.09-26.24,20.29-42.71,24.59-4.98,1.3-10.04,2.16-15.17,2.55-2.37,.18-2.33,.12-3-2.09-1.07-3.54-1.6-7.18-2.01-10.84-.28-2.44-.23-2.52,2.35-2.7,5.61-.39,11.11-1.32,16.49-2.93,.95-.29,1.9-.6,2.84-.93,.81-.29,.9-.82,.27-1.36-5.7-4.56-12.18-8.03-19.05-10.49-2.03-.71-2.05-.74-1.57-2.77,.88-3.74,2.01-7.4,3.57-10.91,.87-1.96,.9-2.01,2.89-1.32,10.82,3.72,20.74,9.8,29.16,17.53,1.47,1.33,1.46,1.3,3.09,.22,4.2-2.83,8.13-6.04,11.6-9.73,1.03-1.24-.9-1.55-1.76-1.74-13.5-3.1-25.07-9.78-34.6-19.81-.57-.58-.68-1.16-.1-1.81,3.01-3.39,6.23-6.55,9.8-9.36,.74-.58,1.24-.24,1.77,.31,5.17,5.46,11.41,9.9,18.42,12.66,.41,.16,.83,.42,1.61,.04" />
            </svg>
            <span className="font-bold text-white tracking-widest text-xs uppercase">Yggdrasil</span>
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-1.5 text-xs text-white/30 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              {BOT_NODES.length} nodes
            </div>
            <Link
              href="/bot"
              className="text-xs font-medium px-3.5 py-1.5 rounded-full border border-white/10 text-white/60 hover:text-white hover:border-white/30 transition-all"
            >
              Dashboard
            </Link>
          </div>
        </nav>

        {/* Hero */}
        <main className="flex-1 flex flex-col items-center justify-end pb-16 md:pb-20 px-6 text-center">
          <div className="space-y-3 mb-8">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-white tracking-tight">
              The World Tree
            </h1>
            <p className="text-sm text-white/30 tracking-[0.25em] uppercase">
              Distributed Systems Framework
            </p>
          </div>

          <Link
            href="/bot"
            className="group inline-flex items-center gap-2 px-7 py-2.5 rounded-full text-sm font-semibold text-white transition-all hover:scale-105 mb-8"
            style={{
              background: "linear-gradient(135deg, #f26b3a, #dc2626)",
              boxShadow: "0 0 30px rgba(242,107,58,0.35)",
            }}
          >
            Enter Network
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" className="group-hover:translate-x-0.5 transition-transform">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>

          {/* Service Links */}
          <div className="flex flex-wrap justify-center gap-1.5">
            {SERVICES.map((s) => (
              <Link
                key={s.id}
                href={s.href}
                onClick={(e) => s.comingSoon && e.preventDefault()}
                className={`px-3 py-1.5 rounded-full text-xs transition-all ${
                  s.comingSoon
                    ? "bg-white/[0.02] border border-white/[0.04] text-white/20 cursor-default"
                    : "bg-primary/10 border border-primary/20 text-white/70 hover:text-white hover:border-primary/40"
                }`}
              >
                {s.name}
                {s.comingSoon && <span className="ml-1 text-[10px] opacity-40">Soon</span>}
              </Link>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
