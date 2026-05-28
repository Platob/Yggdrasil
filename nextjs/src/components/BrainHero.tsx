"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { getTopology } from "@/lib/api";
import type { TopologyNode } from "@/lib/types";

// ── Nordic palette — mirrors globals.css --frost/--emerald/--amber/--rose ──
const COLOR_FROST = "#67e8f9";
const COLOR_EMERALD = "#34d399";
const COLOR_AMBER = "#fbbf24";
const COLOR_ROSE = "#f43f5e";

// Hero scene scale — bigger than BrainMesh: more interneurons, deeper camera
const PEER_RADIUS = 3;
const INTERNEURON_INNER = 1.5;
const INTERNEURON_OUTER = 3.8;
const INTERNEURON_COUNT = 60; // richer "neural density" vs 40 in BrainMesh
const MAX_TOTAL_CELLS = 140;
const MAX_AXONS = 220;
const DUST_PARTICLES = 500; // ambient background dust

// CPU-load coloring — same mapping the SVG topology view uses
function loadColorHex(cpu: number): string {
  if (cpu >= 80) return COLOR_ROSE;
  if (cpu >= 50) return COLOR_AMBER;
  return COLOR_FROST;
}

// ── Cell & Axon types ────────────────────────────────────────────────────
interface Cell {
  id: string;
  position: THREE.Vector3;
  radius: number;
  color: string;
  type: "self" | "peer" | "interneuron";
  activation: number;
}

interface Axon {
  from: number;
  to: number;
  curve: THREE.CatmullRomCurve3;
  color: string;
  pulseOffset: number;
  pulseDuration: number;
}

// ── Build the 3D scene from topology data ───────────────────────────────
function buildBrainScene(nodes: TopologyNode[]): { cells: Cell[]; axons: Axon[] } {
  // Deterministic RNG seeded from node ids so the brain doesn't jitter between
  // 5s topology polls but rebuilds when the actual node set changes.
  const seed = nodes.reduce((h, n) => (h * 31 + n.node_id.charCodeAt(0)) | 0, 17) + nodes.length;
  let rngState = Math.abs(seed) || 1;
  const rand = () => {
    rngState |= 0;
    rngState = (rngState + 0x6d2b79f5) | 0;
    let t = rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const cells: Cell[] = [];
  const selfNode = nodes.find((n) => n.self);
  const peers = nodes.filter((n) => !n.self);

  // Central soma — slightly bigger than BrainMesh for hero impact
  if (selfNode) {
    const activation = Math.min(1, (selfNode.cpu_percent + selfNode.active_runs * 25) / 100);
    cells.push({
      id: selfNode.node_id,
      position: new THREE.Vector3(0, 0, 0),
      radius: 0.32,
      color: COLOR_FROST,
      type: "self",
      activation,
    });
  } else {
    cells.push({
      id: "self-placeholder",
      position: new THREE.Vector3(0, 0, 0),
      radius: 0.3,
      color: COLOR_FROST,
      type: "self",
      activation: 0.5,
    });
  }

  // Peers — fibonacci sphere distribution on shell of PEER_RADIUS
  const peerCount = Math.min(peers.length, MAX_TOTAL_CELLS - INTERNEURON_COUNT - 1);
  const phi = Math.PI * (3 - Math.sqrt(5)); // golden angle
  for (let i = 0; i < peerCount; i++) {
    const peer = peers[i];
    const y = 1 - (i / Math.max(1, peerCount - 1)) * 2;
    const r = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;
    const activation = Math.min(1, (peer.cpu_percent + peer.active_runs * 20) / 100);
    cells.push({
      id: peer.node_id,
      position: new THREE.Vector3(x * PEER_RADIUS, y * PEER_RADIUS, z * PEER_RADIUS),
      radius: 0.17 + Math.min(0.06, peer.active_runs * 0.015),
      color: loadColorHex(peer.cpu_percent),
      type: "peer",
      activation,
    });
  }

  // Synthetic interneurons — fills the brain with dim density (60 cells)
  const remaining = Math.min(INTERNEURON_COUNT, MAX_TOTAL_CELLS - cells.length);
  for (let i = 0; i < remaining; i++) {
    const u = rand();
    const v = rand();
    const cellTheta = 2 * Math.PI * u;
    const cellPhi = Math.acos(2 * v - 1);
    const radius = INTERNEURON_INNER + rand() * (INTERNEURON_OUTER - INTERNEURON_INNER);
    const x = radius * Math.sin(cellPhi) * Math.cos(cellTheta);
    const y = radius * Math.sin(cellPhi) * Math.sin(cellTheta);
    const z = radius * Math.cos(cellPhi);
    cells.push({
      id: `interneuron-${i}`,
      position: new THREE.Vector3(x, y, z),
      radius: 0.05 + rand() * 0.05,
      color: rand() > 0.85 ? COLOR_EMERALD : COLOR_FROST,
      type: "interneuron",
      activation: 0.1 + rand() * 0.3,
    });
  }

  // ── Axons ────────────────────────────────────────────────────────────
  const axons: Axon[] = [];
  const selfIdx = 0;
  const interneuronStart = 1 + peerCount;

  // Every peer → self (always visible, even with zero peers nothing happens)
  for (let i = 1; i <= peerCount; i++) {
    const fromCell = cells[i];
    const toCell = cells[selfIdx];
    const curve = makeOrganicCurve(fromCell.position, toCell.position, rand);
    axons.push({
      from: i,
      to: selfIdx,
      curve,
      color: fromCell.color,
      pulseOffset: rand() * 4,
      pulseDuration: 3 + rand() * 2, // slower than BrainMesh — ambient feel
    });
  }

  // Each peer → ~3 nearest interneurons — sprinkles signal paths through density
  for (let i = 1; i <= peerCount && axons.length < MAX_AXONS; i++) {
    const peerCell = cells[i];
    const dists: Array<{ idx: number; d: number }> = [];
    for (let j = interneuronStart; j < cells.length; j++) {
      dists.push({ idx: j, d: peerCell.position.distanceTo(cells[j].position) });
    }
    dists.sort((a, b) => a.d - b.d);
    for (let k = 0; k < Math.min(3, dists.length) && axons.length < MAX_AXONS; k++) {
      const targetIdx = dists[k].idx;
      const curve = makeOrganicCurve(peerCell.position, cells[targetIdx].position, rand);
      axons.push({
        from: i,
        to: targetIdx,
        curve,
        color: peerCell.color,
        pulseOffset: rand() * 4,
        pulseDuration: 3 + rand() * 2,
      });
    }
  }

  // Cross-links from interneurons → self so dim cells participate in pulsing.
  // Hero scene gets more of these (24 vs BrainMesh's 12) for richer activity.
  const crossLinks = Math.min(24, cells.length - interneuronStart, MAX_AXONS - axons.length);
  for (let k = 0; k < crossLinks; k++) {
    const idx = interneuronStart + Math.floor(rand() * (cells.length - interneuronStart));
    const curve = makeOrganicCurve(cells[idx].position, cells[selfIdx].position, rand);
    axons.push({
      from: idx,
      to: selfIdx,
      curve,
      color: COLOR_FROST,
      pulseOffset: rand() * 5,
      pulseDuration: 3.5 + rand() * 1.5,
    });
  }

  // Interneuron ↔ interneuron — extra ambient sparks unique to the hero scene
  const ambientLinks = Math.min(20, MAX_AXONS - axons.length);
  for (let k = 0; k < ambientLinks; k++) {
    const a = interneuronStart + Math.floor(rand() * (cells.length - interneuronStart));
    const b = interneuronStart + Math.floor(rand() * (cells.length - interneuronStart));
    if (a === b) continue;
    const curve = makeOrganicCurve(cells[a].position, cells[b].position, rand);
    axons.push({
      from: a,
      to: b,
      curve,
      color: COLOR_FROST,
      pulseOffset: rand() * 5,
      pulseDuration: 4 + rand() * 1.5,
    });
  }

  return { cells, axons };
}

// CatmullRomCurve3 with a perpendicular-offset midpoint → organic dendrite curl.
function makeOrganicCurve(
  a: THREE.Vector3,
  b: THREE.Vector3,
  rand: () => number,
): THREE.CatmullRomCurve3 {
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const chord = b.clone().sub(a);
  const upish = Math.abs(chord.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const perp1 = chord.clone().cross(upish).normalize();
  const perp2 = chord.clone().cross(perp1).normalize();
  const offsetMag = chord.length() * 0.2;
  const offset = perp1
    .multiplyScalar((rand() - 0.5) * offsetMag)
    .add(perp2.multiplyScalar((rand() - 0.5) * offsetMag));
  const control = mid.add(offset);
  return new THREE.CatmullRomCurve3([a.clone(), control, b.clone()], false, "catmullrom", 0.5);
}

// ── Soma renderer — purely decorative, no click handlers in hero scene ──
function PulsingCell({ cell }: { cell: Cell }) {
  const coreRef = useRef<THREE.Mesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);
  const ring1Ref = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const tRef = useRef(Math.random() * Math.PI * 2);

  useFrame((_, delta) => {
    tRef.current += delta;
    if (cell.type === "interneuron") return;
    const pulse = 1 + 0.12 * Math.sin(tRef.current * (1.5 + cell.activation));
    if (coreRef.current) coreRef.current.scale.setScalar(pulse);
    if (haloRef.current) {
      const haloPulse = 1 + 0.25 * Math.sin(tRef.current * (1 + cell.activation));
      haloRef.current.scale.setScalar(haloPulse);
    }
    // Multi-ring halo on self — staggered phases for breathing effect
    if (ring1Ref.current) {
      ring1Ref.current.scale.setScalar(1 + 0.2 * Math.sin(tRef.current * 0.8));
    }
    if (ring2Ref.current) {
      ring2Ref.current.scale.setScalar(1 + 0.3 * Math.sin(tRef.current * 0.6 + 1.2));
    }
  });

  const haloOpacity =
    cell.type === "interneuron" ? 0.05 : 0.2 + cell.activation * 0.25;
  const coreOpacity =
    cell.type === "interneuron" ? 0.3 : 0.85 + cell.activation * 0.15;

  return (
    <group position={cell.position}>
      <mesh ref={coreRef}>
        <sphereGeometry args={[cell.radius, 16, 16]} />
        <meshBasicMaterial
          color={cell.color}
          transparent
          opacity={coreOpacity}
          toneMapped={false}
        />
      </mesh>
      <mesh ref={haloRef}>
        <sphereGeometry args={[cell.radius * 2.4, 16, 16]} />
        <meshBasicMaterial
          color={cell.color}
          transparent
          opacity={haloOpacity}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      {/* Multi-ring halo on the central soma — only on self */}
      {cell.type === "self" && (
        <>
          <mesh ref={ring1Ref}>
            <sphereGeometry args={[cell.radius * 4, 24, 24]} />
            <meshBasicMaterial
              color={cell.color}
              transparent
              opacity={0.1}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh ref={ring2Ref}>
            <sphereGeometry args={[cell.radius * 6, 24, 24]} />
            <meshBasicMaterial
              color={cell.color}
              transparent
              opacity={0.05}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
          <mesh>
            <sphereGeometry args={[cell.radius * 8, 24, 24]} />
            <meshBasicMaterial
              color={COLOR_EMERALD}
              transparent
              opacity={0.025}
              blending={THREE.AdditiveBlending}
              depthWrite={false}
              toneMapped={false}
            />
          </mesh>
        </>
      )}
    </group>
  );
}

// ── Axon — curve line + traveling additive pulse along it ──────────────
function AxonComponent({ axon }: { axon: Axon }) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const tRef = useRef(axon.pulseOffset);

  const points = useMemo(() => axon.curve.getPoints(24), [axon.curve]);

  useFrame((_, delta) => {
    tRef.current += delta;
    if (!pulseRef.current) return;
    const phase = (tRef.current % axon.pulseDuration) / axon.pulseDuration;
    const pt = axon.curve.getPoint(phase);
    pulseRef.current.position.copy(pt);
    const fade =
      phase < 0.08
        ? phase / 0.08
        : phase > 0.92
        ? (1 - phase) / 0.08
        : 1;
    const mat = pulseRef.current.material as THREE.MeshBasicMaterial;
    mat.opacity = 0.7 * fade;
  });

  return (
    <group>
      <Line
        points={points}
        color={axon.color}
        lineWidth={0.6}
        transparent
        opacity={0.25}
        toneMapped={false}
      />
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.045, 8, 8]} />
        <meshBasicMaterial
          color={axon.color}
          transparent
          opacity={0.7}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
    </group>
  );
}

// ── Outer breathing ring — slow scale 1 → 1.05 → 1 ─────────────────────
function BreathingRing() {
  const ref = useRef<THREE.Mesh>(null);
  const tRef = useRef(0);

  useFrame((_, delta) => {
    tRef.current += delta;
    if (!ref.current) return;
    // Breathe scale 1.0 -> 1.05 -> 1.0 over ~6s
    const s = 1 + 0.025 * Math.sin(tRef.current * 1.0) + 0.025;
    ref.current.scale.setScalar(s);
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.opacity = 0.06 + 0.04 * Math.sin(tRef.current * 0.8);
  });

  return (
    <mesh ref={ref}>
      <sphereGeometry args={[4, 48, 48]} />
      <meshBasicMaterial
        color={COLOR_FROST}
        transparent
        opacity={0.08}
        wireframe
        blending={THREE.AdditiveBlending}
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  );
}

// ── Background dust field — ~500 ambient particles in a wide sphere ─────
function DustField() {
  const ref = useRef<THREE.Points>(null);

  const positions = useMemo(() => {
    const pos = new Float32Array(DUST_PARTICLES * 3);
    for (let i = 0; i < DUST_PARTICLES; i++) {
      // Random distribution in a large sphere — far behind the brain proper
      const r = 5 + Math.random() * 12;
      const theta = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(p) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(p) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(p);
    }
    return pos;
  }, []);

  // Slow rotation — feels like floating dust mites caught in light
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.015;
      ref.current.rotation.x += delta * 0.005;
    }
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={positions.length / 3}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        color={COLOR_FROST}
        size={0.02}
        transparent
        opacity={0.35}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// ── Scene contents ─────────────────────────────────────────────────────
function BrainHeroScene({ cells, axons }: { cells: Cell[]; axons: Axon[] }) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 0, 5]} intensity={0.6} color={COLOR_FROST} />
      <pointLight position={[5, -3, -2]} intensity={0.3} color={COLOR_EMERALD} />

      <DustField />
      <BreathingRing />

      {axons.map((a, i) => (
        <AxonComponent key={`axon-${i}`} axon={a} />
      ))}
      {cells.map((c) => (
        <PulsingCell key={c.id} cell={c} />
      ))}

      <OrbitControls
        autoRotate
        autoRotateSpeed={1.0}
        enableDamping
        dampingFactor={0.05}
        enablePan={false}
        minDistance={6}
        maxDistance={18}
      />
    </>
  );
}

// ── Exported hero component ────────────────────────────────────────────
interface BrainHeroProps {
  className?: string;
}

export function BrainHero({ className = "" }: BrainHeroProps) {
  // SSR-safe mount — three.js can't render server-side
  const [mounted, setMounted] = useState(false);
  const [nodes, setNodes] = useState<TopologyNode[]>([]);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch topology so the hero shows actual peer cells; refresh every 10s.
  // Errors are ignored — the brain still renders with just the central soma.
  useEffect(() => {
    let active = true;
    const tick = () => {
      getTopology()
        .then((t) => {
          if (active) setNodes(t.nodes);
        })
        .catch(() => {
          // Backend may be down; leave nodes empty for placeholder soma.
        });
    };
    tick();
    const id = setInterval(tick, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  // Rebuild scene only when the set of node ids changes (5/10s polls don't jitter)
  const sceneKey = useMemo(
    () => nodes.map((n) => n.node_id).sort().join(","),
    [nodes],
  );
  const { cells, axons } = useMemo(
    () => buildBrainScene(nodes),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sceneKey],
  );

  if (!mounted) {
    return <div className={`bg-[#050510] ${className}`} />;
  }

  return (
    <Canvas
      className={className}
      camera={{ position: [0, 0, 10], fov: 50 }}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      dpr={[1, 1.5]}
      style={{ background: "transparent" }}
    >
      <BrainHeroScene cells={cells} axons={axons} />
    </Canvas>
  );
}
