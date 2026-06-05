"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import type { TopologyNode, NodeBackend } from "@/lib/types";

// ── Nordic palette (mirrors globals.css --frost/--emerald/--amber/--rose) ──
const COLOR_FROST = "#67e8f9";
const COLOR_EMERALD = "#34d399";
const COLOR_AMBER = "#fbbf24";
const COLOR_ROSE = "#f43f5e";

// Brain layout radii — peers orbit at ~3 units, interneurons fill 1.5 — 4
const PEER_RADIUS = 3;
const INTERNEURON_INNER = 1.5;
const INTERNEURON_OUTER = 3.8;
const INTERNEURON_COUNT = 40;
const MAX_TOTAL_CELLS = 100;
const MAX_AXONS = 150;

// CPU-load coloring; matches the SVG loadColor() in the old page
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
  node?: TopologyNode;
}

interface Axon {
  from: number; // cell index
  to: number;
  curve: THREE.CatmullRomCurve3;
  color: string;
  pulseOffset: number; // seconds, staggers pulses across axons
  pulseDuration: number; // seconds per traversal
}

// ── Build the 3D scene from topology data ───────────────────────────────
function buildBrainScene(nodes: TopologyNode[]): { cells: Cell[]; axons: Axon[] } {
  // Deterministic-ish randomness for stability across re-renders of the same
  // node set. Seed from a hash of joined node ids so the brain doesn't jitter
  // every poll, but new topology shapes still rebuild.
  const seed = nodes.reduce((h, n) => (h * 31 + n.node_id.charCodeAt(0)) | 0, 7) + nodes.length;
  let rngState = Math.abs(seed) || 1;
  const rand = () => {
    // mulberry32
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

  // Self at origin — anchor of the cluster
  if (selfNode) {
    const activation = Math.min(1, (selfNode.cpu_percent + selfNode.active_runs * 25) / 100);
    cells.push({
      id: selfNode.node_id,
      position: new THREE.Vector3(0, 0, 0),
      radius: 0.25,
      color: COLOR_FROST,
      type: "self",
      activation,
      node: selfNode,
    });
  } else {
    // Placeholder soma so the scene still has a center
    cells.push({
      id: "self-placeholder",
      position: new THREE.Vector3(0, 0, 0),
      radius: 0.22,
      color: COLOR_FROST,
      type: "self",
      activation: 0.4,
    });
  }

  // Peers evenly spaced on a sphere of radius PEER_RADIUS using a fibonacci
  // distribution so they look balanced even at small N.
  const peerCount = Math.min(peers.length, MAX_TOTAL_CELLS - INTERNEURON_COUNT - 1);
  const phi = Math.PI * (3 - Math.sqrt(5)); // golden angle
  for (let i = 0; i < peerCount; i++) {
    const peer = peers[i];
    const y = 1 - (i / Math.max(1, peerCount - 1)) * 2; // -1 .. 1
    const r = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;
    const activation = Math.min(1, (peer.cpu_percent + peer.active_runs * 20) / 100);
    cells.push({
      id: peer.node_id,
      position: new THREE.Vector3(x * PEER_RADIUS, y * PEER_RADIUS, z * PEER_RADIUS),
      radius: 0.15 + Math.min(0.05, peer.active_runs * 0.015),
      color: loadColorHex(peer.cpu_percent),
      type: "peer",
      activation,
      node: peer,
    });
  }

  // Synthetic interneurons — fills the brain with dim density
  const remaining = Math.min(INTERNEURON_COUNT, MAX_TOTAL_CELLS - cells.length);
  for (let i = 0; i < remaining; i++) {
    // Random point in spherical shell between INTERNEURON_INNER and OUTER
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
      color: COLOR_FROST,
      type: "interneuron",
      activation: 0.1 + rand() * 0.25,
    });
  }

  // ── Build axons ──────────────────────────────────────────────────────
  const axons: Axon[] = [];
  const selfIdx = 0;
  const interneuronStart = 1 + peerCount;

  // Every peer connects to self with a curved axon
  for (let i = 1; i <= peerCount; i++) {
    const fromCell = cells[i];
    const toCell = cells[selfIdx];
    const curve = makeOrganicCurve(fromCell.position, toCell.position, rand);
    axons.push({
      from: i,
      to: selfIdx,
      curve,
      color: fromCell.color,
      pulseOffset: rand() * 3,
      pulseDuration: 2 + rand() * 2,
    });
  }

  // Each peer wires to ~3 nearby interneurons — sprinkled signal paths
  for (let i = 1; i <= peerCount && axons.length < MAX_AXONS; i++) {
    const peerCell = cells[i];
    const dists: Array<{ idx: number; d: number }> = [];
    for (let j = interneuronStart; j < cells.length; j++) {
      dists.push({ idx: j, d: peerCell.position.distanceTo(cells[j].position) });
    }
    dists.sort((a, b) => a.d - b.d);
    const connectN = 3;
    for (let k = 0; k < Math.min(connectN, dists.length) && axons.length < MAX_AXONS; k++) {
      const targetIdx = dists[k].idx;
      const curve = makeOrganicCurve(peerCell.position, cells[targetIdx].position, rand);
      axons.push({
        from: i,
        to: targetIdx,
        curve,
        color: peerCell.color,
        pulseOffset: rand() * 4,
        pulseDuration: 1.8 + rand() * 2.5,
      });
    }
  }

  // Some interneurons cross-link to self so dim cells participate in pulsing
  const crossLinks = Math.min(12, cells.length - interneuronStart, MAX_AXONS - axons.length);
  for (let k = 0; k < crossLinks; k++) {
    const idx = interneuronStart + Math.floor(rand() * (cells.length - interneuronStart));
    const curve = makeOrganicCurve(cells[idx].position, cells[selfIdx].position, rand);
    axons.push({
      from: idx,
      to: selfIdx,
      curve,
      color: COLOR_FROST,
      pulseOffset: rand() * 4,
      pulseDuration: 2.5 + rand() * 2,
    });
  }

  return { cells, axons };
}

// Build a CatmullRomCurve3 with mid control points offset perpendicular to
// the chord — turns straight lines into organic dendrite curves.
function makeOrganicCurve(
  a: THREE.Vector3,
  b: THREE.Vector3,
  rand: () => number,
): THREE.CatmullRomCurve3 {
  const mid = a.clone().add(b).multiplyScalar(0.5);
  const chord = b.clone().sub(a);
  // Build an arbitrary perpendicular vector
  const upish = Math.abs(chord.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const perp1 = chord.clone().cross(upish).normalize();
  const perp2 = chord.clone().cross(perp1).normalize();
  const offsetMag = chord.length() * 0.18;
  const offset = perp1
    .multiplyScalar((rand() - 0.5) * offsetMag)
    .add(perp2.multiplyScalar((rand() - 0.5) * offsetMag));
  const control = mid.add(offset);
  return new THREE.CatmullRomCurve3([a.clone(), control, b.clone()], false, "catmullrom", 0.5);
}

// ── Soma renderer — sphere + transparent halo, additive blended ─────────
function PulsingCell({
  cell,
  onClick,
}: {
  cell: Cell;
  onClick?: (node: TopologyNode) => void;
}) {
  const coreRef = useRef<THREE.Mesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);
  const tRef = useRef(Math.random() * Math.PI * 2);
  const [hovered, setHovered] = useState(false);

  useFrame((_, delta) => {
    tRef.current += delta;
    // Self & peers breathe; interneurons stay dim & still to keep cost down
    if (cell.type === "interneuron") return;
    const pulse = 1 + 0.12 * Math.sin(tRef.current * (1.5 + cell.activation));
    if (coreRef.current) coreRef.current.scale.setScalar(pulse);
    if (haloRef.current) {
      const haloPulse = 1 + 0.25 * Math.sin(tRef.current * (1 + cell.activation));
      haloRef.current.scale.setScalar(haloPulse);
    }
  });

  const interactive = cell.node !== undefined;
  // Halo opacity tuned so interneurons are nearly invisible, self is bright
  const haloOpacity =
    cell.type === "interneuron" ? 0.05 : 0.18 + cell.activation * 0.25;
  const coreOpacity =
    cell.type === "interneuron" ? 0.3 : 0.85 + cell.activation * 0.15;

  return (
    <group
      position={cell.position}
      onPointerOver={() => interactive && setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onClick={(e) => {
        if (!interactive || !cell.node || !onClick) return;
        e.stopPropagation();
        onClick(cell.node);
      }}
    >
      {/* Bright core soma */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[cell.radius, 16, 16]} />
        <meshBasicMaterial
          color={cell.color}
          transparent
          opacity={coreOpacity}
          toneMapped={false}
        />
      </mesh>
      {/* Soft additive halo — gives the glow */}
      <mesh ref={haloRef}>
        <sphereGeometry args={[cell.radius * 2.4, 16, 16]} />
        <meshBasicMaterial
          color={cell.color}
          transparent
          opacity={hovered ? haloOpacity * 2 : haloOpacity}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      {/* Self gets an extra outer ring for emphasis */}
      {cell.type === "self" && (
        <mesh>
          <sphereGeometry args={[cell.radius * 4, 24, 24]} />
          <meshBasicMaterial
            color={cell.color}
            transparent
            opacity={0.08}
            blending={THREE.AdditiveBlending}
            depthWrite={false}
            toneMapped={false}
          />
        </mesh>
      )}
    </group>
  );
}

// ── Axon — curved line + traveling pulse sphere ────────────────────────
// activityRef.current is in [0..1]; higher values speed up pulses and
// brighten the axon line, simulating a wave of activation rippling
// across the brain when the cluster is busy.
function AxonComponent({ axon, activityRef }: { axon: Axon; activityRef: { current: number } }) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const tRef = useRef(axon.pulseOffset);

  // Precompute sample points for the line (drei <Line> takes Vector3[])
  const points = useMemo(() => axon.curve.getPoints(24), [axon.curve]);

  useFrame((_, delta) => {
    const activity = activityRef.current;
    // Speed scales 1x..3x with activity; quiet cluster ~ slow drift,
    // busy cluster ~ fast traveling waves
    tRef.current += delta * (1 + activity * 2);
    if (!pulseRef.current) return;
    const phase = ((tRef.current % axon.pulseDuration) / axon.pulseDuration);
    const pt = axon.curve.getPoint(phase);
    pulseRef.current.position.copy(pt);
    const fade =
      phase < 0.08 ? phase / 0.08 :
      phase > 0.92 ? (1 - phase) / 0.08 : 1;
    const mat = pulseRef.current.material as THREE.MeshBasicMaterial;
    mat.opacity = (0.55 + 0.4 * activity) * fade;
    // Pulse radius scales too — visible wave-front
    pulseRef.current.scale.setScalar(1 + activity * 0.8);
  });

  return (
    <group>
      <Line
        points={points}
        color={axon.color}
        lineWidth={0.6}
        transparent
        opacity={0.22}
        toneMapped={false}
      />
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.04, 8, 8]} />
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

// ── Ambient drifting dust — wraps the whole scene in particles ─────────
function DustField() {
  const ref = useRef<THREE.Points>(null);

  const { positions, velocities } = useMemo(() => {
    const count = 220;
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Shell from radius 4..6 — outside the brain proper so it feels like atmosphere
      const r = 4 + Math.random() * 2.5;
      const theta = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(p) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(p) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(p);
      vel[i * 3] = (Math.random() - 0.5) * 0.0015;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.0015;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.0015;
    }
    return { positions: pos, velocities: vel };
  }, []);

  useFrame(() => {
    if (!ref.current) return;
    const posAttr = ref.current.geometry.getAttribute("position") as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    for (let i = 0; i < arr.length / 3; i++) {
      arr[i * 3] += velocities[i * 3];
      arr[i * 3 + 1] += velocities[i * 3 + 1];
      arr[i * 3 + 2] += velocities[i * 3 + 2];
      const x = arr[i * 3], y = arr[i * 3 + 1], z = arr[i * 3 + 2];
      const dist = Math.sqrt(x * x + y * y + z * z);
      if (dist > 8 || dist < 3.5) {
        velocities[i * 3] *= -1;
        velocities[i * 3 + 1] *= -1;
        velocities[i * 3 + 2] *= -1;
      }
    }
    posAttr.needsUpdate = true;
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
        size={0.025}
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
function BrainScene({
  cells,
  axons,
  onNodeClick,
  activityRef,
}: {
  cells: Cell[];
  axons: Axon[];
  onNodeClick?: (n: TopologyNode) => void;
  activityRef: { current: number };
}) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[0, 0, 5]} intensity={0.5} color={COLOR_FROST} />
      <pointLight position={[5, -3, -2]} intensity={0.25} color={COLOR_EMERALD} />

      <DustField />

      {axons.map((a, i) => (
        <AxonComponent key={`axon-${i}`} axon={a} activityRef={activityRef} />
      ))}
      {cells.map((c) => (
        <PulsingCell key={c.id} cell={c} onClick={onNodeClick} />
      ))}

      <OrbitControls
        autoRotate
        autoRotateSpeed={0.4}
        enableDamping
        dampingFactor={0.05}
        enablePan={false}
        minDistance={5}
        maxDistance={15}
      />
    </>
  );
}

// ── Exported component ─────────────────────────────────────────────────
interface BrainMeshProps {
  nodes: TopologyNode[];
  onNodeClick?: (node: TopologyNode) => void;
  className?: string;
}

export function BrainMesh({ nodes, onNodeClick, className = "" }: BrainMeshProps) {
  // SSR-safe mount (same pattern as Globe.tsx) — three.js can't render server-side
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // activityRef is read every frame inside useFrame and updated from the
  // /api/v2/backend/stream SSE; using a ref avoids re-rendering the whole
  // Canvas on every snapshot (1Hz).
  const activityRef = useRef(0);
  useEffect(() => {
    if (!mounted) return;
    const es = new EventSource("/api/v2/backend/stream");
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as NodeBackend;
        // Combine CPU% and active_runs into a 0..1 activity factor with a
        // bit of exponential easing so small loads still register visually.
        const cpu01 = data.cpu_percent / 100;
        const runs01 = Math.min(1, data.active_runs / Math.max(1, data.cpu_count));
        const target = Math.min(1, Math.max(cpu01, runs01 * 0.8));
        // EMA smoothing — prevents jitter every snapshot
        activityRef.current = activityRef.current * 0.6 + target * 0.4;
      } catch {
        /* ignore non-JSON keepalive */
      }
    };
    return () => es.close();
  }, [mounted]);

  // Rebuild scene when the set of node ids changes; cell positions are
  // deterministic from the id list, so 5s polls won't jitter the layout.
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
    return (
      <div
        className={className}
        style={{ width: "100%", height: "560px", background: "transparent" }}
      />
    );
  }

  if (nodes.length === 0) {
    return (
      <div
        className={`flex items-center justify-center text-muted/60 text-sm italic ${className}`}
        style={{ height: "560px" }}
      >
        No nodes in the mesh
      </div>
    );
  }

  return (
    <Canvas
      className={className}
      camera={{ position: [0, 0, 8], fov: 50 }}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      dpr={[1, 1.5]}
      style={{ width: "100%", height: "560px", background: "transparent" }}
    >
      <BrainScene cells={cells} axons={axons} onNodeClick={onNodeClick} activityRef={activityRef} />
    </Canvas>
  );
}
