"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface BotNode {
  id: string;
  label: string;
  lat: number;
  lng: number;
  status: "online" | "offline" | "pending";
  version?: string;
  uptime?: number;
}

export interface ArcDef {
  fromId: string;
  toId: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const GLOBE_RADIUS = 2;
const DOT_DENSITY = 0.03;
const DEG2RAD = Math.PI / 180;

// ─── Helpers ─────────────────────────────────────────────────────────────────

export function latLngToVector3(lat: number, lng: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = (lng + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}

function getTimezoneRotation(): number {
  const offset = new Date().getTimezoneOffset() || 0;
  return Math.PI * (offset / (60 * 12));
}

// Rough land mask — continent shapes without texture
function isLand(lat: number, lng: number): boolean {
  if (lat > 25 && lat < 70 && lng > -170 && lng < -50) {
    if (lat > 50 && lng < -140) return true;
    if (lat > 25 && lat < 50 && lng > -130 && lng < -65) return true;
    if (lat > 50 && lng > -140 && lng < -55) return true;
    return false;
  }
  if (lat > -55 && lat < 15 && lng > -80 && lng < -35) return true;
  if (lat > 35 && lat < 70 && lng > -10 && lng < 40) return true;
  if (lat > -35 && lat < 37 && lng > -20 && lng < 52) return true;
  if (lat > 5 && lat < 75 && lng > 40 && lng < 180) return true;
  if (lat > 5 && lat < 45 && lng > 65 && lng < 145) return true;
  if (lat > -45 && lat < -10 && lng > 110 && lng < 155) return true;
  return false;
}

// ─── Earth Dots (GitHub-style) ───────────────────────────────────────────────

function EarthDots({ rotation }: { rotation: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const { matrices, count } = useMemo(() => {
    const matrices: THREE.Matrix4[] = [];
    const rows = Math.floor(180 / (DOT_DENSITY * 100));
    for (let lat = -90; lat <= 90; lat += 180 / rows) {
      const r = Math.cos(Math.abs(lat) * DEG2RAD) * GLOBE_RADIUS;
      const dotsForLat = Math.max(1, Math.floor((r * Math.PI * 2) / (DOT_DENSITY * 3)));
      for (let x = 0; x < dotsForLat; x++) {
        const lng = -180 + (x * 360) / dotsForLat;
        if (!isLand(lat, lng)) continue;
        const pos = latLngToVector3(lat, lng, GLOBE_RADIUS);
        const q = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          pos.clone().normalize()
        );
        matrices.push(new THREE.Matrix4().compose(pos, q, new THREE.Vector3(1, 1, 1)));
      }
    }
    return { matrices, count: matrices.length };
  }, []);

  useEffect(() => {
    if (!meshRef.current) return;
    matrices.forEach((m, i) => meshRef.current!.setMatrixAt(i, m));
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [matrices]);

  useFrame(() => {
    if (meshRef.current) meshRef.current.rotation.y = rotation;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <circleGeometry args={[0.018, 6]} />
      <meshBasicMaterial color="#2a2a3a" transparent opacity={0.85} side={THREE.DoubleSide} />
    </instancedMesh>
  );
}

// ─── Halo (custom GLSL shader — GitHub technique) ────────────────────────────

function Halo() {
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { color: { value: new THREE.Color("#f26b3a") } },
        vertexShader: `
          varying vec3 vNormal;
          void main() {
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 color;
          varying vec3 vNormal;
          void main() {
            float i = pow(0.65 - dot(vNormal, vec3(0,0,1)), 2.0);
            gl_FragColor = vec4(color, i * 0.45);
          }
        `,
        side: THREE.BackSide,
        transparent: true,
        blending: THREE.AdditiveBlending,
      }),
    []
  );
  return (
    <mesh material={mat}>
      <sphereGeometry args={[GLOBE_RADIUS * 1.15, 64, 64]} />
    </mesh>
  );
}

// ─── Globe Core ───────────────────────────────────────────────────────────────

function GlobeCore({ rotation }: { rotation: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => { if (ref.current) ref.current.rotation.y = rotation; });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[GLOBE_RADIUS * 0.995, 64, 64]} />
      <meshStandardMaterial color="#070710" metalness={0.1} roughness={0.9} />
    </mesh>
  );
}

// ─── Node Spikes ─────────────────────────────────────────────────────────────

function NodeSpikes({
  nodes,
  rotation,
  selectedId,
  onSelect,
}: {
  nodes: BotNode[];
  rotation: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const [t, setT] = useState(0);

  useFrame((_, delta) => {
    setT((v) => v + delta);
    if (groupRef.current) groupRef.current.rotation.y = rotation;
  });

  const statusColor = (s: BotNode["status"]) =>
    s === "online" ? "#4ade80" : s === "pending" ? "#fbbf24" : "#ef4444";

  return (
    <group ref={groupRef}>
      {nodes.map((node, i) => {
        const pos = latLngToVector3(node.lat, node.lng, GLOBE_RADIUS);
        const normal = pos.clone().normalize();
        const selected = selectedId === node.id;
        const h = (selected ? 0.22 : 0.14) + 0.04 * Math.sin(t * 2 + i);
        const endPos = pos.clone().add(normal.clone().multiplyScalar(h));
        const col = statusColor(node.status);

        return (
          <group key={node.id}>
            <line>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  count={2}
                  array={new Float32Array([...pos.toArray(), ...endPos.toArray()])}
                  itemSize={3}
                />
              </bufferGeometry>
              <lineBasicMaterial color={col} transparent opacity={selected ? 1 : 0.7} />
            </line>
            <mesh position={endPos} onClick={() => onSelect(node.id)}>
              <sphereGeometry args={[selected ? 0.04 : 0.028, 8, 8]} />
              <meshBasicMaterial color={col} />
            </mesh>
            <mesh position={endPos} rotation={[Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.025, 0.045 + 0.015 * Math.sin(t * 3 + i), 16]} />
              <meshBasicMaterial
                color={col}
                transparent
                opacity={(selected ? 0.5 : 0.25) + 0.15 * Math.sin(t * 3 + i)}
                side={THREE.DoubleSide}
              />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

// ─── Arc Connections ──────────────────────────────────────────────────────────

function ArcConnections({ nodes, arcs, rotation }: { nodes: BotNode[]; arcs: ArcDef[]; rotation: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const [t, setT] = useState(0);

  useFrame((_, delta) => {
    setT((v) => v + delta);
    if (groupRef.current) groupRef.current.rotation.y = rotation;
  });

  const arcGeometries = useMemo(() => {
    return arcs.map(({ fromId, toId }, idx) => {
      const from = nodes.find((n) => n.id === fromId);
      const to = nodes.find((n) => n.id === toId);
      if (!from || !to) return null;
      const start = latLngToVector3(from.lat, from.lng, GLOBE_RADIUS);
      const end = latLngToVector3(to.lat, to.lng, GLOBE_RADIUS);
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      mid.normalize().multiplyScalar(GLOBE_RADIUS + start.distanceTo(end) * 0.22);
      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      return { geometry: new THREE.TubeGeometry(curve, 44, 0.007, 6, false), idx };
    });
  }, [nodes, arcs]);

  return (
    <group ref={groupRef}>
      {arcGeometries.map((arc) => {
        if (!arc) return null;
        const phase = (t * 0.45 + arc.idx * 0.28) % 3;
        const opacity = (phase < 1 ? phase : phase < 2 ? 1 : 3 - phase) * 0.55;
        return (
          <mesh key={arc.idx} geometry={arc.geometry}>
            <meshBasicMaterial color="#f26b3a" transparent opacity={opacity} />
          </mesh>
        );
      })}
    </group>
  );
}

// ─── Landing Pulses ───────────────────────────────────────────────────────────

function LandingPulses({ nodes, rotation }: { nodes: BotNode[]; rotation: number }) {
  const groupRef = useRef<THREE.Group>(null);
  const [effects, setEffects] = useState<{ pos: THREE.Vector3; s: number; o: number; id: number }[]>([]);
  const nextId = useRef(0);

  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y = rotation;
    if (Math.random() < 0.025) {
      const node = nodes[Math.floor(Math.random() * nodes.length)];
      const pos = latLngToVector3(node.lat, node.lng, GLOBE_RADIUS + 0.01);
      setEffects((p) => [...p, { pos, s: 0, o: 1, id: nextId.current++ }]);
    }
    setEffects((p) =>
      p.map((e) => ({ ...e, s: e.s + (1 - e.s) * 0.055, o: 1 - e.s })).filter((e) => e.o > 0.02)
    );
  });

  return (
    <group ref={groupRef}>
      {effects.map((e) => (
        <mesh key={e.id} position={e.pos}>
          <ringGeometry args={[0.03 * e.s, 0.055 * e.s, 16]} />
          <meshBasicMaterial color="#f26b3a" transparent opacity={e.o * 0.45} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}

// ─── Stars ────────────────────────────────────────────────────────────────────

function Stars() {
  const ref = useRef<THREE.Points>(null);

  const geo = useMemo(() => {
    const pos = new Float32Array(2400 * 3);
    for (let i = 0; i < 2400; i++) {
      const r = 22 + Math.random() * 28;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(p) * Math.cos(t);
      pos[i * 3 + 1] = r * Math.sin(p) * Math.sin(t);
      pos[i * 3 + 2] = r * Math.cos(p);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return g;
  }, []);

  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.004;
  });

  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial color="#ffffff" size={0.022} transparent opacity={0.45} sizeAttenuation />
    </points>
  );
}

// ─── Camera fit helper ────────────────────────────────────────────────────────

function CameraSetup({ isMobile }: { isMobile: boolean }) {
  const { camera } = useThree();
  useEffect(() => {
    (camera as THREE.PerspectiveCamera).position.set(0, 0, isMobile ? 6.2 : 5.2);
  }, [camera, isMobile]);
  return null;
}

// ─── Scene ────────────────────────────────────────────────────────────────────

function GlobeScene({
  nodes,
  arcs,
  selectedId,
  onSelect,
  isMobile,
}: {
  nodes: BotNode[];
  arcs: ArcDef[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  isMobile: boolean;
}) {
  const [rotation, setRotation] = useState(() => getTimezoneRotation());
  useFrame((_, delta) => setRotation((r) => r + delta * 0.055));

  return (
    <>
      <CameraSetup isMobile={isMobile} />
      <ambientLight intensity={0.1} />
      <directionalLight position={[5, 3, 5]} intensity={0.35} color="#ffffff" />
      <directionalLight position={[-5, 3, 5]} intensity={0.2} color="#f26b3a" />
      <directionalLight position={[0, -5, 0]} intensity={0.1} color="#4a88c4" />
      <pointLight position={[0, 0, 5]} intensity={0.18} />

      <Stars />
      <Halo />
      <GlobeCore rotation={rotation} />
      <EarthDots rotation={rotation} />
      <NodeSpikes nodes={nodes} rotation={rotation} selectedId={selectedId} onSelect={onSelect} />
      <ArcConnections nodes={nodes} arcs={arcs} rotation={rotation} />
      <LandingPulses nodes={nodes} rotation={rotation} />

      <OrbitControls
        enablePan={false}
        minDistance={3.5}
        maxDistance={9}
        enableDamping
        dampingFactor={0.06}
        rotateSpeed={0.45}
        zoomSpeed={0.7}
      />
    </>
  );
}

// ─── Exported WorldGlobe component ───────────────────────────────────────────

export function WorldGlobe({
  nodes,
  arcs,
  selectedId,
  onSelect,
  className = "",
}: {
  nodes: BotNode[];
  arcs: ArcDef[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  className?: string;
}) {
  const [mounted, setMounted] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    setMounted(true);
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  if (!mounted) return <div className={`bg-[#030306] ${className}`} />;

  return (
    <Canvas
      className={className}
      camera={{ position: [0, 0, 5.2], fov: 45 }}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      dpr={[1, 1.5]}
      style={{ background: "transparent" }}
    >
      <GlobeScene
        nodes={nodes}
        arcs={arcs}
        selectedId={selectedId}
        onSelect={onSelect}
        isMobile={isMobile}
      />
    </Canvas>
  );
}
