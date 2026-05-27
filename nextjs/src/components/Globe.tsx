"use client";

import { useRef, useMemo, useState, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

// ── Constants ────────────────────────────────────────────────
const GLOBE_RADIUS = 2;
const DEG2RAD = Math.PI / 180;
const DOT_DENSITY = 0.03;

// ── Helpers ──────────────────────────────────────────────────
function latLonToVec3(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = (lon + 180) * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

// Rough continent mask (same approach as existing globe)
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

// ── Wireframe Grid ───────────────────────────────────────────
function WireframeGrid({ rotation }: { rotation: number }) {
  const ref = useRef<THREE.LineSegments>(null);

  useFrame(() => {
    if (ref.current) ref.current.rotation.y = rotation;
  });

  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(GLOBE_RADIUS * 0.998, 36, 24);
    return new THREE.WireframeGeometry(geo);
  }, []);

  return (
    <lineSegments ref={ref} geometry={geometry}>
      <lineBasicMaterial color="#1a2a4a" transparent opacity={0.15} />
    </lineSegments>
  );
}

// ── Earth Dots (GitHub-style dot grid of continents) ─────────
function EarthDots({ rotation }: { rotation: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const { matrices, count } = useMemo(() => {
    const mats: THREE.Matrix4[] = [];
    const rows = Math.floor(180 / (DOT_DENSITY * 100));
    for (let lat = -90; lat <= 90; lat += 180 / rows) {
      const r = Math.cos(Math.abs(lat) * DEG2RAD) * GLOBE_RADIUS;
      const dotsForLat = Math.max(1, Math.floor((r * Math.PI * 2) / (DOT_DENSITY * 3)));
      for (let x = 0; x < dotsForLat; x++) {
        const lng = -180 + (x * 360) / dotsForLat;
        if (!isLand(lat, lng)) continue;
        const pos = latLonToVec3(lat, lng, GLOBE_RADIUS);
        const q = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 0, 1),
          pos.clone().normalize(),
        );
        mats.push(new THREE.Matrix4().compose(pos, q, new THREE.Vector3(1, 1, 1)));
      }
    }
    return { matrices: mats, count: mats.length };
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
      <circleGeometry args={[0.016, 6]} />
      <meshBasicMaterial color="#2a3a5a" transparent opacity={0.7} side={THREE.DoubleSide} />
    </instancedMesh>
  );
}

// ── Halo (atmospheric glow ring) ─────────────────────────────
function Halo() {
  const mat = useMemo(
    () =>
      new THREE.ShaderMaterial({
        uniforms: { color: { value: new THREE.Color("#67e8f9") } },
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
            float i = pow(0.6 - dot(vNormal, vec3(0, 0, 1)), 2.0);
            gl_FragColor = vec4(color, i * 0.35);
          }
        `,
        side: THREE.BackSide,
        transparent: true,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  return (
    <mesh material={mat}>
      <sphereGeometry args={[GLOBE_RADIUS * 1.12, 64, 64]} />
    </mesh>
  );
}

// ── Globe Core (dark sphere) ─────────────────────────────────
function GlobeCore({ rotation }: { rotation: number }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (ref.current) ref.current.rotation.y = rotation;
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[GLOBE_RADIUS * 0.995, 64, 64]} />
      <meshStandardMaterial color="#040812" metalness={0.15} roughness={0.85} />
    </mesh>
  );
}

// ── Local node marker (bright pulsing point) ─────────────────
function NodeMarker({
  lat,
  lon,
  rotation,
}: {
  lat: number;
  lon: number;
  rotation: number;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const [t, setT] = useState(0);

  useFrame((_, delta) => {
    setT((v) => v + delta);
    if (groupRef.current) groupRef.current.rotation.y = rotation;
  });

  const pos = useMemo(() => latLonToVec3(lat, lon, GLOBE_RADIUS), [lat, lon]);
  const normal = useMemo(() => pos.clone().normalize(), [pos]);
  const endPos = useMemo(() => pos.clone().add(normal.clone().multiplyScalar(0.05)), [pos, normal]);

  const pulseScale = 1 + 0.3 * Math.sin(t * 3);
  const ringScale = 0.04 + 0.02 * Math.sin(t * 2);

  return (
    <group ref={groupRef}>
      {/* Spike line */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([...pos.toArray(), ...endPos.toArray()]), 3]}
            count={2}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#67e8f9" />
      </line>

      {/* Bright core sphere */}
      <mesh position={endPos}>
        <sphereGeometry args={[0.045 * pulseScale, 12, 12]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>

      {/* Frost glow sphere */}
      <mesh position={endPos}>
        <sphereGeometry args={[0.08 * pulseScale, 12, 12]} />
        <meshBasicMaterial color="#67e8f9" transparent opacity={0.4} />
      </mesh>

      {/* Expanding ring pulse */}
      <mesh position={endPos} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[ringScale, ringScale + 0.02, 24]} />
        <meshBasicMaterial
          color="#67e8f9"
          transparent
          opacity={0.3 + 0.2 * Math.sin(t * 2)}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
}

// ── Floating particles ───────────────────────────────────────
function Particles() {
  const ref = useRef<THREE.Points>(null);

  const { positions, velocities } = useMemo(() => {
    const count = 200;
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      // Scatter particles in a shell around the globe
      const r = GLOBE_RADIUS * (1.3 + Math.random() * 1.5);
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
      // Slow drift velocity
      vel[i * 3] = (Math.random() - 0.5) * 0.002;
      vel[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
      vel[i * 3 + 2] = (Math.random() - 0.5) * 0.002;
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
      // If too far, nudge back
      const x = arr[i * 3], y = arr[i * 3 + 1], z = arr[i * 3 + 2];
      const dist = Math.sqrt(x * x + y * y + z * z);
      if (dist > GLOBE_RADIUS * 3.5 || dist < GLOBE_RADIUS * 1.2) {
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
        color="#67e8f9"
        size={0.02}
        transparent
        opacity={0.4}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// ── Background Stars ─────────────────────────────────────────
function Stars() {
  const ref = useRef<THREE.Points>(null);

  const geo = useMemo(() => {
    const pos = new Float32Array(1800 * 3);
    for (let i = 0; i < 1800; i++) {
      const r = 20 + Math.random() * 30;
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
    if (ref.current) ref.current.rotation.y += delta * 0.003;
  });

  return (
    <points ref={ref} geometry={geo}>
      <pointsMaterial color="#ffffff" size={0.018} transparent opacity={0.35} sizeAttenuation />
    </points>
  );
}

// ── Camera setup ─────────────────────────────────────────────
function CameraSetup() {
  const { camera } = useThree();
  useEffect(() => {
    (camera as THREE.PerspectiveCamera).position.set(0, 0.5, 5);
  }, [camera]);
  return null;
}

// ── Scene ────────────────────────────────────────────────────
function GlobeScene({ lat, lon }: { lat: number | null; lon: number | null }) {
  const [rotation, setRotation] = useState(0);
  useFrame((_, delta) => setRotation((r) => r + delta * 0.04));

  return (
    <>
      <CameraSetup />
      <ambientLight intensity={0.08} />
      <directionalLight position={[5, 3, 5]} intensity={0.3} color="#67e8f9" />
      <directionalLight position={[-5, -2, 3]} intensity={0.15} color="#22d3ee" />
      <pointLight position={[0, 0, 5]} intensity={0.12} color="#ffffff" />

      <Stars />
      <Particles />
      <Halo />
      <GlobeCore rotation={rotation} />
      <WireframeGrid rotation={rotation} />
      <EarthDots rotation={rotation} />

      {lat != null && lon != null && (
        <NodeMarker lat={lat} lon={lon} rotation={rotation} />
      )}

      <OrbitControls
        enablePan={false}
        minDistance={3.5}
        maxDistance={8}
        enableDamping
        dampingFactor={0.06}
        rotateSpeed={0.4}
        zoomSpeed={0.6}
        autoRotate={false}
      />
    </>
  );
}

// ── Exported Globe Component ─────────────────────────────────
interface GlobeProps {
  lat?: number | null;
  lon?: number | null;
  className?: string;
}

export function Globe({ lat = null, lon = null, className = "" }: GlobeProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className={`bg-[#050510] ${className}`} />;
  }

  return (
    <Canvas
      className={className}
      camera={{ position: [0, 0.5, 5], fov: 45 }}
      gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
      dpr={[1, 1.5]}
      style={{ background: "transparent" }}
    >
      <GlobeScene lat={lat} lon={lon} />
    </Canvas>
  );
}
