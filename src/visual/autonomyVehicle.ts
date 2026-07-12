import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

export type AutonomyVehicleKind = 'ego' | 'sedan' | 'suv' | 'van' | 'truck' | 'bus';
export type AutonomyVehicleStyle = 'ego' | 'traffic' | 'service';

type VehicleDimensions = {
  width: number;
  height: number;
  length: number;
};

const COLORS = {
  body: 0xfafcff,
  bodyTraffic: 0xf1f5f9,
  bodyService: 0xe5ebf1,
  glass: 0x101820,
  tire: 0x11151b,
  wheelCap: 0xd8dee6,
  trim: 0x1f2933,
  cyan: 0x18c7ff,
  blue: 0x507cff,
  red: 0xff3c5c,
  amber: 0xffb341,
  shadow: 0x6b7280,
};

export function createAutonomyVehicle(kind: AutonomyVehicleKind, style: AutonomyVehicleStyle): THREE.Group {
  const dims = dimensionsFor(kind);
  const group = new THREE.Group();
  group.name = `${style}_${kind}_autonomy_vehicle`;

  const bodyMat = createBodyMaterial(style);
  const glassMat = new THREE.MeshPhysicalMaterial({
    color: COLORS.glass,
    roughness: 0.18,
    metalness: 0.2,
    clearcoat: 0.6,
    clearcoatRoughness: 0.18,
  });
  const trimMat = new THREE.MeshStandardMaterial({ color: COLORS.trim, roughness: 0.46, metalness: 0.18 });
  const tireMat = new THREE.MeshStandardMaterial({ color: COLORS.tire, roughness: 0.62, metalness: 0.04 });
  const wheelCapMat = new THREE.MeshStandardMaterial({ color: COLORS.wheelCap, roughness: 0.38, metalness: 0.14 });

  addFakeShadow(group, dims);
  addBody(group, dims, kind, bodyMat, trimMat);
  addGlass(group, dims, kind, glassMat);
  addWheels(group, dims, tireMat, wheelCapMat);
  addLightSignature(group, dims, style);
  addPerceptionSignature(group, dims, style);

  return group;
}

function dimensionsFor(kind: AutonomyVehicleKind): VehicleDimensions {
  if (kind === 'bus') return { width: 3.05, height: 2.45, length: 8.6 };
  if (kind === 'truck') return { width: 3, height: 2.35, length: 7.25 };
  if (kind === 'van') return { width: 2.68, height: 1.95, length: 5.85 };
  if (kind === 'suv') return { width: 2.62, height: 1.66, length: 5.22 };
  if (kind === 'ego') return { width: 2.58, height: 1.46, length: 5.34 };
  return { width: 2.38, height: 1.38, length: 4.95 };
}

function createBodyMaterial(style: AutonomyVehicleStyle): THREE.MeshPhysicalMaterial {
  const color = style === 'service' ? COLORS.bodyService : style === 'traffic' ? COLORS.bodyTraffic : COLORS.body;
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness: style === 'ego' ? 0.28 : 0.36,
    metalness: 0.08,
    clearcoat: style === 'ego' ? 0.72 : 0.42,
    clearcoatRoughness: 0.22,
    transparent: style === 'traffic',
    opacity: style === 'traffic' ? 0.78 : 1,
  });
}

function addBody(
  group: THREE.Group,
  dims: VehicleDimensions,
  kind: AutonomyVehicleKind,
  bodyMat: THREE.Material,
  trimMat: THREE.Material,
): void {
  const isLong = kind === 'bus' || kind === 'truck' || kind === 'van';
  const lowerHeight = dims.height * (isLong ? 0.48 : 0.5);
  const body = new THREE.Mesh(
    new RoundedBoxGeometry(dims.width, lowerHeight, dims.length, 6, isLong ? 0.1 : 0.18),
    bodyMat,
  );
  body.position.y = lowerHeight / 2 + 0.2;
  body.castShadow = false;
  body.receiveShadow = false;
  group.add(body);

  const nose = new THREE.Mesh(
    new RoundedBoxGeometry(dims.width * 0.9, lowerHeight * 0.42, dims.length * 0.18, 4, 0.08),
    bodyMat,
  );
  nose.position.set(0, lowerHeight + 0.18, dims.length * 0.38);
  group.add(nose);

  const rocker = new THREE.Mesh(
    new RoundedBoxGeometry(dims.width * 0.96, 0.12, dims.length * 0.92, 3, 0.04),
    trimMat,
  );
  rocker.position.set(0, 0.23, 0);
  group.add(rocker);

  const frontIntake = new THREE.Mesh(new THREE.BoxGeometry(dims.width * 0.55, 0.06, 0.07), trimMat);
  frontIntake.position.set(0, 0.54, dims.length / 2 + 0.035);
  group.add(frontIntake);
}

function addGlass(group: THREE.Group, dims: VehicleDimensions, kind: AutonomyVehicleKind, glassMat: THREE.Material): void {
  const isLong = kind === 'bus' || kind === 'truck' || kind === 'van';
  const cabinLength = dims.length * (isLong ? 0.62 : 0.54);
  const cabinHeight = dims.height * (isLong ? 0.34 : 0.28);
  const cabin = new THREE.Mesh(
    new RoundedBoxGeometry(dims.width * 0.72, cabinHeight, cabinLength, 6, 0.12),
    glassMat,
  );
  cabin.position.set(0, dims.height * 0.68 + 0.22, isLong ? -dims.length * 0.03 : -dims.length * 0.08);
  group.add(cabin);

  const roofGlass = new THREE.Mesh(
    new RoundedBoxGeometry(dims.width * 0.62, 0.07, cabinLength * 0.78, 5, 0.08),
    glassMat,
  );
  roofGlass.position.set(0, dims.height + 0.25, cabin.position.z - cabinLength * 0.03);
  group.add(roofGlass);

  if (!isLong) {
    for (const side of [-1, 1]) {
      const sideGlass = new THREE.Mesh(new THREE.BoxGeometry(0.035, cabinHeight * 0.72, cabinLength * 0.72), glassMat);
      sideGlass.position.set(side * dims.width * 0.37, cabin.position.y, cabin.position.z);
      group.add(sideGlass);
    }
  }
}

function addWheels(
  group: THREE.Group,
  dims: VehicleDimensions,
  tireMat: THREE.Material,
  wheelCapMat: THREE.Material,
): void {
  const radius = Math.min(0.48, dims.height * 0.27);
  const axleZ = dims.length * 0.34;
  const wheelX = dims.width / 2 + 0.035;
  for (const x of [-wheelX, wheelX]) {
    for (const z of [-axleZ, axleZ]) {
      const tire = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.24, 28), tireMat);
      tire.rotation.z = Math.PI / 2;
      tire.position.set(x, radius + 0.13, z);
      group.add(tire);

      const cap = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.72, radius * 0.72, 0.035, 28), wheelCapMat);
      cap.rotation.z = Math.PI / 2;
      cap.position.set(x + Math.sign(x) * 0.13, radius + 0.13, z);
      group.add(cap);
    }
  }
}

function addLightSignature(group: THREE.Group, dims: VehicleDimensions, style: AutonomyVehicleStyle): void {
  const headlightMat = new THREE.MeshBasicMaterial({
    color: style === 'ego' ? COLORS.cyan : COLORS.body,
    transparent: true,
    opacity: style === 'ego' ? 0.9 : 0.58,
  });
  const tailMat = new THREE.MeshBasicMaterial({ color: COLORS.red, transparent: true, opacity: 0.7 });
  for (const x of [-dims.width * 0.26, dims.width * 0.26]) {
    const headlight = new THREE.Mesh(new THREE.BoxGeometry(dims.width * 0.22, 0.045, 0.055), headlightMat);
    headlight.position.set(x, dims.height * 0.46, dims.length / 2 + 0.04);
    group.add(headlight);
  }
  const rear = new THREE.Mesh(new THREE.BoxGeometry(dims.width * 0.68, 0.045, 0.055), tailMat);
  rear.position.set(0, dims.height * 0.54, -dims.length / 2 - 0.04);
  group.add(rear);

  if (style === 'ego') {
    const lidar = new THREE.Mesh(
      new RoundedBoxGeometry(dims.width * 0.34, 0.09, 0.42, 4, 0.045),
      new THREE.MeshBasicMaterial({ color: COLORS.cyan, transparent: true, opacity: 0.9 }),
    );
    lidar.position.set(0, dims.height + 0.38, -dims.length * 0.04);
    group.add(lidar);
  }
}

function addPerceptionSignature(group: THREE.Group, dims: VehicleDimensions, style: AutonomyVehicleStyle): void {
  const color = style === 'ego' ? COLORS.cyan : style === 'service' ? COLORS.amber : COLORS.blue;
  const ringOpacity = style === 'ego' ? 0.24 : 0.12;
  const boxOpacity = style === 'ego' ? 0.58 : 0.32;

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(dims.length * 0.56, dims.length * 0.61, 80),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: ringOpacity, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.08;
  group.add(ring);

  const box = createGroundBox(dims.width + 0.72, dims.length + 0.72, color, boxOpacity);
  box.position.y = 0.1;
  group.add(box);

  if (style === 'ego') {
    for (const radius of [dims.length * 0.86, dims.length * 1.18]) {
      const outerRing = new THREE.Mesh(
        new THREE.RingGeometry(radius, radius + 0.035, 96),
        new THREE.MeshBasicMaterial({ color: COLORS.cyan, transparent: true, opacity: 0.14, side: THREE.DoubleSide }),
      );
      outerRing.rotation.x = -Math.PI / 2;
      outerRing.position.y = 0.07;
      group.add(outerRing);
    }
  }
}

function createGroundBox(width: number, length: number, color: number, opacity: number): THREE.Group {
  const group = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
  const thickness = 0.055;
  const longGeo = new THREE.BoxGeometry(thickness, 0.035, length);
  const shortGeo = new THREE.BoxGeometry(width, 0.035, thickness);
  for (const x of [-width / 2, width / 2]) {
    const edge = new THREE.Mesh(longGeo, mat);
    edge.position.x = x;
    group.add(edge);
  }
  for (const z of [-length / 2, length / 2]) {
    const edge = new THREE.Mesh(shortGeo, mat);
    edge.position.z = z;
    group.add(edge);
  }
  return group;
}

function addFakeShadow(group: THREE.Group, dims: VehicleDimensions): void {
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1, 64),
    new THREE.MeshBasicMaterial({ color: COLORS.shadow, transparent: true, opacity: 0.16, depthWrite: false }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.scale.set(dims.width * 0.74, dims.length * 0.52, 1);
  shadow.position.y = 0.025;
  group.add(shadow);
}
