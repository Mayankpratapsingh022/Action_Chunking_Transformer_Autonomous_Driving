import * as THREE from 'three';
import type { AssetLibrary, VehicleAssetKey } from '../assets/assetLibrary';
import type { ActorState, RoadGraph, ScenarioConfig } from '../types';
import { SeededRng, wrapPi } from '../utils/rng';
import { BLOCK_CENTERS, MAP_EXTENT, nearestRouteIndex, ROAD_WIDTH } from '../world/roadGraph';

type ManagedActor = ActorState & {
  mesh: THREE.Group;
  path: THREE.Vector3[];
  targetIndex: number;
};

const densityCount = { low: 8, medium: 16, high: 28 } as const;
const ROUTE_CLEARANCE = 7.5;
const ROUTE_PATH_OVERLAP_LIMIT = 8;

export class TrafficManager {
  readonly actors: ManagedActor[] = [];
  readonly dynamicColliders: { box: THREE.Box3; actor: ActorState }[] = [];
  private readonly rng: SeededRng;
  private readonly tmpBox = new THREE.Box3();

  constructor(
    private readonly scene: THREE.Scene,
    private readonly config: ScenarioConfig,
    private readonly graph: RoadGraph,
    private readonly assets?: AssetLibrary,
  ) {
    this.rng = new SeededRng(config.trafficSeed ?? config.seed + 1009);
    if (config.kind !== 'curved_loop_drive') {
      this.spawnVehicles();
      this.spawnParkedVehicles();
    }
    this.spawnScenarioActors();
  }

  update(dt: number, elapsed: number, egoPosition?: THREE.Vector3, egoSpeed = 0): void {
    this.dynamicColliders.length = 0;
    for (const actor of this.actors) {
      if (actor.type === 'pedestrian' || actor.type === 'cyclist') {
        actor.position.x += Math.sin(actor.heading) * dt * actor.speed;
        if (Math.abs(actor.position.x) > 17) actor.speed = 0;
      } else {
        const yieldsToEgo = actor.id.startsWith('npc_')
          && egoPosition !== undefined
          && Math.abs(egoSpeed) > 1.2
          && actor.position.distanceTo(egoPosition) < 28;
        if (!yieldsToEgo) this.followPath(actor, dt);
      }
      actor.mesh.position.copy(actor.position);
      actor.mesh.rotation.y = actor.heading;
      this.tmpBox.setFromCenterAndSize(
        new THREE.Vector3(actor.position.x, 0.9, actor.position.z),
        this.colliderSizeFor(actor.type),
      );
      this.dynamicColliders.push({ box: this.tmpBox.clone(), actor });
    }
  }

  getActorStates(): ActorState[] {
    return this.actors.map((actor) => ({
      id: actor.id,
      type: actor.type,
      position: actor.position.clone(),
      heading: actor.heading,
      speed: actor.speed,
      radius: actor.radius,
      laneId: actor.laneId,
    }));
  }

  private spawnVehicles(): void {
    const count = densityCount[this.config.trafficDensity];
    let spawned = 0;
    let attempts = 0;
    while (spawned < count && attempts < count * 12) {
      attempts++;
      const vertical = this.rng.chance(0.55);
      const laneCenter = this.rng.pick(BLOCK_CENTERS);
      const direction = this.rng.chance(0.5) ? 1 : -1;
      const offset = direction === 1 ? 2.4 : -2.4;
      const start = this.rng.range(-MAP_EXTENT, MAP_EXTENT);
      const path = vertical
        ? [new THREE.Vector3(laneCenter + offset, 0, -direction * MAP_EXTENT), new THREE.Vector3(laneCenter + offset, 0, direction * MAP_EXTENT)]
        : [new THREE.Vector3(-direction * MAP_EXTENT, 0, laneCenter - offset), new THREE.Vector3(direction * MAP_EXTENT, 0, laneCenter - offset)];
      const position = vertical
        ? new THREE.Vector3(laneCenter + offset, 0, start)
        : new THREE.Vector3(start, 0, laneCenter - offset);
      if (this.positionInRouteCorridor(position, ROUTE_CLEARANCE)) continue;
      if (this.pathOverlapsRoute(path, ROUTE_CLEARANCE)) continue;
      const type = this.rng.pick(['vehicle', 'vehicle', 'vehicle', 'bus', 'truck'] as const);
      this.addActor({
        id: `npc_${spawned}`,
        type,
        position,
        heading: vertical ? (direction > 0 ? 0 : Math.PI) : (direction > 0 ? Math.PI / 2 : -Math.PI / 2),
        speed: this.rng.range(4, type === 'vehicle' ? 12 : 8),
        radius: type === 'vehicle' ? 1.45 : 2.45,
        laneId: vertical ? `v_${laneCenter}` : `h_${laneCenter}`,
        mesh: this.createVehicleMesh(type),
        path,
        targetIndex: direction > 0 ? 1 : 0,
      });
      spawned++;
    }
  }

  private spawnParkedVehicles(): void {
    const count = this.config.trafficDensity === 'low' ? 3 : this.config.trafficDensity === 'medium' ? 6 : 10;
    for (let i = 0; i < count; i++) {
      const vertical = this.rng.chance(0.55);
      const roadCenter = this.rng.pick(BLOCK_CENTERS);
      const side = this.rng.pick([-1, 1]);
      const curbOffset = ROAD_WIDTH / 2 - 2.2;
      const travel = this.rng.range(-MAP_EXTENT + 18, MAP_EXTENT - 18);
      const position = vertical
        ? new THREE.Vector3(roadCenter + side * curbOffset, 0, travel)
        : new THREE.Vector3(travel, 0, roadCenter + side * curbOffset);
      if (this.positionInRouteCorridor(position, ROUTE_CLEARANCE + 1.5)) continue;
      this.addActor({
        id: `parked_${i}`,
        type: 'vehicle',
        position,
        heading: vertical ? (side > 0 ? 0 : Math.PI) : (side > 0 ? -Math.PI / 2 : Math.PI / 2),
        speed: 0,
        radius: 1.45,
        laneId: vertical ? `parked_v_${roadCenter}` : `parked_h_${roadCenter}`,
        mesh: this.createVehicleMesh('vehicle', 0xf8fafc),
        path: [],
        targetIndex: 0,
      });
    }
  }

  private spawnScenarioActors(): void {
    if (this.config.kind === 'blocked_lane_detour') {
      for (let i = 0; i < 5; i++) {
        this.addActor({
          id: `cone_${i}`,
          type: 'obstacle',
          position: new THREE.Vector3(2.5 + (i % 2) * 1.5, 0, -18 + i * 3.6),
          heading: 0,
          speed: 0,
          radius: 1,
          mesh: this.createConeMesh(),
          path: [],
          targetIndex: 0,
        });
      }
      for (let i = 0; i < 3; i++) {
        this.addActor({
          id: `barrier_${i}`,
          type: 'obstacle',
          position: new THREE.Vector3(0.8, 0, -13 + i * 7),
          heading: Math.PI / 2,
          speed: 0,
          radius: 2,
          mesh: this.createBarrierMesh(),
          path: [],
          targetIndex: 0,
        });
      }
    }

    if (this.config.kind === 'pedestrian_crossing') {
      for (let i = 0; i < 6; i++) {
        this.addActor({
          id: `ped_${i}`,
          type: i % 3 === 0 ? 'cyclist' : 'pedestrian',
          position: new THREE.Vector3(-12 + i * 4, 0, 1.8),
          heading: i % 2 === 0 ? Math.PI / 2 : -Math.PI / 2,
          speed: 1.3 + (i % 2) * 0.4,
          radius: 0.9,
          mesh: this.createPedestrianMesh(i % 3 === 0),
          path: [],
          targetIndex: 0,
        });
      }
    }

    if (this.config.kind === 'lane_change_overtake') {
      this.addActor({
        id: 'slow_lead_vehicle',
        type: 'vehicle',
        position: new THREE.Vector3(56.4, 0, -42),
        heading: 0,
        speed: 3.6,
        radius: 1.45,
        mesh: this.createVehicleMesh('vehicle', 0xe8edf2),
        path: [
          new THREE.Vector3(56.4, 0, -80),
          new THREE.Vector3(56.4, 0, MAP_EXTENT + ROAD_WIDTH + 48),
        ],
        targetIndex: 1,
      });
    }

    if (this.config.kind === 'cut_in_vehicle') {
      this.addActor({
        id: 'cut_in_vehicle',
        type: 'vehicle',
        position: new THREE.Vector3(-45, 0, -46),
        heading: 0,
        speed: 8.5,
        radius: 1.45,
        mesh: this.createVehicleMesh('vehicle', 0xf8fafc),
        path: [new THREE.Vector3(-45, 0, -56), new THREE.Vector3(-51.6, 0, -22), new THREE.Vector3(-51.6, 0, 128)],
        targetIndex: 1,
      });
    }
  }

  private addActor(actor: ManagedActor): void {
    actor.mesh.position.copy(actor.position);
    actor.mesh.rotation.y = actor.heading;
    this.scene.add(actor.mesh);
    this.actors.push(actor);
  }

  private followPath(actor: ManagedActor, dt: number): void {
    if (actor.path.length < 2 || actor.speed <= 0) return;
    const target = actor.path[actor.targetIndex];
    const delta = target.clone().sub(actor.position);
    if (delta.length() < 4) {
      actor.targetIndex = (actor.targetIndex + 1) % actor.path.length;
      return;
    }
    const desiredHeading = Math.atan2(delta.x, delta.z);
    actor.heading = wrapPi(actor.heading + wrapPi(desiredHeading - actor.heading) * Math.min(1, dt * 2.2));
    actor.position.x += Math.sin(actor.heading) * actor.speed * dt;
    actor.position.z += Math.cos(actor.heading) * actor.speed * dt;
    if (actor.id.startsWith('npc_')
      && (Math.abs(actor.position.x) > MAP_EXTENT + ROAD_WIDTH || Math.abs(actor.position.z) > MAP_EXTENT + ROAD_WIDTH)) {
      actor.targetIndex = (actor.targetIndex + 1) % actor.path.length;
    }
  }

  private colliderSizeFor(type: ActorState['type']): THREE.Vector3 {
    if (type === 'bus') return new THREE.Vector3(2.45, 2.15, 6.35);
    if (type === 'truck') return new THREE.Vector3(2.45, 2.15, 5.95);
    if (type === 'obstacle') return new THREE.Vector3(1.8, 1.5, 2.8);
    if (type === 'pedestrian' || type === 'cyclist') return new THREE.Vector3(0.75, 1.8, 0.75);
    return new THREE.Vector3(1.95, 1.3, 3.95);
  }

  private positionInRouteCorridor(position: THREE.Vector3, clearance: number): boolean {
    const destination = this.graph.route[this.graph.route.length - 1];
    if (position.distanceTo(this.graph.start) < 24) return true;
    if (position.distanceTo(destination) < 26) return true;
    return nearestRouteIndex(this.graph.route, position).distance < clearance;
  }

  private pathOverlapsRoute(path: THREE.Vector3[], clearance: number): boolean {
    if (path.length < 2) return false;
    let overlap = 0;
    const start = path[0];
    const end = path[path.length - 1];
    for (let i = 0; i < this.graph.route.length; i += 6) {
      const point = this.graph.route[i];
      if (distanceToSegment2D(point, start, end) < clearance) {
        overlap++;
        if (overlap > ROUTE_PATH_OVERLAP_LIMIT) return true;
      }
    }
    return false;
  }

  private createVehicleMesh(type: ActorState['type'], color = 0xf5f7fa): THREE.Group {
    const assetKey = this.assetKeyForVehicle(type);
    const asset = this.assets?.createVehicle(assetKey, type === 'truck' || type === 'bus' ? 'service' : 'traffic');
    if (asset) return asset;

    const group = new THREE.Group();
    const scale = type === 'bus' ? [2.7, 1.45, 7.3] : type === 'truck' ? [2.9, 1.65, 6.4] : [2.25, 0.85, 4.4];
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(scale[0], scale[1], scale[2]),
      new THREE.MeshStandardMaterial({ color, roughness: 0.42, metalness: 0.05, transparent: true, opacity: type === 'vehicle' ? 0.82 : 0.92 }),
    );
    body.position.y = scale[1] / 2 + 0.22;
    const glass = new THREE.Mesh(
      new THREE.BoxGeometry(scale[0] * 0.68, 0.42, scale[2] * 0.42),
      new THREE.MeshBasicMaterial({ color: 0xd6dee7, transparent: true, opacity: 0.55 }),
    );
    glass.position.y = scale[1] + 0.55;
    glass.position.z = -scale[2] * 0.08;
    group.add(body, glass);
    return group;
  }

  private assetKeyForVehicle(type: ActorState['type']): VehicleAssetKey {
    if (type === 'bus') return 'delivery';
    if (type === 'truck') return 'truck';
    return this.rng.pick(['sedan', 'suv', 'van', 'police'] as const);
  }

  private createConeMesh(): THREE.Group {
    const asset = this.assets?.createProp('construction-cone', 'warning') ?? this.assets?.createProp('cone', 'warning');
    if (asset) return asset;

    const group = new THREE.Group();
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.55, 1.4, 12), new THREE.MeshBasicMaterial({ color: 0xff7a2f }));
    cone.position.y = 0.7;
    const stripe = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.42, 0.08, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    stripe.position.y = 0.75;
    group.add(cone, stripe);
    return group;
  }

  private createBarrierMesh(): THREE.Group {
    const asset = this.assets?.createProp('construction-barrier', 'warning');
    if (asset) return asset;
    const group = new THREE.Group();
    const barrier = new THREE.Mesh(
      new THREE.BoxGeometry(3.6, 0.9, 0.35),
      new THREE.MeshStandardMaterial({ color: 0xff7a2f, roughness: 0.7 }),
    );
    barrier.position.y = 0.45;
    group.add(barrier);
    return group;
  }

  private createPedestrianMesh(cyclist: boolean): THREE.Group {
    const group = new THREE.Group();
    const torsoMat = new THREE.MeshStandardMaterial({
      color: cyclist ? 0x4f6dff : 0xf8fafc,
      roughness: 0.48,
      metalness: 0.08,
    });
    const limbMat = new THREE.MeshStandardMaterial({ color: 0x273241, roughness: 0.62, metalness: 0.02 });
    const headMat = new THREE.MeshStandardMaterial({ color: 0xd8c7b8, roughness: 0.58, metalness: 0.02 });
    const markerMat = new THREE.MeshBasicMaterial({
      color: cyclist ? 0x6d7dff : 0x29d8ff,
      transparent: true,
      opacity: 0.24,
      side: THREE.DoubleSide,
      depthWrite: false,
    });

    const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.82, 8, 16), torsoMat);
    torso.position.y = 1.05;
    torso.scale.set(0.82, 1, 0.58);
    const shoulders = new THREE.Mesh(new THREE.CapsuleGeometry(0.08, 0.64, 6, 10), torsoMat);
    shoulders.position.y = 1.42;
    shoulders.rotation.z = Math.PI / 2;
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), headMat);
    head.position.y = 1.82;

    const leftArm = this.createBodyCapsule(0.055, 0.62, limbMat);
    leftArm.position.set(-0.36, 1.1, 0.02);
    leftArm.rotation.z = -0.28;
    const rightArm = this.createBodyCapsule(0.055, 0.62, limbMat);
    rightArm.position.set(0.36, 1.1, 0.02);
    rightArm.rotation.z = 0.28;
    const leftLeg = this.createBodyCapsule(0.075, 0.7, limbMat);
    leftLeg.position.set(-0.13, 0.43, 0.02);
    leftLeg.rotation.z = cyclist ? -0.48 : -0.1;
    const rightLeg = this.createBodyCapsule(0.075, 0.7, limbMat);
    rightLeg.position.set(0.13, 0.43, -0.02);
    rightLeg.rotation.z = cyclist ? 0.48 : 0.1;

    const footGeo = new THREE.BoxGeometry(0.18, 0.08, 0.34);
    const leftFoot = new THREE.Mesh(footGeo, limbMat);
    leftFoot.position.set(-0.16, 0.08, 0.12);
    const rightFoot = new THREE.Mesh(footGeo, limbMat);
    rightFoot.position.set(0.16, 0.08, 0.12);
    const marker = new THREE.Mesh(new THREE.RingGeometry(0.44, 0.5, 32), markerMat);
    marker.rotation.x = -Math.PI / 2;
    marker.position.y = 0.035;

    group.add(marker, torso, shoulders, head, leftArm, rightArm, leftLeg, rightLeg, leftFoot, rightFoot);
    if (cyclist) {
      const frameMat = new THREE.LineBasicMaterial({ color: 0x17202c, transparent: true, opacity: 0.9 });
      const frame = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-0.48, 0.42, 0), new THREE.Vector3(0, 0.78, 0),
          new THREE.Vector3(0, 0.78, 0), new THREE.Vector3(0.48, 0.42, 0),
          new THREE.Vector3(-0.48, 0.42, 0), new THREE.Vector3(0.48, 0.42, 0),
          new THREE.Vector3(0, 0.78, 0), new THREE.Vector3(0.08, 1.02, 0),
          new THREE.Vector3(0.48, 0.42, 0), new THREE.Vector3(0.62, 0.72, 0),
        ]),
        frameMat,
      );
      group.add(frame);
      for (const x of [-0.48, 0.48]) {
        const wheel = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.03, 8, 24), new THREE.MeshBasicMaterial({ color: 0x111827 }));
        wheel.position.set(x, 0.28, 0);
        wheel.rotation.y = Math.PI / 2;
        group.add(wheel);
      }
    }
    return group;
  }

  private createBodyCapsule(radius: number, length: number, material: THREE.Material): THREE.Mesh {
    return new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 6, 12), material);
  }
}

function distanceToSegment2D(point: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3): number {
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const apx = point.x - a.x;
  const apz = point.z - a.z;
  const lengthSq = abx * abx + abz * abz;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, (apx * abx + apz * abz) / lengthSq));
  const x = a.x + abx * t;
  const z = a.z + abz * t;
  return Math.hypot(point.x - x, point.z - z);
}
