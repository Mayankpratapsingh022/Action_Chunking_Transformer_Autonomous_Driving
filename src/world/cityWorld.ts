import * as THREE from 'three';
import type { AssetLibrary } from '../assets/assetLibrary';
import type { ActorState, RoadGraph, ScenarioConfig, TrafficLightState } from '../types';
import { SeededRng } from '../utils/rng';
import { setPresentationLayer } from '../visual/layers';
import { QUALITY_PROFILES, type RenderQualityProfile } from '../visual/renderQuality';
import { BLOCK_CENTERS, MAP_EXTENT, ROAD_WIDTH, offsetRoute, routeTangent } from './roadGraph';

type Collider = { id: string; box: THREE.Box3; type: ActorState['type'] | 'building' | 'boundary' };

const MATERIALS = {
  ground: new THREE.MeshStandardMaterial({ color: 0xf1f4f6, roughness: 0.94, metalness: 0.01 }),
  road: new THREE.MeshStandardMaterial({ color: 0x9ba5af, roughness: 0.8, metalness: 0.02 }),
  roadDark: new THREE.MeshStandardMaterial({ color: 0x858f9a, roughness: 0.82, metalness: 0.02 }),
  roadShoulder: new THREE.MeshStandardMaterial({ color: 0xd4dbe1, roughness: 0.88, metalness: 0.01 }),
  lane: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.92 }),
  laneSoft: new THREE.MeshBasicMaterial({ color: 0xeef4fb, transparent: true, opacity: 0.44 }),
  crosswalk: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75 }),
  curb: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.85 }),
  glass: new THREE.MeshStandardMaterial({ color: 0xdfe8f1, roughness: 0.3, metalness: 0.05, transparent: true, opacity: 0.62 }),
  building: new THREE.MeshStandardMaterial({ color: 0xe9edf2, roughness: 0.86 }),
  routeGlow: new THREE.MeshBasicMaterial({ color: 0x18c7f2, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }),
  route: new THREE.MeshBasicMaterial({ color: 0x079fe8, transparent: true, opacity: 0.66, side: THREE.DoubleSide, depthWrite: false }),
  routeCore: new THREE.MeshBasicMaterial({ color: 0x1677d2, transparent: true, opacity: 0.86, side: THREE.DoubleSide, depthWrite: false }),
  routeArrow: new THREE.MeshBasicMaterial({ color: 0xe9fbff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }),
  destination: new THREE.MeshBasicMaterial({ color: 0x2fffd0, transparent: true, opacity: 0.62, side: THREE.DoubleSide, depthWrite: false }),
  destinationCore: new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }),
  sensor: new THREE.MeshBasicMaterial({ color: 0x25c8ed, transparent: true, opacity: 0.075, side: THREE.DoubleSide, depthWrite: false }),
  sensorLine: new THREE.MeshBasicMaterial({ color: 0x13acd6, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false }),
  warning: new THREE.MeshBasicMaterial({ color: 0xff5268, transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false }),
};

const BUILDING_MATERIALS = [
  new THREE.MeshStandardMaterial({ color: 0xf7f8fa, roughness: 0.78 }),
  new THREE.MeshStandardMaterial({ color: 0xe7ebef, roughness: 0.84 }),
  new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.86 }),
  new THREE.MeshStandardMaterial({ color: 0xdce2e8, roughness: 0.8 }),
  new THREE.MeshStandardMaterial({ color: 0xeff2f5, roughness: 0.82 }),
];

const TREE_MATERIALS = {
  trunk: new THREE.MeshStandardMaterial({ color: 0x8a7259, roughness: 0.86 }),
  canopy: new THREE.MeshStandardMaterial({ color: 0x86caa0, roughness: 0.92 }),
  canopyDark: new THREE.MeshStandardMaterial({ color: 0x55a47b, roughness: 0.94 }),
  base: new THREE.MeshBasicMaterial({ color: 0x74bd92, transparent: true, opacity: 0.12, side: THREE.DoubleSide }),
};

const SHARED_GEOMETRIES = {
  box: new THREE.BoxGeometry(1, 1, 1),
  treeTrunkHigh: new THREE.CylinderGeometry(0.16, 0.26, 3.1, 10),
  treeTrunkLow: new THREE.CylinderGeometry(0.16, 0.26, 3.1, 6),
  treeCanopyHigh: new THREE.SphereGeometry(1, 18, 10),
  treeCanopyLow: new THREE.SphereGeometry(1, 10, 8),
  treeBase: new THREE.CircleGeometry(0.8, 24),
};

const PERSISTENT_GEOMETRIES = new Set<THREE.BufferGeometry>(Object.values(SHARED_GEOMETRIES));

const PERSISTENT_MATERIALS = new Set<THREE.Material>([
  ...Object.values(MATERIALS),
  ...BUILDING_MATERIALS,
  ...Object.values(TREE_MATERIALS),
]);

type SignalVisual = {
  lens: THREE.Mesh;
  halo: THREE.Mesh;
  color: number;
};

export class CityWorld {
  readonly scene: THREE.Scene;
  readonly colliders: Collider[] = [];
  readonly trafficLights: TrafficLightState[] = [];
  readonly sensorGroup = new THREE.Group();
  readonly routeGroup = new THREE.Group();
  private readonly rng: SeededRng;
  private readonly signalVisuals = new Map<string, SignalVisual[]>();
  private rainGeometry: THREE.BufferGeometry | null = null;
  private rainBaseY: Float32Array | null = null;

  constructor(
    scene: THREE.Scene,
    private readonly graph: RoadGraph,
    private readonly config: ScenarioConfig,
    private readonly assets?: AssetLibrary,
    private readonly quality: RenderQualityProfile = QUALITY_PROFILES.balanced,
  ) {
    this.scene = scene;
    this.rng = new SeededRng(config.seed);
    this.build();
  }

  update(time: number, egoPosition: THREE.Vector3, egoHeading: number, hazard: boolean): void {
    for (const light of this.trafficLights) {
      const phase = (time + Math.abs(light.position.x) * 0.01 + Math.abs(light.position.z) * 0.02) % 18;
      light.state = phase < 9 ? 'green' : phase < 12 ? 'yellow' : 'red';
      for (const [index, state] of (['red', 'yellow', 'green'] as const).entries()) {
        const visual = this.signalVisuals.get(light.id)?.[index];
        if (!visual) continue;
        const mesh = visual.lens;
        const active = light.state === state;
        const color = visual.color;
        if (mesh.material instanceof THREE.MeshStandardMaterial) {
          mesh.material.color.setHex(active ? color : 0x7f8b98);
          mesh.material.emissive.setHex(active ? color : 0x050607);
          mesh.material.emissiveIntensity = active ? (mesh.geometry instanceof THREE.CircleGeometry ? 2.7 : 1.45) : 0.05;
        } else if (mesh.material instanceof THREE.MeshBasicMaterial) {
          mesh.material.color.setHex(active ? color : 0x7f8b98);
        }
        const glow = visual.halo;
        if (glow.material instanceof THREE.MeshBasicMaterial) {
          glow.material.color.setHex(color);
          glow.material.opacity = active ? 0.72 : 0.05;
        }
      }
    }

    this.sensorGroup.position.set(
      egoPosition.x - Math.sin(egoHeading) * 2.3,
      egoPosition.y,
      egoPosition.z - Math.cos(egoHeading) * 2.3,
    );
    this.sensorGroup.rotation.y = egoHeading;
    const warning = this.sensorGroup.getObjectByName('sensor-warning');
    if (warning) warning.visible = hazard;
    this.updateRain(time, egoPosition);
  }

  setPresentationVisibility(showSensors: boolean, showRoute = true): void {
    this.sensorGroup.visible = showSensors;
    this.routeGroup.visible = showRoute;
  }

  dispose(): void {
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    this.scene.traverse((object) => {
      if (object instanceof THREE.DirectionalLight) object.shadow.map?.dispose();
      if (!(object instanceof THREE.Mesh || object instanceof THREE.LineSegments || object instanceof THREE.Points)) return;
      if (!object.userData.sharedGeometry
        && object.geometry instanceof THREE.BufferGeometry
        && !PERSISTENT_GEOMETRIES.has(object.geometry)) geometries.add(object.geometry);
      const objectMaterials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of objectMaterials) {
        if (!PERSISTENT_MATERIALS.has(material)) materials.add(material);
      }
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    this.rainGeometry = null;
    this.rainBaseY = null;
    this.signalVisuals.clear();
    this.scene.clear();
  }

  private build(): void {
    MATERIALS.roadDark.color.setHex(this.config.weather === 'rain' ? 0x707b86 : 0x858f9a);
    MATERIALS.roadDark.roughness = this.config.weather === 'rain' ? 0.46 : 0.82;
    MATERIALS.roadShoulder.color.setHex(this.config.weather === 'rain' ? 0xc8d0d7 : 0xd4dbe1);
    this.scene.background = new THREE.Color(this.config.weather === 'rain' ? 0xe8edf1 : 0xf5f7f9);
    this.scene.fog = this.config.weather === 'fog'
      ? new THREE.Fog(0xf8fafc, 70, 245)
      : new THREE.Fog(0xf8fafc, 170, 380);

    this.addLighting();
    this.addGround();
    if (this.config.kind === 'curved_loop_drive') {
      this.addLoopRoad();
      this.addRouteOverlay();
      this.addDestinationMarker();
      this.addLoopSceneDetails();
      this.addSensorOverlay();
      this.addWeather();
      this.addBoundaries();
      return;
    }

    this.addRoadGrid();
    this.addRouteOverlay();
    this.addDestinationMarker();
    this.addBuildings();
    this.addStreetDetails();
    this.addTrafficLights();
    this.addSensorOverlay();
    this.addWeather();
    this.addBoundaries();
  }

  private addLighting(): void {
    const rainy = this.config.weather === 'rain';
    this.scene.add(new THREE.HemisphereLight(0xffffff, rainy ? 0xaebac5 : 0xd3dbe2, rainy ? 1.25 : 1.55));
    const sun = new THREE.DirectionalLight(rainy ? 0xdce9f3 : 0xffffff, rainy ? 1.15 : 1.9);
    sun.position.set(-80, 120, 70);
    sun.castShadow = this.quality.shadows;
    if (this.quality.shadows) {
      sun.shadow.mapSize.set(this.quality.shadowMapSize, this.quality.shadowMapSize);
      sun.shadow.camera.left = -110;
      sun.shadow.camera.right = 110;
      sun.shadow.camera.top = 110;
      sun.shadow.camera.bottom = -110;
      sun.shadow.camera.near = 10;
      sun.shadow.camera.far = 260;
      sun.shadow.bias = -0.00025;
    }
    this.scene.add(sun);
  }

  private addGround(): void {
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(360, 360), MATERIALS.ground);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.03;
    ground.receiveShadow = this.quality.shadows;
    this.scene.add(ground);
  }

  private addRoadGrid(): void {
    const roadExtent = MAP_EXTENT * 1.1;
    const roadY = 0.035;
    for (const x of BLOCK_CENTERS) {
      for (const segment of this.roadSurfaceSegments(roadExtent, ROAD_WIDTH / 2)) {
        const length = segment.end - segment.start;
        const mid = (segment.start + segment.end) / 2;
        const road = new THREE.Mesh(new THREE.PlaneGeometry(ROAD_WIDTH, length), MATERIALS.roadDark);
        road.rotation.x = -Math.PI / 2;
        road.position.set(x, roadY, mid);
        this.scene.add(road);
      }
      this.addLaneLines('vertical', x);
      this.addLaneDirectionMarkers('vertical', x);
      this.addCurbs('vertical', x);
    }

    for (const z of BLOCK_CENTERS) {
      const road = new THREE.Mesh(new THREE.PlaneGeometry(MAP_EXTENT * 2.2, ROAD_WIDTH), MATERIALS.roadDark);
      road.rotation.x = -Math.PI / 2;
      road.position.set(0, roadY, z);
      this.scene.add(road);
      this.addLaneLines('horizontal', z);
      this.addLaneDirectionMarkers('horizontal', z);
      this.addCurbs('horizontal', z);
    }

    for (const x of BLOCK_CENTERS) {
      for (const z of BLOCK_CENTERS) {
        if (this.shouldAddCrosswalks(x, z)) this.addCrosswalks(x, z);
      }
    }
  }

  private shouldAddCrosswalks(x: number, z: number): boolean {
    return this.config.kind === 'pedestrian_crossing' && Math.abs(x) < 1 && Math.abs(z) < 1;
  }

  private addLaneLines(kind: 'vertical' | 'horizontal', center: number): void {
    const dashGeo = kind === 'vertical'
      ? new THREE.BoxGeometry(0.24, 0.025, 5.5)
      : new THREE.BoxGeometry(5.5, 0.025, 0.24);
    for (let t = -MAP_EXTENT; t <= MAP_EXTENT; t += 12) {
      if (this.nearIntersectionAxis(t, ROAD_WIDTH / 2 + 6)) continue;
      const dash = new THREE.Mesh(dashGeo, MATERIALS.lane);
      if (kind === 'vertical') dash.position.set(center, 0.09, t);
      else dash.position.set(t, 0.09, center);
      this.scene.add(dash);
    }
  }

  private addLaneDirectionMarkers(kind: 'vertical' | 'horizontal', center: number): void {
    for (let t = -MAP_EXTENT + 18; t <= MAP_EXTENT - 18; t += 36) {
      if (this.nearIntersectionAxis(t, ROAD_WIDTH / 2 + 10)) continue;
      for (const side of [-1, 1]) {
        const arrow = this.createRoadArrow();
        arrow.position.y = 0.125;
        if (kind === 'vertical') {
          arrow.position.x = center + side * 2.6;
          arrow.position.z = t;
          arrow.rotation.y = side > 0 ? 0 : Math.PI;
        } else {
          arrow.position.x = t;
          arrow.position.z = center - side * 2.6;
          arrow.rotation.y = side > 0 ? Math.PI / 2 : -Math.PI / 2;
        }
        this.scene.add(arrow);
      }
    }
  }

  private createRoadArrow(): THREE.Mesh {
    const shape = new THREE.Shape();
    shape.moveTo(0, 2.2);
    shape.lineTo(0.65, 0.75);
    shape.lineTo(0.25, 0.75);
    shape.lineTo(0.25, -1.9);
    shape.lineTo(-0.25, -1.9);
    shape.lineTo(-0.25, 0.75);
    shape.lineTo(-0.65, 0.75);
    shape.lineTo(0, 2.2);
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shape), MATERIALS.laneSoft);
    mesh.rotation.x = -Math.PI / 2;
    return mesh;
  }

  private addCurbs(kind: 'vertical' | 'horizontal', center: number): void {
    const segments = this.roadClearSegments();
    for (const side of [-1, 1]) {
      for (const segment of segments) {
        const length = segment.end - segment.start;
        const mid = (segment.start + segment.end) / 2;
        const curbGeo = kind === 'vertical'
          ? new THREE.BoxGeometry(0.48, 0.2, length)
          : new THREE.BoxGeometry(length, 0.2, 0.48);
        const curb = new THREE.Mesh(curbGeo, MATERIALS.curb);
        if (kind === 'vertical') curb.position.set(center + side * (ROAD_WIDTH / 2 + 0.4), 0.11, mid);
        else curb.position.set(mid, 0.11, center + side * (ROAD_WIDTH / 2 + 0.4));
        this.scene.add(curb);
      }
    }
  }

  private nearIntersectionAxis(value: number, margin: number): boolean {
    return BLOCK_CENTERS.some((center) => Math.abs(value - center) < margin);
  }

  private roadClearSegments(): Array<{ start: number; end: number }> {
    const gap = ROAD_WIDTH / 2 + 4;
    const segments: Array<{ start: number; end: number }> = [];
    let cursor = -MAP_EXTENT;
    for (const center of BLOCK_CENTERS) {
      const end = center - gap;
      if (end > cursor) segments.push({ start: cursor, end });
      cursor = center + gap;
    }
    if (cursor < MAP_EXTENT) segments.push({ start: cursor, end: MAP_EXTENT });
    return segments;
  }

  private roadSurfaceSegments(extent: number, halfGap: number): Array<{ start: number; end: number }> {
    const segments: Array<{ start: number; end: number }> = [];
    let cursor = -extent;
    for (const center of BLOCK_CENTERS) {
      const end = center - halfGap;
      if (end > cursor) segments.push({ start: cursor, end });
      cursor = center + halfGap;
    }
    if (cursor < extent) segments.push({ start: cursor, end: extent });
    return segments;
  }

  private addCrosswalks(x: number, z: number): void {
    for (const side of [-1, 1]) {
      for (let stripe = -3; stripe <= 3; stripe++) {
        const north = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.03, 5.8), MATERIALS.crosswalk);
        north.position.set(x + stripe * 2.2, 0.14, z + side * 14);
        this.scene.add(north);

        const east = new THREE.Mesh(new THREE.BoxGeometry(5.8, 0.03, 1.4), MATERIALS.crosswalk);
        east.position.set(x + side * 14, 0.14, z + stripe * 2.2);
        this.scene.add(east);
      }
    }
  }

  private addRouteOverlay(): void {
    const glow = this.createRibbon(this.graph.route, 2.8, MATERIALS.routeGlow, 0.205);
    const main = this.createRibbon(this.graph.route, 1.65, MATERIALS.route, 0.215);
    const core = this.createRibbon(this.graph.route, 0.72, MATERIALS.routeCore, 0.225);
    glow.name = 'route-glow';
    main.name = 'route-ribbon';
    core.name = 'route-core';
    this.routeGroup.add(glow, main, core);
    this.addRouteChevrons(this.graph.route);
    setPresentationLayer(this.routeGroup);
    this.scene.add(this.routeGroup);
  }

  private addRouteChevrons(points: THREE.Vector3[]): void {
    for (let i = 10; i < points.length - 10; i += 22) {
      const point = points[i];
      const chevron = this.createRouteChevronAt(point, routeTangent(points, i));
      this.routeGroup.add(chevron);
    }
  }

  private addDestinationMarker(): void {
    const destination = this.graph.route[this.graph.route.length - 1];
    const ring = new THREE.Mesh(new THREE.RingGeometry(2.6, 2.88, 64), MATERIALS.destination);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(destination.x, 0.34, destination.z);

    const core = new THREE.Mesh(new THREE.CircleGeometry(0.9, 40), MATERIALS.destinationCore);
    core.rotation.x = -Math.PI / 2;
    core.position.set(destination.x, 0.36, destination.z);

    const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.09, 5.6, 12), MATERIALS.destination);
    beacon.position.set(destination.x, 2.95, destination.z);

    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.44, 18, 12), MATERIALS.destinationCore);
    cap.position.set(destination.x, 5.9, destination.z);
    this.routeGroup.add(ring, core, beacon, cap);
  }

  private createRouteChevronAt(point: THREE.Vector3, tangent: THREE.Vector3): THREE.Mesh {
    const forward = tangent.clone().setY(0).normalize();
    const perp = new THREE.Vector3(forward.z, 0, -forward.x);
    const tip = point.clone().add(forward.clone().multiplyScalar(1.1));
    const leftOuter = point.clone().add(forward.clone().multiplyScalar(-0.68)).add(perp.clone().multiplyScalar(0.56));
    const leftInner = point.clone().add(forward.clone().multiplyScalar(-0.48)).add(perp.clone().multiplyScalar(0.2));
    const center = point.clone().add(forward.clone().multiplyScalar(0.2));
    const rightInner = point.clone().add(forward.clone().multiplyScalar(-0.48)).add(perp.clone().multiplyScalar(-0.2));
    const rightOuter = point.clone().add(forward.clone().multiplyScalar(-0.68)).add(perp.clone().multiplyScalar(-0.56));
    const y = 0.245;
    const positions = [
      tip.x, y, tip.z, leftOuter.x, y, leftOuter.z, leftInner.x, y, leftInner.z,
      tip.x, y, tip.z, leftInner.x, y, leftInner.z, center.x, y, center.z,
      tip.x, y, tip.z, center.x, y, center.z, rightInner.x, y, rightInner.z,
      tip.x, y, tip.z, rightInner.x, y, rightInner.z, rightOuter.x, y, rightOuter.z,
    ];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.computeVertexNormals();
    return new THREE.Mesh(geometry, MATERIALS.routeArrow);
  }

  private createRibbon(points: THREE.Vector3[], width: number, material: THREE.Material, y: number): THREE.Mesh {
    const positions: number[] = [];
    const left: THREE.Vector3[] = [];
    const right: THREE.Vector3[] = [];
    for (let index = 0; index < points.length; index++) {
      const previous = points[Math.max(0, index - 1)];
      const next = points[Math.min(points.length - 1, index + 1)];
      const tangent = new THREE.Vector3().subVectors(next, previous).setY(0).normalize();
      const perpendicular = new THREE.Vector3(-tangent.z, 0, tangent.x).multiplyScalar(width / 2);
      left.push(points[index].clone().add(perpendicular));
      right.push(points[index].clone().sub(perpendicular));
    }
    for (let i = 0; i < points.length - 1; i++) {
      const al = left[i];
      const ar = right[i];
      const bl = left[i + 1];
      const br = right[i + 1];
      positions.push(
        al.x, y, al.z, br.x, y, br.z, ar.x, y, ar.z,
        al.x, y, al.z, bl.x, y, bl.z, br.x, y, br.z,
      );
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.computeVertexNormals();
    return new THREE.Mesh(geometry, material);
  }

  private addBuildings(): void {
    const blockCenters = [-108, -27, 27, 108];
    for (const bx of blockCenters) {
      for (const bz of blockCenters) {
        if (Math.abs(bx) < 22 || Math.abs(bz) < 22) continue;
        const count = Math.max(1, Math.round((2 + this.rng.int(0, 3)) * this.quality.worldDetail));
        for (let i = 0; i < count; i++) {
          const w = this.rng.range(9, 19);
          const d = this.rng.range(9, 18);
          const h = this.rng.range(8, this.quality.id === 'low' ? 25 : 34);
          const x = bx + this.rng.range(-12, 12);
          const z = bz + this.rng.range(-12, 12);
          if (this.nearRoad(x, z, Math.max(w, d) / 2 + 4)) continue;
          const mesh = new THREE.Mesh(SHARED_GEOMETRIES.box, this.rng.pick(BUILDING_MATERIALS));
          mesh.position.set(x, h / 2, z);
          mesh.scale.set(w, h, d);
          mesh.castShadow = this.quality.shadows;
          mesh.receiveShadow = this.quality.shadows;
          this.scene.add(mesh);
          this.addWindowStrips(mesh, w, h, d);
          if (this.quality.id !== 'low') this.addBuildingCrown(mesh, w, h, d);
          this.colliders.push({ id: `building_${x.toFixed(1)}_${z.toFixed(1)}`, box: new THREE.Box3().setFromObject(mesh), type: 'building' });
        }
      }
    }
  }

  private addWindowStrips(mesh: THREE.Mesh, w: number, h: number, d: number): void {
    if (this.quality.id === 'low') return;
    const rows = Math.max(1, Math.floor(h / 6.5));
    for (let row = 1; row <= rows; row++) {
      const y = Math.min(h - 1.2, row * 5.8);
      const front = new THREE.Mesh(SHARED_GEOMETRIES.box, MATERIALS.glass);
      front.position.set(mesh.position.x, y, mesh.position.z + d / 2 + 0.045);
      front.scale.set(w * 0.72, 0.62, 0.07);
      const side = new THREE.Mesh(SHARED_GEOMETRIES.box, MATERIALS.glass);
      side.position.set(mesh.position.x + w / 2 + 0.045, y, mesh.position.z);
      side.scale.set(0.07, 0.62, d * 0.68);
      this.scene.add(front, side);
    }
  }

  private addBuildingCrown(mesh: THREE.Mesh, w: number, h: number, d: number): void {
    const roof = new THREE.Mesh(
      SHARED_GEOMETRIES.box,
      BUILDING_MATERIALS[(BUILDING_MATERIALS.indexOf(mesh.material as THREE.MeshStandardMaterial) + 1) % BUILDING_MATERIALS.length],
    );
    roof.position.set(mesh.position.x, h + 0.22, mesh.position.z);
    roof.scale.set(w * 0.72, 0.45, d * 0.72);
    roof.castShadow = this.quality.shadows;
    this.scene.add(roof);
  }

  private addStreetDetails(): void {
    for (let i = 0; i < Math.round(42 * this.quality.worldDetail); i++) {
      const nearXRoad = this.rng.pick(BLOCK_CENTERS);
      const z = this.rng.range(-MAP_EXTENT, MAP_EXTENT);
      const side = this.rng.pick([-1, 1]);
      this.addTree(nearXRoad + side * (ROAD_WIDTH / 2 + this.rng.range(5, 10)), z);
    }
    for (let i = 0; i < Math.round(32 * this.quality.worldDetail); i++) {
      const nearZRoad = this.rng.pick(BLOCK_CENTERS);
      const x = this.rng.range(-MAP_EXTENT, MAP_EXTENT);
      const side = this.rng.pick([-1, 1]);
      this.addLamp(x, nearZRoad + side * (ROAD_WIDTH / 2 + 3));
    }
    for (let i = 0; i < Math.round(9 * this.quality.worldDetail); i++) {
      const nearZRoad = this.rng.pick(BLOCK_CENTERS);
      const x = this.rng.range(-MAP_EXTENT + 18, MAP_EXTENT - 18);
      const side = this.rng.pick([-1, 1]);
      this.addRoadSign(x, nearZRoad + side * (ROAD_WIDTH / 2 + 5), side > 0 ? Math.PI : 0);
    }
  }

  private addLoopRoad(): void {
    const shoulder = this.createRibbon(this.graph.route, ROAD_WIDTH + 4.4, MATERIALS.roadShoulder, 0.09);
    const road = this.createRibbon(this.graph.route, ROAD_WIDTH + 0.7, MATERIALS.roadDark, 0.13);
    const innerEdge = this.createRibbon(offsetRoute(this.graph.route, -(ROAD_WIDTH / 2 + 1.15)), 0.36, MATERIALS.roadShoulder, 0.155);
    const outerEdge = this.createRibbon(offsetRoute(this.graph.route, ROAD_WIDTH / 2 + 1.15), 0.36, MATERIALS.roadShoulder, 0.155);
    this.scene.add(shoulder, road, innerEdge, outerEdge);
    this.addLoopLaneDashes(this.graph.route);
  }

  private addLoopLaneDashes(points: THREE.Vector3[]): void {
    const dashGeo = new THREE.BoxGeometry(0.26, 0.028, 4.8);
    for (let i = 8; i < points.length - 8; i += 15) {
      const tangent = routeTangent(points, i).setY(0).normalize();
      const dash = new THREE.Mesh(dashGeo, MATERIALS.lane);
      dash.position.set(points[i].x, 0.18, points[i].z);
      dash.rotation.y = Math.atan2(tangent.x, tangent.z);
      this.scene.add(dash);
    }
  }

  private addLoopSceneDetails(): void {
    const island = new THREE.Mesh(
      new THREE.CircleGeometry(24, 72),
      new THREE.MeshBasicMaterial({ color: 0xe9f4ef, transparent: true, opacity: 0.92, side: THREE.DoubleSide }),
    );
    island.rotation.x = -Math.PI / 2;
    island.position.y = 0.045;
    this.scene.add(island);

    for (const [x, z] of [
      [-18, -8],
      [16, -12],
      [6, 16],
      [-80, -34],
      [-78, 22],
      [-48, 60],
      [4, 69],
      [52, 55],
      [82, 20],
      [78, -30],
      [36, -66],
      [-42, -66],
    ] as const) {
      this.addTree(x, z);
    }

    for (const angle of [-2.65, -1.78, -0.9, 0.08, 0.94, 1.82, 2.7] as const) {
      const x = Math.cos(angle) * 76;
      const z = Math.sin(angle) * 56;
      this.addLoopLamp(x, z, this.headingForLocalNegativeZ(x, z, 0, 0));
    }
  }

  private addLoopLamp(x: number, z: number, heading: number): void {
    if (this.nearRoad(x, z, 2)) return;
    const group = new THREE.Group();
    group.position.set(x, 0, z);
    group.rotation.y = heading;
    const metalMat = new THREE.MeshStandardMaterial({ color: 0x7f8b98, roughness: 0.42, metalness: 0.34 });
    const lightMat = new THREE.MeshBasicMaterial({ color: 0xdff6ff, transparent: true, opacity: 0.78 });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.12, 4.8, 12), metalMat);
    pole.position.y = 2.4;
    const armCurve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(0, 4.55, 0),
      new THREE.Vector3(0, 4.9, -0.82),
      new THREE.Vector3(0, 4.55, -1.65),
    );
    const arm = new THREE.Mesh(new THREE.TubeGeometry(armCurve, 16, 0.04, 8), metalMat);
    const lampHead = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.16, 0.38), metalMat);
    lampHead.position.set(0, 4.5, -1.86);
    const lampLens = new THREE.Mesh(new THREE.SphereGeometry(0.16, 12, 8), lightMat);
    lampLens.position.set(0, 4.38, -1.95);
    lampLens.scale.set(1.25, 0.38, 0.7);
    group.add(pole, arm, lampHead, lampLens);
    this.scene.add(group);
    this.colliders.push({
      id: `loop_lamp_${x.toFixed(1)}_${z.toFixed(1)}`,
      box: new THREE.Box3(
        new THREE.Vector3(x - 0.35, 0, z - 0.35),
        new THREE.Vector3(x + 0.35, 5.2, z + 0.35),
      ),
      type: 'building',
    });
  }

  private headingForLocalNegativeZ(x: number, z: number, targetX: number, targetZ: number): number {
    return Math.atan2(x - targetX, z - targetZ);
  }

  private addTree(x: number, z: number): void {
    if (this.nearRoad(x, z, 2)) return;
    const group = new THREE.Group();
    group.position.set(x, 0, z);
    group.rotation.y = this.rng.range(0, Math.PI * 2);
    const trunk = new THREE.Mesh(
      this.quality.id === 'low' ? SHARED_GEOMETRIES.treeTrunkLow : SHARED_GEOMETRIES.treeTrunkHigh,
      TREE_MATERIALS.trunk,
    );
    trunk.position.y = 1.55;
    const canopyGeometry = this.quality.id === 'low' ? SHARED_GEOMETRIES.treeCanopyLow : SHARED_GEOMETRIES.treeCanopyHigh;
    const canopyBase = new THREE.Mesh(canopyGeometry, TREE_MATERIALS.canopy);
    canopyBase.position.set(0, 3.6, 0);
    canopyBase.scale.set(1.26, 0.9, 1.21);
    const canopyTop = new THREE.Mesh(canopyGeometry, TREE_MATERIALS.canopyDark);
    canopyTop.position.set(0.12, 4.35, -0.08);
    canopyTop.scale.set(0.84, 0.81, 0.81);
    const canopySide = new THREE.Mesh(canopyGeometry, TREE_MATERIALS.canopy);
    canopySide.position.set(-0.72, 3.75, 0.28);
    canopySide.scale.set(0.64, 0.53, 0.64);
    const base = new THREE.Mesh(
      SHARED_GEOMETRIES.treeBase,
      TREE_MATERIALS.base,
    );
    base.rotation.x = -Math.PI / 2;
    base.position.y = 0.08;
    group.add(base, trunk, canopyBase, canopyTop, canopySide);
    group.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.castShadow = this.quality.shadows;
        object.receiveShadow = this.quality.shadows;
      }
    });
    this.scene.add(group);
    this.colliders.push({
      id: `tree_${x.toFixed(1)}_${z.toFixed(1)}`,
      box: new THREE.Box3(
        new THREE.Vector3(x - 1.05, 0, z - 1.05),
        new THREE.Vector3(x + 1.05, 4.9, z + 1.05),
      ),
      type: 'building',
    });
  }

  private addLamp(x: number, z: number): void {
    if (this.nearRoad(x, z, 1)) return;
    const nearestZ = BLOCK_CENTERS.reduce((best, center) => (Math.abs(z - center) < Math.abs(z - best) ? center : best), BLOCK_CENTERS[0]);
    const side = z > nearestZ ? -1 : 1;
    const asset = this.assets?.createProp(this.rng.chance(0.65) ? 'light-curved' : 'light-square', 'street', new THREE.Vector3(2.7, 5.6, 2.7));
    if (asset) {
      asset.position.set(x, 0, z);
      asset.rotation.y = side > 0 ? 0 : Math.PI;
      this.scene.add(asset);
      this.colliders.push({
        id: `lamp_${x.toFixed(1)}_${z.toFixed(1)}`,
        box: new THREE.Box3(
          new THREE.Vector3(x - 0.35, 0, z - 0.35),
          new THREE.Vector3(x + 0.35, 5.1, z + 0.35),
        ),
        type: 'building',
      });
      return;
    }

    const group = new THREE.Group();
    group.position.set(x, 0, z);
    group.rotation.y = side > 0 ? 0 : Math.PI;
    const metalMat = new THREE.MeshStandardMaterial({ color: 0x8d97a3, roughness: 0.42, metalness: 0.32 });
    const lightMat = new THREE.MeshStandardMaterial({
      color: 0xfff3c4,
      roughness: 0.22,
      metalness: 0.02,
      emissive: 0xffd778,
      emissiveIntensity: 0.7,
    });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.12, 4.7, 12), metalMat);
    pole.position.y = 2.35;
    const armCurve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(0, 4.45, 0),
      new THREE.Vector3(0, 4.85, -0.75),
      new THREE.Vector3(0, 4.58, -1.55),
    );
    const arm = new THREE.Mesh(new THREE.TubeGeometry(armCurve, 16, 0.045, 8), metalMat);
    const lampHead = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.18, 0.42), metalMat);
    lampHead.position.set(0, 4.54, -1.82);
    const lampLens = new THREE.Mesh(new THREE.SphereGeometry(0.18, 12, 8), lightMat);
    lampLens.position.set(0, 4.42, -1.92);
    lampLens.scale.set(1.25, 0.38, 0.7);
    group.add(pole, arm, lampHead, lampLens);
    this.scene.add(group);
    this.colliders.push({
      id: `lamp_${x.toFixed(1)}_${z.toFixed(1)}`,
      box: new THREE.Box3(
        new THREE.Vector3(x - 0.35, 0, z - 0.35),
        new THREE.Vector3(x + 0.35, 5.1, z + 0.35),
      ),
      type: 'building',
    });
  }

  private addRoadSign(x: number, z: number, heading: number): void {
    if (this.nearRoad(x, z, 1)) return;
    const asset = this.assets?.createProp('sign-highway', 'street', new THREE.Vector3(3.6, 3.2, 0.9));
    if (!asset) return;
    asset.position.set(x, 0, z);
    asset.rotation.y = heading;
    this.scene.add(asset);
    this.colliders.push({ id: `sign_${x.toFixed(1)}_${z.toFixed(1)}`, box: new THREE.Box3().setFromObject(asset), type: 'building' });
  }

  private addTrafficLights(): void {
    let id = 0;
    for (const x of BLOCK_CENTERS) {
      for (const z of BLOCK_CENTERS) {
        const routeRadius = this.quality.id === 'low' ? 16 : 22;
        if (!this.routeNearPoint(x, z, routeRadius)) continue;
        for (const [dx, dz] of [[-12, -12], [12, -12], [-12, 12], [12, 12]]) {
          const group = this.createTrafficLightAssembly(`traffic_light_${id}`);
          group.position.set(x + dx, 0, z + dz);
          group.rotation.y = Math.atan2(-dx, -dz);
          this.scene.add(group);
          this.colliders.push({
            id: `traffic_light_pole_${id}`,
            box: new THREE.Box3(
              new THREE.Vector3(x + dx - 0.35, 0, z + dz - 0.35),
              new THREE.Vector3(x + dx + 0.35, 5.3, z + dz + 0.35),
            ),
            type: 'building',
          });
          this.trafficLights.push({ id: `traffic_light_${id}`, position: new THREE.Vector3(x + dx, 4.3, z + dz), state: 'green' });
          id++;
        }
      }
    }
    if (this.config.kind === 'traffic_light_stop_go') {
      const lightId = `traffic_light_${id}`;
      const gantry = this.createRouteSignalGantry(lightId);
      gantry.position.set(0, 0, -108);
      this.scene.add(gantry);
      this.colliders.push({
        id: `traffic_light_pole_${id}`,
        box: new THREE.Box3(
          new THREE.Vector3(-5.35, 0, -108.45),
          new THREE.Vector3(-4.45, 5.4, -107.55),
        ),
        type: 'building',
      });
      this.trafficLights.push({ id: lightId, position: new THREE.Vector3(2.4, 4.3, -108), state: 'green' });
    }
  }

  private createRouteSignalGantry(id: string): THREE.Group {
    const group = new THREE.Group();
    const metal = new THREE.MeshStandardMaterial({ color: 0x828d99, roughness: 0.42, metalness: 0.38 });
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.17, 5.15, 16), metal);
    pole.position.set(-4.9, 2.58, 0);
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 7.35, 14), metal);
    arm.rotation.z = Math.PI / 2;
    arm.position.set(-1.35, 4.94, 0);
    const hanger = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.64, 12), metal);
    hanger.position.set(2.4, 4.62, 0);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.46, 0.14, 22), metal);
    base.position.set(-4.9, 0.07, 0);
    const signal = this.createRouteSignalHead(id);
    signal.position.set(2.4, 3.92, -0.02);
    group.add(base, pole, arm, hanger, signal);
    return group;
  }

  private createRouteSignalHead(id: string): THREE.Group {
    const group = new THREE.Group();
    const mountMat = new THREE.MeshStandardMaterial({ color: 0x596675, roughness: 0.46, metalness: 0.32 });
    const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 2.35, 12), mountMat);
    spine.position.set(0, 0, -0.08);
    const topCap = new THREE.Mesh(new THREE.SphereGeometry(0.12, 14, 10), mountMat);
    topCap.position.set(0, 1.28, -0.08);
    const bottomCap = topCap.clone();
    bottomCap.position.y = -1.28;
    group.add(spine, topCap, bottomCap);

    const states = [
      ['red', 0xff3048, 0.92],
      ['yellow', 0xffc129, 0],
      ['green', 0x28f06e, -0.92],
    ] as const;
    const visuals: SignalVisual[] = [];
    for (const [state, color, y] of states) {
      const bracket = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.48, 10), mountMat);
      bracket.rotation.x = Math.PI / 2;
      bracket.position.set(0, y, -0.24);
      const lens = new THREE.Mesh(
        new THREE.SphereGeometry(0.42, 30, 20),
        new THREE.MeshBasicMaterial({
          color: state === 'green' ? color : 0x7f8b98,
        }),
      );
      lens.name = `${id}_${state}`;
      lens.position.set(0, y, -0.52);
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(0.48, 0.64, 40),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: state === 'green' ? 0.72 : 0.05,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
      halo.name = `${id}_${state}_glow`;
      halo.rotation.y = Math.PI;
      halo.position.set(0, y, -0.72);
      const rim = new THREE.Mesh(new THREE.TorusGeometry(0.44, 0.035, 8, 34), mountMat);
      rim.position.set(0, y, -0.52);
      group.add(bracket, rim, lens, halo);
      visuals.push({ lens, halo, color });
    }
    this.signalVisuals.set(id, visuals);
    return group;
  }

  private createTrafficLightAssembly(id: string): THREE.Group {
    const group = new THREE.Group();
    const metalMat = new THREE.MeshStandardMaterial({ color: 0x8d97a3, roughness: 0.48, metalness: 0.28 });
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.36, 0.14, 18), metalMat);
    base.position.y = 0.07;
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.14, 4.7, 14), metalMat);
    pole.position.y = 2.35;
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.052, 0.052, 1.9, 12), metalMat);
    mast.rotation.x = Math.PI / 2;
    mast.position.set(0, 4.62, -0.9);
    const hanger = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.6, 10), metalMat);
    hanger.position.set(0, 4.32, -1.8);
    const signal = this.createRouteSignalHead(id);
    signal.position.set(0, 3.62, -1.8);
    group.add(base, pole, mast, hanger, signal);
    return group;
  }

  private addSensorOverlay(): void {
    for (const [inner, outer, opacity] of [
      [4, 4.12, 0.2],
      [8.5, 8.62, 0.13],
      [13, 13.14, 0.09],
    ]) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(inner, outer, 64),
        new THREE.MeshBasicMaterial({ color: 0x18badf, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.35;
      this.sensorGroup.add(ring);
    }

    const forwardArc = new THREE.Mesh(new THREE.CircleGeometry(21, 44, Math.PI * 0.2, Math.PI * 0.6), MATERIALS.sensor);
    forwardArc.rotation.x = -Math.PI / 2;
    forwardArc.position.y = 0.36;
    forwardArc.position.z = 5;
    const nearArc = new THREE.Mesh(new THREE.RingGeometry(6, 14, 48, 1, Math.PI * 0.22, Math.PI * 0.56), MATERIALS.sensorLine);
    nearArc.rotation.x = -Math.PI / 2;
    nearArc.position.y = 0.38;
    nearArc.position.z = 3;
    const warning = new THREE.Mesh(new THREE.RingGeometry(4, 15, 48, 1, Math.PI * 0.22, Math.PI * 0.56), MATERIALS.warning);
    warning.rotation.x = -Math.PI / 2;
    warning.position.y = 0.39;
    warning.name = 'sensor-warning';
    warning.visible = false;
    this.sensorGroup.add(forwardArc, nearArc, warning);
    setPresentationLayer(this.sensorGroup);
    this.scene.add(this.sensorGroup);
  }

  private addWeather(): void {
    if (this.config.weather !== 'rain') return;
    MATERIALS.roadDark.color.setHex(0x707b86);
    MATERIALS.roadDark.roughness = 0.46;
    MATERIALS.roadShoulder.color.setHex(0xc8d0d7);

    const count = this.quality.rainDrops;
    const positions = new Float32Array(count * 2 * 3);
    this.rainBaseY = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const x = this.rng.range(-115, 115);
      const y = this.rng.range(4, 50);
      const z = this.rng.range(-115, 115);
      this.rainBaseY[i] = y;
      const index = i * 6;
      positions[index] = x;
      positions[index + 1] = y;
      positions[index + 2] = z;
      positions[index + 3] = x - 0.18;
      positions[index + 4] = y - 1.25;
      positions[index + 5] = z + 0.14;
    }
    this.rainGeometry = new THREE.BufferGeometry();
    this.rainGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const rain = new THREE.LineSegments(
      this.rainGeometry,
      new THREE.LineBasicMaterial({ color: 0x9bcce2, transparent: true, opacity: 0.48, depthWrite: false }),
    );
    rain.name = 'rain-field';
    rain.frustumCulled = false;
    this.scene.add(rain);
  }

  private updateRain(time: number, egoPosition: THREE.Vector3): void {
    if (!this.rainGeometry || !this.rainBaseY) return;
    const attribute = this.rainGeometry.getAttribute('position') as THREE.BufferAttribute;
    const positions = attribute.array as Float32Array;
    for (let i = 0; i < this.rainBaseY.length; i++) {
      const y = 3 + modulo(this.rainBaseY[i] - time * 24, 47);
      const index = i * 6;
      positions[index + 1] = y;
      positions[index + 4] = y - 1.25;
    }
    attribute.needsUpdate = true;
    const rain = this.scene.getObjectByName('rain-field');
    if (rain) rain.position.set(egoPosition.x, 0, egoPosition.z);
  }

  private addBoundaries(): void {
    const extent = MAP_EXTENT + 12;
    for (const [x, z, sx, sz] of [
      [0, extent, extent * 2, 2],
      [0, -extent, extent * 2, 2],
      [extent, 0, 2, extent * 2],
      [-extent, 0, 2, extent * 2],
    ]) {
      const box = new THREE.Box3(
        new THREE.Vector3(x - sx / 2, 0, z - sz / 2),
        new THREE.Vector3(x + sx / 2, 8, z + sz / 2),
      );
      this.colliders.push({ id: `boundary_${x}_${z}`, box, type: 'boundary' });
    }
  }

  private nearRoad(x: number, z: number, margin: number): boolean {
    if (this.config.kind === 'curved_loop_drive') {
      const clearance = ROAD_WIDTH / 2 + margin;
      for (let i = 0; i < this.graph.route.length; i += 6) {
        if (Math.hypot(x - this.graph.route[i].x, z - this.graph.route[i].z) < clearance) return true;
      }
      return false;
    }

    for (const center of BLOCK_CENTERS) {
      if (Math.abs(x - center) < ROAD_WIDTH / 2 + margin) return true;
      if (Math.abs(z - center) < ROAD_WIDTH / 2 + margin) return true;
    }
    return false;
  }

  private routeNearPoint(x: number, z: number, radius: number): boolean {
    for (let index = 0; index < this.graph.route.length; index += 5) {
      const point = this.graph.route[index];
      if (Math.hypot(point.x - x, point.z - z) <= radius) return true;
    }
    return false;
  }
}

function modulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}
