import * as THREE from 'three';
import type { RoadGraph, RoadLane, RouteVariant, ScenarioConfig, ScenarioKind } from '../types';

export const LANE_WIDTH = 4;
export const ROAD_WIDTH = 18;
export const MAP_EXTENT = 142;
export const BLOCK_CENTERS = [-54, 0, 54];

const UP = new THREE.Vector3(0, 1, 0);

function v(x: number, z: number): THREE.Vector3 {
  return new THREE.Vector3(x, 0, z);
}

function headingFromTo(a: THREE.Vector3, b: THREE.Vector3): number {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  return Math.atan2(dx, dz);
}

function clonePoints(points: THREE.Vector3[]): THREE.Vector3[] {
  return points.map((p) => p.clone());
}

function smoothRoute(points: THREE.Vector3[], samples = 240): THREE.Vector3[] {
  if (points.length < 3) return clonePoints(points);
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.35);
  return curve.getPoints(samples);
}

function ellipseRoute(centerX: number, centerZ: number, radiusX: number, radiusZ: number, start: number, end: number, samples = 300): THREE.Vector3[] {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const angle = start + (end - start) * t;
    points.push(v(centerX + Math.cos(angle) * radiusX, centerZ + Math.sin(angle) * radiusZ));
  }
  return points;
}

function buildBaseLanes(): RoadLane[] {
  const lanes: RoadLane[] = [];

  for (const x of BLOCK_CENTERS) {
    lanes.push({
      id: `v_${x}_north`,
      kind: 'vertical',
      center: [v(x + LANE_WIDTH * 0.6, -MAP_EXTENT), v(x + LANE_WIDTH * 0.6, MAP_EXTENT)],
      width: LANE_WIDTH,
      speedLimit: 16,
    });
    lanes.push({
      id: `v_${x}_south`,
      kind: 'vertical',
      center: [v(x - LANE_WIDTH * 0.6, MAP_EXTENT), v(x - LANE_WIDTH * 0.6, -MAP_EXTENT)],
      width: LANE_WIDTH,
      speedLimit: 16,
    });
  }

  for (const z of BLOCK_CENTERS) {
    lanes.push({
      id: `h_${z}_east`,
      kind: 'horizontal',
      center: [v(-MAP_EXTENT, z - LANE_WIDTH * 0.6), v(MAP_EXTENT, z - LANE_WIDTH * 0.6)],
      width: LANE_WIDTH,
      speedLimit: 16,
    });
    lanes.push({
      id: `h_${z}_west`,
      kind: 'horizontal',
      center: [v(MAP_EXTENT, z + LANE_WIDTH * 0.6), v(-MAP_EXTENT, z + LANE_WIDTH * 0.6)],
      width: LANE_WIDTH,
      speedLimit: 16,
    });
  }

  return lanes;
}

function intersectionRoute(variant: RouteVariant): { route: THREE.Vector3[]; intentText: string } {
  if (variant === 'straight') {
    return {
      route: smoothRoute([v(2.4, -128), v(2.4, -54), v(2.4, -14), v(2.4, 18), v(2.4, 58), v(2.4, 128)]),
      intentText: 'Continue straight through the main intersection.',
    };
  }
  if (variant === 'right') {
    return {
      route: smoothRoute([v(2.4, -128), v(2.4, -54), v(2.4, -14), v(18, -2.4), v(58, -2.4), v(128, -2.4)]),
      intentText: 'Make a safe right turn at the main intersection.',
    };
  }
  return {
    route: smoothRoute([v(2.4, -128), v(2.4, -54), v(2.4, -14), v(-18, 2.4), v(-58, 2.4), v(-128, 2.4)]),
    intentText: 'Proceed through the city and make the protected left turn at the main intersection.',
  };
}

function routeFor(kind: ScenarioKind, routeVariant: RouteVariant = 'default'): { route: THREE.Vector3[]; intentText: string } {
  switch (kind) {
    case 'intersection_unprotected_left':
      return intersectionRoute(routeVariant);
    case 'lane_change_overtake':
      return {
        route: smoothRoute([
          v(56.4, -128),
          v(56.4, -76),
          v(49.2, -62),
          v(49.2, 24),
          v(56.4, 46),
          v(56.4, 128),
        ]),
        intentText: 'Overtake the slow vehicle, return to lane, and continue north.',
      };
    case 'cut_in_vehicle':
      return {
        route: smoothRoute([v(-51.6, -128), v(-51.6, -72), v(-51.6, -20), v(-51.6, 54), v(-51.6, 128)]),
        intentText: 'Maintain lane and yield smoothly if a vehicle cuts in.',
      };
    case 'blocked_lane_detour':
      return {
        route: smoothRoute([v(2.4, -128), v(2.4, -54), v(-4.8, -34), v(-5.2, -4), v(-5.2, 34), v(2.4, 58), v(2.4, 128)]),
        intentText: 'Detour around the blocked lane and merge back safely.',
      };
    case 'pedestrian_crossing':
      return {
        route: smoothRoute([v(-2.4, 128), v(-2.4, 72), v(-2.4, 20), v(-2.4, -54), v(-2.4, -128)]),
        intentText: 'Drive south and stop for pedestrians in the crosswalk.',
      };
    case 'traffic_light_stop_go':
      return {
        route: smoothRoute([v(2.4, -128), v(2.4, -58), v(2.4, -8), v(2.4, 54), v(2.4, 128)]),
        intentText: 'Follow traffic lights and continue straight when safe.',
      };
    case 'curved_loop_drive':
      return {
        route: ellipseRoute(0, 0, 58, 42, -Math.PI / 2, Math.PI * 1.46),
        intentText: 'Follow the curved loop road smoothly and drive to the marked destination.',
      };
  }
}

export function createRoadGraph(config: ScenarioConfig): RoadGraph {
  const { route, intentText } = routeFor(config.kind, config.routeVariant);
  return {
    lanes: buildBaseLanes(),
    route,
    intentText,
    start: route[0].clone(),
    startHeading: headingFromTo(route[0], route[5] ?? route[1]),
  };
}

export function pointHeadingAtRoute(route: THREE.Vector3[], index: number): number {
  const pointIndex = Math.max(0, Math.min(route.length - 1, index));
  const forwardIndex = Math.min(route.length - 1, pointIndex + 3);
  if (forwardIndex > pointIndex) return headingFromTo(route[pointIndex], route[forwardIndex]);

  // At the destination there is no forward point. Use the incoming tangent
  // instead of treating a zero-length vector as a north-facing heading.
  const backwardIndex = Math.max(0, pointIndex - 3);
  if (backwardIndex < pointIndex) return headingFromTo(route[backwardIndex], route[pointIndex]);
  return 0;
}

export function nearestRouteIndex(route: THREE.Vector3[], position: THREE.Vector3): { index: number; distance: number } {
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < route.length; i++) {
    const distance = route[i].distanceTo(position);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return { index: bestIndex, distance: bestDistance };
}

export function routeTangent(route: THREE.Vector3[], index: number): THREE.Vector3 {
  const a = route[Math.max(0, index - 2)];
  const b = route[Math.min(route.length - 1, index + 2)];
  return new THREE.Vector3().subVectors(b, a).normalize();
}

export function offsetRoute(route: THREE.Vector3[], offset: number): THREE.Vector3[] {
  return route.map((point, i) => {
    const tangent = routeTangent(route, i);
    const perp = tangent.clone().applyAxisAngle(UP, Math.PI / 2);
    return point.clone().add(perp.multiplyScalar(offset));
  });
}
