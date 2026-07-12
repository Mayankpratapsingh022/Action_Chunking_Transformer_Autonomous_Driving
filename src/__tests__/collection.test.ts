import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import type { ScenarioConfig } from '../types';
import { ExpertDriver } from '../vla/expertDriver';
import { createRoadGraph, nearestRouteIndex, pointHeadingAtRoute, routeTangent } from '../world/roadGraph';

const base: ScenarioConfig = {
  id: 'collection-test',
  kind: 'intersection_unprotected_left',
  routeVariant: 'left',
  seed: 44,
  trafficSeed: 91,
  trafficDensity: 'low',
  routeIntent: 'turn left',
  weather: 'clear',
};

describe('dataset routes and recovery expert', () => {
  it('uses the same intersection start for three language-controlled routes', () => {
    const left = createRoadGraph({ ...base, routeVariant: 'left' });
    const straight = createRoadGraph({ ...base, routeVariant: 'straight' });
    const right = createRoadGraph({ ...base, routeVariant: 'right' });
    expect(left.start.distanceTo(straight.start)).toBeLessThan(0.001);
    expect(left.start.distanceTo(right.start)).toBeLessThan(0.001);
    expect(left.route.at(-1)!.distanceTo(straight.route.at(-1)!)).toBeGreaterThan(100);
    expect(right.route.at(-1)!.distanceTo(straight.route.at(-1)!)).toBeGreaterThan(100);
  });

  it('steers a road-edge pose back toward the remaining route', () => {
    const graph = createRoadGraph({ ...base, routeVariant: 'straight' });
    const index = 80;
    const tangent = routeTangent(graph.route, index).setY(0).normalize();
    const side = new THREE.Vector3(tangent.z, 0, -tangent.x);
    const position = graph.route[index].clone().addScaledVector(side, 7);
    let heading = pointHeadingAtRoute(graph.route, index);
    let speed = 7;
    const expert = new ExpertDriver(graph);
    expert.reset(index);
    const initialDistance = nearestRouteIndex(graph.route, position).distance;
    for (let step = 0; step < 120; step++) {
      const command = expert.compute(position, heading, speed, [], []);
      speed += (command.throttle * 5 - command.brake * 8 - 0.4) * 0.05;
      speed = Math.max(1, Math.min(10, speed));
      heading -= command.steer * 1.1 * 0.05;
      position.x += Math.sin(heading) * speed * 0.05;
      position.z += Math.cos(heading) * speed * 0.05;
    }
    expect(nearestRouteIndex(graph.route, position).distance).toBeLessThan(initialDistance * 0.35);
    expect(expert.routeIndex).toBeGreaterThanOrEqual(index);
  });

  it('keeps the incoming road heading at every route destination', () => {
    const left = createRoadGraph({ ...base, routeVariant: 'left' });
    const right = createRoadGraph({ ...base, routeVariant: 'right' });
    const loop = createRoadGraph({ ...base, kind: 'curved_loop_drive', routeVariant: 'default' });

    expect(pointHeadingAtRoute(left.route, left.route.length - 1)).toBeCloseTo(-Math.PI / 2, 2);
    expect(pointHeadingAtRoute(right.route, right.route.length - 1)).toBeCloseTo(Math.PI / 2, 2);
    expect(Math.abs(pointHeadingAtRoute(loop.route, loop.route.length - 1))).toBeGreaterThan(1);
  });
});
