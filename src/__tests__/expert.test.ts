import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import type { ScenarioConfig } from '../types';
import { ExpertDriver } from '../vla/expertDriver';
import { createRoadGraph } from '../world/roadGraph';

const config: ScenarioConfig = {
  id: 'test',
  kind: 'traffic_light_stop_go',
  seed: 12,
  trafficDensity: 'low',
  routeIntent: 'test',
  weather: 'clear',
};

describe('expert driver', () => {
  it('produces bounded controls', () => {
    const graph = createRoadGraph(config);
    const expert = new ExpertDriver(graph);
    const command = expert.compute(graph.start, graph.startHeading, 0, [], []);
    expect(command.throttle).toBeGreaterThanOrEqual(0);
    expect(command.throttle).toBeLessThanOrEqual(1);
    expect(command.brake).toBeGreaterThanOrEqual(0);
    expect(command.brake).toBeLessThanOrEqual(1);
    expect(command.steer).toBeGreaterThanOrEqual(-1);
    expect(command.steer).toBeLessThanOrEqual(1);
  });

  it('brakes for a close forward actor', () => {
    const graph = createRoadGraph(config);
    const expert = new ExpertDriver(graph);
    const actor = {
      id: 'hazard',
      type: 'vehicle' as const,
      position: graph.start.clone().add(new THREE.Vector3(0, 0, 5)),
      heading: 0,
      speed: 0,
      radius: 2,
    };
    const command = expert.compute(graph.start, graph.startHeading, 8, [actor], []);
    expect(command.brake).toBeGreaterThan(0);
  });
});
