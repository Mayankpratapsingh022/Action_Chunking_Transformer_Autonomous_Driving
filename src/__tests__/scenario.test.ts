import { describe, expect, it } from 'vitest';
import { createRoadGraph } from '../world/roadGraph';
import type { ScenarioConfig } from '../types';

const base: ScenarioConfig = {
  id: 'test',
  kind: 'intersection_unprotected_left',
  seed: 7,
  trafficDensity: 'medium',
  routeIntent: 'test',
  weather: 'clear',
};

describe('road graph generation', () => {
  it('is deterministic for the same scenario', () => {
    const a = createRoadGraph(base);
    const b = createRoadGraph(base);
    expect(a.route.length).toBe(b.route.length);
    expect(a.start.x).toBe(b.start.x);
    expect(a.start.z).toBe(b.start.z);
    expect(a.route[20].x).toBeCloseTo(b.route[20].x);
  });

  it('creates an urban lane graph and route', () => {
    const graph = createRoadGraph(base);
    expect(graph.lanes.length).toBeGreaterThanOrEqual(12);
    expect(graph.route.length).toBeGreaterThan(100);
    expect(graph.intentText).toContain('left');
  });

  it('creates a curved loop route for loop-driving data', () => {
    const graph = createRoadGraph({ ...base, kind: 'curved_loop_drive' });
    const start = graph.route[0];
    const end = graph.route[graph.route.length - 1];
    expect(graph.route.length).toBeGreaterThan(250);
    expect(start.distanceTo(end)).toBeLessThan(10);
    expect(graph.intentText).toContain('curved loop');
  });
});
