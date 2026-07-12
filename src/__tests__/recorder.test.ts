import { describe, expect, it } from 'vitest';
import type { DatasetExport } from '../types';
import { validateDataset } from '../vla/recorder';

describe('dataset validation', () => {
  it('accepts the vla urban schema', () => {
    const dataset: DatasetExport = {
      metadata: {
        image_width: 128,
        image_height: 128,
        frame_stack: 1,
        num_intents: 1,
        intent_labels: ['drive'],
        intent_texts: ['Drive'],
        num_samples: 1,
        capture_rate_ms: 90,
        capture_resolution: 128,
        observation_keys: ['image', 'bev_image', 'language_text', 'ego', 'task'],
        schema_version: 'vla-urban-3',
        created: new Date(0).toISOString(),
      },
      samples: [{
        timestamp: 0,
        scenario_id: 's',
        seed: 1,
        run_mode: 'training',
        capture_resolution: 128,
        image: 'data:image/png;base64,abc',
        language_id: 0,
        language_text: 'Drive',
        actions: { forward: 1, backward: 0, left: 0, right: 0 },
        control: { throttle: 1, brake: 0, steer: 0 },
        ego: { x: 0, z: 0, heading: 0, speed: 0, steering: 0, laneId: 'l' },
        events: {
          collision: false,
          offRoute: false,
          redLightViolation: false,
          goalReached: false,
          episodeDone: false,
        },
        task: {
          destination: { x: 0, z: 40 },
          routeProgress: 0.25,
          distanceToDestination: 40,
          reachedDestination: false,
          episodeDone: false,
          outcome: 'in_progress',
        },
      }],
    };
    expect(validateDataset(dataset)).toBe(true);
  });
});
