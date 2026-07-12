import { describe, expect, it } from 'vitest';
import { isStableRecovery, recoveryAdjustment } from '../collection/recovery';
import type { PerturbationSpec } from '../collection/protocol';

const perturbation = (type: PerturbationSpec['type'], severity: PerturbationSpec['severity'] = 'severe'): PerturbationSpec => ({
  type,
  direction: 'right',
  severity,
  triggerProgress: 0.3,
});

describe('dataset recovery perturbations', () => {
  it('does not add a lateral teleport to speed and heading-only failures', () => {
    expect(recoveryAdjustment(perturbation('overspeed'), 8).lateralOffset).toBe(0);
    expect(recoveryAdjustment(perturbation('late_braking'), 8).lateralOffset).toBe(0);
    expect(recoveryAdjustment(perturbation('heading_error'), 8).lateralOffset).toBe(0);
    expect(recoveryAdjustment(perturbation('steering_oscillation'), 8).lateralOffset).toBe(0);
  });

  it('requires a stable road-corridor pose and a recovered speed', () => {
    const overspeed = perturbation('overspeed');
    expect(isStableRecovery({ distance: 4.8, headingError: 0.1, speed: 10 }, overspeed)).toBe(true);
    expect(isStableRecovery({ distance: 6.2, headingError: 0.1, speed: 10 }, overspeed)).toBe(false);
    expect(isStableRecovery({ distance: 4.8, headingError: 0.3, speed: 10 }, overspeed)).toBe(false);
    expect(isStableRecovery({ distance: 4.8, headingError: 0.1, speed: 15 }, overspeed)).toBe(false);
  });
});
