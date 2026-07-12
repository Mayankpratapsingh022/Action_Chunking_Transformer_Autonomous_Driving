import type { PerturbationSpec } from './protocol';

export interface RecoveryAdjustment {
  lateralOffset: number;
  headingOffset: number;
  speed: number;
  steering: number;
}

export interface RecoveryMetrics {
  distance: number;
  headingError: number;
  speed: number;
}

const HEADING_BY_SEVERITY = [0.14, 0.3, 0.5] as const;
const LATERAL_BY_SEVERITY = [1.8, 4.2, 7.2] as const;
const ROAD_EDGE_BY_SEVERITY = [5.8, 7.4, 9.2] as const;
const SPEED_BY_SEVERITY = [13, 16, 19] as const;

export function recoveryAdjustment(
  perturbation: PerturbationSpec,
  currentSpeed: number,
): RecoveryAdjustment {
  const severityIndex = perturbation.severity === 'mild' ? 0 : perturbation.severity === 'medium' ? 1 : 2;
  const direction = perturbation.direction === 'left' ? -1 : 1;
  const heading = HEADING_BY_SEVERITY[severityIndex] * direction;
  const lateral = LATERAL_BY_SEVERITY[severityIndex] * direction;
  const speed = Math.max(5.5, currentSpeed);

  switch (perturbation.type) {
    case 'road_edge':
      return { lateralOffset: ROAD_EDGE_BY_SEVERITY[severityIndex] * direction, headingOffset: 0, speed, steering: 0 };
    case 'lateral_offset':
      return { lateralOffset: lateral, headingOffset: 0, speed, steering: 0 };
    case 'heading_error':
      return { lateralOffset: 0, headingOffset: heading, speed, steering: 0 };
    case 'overspeed':
    case 'late_braking':
    case 'close_following':
      return { lateralOffset: 0, headingOffset: 0, speed: SPEED_BY_SEVERITY[severityIndex], steering: 0 };
    case 'late_steering':
    case 'intersection_misalignment':
      return { lateralOffset: lateral * 0.75, headingOffset: heading, speed, steering: 0 };
    case 'aborted_lane_change':
      return { lateralOffset: lateral * 0.55, headingOffset: 0, speed, steering: 0 };
    case 'steering_oscillation':
      return { lateralOffset: 0, headingOffset: heading, speed, steering: direction * 0.8 };
  }
}

export function isStableRecovery(metrics: RecoveryMetrics, perturbation: PerturbationSpec): boolean {
  const recoveringFromSpeed = perturbation.type === 'overspeed'
    || perturbation.type === 'late_braking'
    || perturbation.type === 'close_following';
  const speedLimit = recoveringFromSpeed ? 12.5 : 13.5;
  return metrics.distance < 5.5
    && metrics.headingError < 0.22
    && Math.abs(metrics.speed) < speedLimit;
}
