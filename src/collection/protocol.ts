import type { RouteVariant, ScenarioKind, TrafficDensity, WeatherKind } from '../types';

export type EpisodeKind = 'nominal' | 'recovery' | 'failure';
export type DatasetSplit = 'train' | 'validation' | 'test' | 'analysis';
export type RecoveryType =
  | 'road_edge'
  | 'lateral_offset'
  | 'heading_error'
  | 'overspeed'
  | 'late_steering'
  | 'aborted_lane_change'
  | 'close_following'
  | 'late_braking'
  | 'steering_oscillation'
  | 'intersection_misalignment';
export type Severity = 'mild' | 'medium' | 'severe';
export type FailureLabel =
  | 'collision_vehicle'
  | 'collision_obstacle'
  | 'collision_pedestrian'
  | 'offroad_timeout'
  | 'wrong_route'
  | 'red_light_violation'
  | 'lane_departure'
  | 'unsafe_overspeed'
  | 'recovery_failed'
  | 'episode_timeout';

export interface PerturbationSpec {
  type: RecoveryType;
  direction?: 'left' | 'right';
  severity: Severity;
  triggerProgress: number;
}

export interface ExpertProfile {
  id: 'cautious' | 'normal' | 'assertive';
  speedScale: number;
  followingDistanceScale: number;
  steeringGain: number;
}

export interface CollectorEpisodeConfig {
  id: string;
  kind: EpisodeKind;
  split: DatasetSplit;
  taskId: string;
  instructionId: string;
  instruction: string;
  scenario: ScenarioKind;
  routeVariant: RouteVariant;
  worldSeed: number;
  trafficSeed: number;
  weather: WeatherKind;
  trafficDensity: TrafficDensity;
  expertProfile: ExpertProfile;
  perturbation?: PerturbationSpec;
  failureLabel?: FailureLabel;
  maxDurationSeconds: number;
  frameEndpoint: string;
}

export interface FrameTelemetry {
  frameIndex: number;
  timestamp: number;
  observation: {
    state: [number, number, number, number];
  };
  action: [number, number, number];
  routeProgress: number;
  routeError: number;
  headingError: number;
  expertRecovering: boolean;
  expertStopReason?: string;
  collision: boolean;
  offRoute: boolean;
}

export interface CollectorEpisodeResult {
  id: string;
  valid: boolean;
  success: boolean;
  frames: number;
  simulatedSeconds: number;
  routeProgress: number;
  outcome: 'success' | 'collision' | 'off_route' | 'timeout' | 'labelled_failure';
  failureLabel?: FailureLabel;
  recovered?: boolean;
  recoveryTimeSeconds?: number;
  telemetry: FrameTelemetry[];
  error?: string;
}

export interface BrowserCollectorApi {
  health(): { ready: true; version: string };
  runEpisode(config: CollectorEpisodeConfig): Promise<CollectorEpisodeResult>;
}
