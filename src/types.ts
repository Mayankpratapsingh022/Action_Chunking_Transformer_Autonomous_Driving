import * as THREE from 'three';
import type { LanguageIntentOption } from './vla/languageIntents';

export type { LanguageIntentOption };

export type ScenarioKind =
  | 'intersection_unprotected_left'
  | 'lane_change_overtake'
  | 'cut_in_vehicle'
  | 'blocked_lane_detour'
  | 'pedestrian_crossing'
  | 'traffic_light_stop_go'
  | 'curved_loop_drive';

export type RouteVariant = 'left' | 'straight' | 'right' | 'default';

export type TrafficDensity = 'low' | 'medium' | 'high';
export type WeatherKind = 'clear' | 'fog' | 'rain';
export type CameraMode = 'autonomy' | 'chase' | 'front' | 'bev';
export type RunMode = 'training' | 'inference';
export type CaptureResolution = 64 | 128 | 256;
export type RenderQuality = 'auto' | 'high' | 'balanced' | 'low';

export interface ScenarioConfig {
  id: string;
  kind: ScenarioKind;
  routeVariant?: RouteVariant;
  seed: number;
  trafficSeed?: number;
  trafficDensity: TrafficDensity;
  routeIntent: string;
  weather: WeatherKind;
}

export interface EgoState {
  x: number;
  z: number;
  heading: number;
  speed: number;
  steering: number;
  laneId: string;
}

export interface ControlCommand {
  throttle: number;
  brake: number;
  steer: number;
}

export interface BinaryActions {
  forward: 0 | 1;
  backward: 0 | 1;
  left: 0 | 1;
  right: 0 | 1;
}

export interface ActionVector {
  forward: number;
  backward: number;
  left: number;
  right: number;
}

export interface SimEvents {
  collision: boolean;
  offRoute: boolean;
  redLightViolation: boolean;
  goalReached: boolean;
  episodeDone: boolean;
}

export interface TaskProgress {
  destination: {
    x: number;
    z: number;
  };
  routeProgress: number;
  distanceToDestination: number;
  reachedDestination: boolean;
  episodeDone: boolean;
  outcome: 'in_progress' | 'success' | 'collision' | 'off_route';
  warning?: string | null;
}

export interface VLADatasetSample {
  timestamp: number;
  scenario_id: string;
  seed: number;
  run_mode: RunMode;
  capture_resolution: CaptureResolution;
  image: string;
  bev_image?: string;
  language_id: number;
  language_text: string;
  actions: BinaryActions;
  control: ControlCommand;
  ego: EgoState;
  events: SimEvents;
  task: TaskProgress;
}

export interface RoadLane {
  id: string;
  kind: 'vertical' | 'horizontal' | 'connector';
  center: THREE.Vector3[];
  width: number;
  speedLimit: number;
}

export interface RoadGraph {
  lanes: RoadLane[];
  route: THREE.Vector3[];
  intentText: string;
  start: THREE.Vector3;
  startHeading: number;
}

export interface ActorState {
  id: string;
  type: 'vehicle' | 'bus' | 'truck' | 'pedestrian' | 'cyclist' | 'obstacle';
  position: THREE.Vector3;
  heading: number;
  speed: number;
  radius: number;
  laneId?: string;
}

export interface TrafficLightState {
  id: string;
  position: THREE.Vector3;
  state: 'red' | 'yellow' | 'green';
}

export interface DatasetExport {
  metadata: {
    image_width: number;
    image_height: number;
    frame_stack: number;
    num_intents: number;
    intent_labels: string[];
    intent_texts: string[];
    num_samples: number;
    capture_rate_ms: number;
    capture_resolution: CaptureResolution;
    observation_keys: string[];
    schema_version: string;
    created: string;
  };
  samples: VLADatasetSample[];
}
