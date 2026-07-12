import * as THREE from 'three';
import { AssetLibrary } from './assets/assetLibrary';
import type {
  CollectorEpisodeConfig,
  CollectorEpisodeResult,
  FailureLabel,
  FrameTelemetry,
  PerturbationSpec,
} from './collection/protocol';
import { isStableRecovery, recoveryAdjustment } from './collection/recovery';
import { EgoVehicle } from './entities/egoVehicle';
import { TrafficManager } from './entities/trafficManager';
import type { ControlCommand, RoadGraph, ScenarioConfig } from './types';
import { wrapPi } from './utils/rng';
import { configureSensorCamera } from './visual/layers';
import type { RenderQualityProfile } from './visual/renderQuality';
import { ExpertDriver } from './vla/expertDriver';
import { CityWorld } from './world/cityWorld';
import { createRoadGraph, nearestRouteIndex, pointHeadingAtRoute, routeTangent } from './world/roadGraph';

const CAPTURE_SIZE = 256;
const CAPTURE_FPS = 10;
const CONTROL_DT = 1 / CAPTURE_FPS;
const PHYSICS_DT = 0.02;
const PHYSICS_STEPS = Math.round(CONTROL_DT / PHYSICS_DT);
const PERTURBATION_CLEARANCE = 2.5;
const ROAD_EDGE_PERTURBATION_CLEARANCE = 5.5;
const VERSION = 'urban-collector-1';

const DATASET_QUALITY: RenderQualityProfile = {
  id: 'low',
  pixelRatioCap: 1,
  shadowMapSize: 0,
  shadows: false,
  sensorPreviewFps: 0,
  worldDetail: 0.68,
  rainDrops: 320,
};

const canvas = document.getElementById('collector-canvas') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  alpha: false,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: false,
});
renderer.setSize(1, 1, false);
renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.94;
renderer.shadowMap.enabled = false;

const frontCamera = new THREE.PerspectiveCamera(72, 1, 0.1, 180);
configureSensorCamera(frontCamera);
const renderTarget = new THREE.WebGLRenderTarget(CAPTURE_SIZE, CAPTURE_SIZE, {
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat,
  type: THREE.UnsignedByteType,
  depthBuffer: true,
  stencilBuffer: false,
});
renderTarget.texture.colorSpace = THREE.SRGBColorSpace;
const pixels = new Uint8Array(CAPTURE_SIZE * CAPTURE_SIZE * 4);

const assets = new AssetLibrary();
await assets.loadAll();

type Environment = {
  scene: THREE.Scene;
  graph: RoadGraph;
  world: CityWorld;
  ego: EgoVehicle;
  traffic: TrafficManager;
  expert: ExpertDriver;
};

function createEnvironment(episode: CollectorEpisodeConfig): Environment {
  const config: ScenarioConfig = {
    id: episode.id,
    kind: episode.scenario,
    routeVariant: episode.routeVariant,
    seed: episode.worldSeed,
    trafficSeed: episode.trafficSeed,
    trafficDensity: episode.trafficDensity,
    routeIntent: episode.instruction,
    weather: episode.weather,
  };
  const scene = new THREE.Scene();
  const graph = createRoadGraph(config);
  const world = new CityWorld(scene, graph, config, assets, DATASET_QUALITY);
  world.setPresentationVisibility(false, false);
  const ego = new EgoVehicle(graph, assets);
  ego.addTo(scene);
  const traffic = new TrafficManager(scene, config, graph, assets);
  const expert = new ExpertDriver(graph, episode.expertProfile);
  return { scene, graph, world, ego, traffic, expert };
}

function disposeEnvironment(environment: Environment): void {
  environment.ego.dispose();
  environment.world.dispose();
}

function routeProgress(environment: Environment): number {
  return environment.expert.routeIndex / Math.max(1, environment.graph.route.length - 1);
}

function routeMetrics(environment: Environment): { distance: number; headingError: number } {
  const nearest = nearestRouteIndex(environment.graph.route, environment.ego.position);
  return {
    distance: nearest.distance,
    headingError: Math.abs(wrapPi(pointHeadingAtRoute(environment.graph.route, nearest.index) - environment.ego.heading)),
  };
}

function updateSensorCamera(ego: EgoVehicle): void {
  const pose = ego.getFrontCameraPose();
  frontCamera.position.copy(pose.position);
  frontCamera.lookAt(pose.lookAt);
  frontCamera.updateMatrixWorld(true);
}

async function captureFrame(scene: THREE.Scene, endpoint: string): Promise<void> {
  const previousTarget = renderer.getRenderTarget();
  renderer.setRenderTarget(renderTarget);
  renderer.render(scene, frontCamera);
  renderer.setRenderTarget(previousTarget);
  await renderer.readRenderTargetPixelsAsync(renderTarget, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE, pixels);
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'content-type': 'application/octet-stream' },
    body: pixels.slice().buffer,
  });
  if (!response.ok) throw new Error(`Frame sink rejected frame: ${response.status}`);
}

function applyPerturbation(environment: Environment, perturbation: PerturbationSpec): void {
  const triggerIndex = Math.max(4, Math.min(environment.graph.route.length - 5, environment.expert.routeIndex));
  const adjustment = recoveryAdjustment(perturbation, environment.ego.speed);
  const clearance = perturbation.type === 'road_edge'
    ? ROAD_EDGE_PERTURBATION_CLEARANCE
    : PERTURBATION_CLEARANCE;
  const taskActorColliders = environment.traffic.dynamicColliders.filter(({ actor }) => !actor.id.startsWith('npc_'));
  const colliders = [...environment.world.colliders, ...taskActorColliders];
  let placed = false;
  let selectedIndex = triggerIndex;

  // A trigger can land beside street furniture or a scenario actor. Search a
  // short route window, then reduce only the lateral magnitude if necessary.
  for (const indexOffset of [0, 6, -6, 12, -12, 18, -18]) {
    const index = Math.max(4, Math.min(environment.graph.route.length - 5, triggerIndex + indexOffset));
    const point = environment.graph.route[index];
    const tangent = routeTangent(environment.graph.route, index).setY(0).normalize();
    const side = new THREE.Vector3(tangent.z, 0, -tangent.x);
    const heading = pointHeadingAtRoute(environment.graph.route, index) + adjustment.headingOffset;
    for (const scale of [1, 0.82, 0.64, 0.46, 0]) {
      const position = point.clone().addScaledVector(side, adjustment.lateralOffset * scale);
      environment.ego.setPose(position, heading, adjustment.speed, adjustment.steering);
      const unsafe = colliders.some(({ box }) => environment.ego.collisionBox.intersectsBox(
        box.clone().expandByScalar(clearance),
      ));
      if (!unsafe) {
        placed = true;
        selectedIndex = index;
        break;
      }
    }
    if (placed) break;
  }
  if (!placed) throw new Error(`Could not place collision-free ${perturbation.type} perturbation`);
  environment.expert.reset(selectedIndex);
}

function failureCommand(
  label: FailureLabel,
  expertCommand: ControlCommand,
  elapsed: number,
  direction: number,
): ControlCommand {
  if (label === 'unsafe_overspeed' || label === 'red_light_violation') {
    return { throttle: 1, brake: 0, steer: expertCommand.steer * 0.35 };
  }
  if (label === 'episode_timeout' || label === 'offroad_timeout') {
    return { throttle: 0, brake: 1, steer: 0 };
  }
  if (label.startsWith('collision_')) {
    return { throttle: 1, brake: 0, steer: elapsed < 1 ? expertCommand.steer : direction * 0.9 };
  }
  return { throttle: 0.85, brake: 0, steer: direction * (elapsed < 0.8 ? 0.65 : 1) };
}

async function fastForwardToTrigger(environment: Environment, triggerProgress: number): Promise<number> {
  let elapsed = 0;
  const target = Math.max(0.08, Math.min(0.72, triggerProgress));
  while (routeProgress(environment) < target && elapsed < 32) {
    environment.world.update(elapsed, environment.ego.position, environment.ego.heading, false);
    const actors = environment.traffic.getActorStates();
    const command = environment.expert.compute(
      environment.ego.position,
      environment.ego.heading,
      environment.ego.speed,
      actors,
      environment.world.trafficLights,
    );
    environment.traffic.update(PHYSICS_DT, elapsed, environment.ego.position, environment.ego.speed);
    // Warm-up is not recorded. Ignore dynamic contacts so a crossing NPC cannot
    // invalidate a deterministic recovery setup before the labelled segment begins.
    environment.ego.update(PHYSICS_DT, command, environment.world.colliders);
    elapsed += PHYSICS_DT;
    if (environment.ego.events.collision) {
      environment.ego.respawnOnRoute(environment.expert.routeIndex + 4);
      environment.expert.reset(environment.expert.routeIndex + 4);
    }
  }
  return elapsed;
}

async function runEpisode(config: CollectorEpisodeConfig): Promise<CollectorEpisodeResult> {
  const environment = createEnvironment(config);
  const telemetry: FrameTelemetry[] = [];
  let simulatedTime = 0;
  let recordingTime = 0;
  let recoveryStableSeconds = 0;
  let recoveryTimeSeconds: number | undefined;
  let recovered = config.kind !== 'recovery';
  let episodeCollision = false;
  let outcome: CollectorEpisodeResult['outcome'] = 'timeout';
  let success = false;
  let previousCommand: ControlCommand = { throttle: 0, brake: 0, steer: 0 };
  const direction = config.perturbation?.direction === 'left' ? -1 : 1;

  try {
    if (config.kind === 'recovery' && config.perturbation) {
      simulatedTime = await fastForwardToTrigger(environment, config.perturbation.triggerProgress);
      applyPerturbation(environment, config.perturbation);
    } else if (config.kind === 'failure') {
      simulatedTime = await fastForwardToTrigger(environment, config.perturbation?.triggerProgress ?? 0.22);
      if (config.failureLabel === 'offroad_timeout' || config.failureLabel === 'recovery_failed') {
        applyPerturbation(environment, {
          type: 'road_edge',
          direction: direction < 0 ? 'left' : 'right',
          severity: 'severe',
          triggerProgress: routeProgress(environment),
        });
      }
    }

    const effectiveDurationSeconds = config.kind === 'recovery'
      ? Math.max(60, config.maxDurationSeconds)
      : config.maxDurationSeconds;
    const maxFrames = Math.ceil(effectiveDurationSeconds * CAPTURE_FPS);
    for (let frameIndex = 0; frameIndex < maxFrames; frameIndex++) {
      environment.world.update(simulatedTime, environment.ego.position, environment.ego.heading, false);
      const actors = environment.traffic.getActorStates();
      const expertCommand = environment.expert.compute(
        environment.ego.position,
        environment.ego.heading,
        environment.ego.speed,
        actors,
        environment.world.trafficLights,
      );
      const command = config.kind === 'failure'
        ? failureCommand(config.failureLabel ?? 'lane_departure', expertCommand, recordingTime, direction)
        : expertCommand;
      const metrics = routeMetrics(environment);
      updateSensorCamera(environment.ego);
      await captureFrame(environment.scene, config.frameEndpoint);
      telemetry.push({
        frameIndex,
        timestamp: Number(recordingTime.toFixed(4)),
        observation: {
          state: [environment.ego.speed, environment.ego.steering, previousCommand.throttle, previousCommand.brake],
        },
        action: [command.throttle, command.brake, command.steer],
        routeProgress: routeProgress(environment),
        routeError: metrics.distance,
        headingError: metrics.headingError,
        expertRecovering: environment.expert.isRecovering,
        expertStopReason: environment.expert.lastStopReason ?? undefined,
        collision: environment.ego.events.collision,
        offRoute: environment.ego.events.offRoute,
      });
      previousCommand = command;

      for (let step = 0; step < PHYSICS_STEPS; step++) {
        environment.traffic.update(PHYSICS_DT, simulatedTime, environment.ego.position, environment.ego.speed);
        const taskActorColliders = environment.traffic.dynamicColliders.filter(({ actor }) => !actor.id.startsWith('npc_'));
        environment.ego.update(PHYSICS_DT, command, [...environment.world.colliders, ...taskActorColliders]);
        episodeCollision ||= environment.ego.events.collision;
        simulatedTime += PHYSICS_DT;
        recordingTime += PHYSICS_DT;
        environment.world.update(simulatedTime, environment.ego.position, environment.ego.heading, false);
        if (episodeCollision && config.kind !== 'failure') break;
      }

      const after = routeMetrics(environment);
      if (config.kind === 'recovery' && config.perturbation && !recovered) {
        if (isStableRecovery({ ...after, speed: environment.ego.speed }, config.perturbation)) {
          recoveryStableSeconds += CONTROL_DT;
          if (recoveryStableSeconds >= 1) {
            recovered = true;
            recoveryTimeSeconds = recordingTime;
          }
        } else {
          recoveryStableSeconds = 0;
        }
      }

      const progress = routeProgress(environment);
      const destination = environment.graph.route[environment.graph.route.length - 1];
      if (config.kind !== 'failure' && episodeCollision) {
        outcome = 'collision';
        break;
      }
      if (progress >= 0.94 && environment.ego.position.distanceTo(destination) < 16) {
        success = config.kind !== 'recovery' || recovered;
        outcome = success ? 'success' : 'off_route';
        break;
      }
      if (config.kind !== 'failure' && after.distance > 16 && recordingTime > 5) {
        outcome = 'off_route';
        break;
      }
      if (config.kind === 'failure') {
        const label = config.failureLabel ?? 'lane_departure';
        const labelled = episodeCollision
          || (label === 'unsafe_overspeed' && environment.ego.speed >= 18)
          || ((label === 'lane_departure' || label === 'wrong_route' || label === 'recovery_failed') && after.distance >= 11)
          || recordingTime >= (label === 'episode_timeout' ? 8 : 4);
        if (labelled) {
          outcome = 'labelled_failure';
          break;
        }
      }
    }

    const valid = config.kind === 'failure'
      ? outcome === 'labelled_failure'
      : success && !episodeCollision;
    return {
      id: config.id,
      valid,
      success,
      frames: telemetry.length,
      simulatedSeconds: recordingTime,
      routeProgress: routeProgress(environment),
      outcome,
      failureLabel: config.kind === 'failure' ? config.failureLabel : undefined,
      recovered: config.kind === 'recovery' ? recovered : undefined,
      recoveryTimeSeconds,
      telemetry,
      error: valid ? undefined : `Episode ended with ${outcome} at ${(routeProgress(environment) * 100).toFixed(1)}%`,
    };
  } catch (error) {
    return {
      id: config.id,
      valid: false,
      success: false,
      frames: telemetry.length,
      simulatedSeconds: recordingTime,
      routeProgress: routeProgress(environment),
      outcome,
      failureLabel: config.failureLabel,
      recovered: config.kind === 'recovery' ? recovered : undefined,
      recoveryTimeSeconds,
      telemetry,
      error: error instanceof Error ? error.message : String(error),
    };
  } finally {
    disposeEnvironment(environment);
  }
}

window.__VLA_COLLECTOR__ = {
  health: () => ({ ready: true, version: VERSION }),
  runEpisode,
};
