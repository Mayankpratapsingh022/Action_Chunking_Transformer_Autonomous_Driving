import * as THREE from 'three';
import './style.css';
import { AssetLibrary } from './assets/assetLibrary';
import { EgoVehicle } from './entities/egoVehicle';
import { TrafficManager } from './entities/trafficManager';
import type {
  ActionVector,
  ActorState,
  CameraMode,
  CaptureResolution,
  ControlCommand,
  RenderQuality,
  RunMode,
  ScenarioConfig,
  ScenarioKind,
  TaskProgress,
  TrafficDensity,
} from './types';
import { Hud } from './ui/hud';
import { EpisodeVideoRecorder } from './vla/episodeVideoRecorder';
import { ExpertDriver } from './vla/expertDriver';
import { InferenceClient } from './vla/inferenceClient';
import {
  LANGUAGE_INTENTS,
  languageIntentById,
  languageIntentForScenario,
  languageIntentIndex,
} from './vla/languageIntents';
import { VLARecorder } from './vla/recorder';
import { configurePresentationCamera, configureSensorCamera } from './visual/layers';
import { RenderQualityManager } from './visual/renderQuality';
import { CityWorld } from './world/cityWorld';
import { createRoadGraph, nearestRouteIndex } from './world/roadGraph';

const canvas = document.getElementById('viewport') as HTMLCanvasElement;
const preview = document.getElementById('sensor-preview') as HTMLCanvasElement;
const hudRoot = document.getElementById('hud-root') as HTMLElement;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: false,
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.94;
const qualityManager = new RenderQualityManager(renderer);

const defaultLanguageIntent = languageIntentForScenario('intersection_unprotected_left');
let selectedLanguageIntentId = defaultLanguageIntent.id;

let config: ScenarioConfig = {
  id: 'urban-autonomy-001',
  kind: 'intersection_unprotected_left',
  routeVariant: defaultLanguageIntent.routeVariant,
  seed: 42,
  trafficDensity: 'medium',
  routeIntent: defaultLanguageIntent.text,
  weather: 'clear',
};

const assets = new AssetLibrary();
await assets.loadAll();

let scene = new THREE.Scene();
let graph = createRoadGraph(config);
let world = new CityWorld(scene, graph, config, assets, qualityManager.current);
let ego = new EgoVehicle(graph, assets);
ego.addTo(scene);
let traffic = new TrafficManager(scene, config, graph, assets);
let expert = new ExpertDriver(graph);
let inference = new InferenceClient();

const mainCamera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 600);
const frontCamera = new THREE.PerspectiveCamera(72, 1, 0.1, 180);
const bevCamera = new THREE.OrthographicCamera(-72, 72, 72, -72, 0.1, 260);
configurePresentationCamera(mainCamera);
configureSensorCamera(frontCamera);
configureSensorCamera(bevCamera);
let cameraMode: CameraMode = 'autonomy';
let runMode: RunMode = 'training';
let captureResolution: CaptureResolution = 128;
let expertEnabled = false;
let inferenceEnabled = false;
let awaitingStart = true;
let episodeFinishing = false;
let collisions = 0;
let offRoute = 0;
let lastCollision = false;
let lastOffRoute = false;
let status = 'Press WASD/arrows to start';
let latestCommand: ControlCommand = { throttle: 0, brake: 0, steer: 0 };
let fps = 0;
let frames = 0;
let fpsTimer = 0;
let maxRouteProgress = 0;
let taskWarning: string | null = null;
let taskProgress: TaskProgress = computeTaskProgress();
let lastGoalReached = false;
let pendingRespawnIndex: number | null = null;
let previewTimer = 0;
let hudTimer = 0;
let inferenceTimer = 0;

let recorder = createRecorder();
const episodeVideo = new EpisodeVideoRecorder();

const hud = new Hud(hudRoot, {
  onScenarioChange: (kind) => {
    const intent = languageIntentForScenario(kind);
    selectedLanguageIntentId = intent.id;
    reloadScenario({ kind, routeVariant: intent.routeVariant });
  },
  onSeedChange: (seed) => reloadScenario({ seed }),
  onDensityChange: (trafficDensity) => reloadScenario({ trafficDensity }),
  onWeatherChange: (weather) => reloadScenario({ weather }),
  onQualityChange: (quality: RenderQuality) => {
    qualityManager.setMode(quality);
    reloadScenario({});
  },
  onResolutionChange: (resolution) => setCaptureResolution(resolution),
  onModeChange: (mode) => setRunMode(mode),
  onLanguageIntentChange: (id) => setLanguageIntent(id),
  onCameraChange: (mode) => {
    cameraMode = mode;
  },
  onToggleExpert: toggleExpertDriver,
  onToggleInference: toggleInferenceDriver,
  onToggleRecording: toggleDatasetRecording,
  onSaveVideo: saveFinalVideo,
  onDownload: exportDataset,
  onReset: () => resetIteration('Scenario reset. Press WASD/arrows to start'),
  onVirtualControl: (code, pressed) => {
    ego.setVirtualControl(code, pressed);
    if (pressed && runMode !== 'inference') startEpisode('manual');
  },
}, hudState());

function reloadScenario(partial: Partial<ScenarioConfig>): void {
  recorder.dispose();
  ego.dispose();
  world.dispose();
  config = {
    ...config,
    ...partial,
    id: `urban-${selectedLanguageIntentId}-${partial.seed ?? config.seed}`,
    routeIntent: selectedLanguageText(),
  };
  scene = new THREE.Scene();
  graph = createRoadGraph(config);
  world = new CityWorld(scene, graph, config, assets, qualityManager.current);
  ego = new EgoVehicle(graph, assets);
  ego.addTo(scene);
  traffic = new TrafficManager(scene, config, graph, assets);
  expert = new ExpertDriver(graph);
  recorder = createRecorder();
  previewTimer = 0;
  hudTimer = 0;
  collisions = 0;
  offRoute = 0;
  episodeVideo.clear();
  resetEpisode('Scenario loaded. Press WASD/arrows to start');
}

function createRecorder(): VLARecorder {
  preview.width = captureResolution;
  preview.height = captureResolution;
  return new VLARecorder(renderer, scene, frontCamera, bevCamera, config, captureResolution);
}

function selectedLanguageText(): string {
  return languageIntentById(selectedLanguageIntentId).text;
}

function intentFor(kind: ScenarioKind): string {
  const intents: Record<ScenarioKind, string> = {
    intersection_unprotected_left: 'Make a safe left turn at the main intersection.',
    lane_change_overtake: 'Overtake the slow vehicle and continue north.',
    cut_in_vehicle: 'Yield smoothly to a vehicle cutting in.',
    blocked_lane_detour: 'Detour around the blocked lane.',
    pedestrian_crossing: 'Stop for pedestrians crossing.',
    traffic_light_stop_go: 'Follow the traffic lights.',
    curved_loop_drive: 'Follow the curved loop road to the destination.',
  };
  return intents[kind];
}

function updateCamera(camera: THREE.Camera, mode: CameraMode): void {
  const pose = mode === 'front'
    ? ego.getFrontCameraPose()
    : mode === 'chase'
      ? ego.getChaseCameraPose()
      : mode === 'bev'
        ? ego.getBevCameraPose()
        : ego.getAutonomyCameraPose();

  const resolved = resolveCameraOcclusion(pose, mode);
  camera.position.lerp(resolved.position, mode === 'front' || mode === 'bev' ? 1 : 0.14);
  camera.lookAt(resolved.lookAt);
}

function resolveCameraOcclusion(
  pose: { position: THREE.Vector3; lookAt: THREE.Vector3 },
  mode: CameraMode,
): { position: THREE.Vector3; lookAt: THREE.Vector3 } {
  if (mode === 'front' || mode === 'bev') return pose;
  const origin = ego.position.clone().add(new THREE.Vector3(0, 2.2, 0));
  const offset = pose.position.clone().sub(origin);
  const distance = offset.length();
  if (distance < 0.01) return pose;
  const ray = new THREE.Ray(origin, offset.clone().normalize());
  const hit = new THREE.Vector3();
  let safeDistance = distance;
  for (const collider of world.colliders) {
    if (collider.type !== 'building') continue;
    const point = ray.intersectBox(collider.box, hit);
    if (!point) continue;
    const hitDistance = point.distanceTo(origin);
    if (hitDistance < safeDistance) safeDistance = Math.max(4.5, hitDistance - 1.5);
  }
  if (safeDistance >= distance) return pose;
  return {
    position: origin.add(ray.direction.clone().multiplyScalar(safeDistance)),
    lookAt: pose.lookAt,
  };
}

function updateSensorCameras(): void {
  const fp = ego.getFrontCameraPose();
  frontCamera.position.copy(fp.position);
  frontCamera.lookAt(fp.lookAt);
  const bev = ego.getBevCameraPose();
  bevCamera.position.copy(bev.position);
  bevCamera.lookAt(bev.lookAt);
}

function commandFromInference(): ControlCommand | null {
  const result = inference.predict(recorder.captureFrontImage(), languageIdForIntent());
  if (!result) return null;
  const actions = result.actions;
  return {
    throttle: actions.forward ? 0.8 : 0,
    brake: actions.backward ? 0.5 : 0,
    steer: actions.left ? -0.75 : actions.right ? 0.75 : 0,
  };
}

function actionVectorFromCommand(command: ControlCommand): ActionVector {
  return {
    forward: clamp01(command.throttle),
    backward: clamp01(command.brake),
    left: clamp01(-command.steer),
    right: clamp01(command.steer),
  };
}

function setLanguageIntent(id: string): void {
  const intent = languageIntentById(id);
  selectedLanguageIntentId = intent.id;
  if (config.kind !== intent.scenario || config.routeVariant !== intent.routeVariant) {
    reloadScenario({ kind: intent.scenario, routeVariant: intent.routeVariant });
  } else {
    config = {
      ...config,
      id: `urban-${selectedLanguageIntentId}-${config.seed}`,
      routeIntent: intent.text,
    };
    status = `Language intent: ${intent.label}`;
  }
}

function setRunMode(mode: RunMode): void {
  runMode = mode;
  expertEnabled = false;
  inferenceEnabled = false;
  awaitingStart = true;
  latestCommand = { throttle: 0, brake: 0, steer: 0 };
  inferenceTimer = 0;
  status = mode === 'training'
    ? 'Training mode: press WASD/arrows to collect imitation data'
    : 'Inference mode: press I to connect or use Inference after connecting';
}

function toggleExpertDriver(): void {
  const wasExpertEnabled = expertEnabled;
  setRunMode('training');
  if (wasExpertEnabled) {
    latestCommand = { throttle: 0, brake: 0, steer: 0 };
    status = 'Expert stopped. Press WASD/arrows to start';
    return;
  }
  startEpisode('expert');
}

function toggleInferenceDriver(): void {
  const wasInferenceEnabled = inferenceEnabled;
  setRunMode('inference');
  if (wasInferenceEnabled) {
    latestCommand = { throttle: 0, brake: 0, steer: 0 };
    status = 'Inference stopped. Press WASD/arrows to start';
    return;
  }
  startInferenceEpisode();
}

function toggleDatasetRecording(): void {
  if (recorder.recording) recorder.stop();
  else recorder.start();
  status = recorder.recording ? 'Recording VLA dataset' : 'Recording stopped';
}

function saveFinalVideo(): void {
  const saved = episodeVideo.download(`vla_episode_${config.kind}_${Date.now()}.webm`);
  status = saved ? 'Saved final episode video' : 'No completed episode video yet';
}

function exportDataset(): void {
  recorder.download();
  status = `Exported ${recorder.samples.length} samples`;
}

function resetIteration(message = 'Next iteration. Press WASD/arrows to start'): void {
  void episodeVideo.stop().then(() => {
    resetEpisode(message);
  });
}

function setCaptureResolution(resolution: CaptureResolution): void {
  if (captureResolution === resolution) return;
  captureResolution = resolution;
  const wasRecording = recorder.recording;
  if (wasRecording) recorder.stop();
  recorder.dispose();
  recorder = createRecorder();
  status = `Model view set to ${resolution} x ${resolution}. Dataset buffer reset`;
}

function startInferenceEpisode(): void {
  if (!inference.connected) {
    status = 'Inference mode selected. Press I to connect model server';
    return;
  }
  startEpisode('inference');
}

function startEpisode(mode: 'manual' | 'expert' | 'inference'): void {
  if (episodeFinishing) return;
  runMode = mode === 'inference' ? 'inference' : 'training';
  awaitingStart = false;
  expertEnabled = mode === 'expert';
  inferenceEnabled = mode === 'inference';
  if (episodeVideo.ready) episodeVideo.clear();
  if (!episodeVideo.recording && !episodeVideo.start(canvas)) {
    status = 'Video recording unavailable in this browser';
    return;
  }
  status = mode === 'expert'
    ? 'Expert driver active'
    : mode === 'inference'
      ? 'Inference mode uses last server prediction'
      : 'Manual control';
}

function resetEpisode(message: string): void {
  ego.reset();
  expert.reset();
  maxRouteProgress = 0;
  taskWarning = null;
  taskProgress = computeTaskProgress();
  ego.setTaskEvents(taskProgress.reachedDestination, taskProgress.episodeDone);
  lastCollision = false;
  lastOffRoute = false;
  lastGoalReached = false;
  pendingRespawnIndex = null;
  awaitingStart = true;
  episodeFinishing = false;
  expertEnabled = false;
  inferenceEnabled = false;
  latestCommand = { throttle: 0, brake: 0, steer: 0 };
  inferenceTimer = 0;
  status = message;
}

function respawnEgoOnRoute(index: number): void {
  ego.respawnOnRoute(Math.min(graph.route.length - 1, index + 4));
  taskWarning = null;
  taskProgress = computeTaskProgress();
  ego.setTaskEvents(taskProgress.reachedDestination, taskProgress.episodeDone);
  lastCollision = false;
  lastOffRoute = false;
  pendingRespawnIndex = null;
  status = 'Collision: respawned on route';
}

function finishEpisodeAtDestination(): void {
  if (episodeFinishing) return;
  episodeFinishing = true;
  awaitingStart = true;
  expertEnabled = false;
  inferenceEnabled = false;
  latestCommand = { throttle: 0, brake: 0, steer: 0 };
  if (recorder.recording) recorder.stop();
  status = episodeVideo.recording ? 'Destination reached. Preparing final video' : 'Destination reached';
  void episodeVideo.stop().then((videoReady) => {
    resetEpisode(videoReady
      ? 'Destination reached. Save final video or press WASD/arrows to start'
      : 'Destination reached. Press WASD/arrows to start');
  });
}

function computeTaskProgress(): TaskProgress {
  const nearest = nearestRouteIndex(graph.route, ego.position);
  const destination = graph.route[graph.route.length - 1];
  const rawProgress = nearest.index / Math.max(1, graph.route.length - 1);
  const routeProgress = Math.max(maxRouteProgress, rawProgress);
  maxRouteProgress = routeProgress;
  const distanceToDestination = ego.position.distanceTo(destination);
  const reachedDestination = routeProgress > 0.94 && distanceToDestination < 10;
  const episodeDone = reachedDestination || ego.events.collision;
  const outcome = reachedDestination
    ? 'success'
    : ego.events.collision
      ? 'collision'
      : ego.events.offRoute
        ? 'off_route'
        : 'in_progress';

  return {
    destination: {
      x: destination.x,
      z: destination.z,
    },
    routeProgress,
    distanceToDestination,
    reachedDestination,
    episodeDone,
    outcome,
    warning: taskWarning,
  };
}

function computeTaskWarning(actorStates: ActorState[]): string | null {
  if (awaitingStart || episodeFinishing) return null;
  if (ego.events.collision) return 'Collision detected. Respawning on the route';

  const nearest = nearestRouteIndex(graph.route, ego.position);
  if (nearest.distance > 8.5) return 'Return to the highlighted route';

  if (selectedLanguageIntentId === 'stop_pedestrians' && pedestrianStopWarning(actorStates)) {
    return 'Stop: pedestrian or cyclist crossing ahead';
  }

  if (selectedLanguageIntentId === 'obey_traffic_lights' && trafficLightStopWarning()) {
    return 'Stop for the red or yellow traffic light';
  }

  if (selectedLanguageIntentId === 'detour_blocked_lane' && blockedLaneWarning(actorStates)) {
    return 'Move left: the blocked lane is still in your path';
  }

  if (selectedLanguageIntentId === 'yield_cut_in' && cutInWarning(actorStates)) {
    return 'Yield: cut-in vehicle is too close ahead';
  }

  if (selectedLanguageIntentId === 'overtake_slow_vehicle' && overtakeWarning(actorStates)) {
    return 'Do not tailgate: change lane before overtaking';
  }

  return null;
}

function pedestrianStopWarning(actorStates: ActorState[]): boolean {
  if (ego.speed < 1.1) return false;
  return actorStates.some((actor) => {
    if (actor.type !== 'pedestrian' && actor.type !== 'cyclist') return false;
    const metrics = forwardMetrics(actor.position);
    return metrics.longitudinal > 0 && metrics.longitudinal < 30 && metrics.lateral < 8;
  });
}

function trafficLightStopWarning(): boolean {
  if (ego.speed < 1.1) return false;
  return world.trafficLights.some((light) => {
    if (light.state === 'green') return false;
    const metrics = forwardMetrics(light.position);
    return metrics.longitudinal > 0 && metrics.longitudinal < 32 && metrics.lateral < 18;
  });
}

function blockedLaneWarning(actorStates: ActorState[]): boolean {
  if (ego.speed < 1.1) return false;
  const obstacleAhead = actorStates.some((actor) => {
    if (actor.type !== 'obstacle') return false;
    const metrics = forwardMetrics(actor.position);
    return metrics.longitudinal > 0 && metrics.longitudinal < 28 && metrics.lateral < 6.2;
  });
  const stillInBlockedLane = ego.position.x > -0.8 && ego.position.z > -46 && ego.position.z < 22;
  return obstacleAhead || stillInBlockedLane;
}

function cutInWarning(actorStates: ActorState[]): boolean {
  const actor = actorStates.find((candidate) => candidate.id === 'cut_in_vehicle');
  if (!actor || ego.speed < 1.2) return false;
  const metrics = forwardMetrics(actor.position);
  const closingFast = ego.speed > actor.speed + 1.4;
  return metrics.longitudinal > 0 && metrics.longitudinal < 24 && metrics.lateral < 5.8 && (closingFast || ego.speed > 5.5);
}

function overtakeWarning(actorStates: ActorState[]): boolean {
  const actor = actorStates.find((candidate) => candidate.id === 'slow_lead_vehicle');
  if (!actor || ego.speed < 1.2) return false;
  const metrics = forwardMetrics(actor.position);
  return metrics.longitudinal > 0 && metrics.longitudinal < 20 && metrics.lateral < 4.5 && ego.speed > actor.speed + 1.0;
}

function forwardMetrics(target: THREE.Vector3): { longitudinal: number; lateral: number } {
  const dx = target.x - ego.position.x;
  const dz = target.z - ego.position.z;
  const forwardX = Math.sin(ego.heading);
  const forwardZ = Math.cos(ego.heading);
  return {
    longitudinal: dx * forwardX + dz * forwardZ,
    lateral: Math.abs(dx * forwardZ - dz * forwardX),
  };
}

function hudState() {
  return {
    config,
    runMode,
    renderQuality: qualityManager.mode,
    activeQuality: qualityManager.current.id,
    captureResolution,
    languageIntentId: selectedLanguageIntentId,
    languageIntents: LANGUAGE_INTENTS,
    languageIntent: selectedLanguageText(),
    cameraMode,
    expertEnabled,
    inferenceEnabled,
    awaitingStart,
    recording: recorder.recording,
    videoRecording: episodeVideo.recording,
    videoReady: episodeVideo.ready,
    samples: recorder.samples.length,
    speed: ego.speed,
    steering: ego.steering,
    task: taskProgress,
    fps,
    status,
    taskWarning,
    collisions,
    offRoute,
    actions: actionVectorFromCommand(latestCommand),
  };
}

function animate(): void {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const elapsed = clock.elapsedTime;

  frames++;
  fpsTimer += dt;
  if (fpsTimer > 1) {
    fps = frames;
    frames = 0;
    fpsTimer = 0;
    qualityManager.reportFps(fps);
  }

  if (!awaitingStart && !episodeFinishing) traffic.update(dt, elapsed, ego.position, ego.speed);
  const actorStates = traffic.getActorStates();
  updateSensorCameras();

  if (awaitingStart || episodeFinishing) {
    latestCommand = { throttle: 0, brake: 0, steer: 0 };
  } else if (expertEnabled) {
    latestCommand = expert.compute(ego.position, ego.heading, ego.speed, actorStates, world.trafficLights);
  } else if (inferenceEnabled) {
    inferenceTimer += dt;
    if (inferenceTimer >= 0.1) {
      inferenceTimer = 0;
      latestCommand = commandFromInference() ?? latestCommand;
    }
  } else {
    latestCommand = ego.manualCommand();
  }

  const colliders = [...world.colliders, ...traffic.dynamicColliders];
  ego.update(dt, latestCommand, colliders);
  if (ego.events.collision && !lastCollision) {
    collisions++;
    pendingRespawnIndex = nearestRouteIndex(graph.route, ego.position).index;
  }
  if (ego.events.offRoute && !lastOffRoute) {
    offRoute++;
    status = 'Off route: steer back to the highlighted path';
  }
  taskProgress = computeTaskProgress();
  taskWarning = computeTaskWarning(actorStates);
  taskProgress.warning = taskWarning;
  ego.events.redLightViolation = taskWarning === 'Stop for the red or yellow traffic light';
  ego.setTaskEvents(taskProgress.reachedDestination, taskProgress.episodeDone);
  if (taskProgress.reachedDestination && !lastGoalReached) finishEpisodeAtDestination();
  lastCollision = ego.events.collision;
  lastOffRoute = ego.events.offRoute;
  lastGoalReached = taskProgress.reachedDestination;

  const hazard = actorStates.some((actor) => actor.position.distanceTo(ego.position) < actor.radius + 3.5);
  world.update(elapsed, ego.position, ego.heading, hazard || ego.events.collision || Boolean(taskWarning));
  world.setPresentationVisibility(
    cameraMode === 'autonomy' || cameraMode === 'bev',
    cameraMode !== 'front',
  );

  const actions = ego.commandToActions(latestCommand);
  if (!awaitingStart && !episodeFinishing) {
    recorder.capture(
      performance.now(),
      languageIdForIntent(),
      selectedLanguageText(),
      actions,
      latestCommand,
      ego.getState(),
      ego.events,
      taskProgress,
      runMode,
    );
  }
  previewTimer += dt;
  const previewInterval = 1 / qualityManager.current.sensorPreviewFps;
  if (hud.previewVisible && previewTimer >= previewInterval) {
    previewTimer = 0;
    recorder.drawPreview(preview);
  }
  if (pendingRespawnIndex !== null) respawnEgoOnRoute(pendingRespawnIndex);

  updateCamera(mainCamera, cameraMode);
  renderer.render(scene, mainCamera);
  hudTimer += dt;
  if (hudTimer >= 0.1) {
    hudTimer = 0;
    hud.render(hudState());
  }
}

function languageIdForIntent(): number {
  return languageIntentIndex(selectedLanguageIntentId);
}

function isFormControlTarget(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement
    || target instanceof HTMLSelectElement
    || target instanceof HTMLTextAreaElement
    || (target instanceof HTMLElement && target.isContentEditable);
}

async function connectInferenceServer(): Promise<void> {
  try {
    await inference.connect('http://localhost:8000');
    setRunMode('inference');
    startEpisode('inference');
    status = 'Connected to inference server';
  } catch (error) {
    status = error instanceof Error ? error.message : 'Inference connection failed';
  }
}

window.addEventListener('resize', () => {
  const width = window.innerWidth;
  const height = window.innerHeight;
  qualityManager.resize(width, height);
  mainCamera.aspect = width / height;
  mainCamera.updateProjectionMatrix();
});

window.addEventListener('keydown', async (event) => {
  const editingControl = isFormControlTarget(event.target);
  if (!editingControl && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(event.code)) {
    event.preventDefault();
  }
  if (!editingControl && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(event.code)) {
    if (runMode === 'inference') startInferenceEpisode();
    else startEpisode('manual');
  }
  if (editingControl || event.repeat) return;
  if (event.code === 'Digit1') cameraMode = 'autonomy';
  if (event.code === 'Digit2') cameraMode = 'chase';
  if (event.code === 'Digit3') cameraMode = 'front';
  if (event.code === 'Digit4') cameraMode = 'bev';
  if (event.code === 'KeyR') {
    event.preventDefault();
    toggleDatasetRecording();
  }
  if (event.code === 'KeyN') {
    event.preventDefault();
    resetIteration();
  }
  if (event.code === 'KeyV') {
    event.preventDefault();
    saveFinalVideo();
  }
  if (event.code === 'KeyE') {
    event.preventDefault();
    exportDataset();
  }
  if (event.code === 'KeyX') {
    event.preventDefault();
    toggleExpertDriver();
  }
  if (event.code === 'KeyI') {
    event.preventDefault();
    if (inference.connected) toggleInferenceDriver();
    else await connectInferenceServer();
  }
});

const clock = new THREE.Clock();
updateCamera(mainCamera, cameraMode);

Object.assign(window, {
  __VLA_DEBUG__: {
    getRenderStats: () => ({
      geometries: renderer.info.memory.geometries,
      textures: renderer.info.memory.textures,
      calls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      quality: qualityManager.current.id,
      pixelRatio: renderer.getPixelRatio(),
    }),
  },
});

animate();

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
