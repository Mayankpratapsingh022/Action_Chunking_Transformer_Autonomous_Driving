import { icon, type IconDefinition } from '@fortawesome/fontawesome-svg-core';
import {
  faArrowLeft,
  faArrowRight,
  faArrowUp,
  faBullseye,
  faCamera,
  faCarSide,
  faChartLine,
  faCircle,
  faDownload,
  faEye,
  faFloppyDisk,
  faGaugeHigh,
  faMap,
  faRobot,
  faRotateRight,
  faRoute,
  faSliders,
  faStop,
  faVideo,
} from '@fortawesome/free-solid-svg-icons';
import type {
  ActionVector,
  CameraMode,
  CaptureResolution,
  LanguageIntentOption,
  RenderQuality,
  RunMode,
  ScenarioConfig,
  ScenarioKind,
  TaskProgress,
  TrafficDensity,
  WeatherKind,
} from '../types';

const scenarioOptions: Array<{ value: ScenarioKind; label: string }> = [
  { value: 'intersection_unprotected_left', label: 'Urban Intersection' },
  { value: 'lane_change_overtake', label: 'Lane Change' },
  { value: 'cut_in_vehicle', label: 'Cut In' },
  { value: 'blocked_lane_detour', label: 'Blocked Lane' },
  { value: 'pedestrian_crossing', label: 'Pedestrian Crossing' },
  { value: 'traffic_light_stop_go', label: 'Traffic Light' },
  { value: 'curved_loop_drive', label: 'Curved Loop' },
];

const actionKeys = ['forward', 'left', 'right', 'backward'] as const;
type ActionKey = (typeof actionKeys)[number];

function svgIcon(definition: IconDefinition, className: string): string {
  return icon(definition, { classes: className.split(' ') }).html.join('');
}

export interface HudState {
  config: ScenarioConfig;
  runMode: RunMode;
  renderQuality: RenderQuality;
  activeQuality: string;
  captureResolution: CaptureResolution;
  languageIntentId: string;
  languageIntents: LanguageIntentOption[];
  languageIntent: string;
  cameraMode: CameraMode;
  expertEnabled: boolean;
  inferenceEnabled: boolean;
  awaitingStart: boolean;
  recording: boolean;
  videoRecording: boolean;
  videoReady: boolean;
  samples: number;
  speed: number;
  steering: number;
  task: TaskProgress;
  fps: number;
  status: string;
  taskWarning: string | null;
  collisions: number;
  offRoute: number;
  actions: ActionVector;
}

export interface HudCallbacks {
  onScenarioChange: (kind: ScenarioKind) => void;
  onSeedChange: (seed: number) => void;
  onDensityChange: (density: TrafficDensity) => void;
  onWeatherChange: (weather: WeatherKind) => void;
  onQualityChange: (quality: RenderQuality) => void;
  onResolutionChange: (resolution: CaptureResolution) => void;
  onModeChange: (mode: RunMode) => void;
  onLanguageIntentChange: (id: string) => void;
  onCameraChange: (mode: CameraMode) => void;
  onToggleExpert: () => void;
  onToggleInference: () => void;
  onToggleRecording: () => void;
  onSaveVideo: () => void;
  onDownload: () => void;
  onReset: () => void;
  onVirtualControl: (code: 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight', pressed: boolean) => void;
}

export class Hud {
  private readonly status = new Map<string, HTMLElement>();
  private readonly controls = new Map<string, HTMLInputElement | HTMLSelectElement>();
  private readonly buttons = new Map<string, HTMLButtonElement>();
  private readonly actionBars = new Map<ActionKey, HTMLElement>();
  private readonly actionValues = new Map<ActionKey, HTMLElement>();
  private readonly actionLines = new Map<ActionKey, SVGPolylineElement>();
  private readonly actionHistory: ActionVector[] = [];
  private readonly modeButtons: HTMLButtonElement[];
  private readonly cameraButtons: HTMLButtonElement[];
  private readonly warning: HTMLElement;
  private inspectorOpen = true;
  private telemetryOpen = true;
  private previewOpen = true;

  constructor(private readonly root: HTMLElement, callbacks: HudCallbacks, initial: HudState) {
    this.root.innerHTML = `
      <header class="topbar">
        <div class="brand">
          <div class="brand-mark" aria-hidden="true"></div>
          <div><strong>VLA Urban Autonomy</strong><span>Closed-loop simulator</span></div>
        </div>
        <div class="top-actions">
          <div class="mode-switch" aria-label="Run mode">
            <button data-mode="training">Training</button>
            <button data-mode="inference">Inference</button>
          </div>
          <div class="segmented" data-role="camera" aria-label="Camera view">
            <button data-camera="autonomy" title="Autonomy view" aria-label="Autonomy view">${svgIcon(faBullseye, 'view-icon')}</button>
            <button data-camera="chase" title="Chase view" aria-label="Chase view">${svgIcon(faCarSide, 'view-icon')}</button>
            <button data-camera="front" title="Front sensor view" aria-label="Front sensor view">${svgIcon(faCamera, 'view-icon')}</button>
            <button data-camera="bev" title="Bird's-eye view" aria-label="Bird's-eye view">${svgIcon(faMap, 'view-icon')}</button>
          </div>
          <div class="segmented workspace-toggles" aria-label="Workspace panels">
            <button id="inspector-toggle" class="active" title="Scenario controls" aria-label="Toggle scenario controls">${svgIcon(faSliders, 'view-icon')}</button>
            <button id="telemetry-toggle" class="active" title="Driving telemetry" aria-label="Toggle driving telemetry">${svgIcon(faChartLine, 'view-icon')}</button>
            <button id="preview-toggle" class="active" title="Model camera preview" aria-label="Toggle model camera preview">${svgIcon(faEye, 'view-icon')}</button>
          </div>
        </div>
      </header>

      <aside class="panel left-panel" aria-label="Scenario controls">
        <div class="panel-heading"><span>Scenario</span><strong data-status="activeQuality"></strong></div>
        <label>Scenario<select id="scenario-select"></select></label>
        <div class="field-grid">
          <label>Seed<input id="seed-input" type="number" min="0" step="1"></label>
          <label>Traffic<select id="density-select"><option>low</option><option>medium</option><option>high</option></select></label>
        </div>
        <div class="field-grid">
          <label>Weather<select id="weather-select"><option>clear</option><option>fog</option><option>rain</option></select></label>
          <label>Quality<select id="quality-select"><option>auto</option><option>high</option><option>balanced</option><option>low</option></select></label>
        </div>
        <label>Model View<select id="resolution-select"><option value="64">64 x 64</option><option value="128">128 x 128</option><option value="256">256 x 256</option></select></label>
        <label>Language Intent<select id="intent-select"></select></label>
        <div class="intent-box"><span>Instruction</span><strong data-status="intent"></strong></div>
      </aside>

      <aside class="panel right-panel" aria-label="Driving telemetry">
        <div class="panel-heading"><span>Telemetry</span>${svgIcon(faGaugeHigh, 'panel-icon')}</div>
        <div class="metric-grid">
          <div class="metric"><span>Speed</span><strong data-status="speed">0.00</strong></div>
          <div class="metric"><span>Steer</span><strong data-status="steering">0.00</strong></div>
          <div class="metric"><span>Progress</span><strong data-status="progress">0%</strong></div>
          <div class="metric"><span>Goal m</span><strong data-status="goalDistance">0.0</strong></div>
          <div class="metric"><span>Samples</span><strong data-status="samples">0</strong></div>
          <div class="metric"><span>FPS</span><strong data-status="fps">0</strong></div>
          <div class="metric"><span>Collisions</span><strong data-status="collisions">0</strong></div>
          <div class="metric"><span>Off route</span><strong data-status="offRoute">0</strong></div>
        </div>
      </aside>

      <aside class="action-panel" aria-label="Action telemetry">
        <div class="action-head"><span>ACT Action Vector</span><strong>[F, L, R, B]</strong></div>
        <div class="action-bars">
          ${actionKeys.map((key) => `<div class="action-row" data-action-row="${key}"><span>${key === 'backward' ? 'Brake' : key[0].toUpperCase() + key.slice(1)}</span><div class="action-track"><i data-action-bar="${key}"></i></div><strong data-action-value="${key}">0.00</strong></div>`).join('')}
        </div>
        <svg class="action-spark" viewBox="0 0 220 58" preserveAspectRatio="none" aria-label="Live action history">
          <line x1="0" y1="29" x2="220" y2="29"></line>
          ${actionKeys.map((key) => `<polyline data-action-line="${key}" points=""></polyline>`).join('')}
        </svg>
      </aside>

      <nav class="record-dock" aria-label="Run controls">
        <button id="record-button" class="record-toggle" title="Record dataset (R)" aria-label="Record or stop dataset">
          <span class="record-ring">${svgIcon(faCircle, 'record-icon record-start')}${svgIcon(faStop, 'record-icon record-stop')}</span>
          <span data-status="recordLabel">Record</span>
        </button>
        <button id="reset-button" class="tool-button" title="Reset episode (N)" aria-label="Reset episode">${svgIcon(faRotateRight, 'button-icon')}<span>Reset</span></button>
        <button id="video-button" class="tool-button" title="Save episode video (V)" aria-label="Save episode video">${svgIcon(faVideo, 'button-icon')}<span data-status="videoLabel">Save</span></button>
        <button id="download-button" class="tool-button" title="Export dataset (E)" aria-label="Export dataset">${svgIcon(faDownload, 'button-icon')}<span>Export</span></button>
        <span class="dock-divider" aria-hidden="true"></span>
        <button id="expert-button" class="tool-button" title="Toggle expert driver (X)" aria-label="Toggle expert driver">${svgIcon(faRoute, 'button-icon')}<span>Expert</span></button>
        <button id="inference-button" class="tool-button" title="Toggle inference driver (I)" aria-label="Toggle inference driver">${svgIcon(faRobot, 'button-icon')}<span>Infer</span></button>
        <div class="buffer-status" title="Dataset samples">${svgIcon(faFloppyDisk, 'button-icon')}<span data-status="samplePill">0</span></div>
      </nav>

      <div class="touch-controls" aria-label="Touch driving controls">
        <div class="touch-steer">
          <button id="touch-left" aria-label="Steer left">${svgIcon(faArrowLeft, 'touch-icon')}</button>
          <button id="touch-right" aria-label="Steer right">${svgIcon(faArrowRight, 'touch-icon')}</button>
        </div>
        <div class="touch-pedals">
          <button id="touch-go" aria-label="Accelerate">${svgIcon(faArrowUp, 'touch-icon')}</button>
          <button id="touch-brake" aria-label="Brake">${svgIcon(faStop, 'touch-icon')}</button>
        </div>
      </div>

      <div class="task-warning" data-role="task-warning"><span>Safety event</span><strong data-status="taskWarning"></strong></div>
      <div class="bottom-status"><span class="status-dot"></span><span data-status="status">Ready</span></div>
    `;

    this.root.querySelectorAll<HTMLElement>('[data-status]').forEach((element) => this.status.set(element.dataset.status!, element));
    this.root.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input[id], select[id]').forEach((element) => this.controls.set(element.id, element));
    this.root.querySelectorAll<HTMLButtonElement>('button[id]').forEach((element) => this.buttons.set(element.id, element));
    for (const key of actionKeys) {
      this.actionBars.set(key, this.root.querySelector<HTMLElement>(`[data-action-bar="${key}"]`)!);
      this.actionValues.set(key, this.root.querySelector<HTMLElement>(`[data-action-value="${key}"]`)!);
      this.actionLines.set(key, this.root.querySelector<SVGPolylineElement>(`[data-action-line="${key}"]`)!);
    }
    this.modeButtons = [...this.root.querySelectorAll<HTMLButtonElement>('[data-mode]')];
    this.cameraButtons = [...this.root.querySelectorAll<HTMLButtonElement>('[data-camera]')];
    this.warning = this.root.querySelector<HTMLElement>('[data-role="task-warning"]')!;

    const scenarioSelect = this.control<HTMLSelectElement>('scenario-select');
    scenarioOptions.forEach(({ value, label }) => scenarioSelect.add(new Option(label, value)));
    const intentSelect = this.control<HTMLSelectElement>('intent-select');
    initial.languageIntents.forEach((intent) => intentSelect.add(new Option(intent.label, intent.id)));

    scenarioSelect.addEventListener('change', () => callbacks.onScenarioChange(scenarioSelect.value as ScenarioKind));
    this.control<HTMLInputElement>('seed-input').addEventListener('change', (event) => callbacks.onSeedChange(Math.max(0, Math.round(Number((event.target as HTMLInputElement).value) || 0))));
    this.control<HTMLSelectElement>('density-select').addEventListener('change', (event) => callbacks.onDensityChange((event.target as HTMLSelectElement).value as TrafficDensity));
    this.control<HTMLSelectElement>('weather-select').addEventListener('change', (event) => callbacks.onWeatherChange((event.target as HTMLSelectElement).value as WeatherKind));
    this.control<HTMLSelectElement>('quality-select').addEventListener('change', (event) => callbacks.onQualityChange((event.target as HTMLSelectElement).value as RenderQuality));
    this.control<HTMLSelectElement>('resolution-select').addEventListener('change', (event) => callbacks.onResolutionChange(Number((event.target as HTMLSelectElement).value) as CaptureResolution));
    intentSelect.addEventListener('change', () => callbacks.onLanguageIntentChange(intentSelect.value));
    this.modeButtons.forEach((button) => button.addEventListener('click', () => callbacks.onModeChange(button.dataset.mode as RunMode)));
    this.cameraButtons.forEach((button) => button.addEventListener('click', () => callbacks.onCameraChange(button.dataset.camera as CameraMode)));

    this.button('expert-button').addEventListener('click', callbacks.onToggleExpert);
    this.button('inference-button').addEventListener('click', callbacks.onToggleInference);
    this.button('record-button').addEventListener('click', callbacks.onToggleRecording);
    this.button('video-button').addEventListener('click', callbacks.onSaveVideo);
    this.button('download-button').addEventListener('click', callbacks.onDownload);
    this.button('reset-button').addEventListener('click', callbacks.onReset);
    this.button('inspector-toggle').addEventListener('click', () => this.toggleInspector());
    this.button('telemetry-toggle').addEventListener('click', () => this.toggleTelemetry());
    this.button('preview-toggle').addEventListener('click', () => this.togglePreview());

    this.bindTouchControl('touch-left', 'ArrowLeft', callbacks);
    this.bindTouchControl('touch-right', 'ArrowRight', callbacks);
    this.bindTouchControl('touch-go', 'ArrowUp', callbacks);
    this.bindTouchControl('touch-brake', 'ArrowDown', callbacks);
    if (window.matchMedia('(max-width: 720px)').matches) {
      this.inspectorOpen = false;
      this.root.classList.add('inspector-closed');
      this.button('inspector-toggle').classList.remove('active');
    }
    this.render(initial);
  }

  get previewVisible(): boolean {
    return this.previewOpen;
  }

  render(state: HudState): void {
    this.setControl('scenario-select', state.config.kind);
    this.setControl('seed-input', String(state.config.seed));
    this.setControl('density-select', state.config.trafficDensity);
    this.setControl('weather-select', state.config.weather);
    this.setControl('quality-select', state.renderQuality);
    this.setControl('resolution-select', String(state.captureResolution));
    this.setControl('intent-select', state.languageIntentId);
    this.setText('activeQuality', `${state.activeQuality} quality`);
    this.setText('intent', state.languageIntent);
    this.setText('speed', Math.abs(state.speed).toFixed(1));
    this.setText('steering', state.steering.toFixed(2));
    this.setText('progress', `${Math.round(state.task.routeProgress * 100)}%`);
    this.setText('goalDistance', state.task.distanceToDestination.toFixed(0));
    this.setText('samples', String(state.samples));
    this.setText('samplePill', String(state.samples));
    this.setText('fps', String(state.fps));
    this.setText('status', state.status);
    this.setText('taskWarning', state.taskWarning ?? '');
    this.setText('recordLabel', state.recording ? 'Stop' : 'Record');
    this.setText('videoLabel', state.videoRecording ? 'Recording' : 'Save');
    this.setText('collisions', String(state.collisions));
    this.setText('offRoute', String(state.offRoute));
    this.renderActionPlot(state.actions);
    this.warning.classList.toggle('visible', Boolean(state.taskWarning));
    this.toggleButton('expert-button', state.expertEnabled);
    this.toggleButton('inference-button', state.inferenceEnabled);
    this.toggleButton('record-button', state.recording);
    this.toggleButton('video-button', state.videoReady || state.videoRecording);
    this.button('video-button').disabled = !state.videoReady;
    this.modeButtons.forEach((button) => button.classList.toggle('active', button.dataset.mode === state.runMode));
    this.cameraButtons.forEach((button) => button.classList.toggle('active', button.dataset.camera === state.cameraMode));
    this.root.classList.toggle('automated-driving', state.expertEnabled || state.inferenceEnabled);
  }

  private toggleInspector(): void {
    this.inspectorOpen = !this.inspectorOpen;
    this.root.classList.toggle('inspector-closed', !this.inspectorOpen);
    this.button('inspector-toggle').classList.toggle('active', this.inspectorOpen);
  }

  private toggleTelemetry(): void {
    this.telemetryOpen = !this.telemetryOpen;
    this.root.classList.toggle('telemetry-closed', !this.telemetryOpen);
    this.button('telemetry-toggle').classList.toggle('active', this.telemetryOpen);
  }

  private togglePreview(): void {
    this.previewOpen = !this.previewOpen;
    document.getElementById('sensor-preview')?.classList.toggle('is-hidden', !this.previewOpen);
    document.getElementById('model-view-label')?.classList.toggle('is-hidden', !this.previewOpen);
    this.button('preview-toggle').classList.toggle('active', this.previewOpen);
  }

  private bindTouchControl(id: string, code: 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight', callbacks: HudCallbacks): void {
    const button = this.button(id);
    const release = (event: Event) => {
      event.preventDefault();
      button.classList.remove('pressed');
      callbacks.onVirtualControl(code, false);
    };
    button.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      button.classList.add('pressed');
      callbacks.onVirtualControl(code, true);
    });
    button.addEventListener('pointerup', release);
    button.addEventListener('pointercancel', release);
    button.addEventListener('contextmenu', (event) => event.preventDefault());
  }

  private renderActionPlot(actions: ActionVector): void {
    const sample = {
      forward: clamp01(actions.forward),
      backward: clamp01(actions.backward),
      left: clamp01(actions.left),
      right: clamp01(actions.right),
    };
    this.actionHistory.push(sample);
    if (this.actionHistory.length > 44) this.actionHistory.shift();
    for (const key of actionKeys) {
      const value = sample[key];
      this.actionBars.get(key)!.style.transform = `scaleX(${value.toFixed(3)})`;
      this.actionValues.get(key)!.textContent = value.toFixed(2);
      this.actionLines.get(key)!.setAttribute('points', this.pointsForAction(key));
    }
  }

  private pointsForAction(key: ActionKey): string {
    const count = Math.max(1, this.actionHistory.length - 1);
    return this.actionHistory.map((sample, index) => `${((index / count) * 220).toFixed(1)},${(4 + (1 - sample[key]) * 50).toFixed(1)}`).join(' ');
  }

  private setText(key: string, value: string): void {
    const element = this.status.get(key);
    if (element && element.textContent !== value) element.textContent = value;
  }

  private setControl(id: string, value: string): void {
    const element = this.controls.get(id);
    if (element && element.value !== value) element.value = value;
  }

  private toggleButton(id: string, active: boolean): void {
    this.button(id).classList.toggle('active', active);
  }

  private control<T extends HTMLInputElement | HTMLSelectElement>(id: string): T {
    return this.controls.get(id) as T;
  }

  private button(id: string): HTMLButtonElement {
    return this.buttons.get(id)!;
  }
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}
