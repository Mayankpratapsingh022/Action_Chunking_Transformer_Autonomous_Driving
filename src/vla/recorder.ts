import * as THREE from 'three';
import type {
  BinaryActions,
  CaptureResolution,
  ControlCommand,
  DatasetExport,
  EgoState,
  RunMode,
  ScenarioConfig,
  SimEvents,
  TaskProgress,
  VLADatasetSample,
} from '../types';
import { LANGUAGE_INTENTS } from './languageIntents';
import { configureSensorCamera } from '../visual/layers';

export class VLARecorder {
  readonly renderTarget: THREE.WebGLRenderTarget;
  readonly bevTarget: THREE.WebGLRenderTarget;

  readonly samples: VLADatasetSample[] = [];
  recording = false;
  captureRateMs = 90;
  private lastCapture = 0;
  private readonly pixels: Uint8Array;
  private readonly canvas = document.createElement('canvas');
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly scene: THREE.Scene,
    private readonly frontCamera: THREE.Camera,
    private readonly bevCamera: THREE.Camera,
    private readonly config: ScenarioConfig,
    readonly captureSize: CaptureResolution = 128,
  ) {
    configureSensorCamera(this.frontCamera);
    configureSensorCamera(this.bevCamera);
    this.renderTarget = new THREE.WebGLRenderTarget(captureSize, captureSize, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });
    this.bevTarget = new THREE.WebGLRenderTarget(captureSize, captureSize, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    });
    this.pixels = new Uint8Array(captureSize * captureSize * 4);
    this.canvas.width = captureSize;
    this.canvas.height = captureSize;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not create capture canvas');
    this.ctx = ctx;
  }

  start(): void {
    this.recording = true;
    this.lastCapture = 0;
  }

  stop(): void {
    this.recording = false;
  }

  clear(): void {
    this.samples.length = 0;
  }

  capture(
    timestamp: number,
    languageId: number,
    languageText: string,
    actions: BinaryActions,
    control: ControlCommand,
    ego: EgoState,
    events: SimEvents,
    task: TaskProgress,
    runMode: RunMode,
  ): boolean {
    if (!this.recording) return false;
    if (timestamp - this.lastCapture < this.captureRateMs) return false;
    this.lastCapture = timestamp;

    const image = this.renderCameraToBase64(this.frontCamera, this.renderTarget);
    const bevImage = this.renderCameraToBase64(this.bevCamera, this.bevTarget);
    this.samples.push({
      timestamp,
      scenario_id: this.config.id,
      seed: this.config.seed,
      run_mode: runMode,
      capture_resolution: this.captureSize,
      image,
      bev_image: bevImage,
      language_id: languageId,
      language_text: languageText,
      actions,
      control: { ...control },
      ego: { ...ego },
      events: { ...events },
      task: {
        ...task,
        destination: { ...task.destination },
      },
    });
    return true;
  }

  exportDataset(): DatasetExport {
    return {
      metadata: {
        image_width: this.captureSize,
        image_height: this.captureSize,
        frame_stack: 1,
        num_intents: LANGUAGE_INTENTS.length,
        intent_labels: LANGUAGE_INTENTS.map((intent) => intent.id),
        intent_texts: LANGUAGE_INTENTS.map((intent) => intent.text),
        num_samples: this.samples.length,
        capture_rate_ms: this.captureRateMs,
        capture_resolution: this.captureSize,
        observation_keys: ['image', 'bev_image', 'language_text', 'ego', 'task'],
        schema_version: 'vla-urban-3',
        created: new Date().toISOString(),
      },
      samples: this.samples.slice(),
    };
  }

  download(): void {
    const dataset = this.exportDataset();
    const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vla_urban_dataset_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  drawPreview(targetCanvas: HTMLCanvasElement): void {
    const ctx = targetCanvas.getContext('2d');
    if (!ctx) return;
    this.captureFrontImage();
    ctx.drawImage(this.canvas, 0, 0, targetCanvas.width, targetCanvas.height);
  }

  dispose(): void {
    this.stop();
    this.renderTarget.dispose();
    this.bevTarget.dispose();
  }

  captureFrontImage(): string {
    return this.renderCameraToBase64(this.frontCamera, this.renderTarget);
  }

  private renderCameraToBase64(camera: THREE.Camera, target: THREE.WebGLRenderTarget): string {
    const currentTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, camera);
    this.renderer.setRenderTarget(currentTarget);
    this.renderer.readRenderTargetPixels(target, 0, 0, this.captureSize, this.captureSize, this.pixels);

    const imageData = this.ctx.createImageData(this.captureSize, this.captureSize);
    for (let y = 0; y < this.captureSize; y++) {
      const srcRow = (this.captureSize - 1 - y) * this.captureSize * 4;
      const dstRow = y * this.captureSize * 4;
      for (let x = 0; x < this.captureSize * 4; x++) {
        imageData.data[dstRow + x] = this.pixels[srcRow + x];
      }
    }
    this.ctx.putImageData(imageData, 0, 0);
    return this.canvas.toDataURL('image/png');
  }
}

export function validateDataset(dataset: DatasetExport): boolean {
  if (!dataset.metadata || !Array.isArray(dataset.samples)) return false;
  return dataset.samples.every((sample) => (
    typeof sample.image === 'string'
    && typeof sample.language_id === 'number'
    && (sample.run_mode === 'training' || sample.run_mode === 'inference')
    && [64, 128, 256].includes(sample.capture_resolution)
    && sample.actions.forward !== undefined
    && typeof sample.control.steer === 'number'
    && typeof sample.ego.x === 'number'
    && typeof sample.events.collision === 'boolean'
    && typeof sample.events.goalReached === 'boolean'
    && typeof sample.task?.routeProgress === 'number'
    && typeof sample.task?.distanceToDestination === 'number'
  ));
}
