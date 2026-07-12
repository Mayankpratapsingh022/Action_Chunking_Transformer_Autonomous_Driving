import * as THREE from 'three';
import type { RenderQuality } from '../types';

export type ActiveQuality = Exclude<RenderQuality, 'auto'>;

export interface RenderQualityProfile {
  id: ActiveQuality;
  pixelRatioCap: number;
  shadowMapSize: number;
  shadows: boolean;
  sensorPreviewFps: number;
  worldDetail: number;
  rainDrops: number;
}

export interface DeviceCapability {
  coarsePointer: boolean;
  deviceMemory?: number;
  hardwareConcurrency?: number;
  viewportWidth: number;
}

export const QUALITY_PROFILES: Record<ActiveQuality, RenderQualityProfile> = {
  high: {
    id: 'high',
    pixelRatioCap: 2,
    shadowMapSize: 1024,
    shadows: true,
    sensorPreviewFps: 12,
    worldDetail: 1,
    rainDrops: 720,
  },
  balanced: {
    id: 'balanced',
    pixelRatioCap: 1.5,
    shadowMapSize: 768,
    shadows: true,
    sensorPreviewFps: 10,
    worldDetail: 0.78,
    rainDrops: 480,
  },
  low: {
    id: 'low',
    pixelRatioCap: 1,
    shadowMapSize: 0,
    shadows: false,
    sensorPreviewFps: 6,
    worldDetail: 0.5,
    rainDrops: 260,
  },
};

export function chooseAutomaticQuality(capability: DeviceCapability): ActiveQuality {
  const memory = capability.deviceMemory ?? 4;
  const cores = capability.hardwareConcurrency ?? 4;
  if (capability.coarsePointer || capability.viewportWidth < 760 || memory <= 3 || cores <= 4) return 'low';
  if (memory >= 8 && cores >= 8 && capability.viewportWidth >= 1280) return 'high';
  return 'balanced';
}

export function profileForQuality(
  quality: RenderQuality,
  capability: DeviceCapability = readDeviceCapability(),
): RenderQualityProfile {
  return QUALITY_PROFILES[quality === 'auto' ? chooseAutomaticQuality(capability) : quality];
}

export function readDeviceCapability(): DeviceCapability {
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    coarsePointer: window.matchMedia('(pointer: coarse)').matches,
    deviceMemory: nav.deviceMemory,
    hardwareConcurrency: navigator.hardwareConcurrency,
    viewportWidth: window.innerWidth,
  };
}

export class RenderQualityManager {
  private quality: RenderQuality = 'auto';
  private profile = profileForQuality('auto');
  private lowFpsWindows = 0;
  private highFpsWindows = 0;

  constructor(private readonly renderer: THREE.WebGLRenderer) {
    this.applyRendererProfile();
  }

  get mode(): RenderQuality {
    return this.quality;
  }

  get current(): RenderQualityProfile {
    return this.profile;
  }

  setMode(quality: RenderQuality): boolean {
    this.quality = quality;
    const next = profileForQuality(quality);
    const changed = next.id !== this.profile.id;
    this.profile = next;
    this.lowFpsWindows = 0;
    this.highFpsWindows = 0;
    this.applyRendererProfile();
    return changed;
  }

  reportFps(fps: number): boolean {
    if (this.quality !== 'auto' || fps <= 0) return false;
    this.lowFpsWindows = fps < 48 ? this.lowFpsWindows + 1 : 0;
    this.highFpsWindows = fps > 74 ? this.highFpsWindows + 1 : 0;

    let next: ActiveQuality | null = null;
    if (this.lowFpsWindows >= 4) {
      next = this.profile.id === 'high' ? 'balanced' : this.profile.id === 'balanced' ? 'low' : null;
    } else if (this.highFpsWindows >= 12) {
      next = this.profile.id === 'low' ? 'balanced' : null;
    }
    if (!next) return false;

    this.profile = QUALITY_PROFILES[next];
    this.lowFpsWindows = 0;
    this.highFpsWindows = 0;
    this.applyRendererProfile();
    return true;
  }

  resize(width: number, height: number): void {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, this.profile.pixelRatioCap));
    this.renderer.setSize(width, height);
  }

  private applyRendererProfile(): void {
    this.renderer.shadowMap.enabled = this.profile.shadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.resize(window.innerWidth, window.innerHeight);
  }
}
