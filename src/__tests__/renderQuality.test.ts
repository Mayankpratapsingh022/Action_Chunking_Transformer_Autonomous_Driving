import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
  PRESENTATION_LAYER,
  WORLD_LAYER,
  configurePresentationCamera,
  configureSensorCamera,
  setPresentationLayer,
} from '../visual/layers';
import { chooseAutomaticQuality, profileForQuality } from '../visual/renderQuality';

describe('render quality', () => {
  it('selects low quality for mobile and constrained devices', () => {
    expect(chooseAutomaticQuality({
      coarsePointer: true,
      deviceMemory: 8,
      hardwareConcurrency: 8,
      viewportWidth: 1280,
    })).toBe('low');
    expect(chooseAutomaticQuality({
      coarsePointer: false,
      deviceMemory: 2,
      hardwareConcurrency: 4,
      viewportWidth: 1440,
    })).toBe('low');
  });

  it('selects high quality only for capable desktop devices', () => {
    const capability = {
      coarsePointer: false,
      deviceMemory: 8,
      hardwareConcurrency: 10,
      viewportWidth: 1440,
    };
    expect(chooseAutomaticQuality(capability)).toBe('high');
    expect(profileForQuality('auto', capability).pixelRatioCap).toBe(2);
  });
});

describe('visual layers', () => {
  it('keeps presentation graphics out of sensor cameras', () => {
    const root = new THREE.Group();
    const child = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1));
    root.add(child);
    setPresentationLayer(root);

    const presentationCamera = new THREE.PerspectiveCamera();
    const sensorCamera = new THREE.PerspectiveCamera();
    configurePresentationCamera(presentationCamera);
    configureSensorCamera(sensorCamera);

    expect(child.layers.test(presentationCamera.layers)).toBe(true);
    expect(child.layers.test(sensorCamera.layers)).toBe(false);
    expect(sensorCamera.layers.isEnabled(WORLD_LAYER)).toBe(true);
    expect(sensorCamera.layers.isEnabled(PRESENTATION_LAYER)).toBe(false);
  });
});
