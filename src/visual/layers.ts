import * as THREE from 'three';

export const WORLD_LAYER = 0;
export const PRESENTATION_LAYER = 1;

export function setPresentationLayer(root: THREE.Object3D): void {
  root.traverse((object) => object.layers.set(PRESENTATION_LAYER));
}

export function configurePresentationCamera(camera: THREE.Camera): void {
  camera.layers.enable(WORLD_LAYER);
  camera.layers.enable(PRESENTATION_LAYER);
}

export function configureSensorCamera(camera: THREE.Camera): void {
  camera.layers.set(WORLD_LAYER);
}
