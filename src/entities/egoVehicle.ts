import * as THREE from 'three';
import type { AssetLibrary } from '../assets/assetLibrary';
import type { ControlCommand, EgoState, RoadGraph, SimEvents } from '../types';
import { clamp, wrapPi } from '../utils/rng';
import { nearestRouteIndex, pointHeadingAtRoute } from '../world/roadGraph';

const MAX_SPEED = 24;
const MAX_REVERSE = 8;
const ACCEL = 18;
const REVERSE_ACCEL = 10;
const BRAKE = 22;
const DRAG = 1.7;
const MAX_STEER_RATE = 2.55;
const STEER_RESPONSE = 7.2;
const INPUT_RISE_RATE = 4.2;
const INPUT_FALL_RATE = 5.6;
const STEER_INPUT_RATE = 6.2;

export class EgoVehicle {
  readonly group = new THREE.Group();
  readonly collisionBox = new THREE.Box3();
  position = new THREE.Vector3();
  heading = 0;
  speed = 0;
  steering = 0;
  currentLaneId = 'route';
  events: SimEvents = {
    collision: false,
    offRoute: false,
    redLightViolation: false,
    goalReached: false,
    episodeDone: false,
  };

  private readonly keyState = new Set<string>();
  private readonly onKeyDown = (event: KeyboardEvent) => this.keyState.add(event.code);
  private readonly onKeyUp = (event: KeyboardEvent) => this.keyState.delete(event.code);
  private lastManual: ControlCommand = { throttle: 0, brake: 0, steer: 0 };
  private smoothedManual: ControlCommand = { throttle: 0, brake: 0, steer: 0 };
  private lastManualTime = 0;
  private readonly previous = new THREE.Vector3();

  constructor(private readonly graph: RoadGraph, private readonly assets?: AssetLibrary) {
    this.buildMesh();
    this.reset();
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
  }

  addTo(scene: THREE.Scene): void {
    scene.add(this.group);
  }

  reset(): void {
    this.position.copy(this.graph.start);
    this.heading = this.graph.startHeading;
    this.speed = 0;
    this.steering = 0;
    this.clearControlState();
    this.events = {
      collision: false,
      offRoute: false,
      redLightViolation: false,
      goalReached: false,
      episodeDone: false,
    };
    this.syncMesh();
  }

  respawnOnRoute(index: number): void {
    const safeIndex = Math.max(0, Math.min(this.graph.route.length - 1, Math.round(index)));
    this.position.copy(this.graph.route[safeIndex]);
    this.heading = pointHeadingAtRoute(this.graph.route, safeIndex);
    this.speed = 0;
    this.steering = 0;
    this.clearControlState();
    this.events = {
      collision: false,
      offRoute: false,
      redLightViolation: false,
      goalReached: false,
      episodeDone: false,
    };
    this.syncMesh();
  }

  setPose(position: THREE.Vector3, heading: number, speed = 0, steering = 0): void {
    this.position.copy(position);
    this.heading = wrapPi(heading);
    this.speed = clamp(speed, -MAX_REVERSE, MAX_SPEED);
    this.steering = clamp(steering, -1, 1);
    this.clearControlState();
    this.events = {
      collision: false,
      offRoute: false,
      redLightViolation: false,
      goalReached: false,
      episodeDone: false,
    };
    this.syncMesh();
  }

  manualCommand(): ControlCommand {
    const now = performance.now() / 1000;
    const dt = this.lastManualTime > 0 ? Math.min(now - this.lastManualTime, 0.08) : 1 / 60;
    this.lastManualTime = now;

    const targetThrottle = this.keyState.has('ArrowUp') || this.keyState.has('KeyW') ? 1 : 0;
    const targetBrake = this.keyState.has('Space') || this.keyState.has('ArrowDown') || this.keyState.has('KeyS') ? 1 : 0;
    const targetSteer = (this.keyState.has('ArrowLeft') || this.keyState.has('KeyA') ? -1 : 0)
      + (this.keyState.has('ArrowRight') || this.keyState.has('KeyD') ? 1 : 0);

    this.smoothedManual.throttle = approach(
      this.smoothedManual.throttle,
      targetThrottle,
      targetThrottle > this.smoothedManual.throttle ? INPUT_RISE_RATE : INPUT_FALL_RATE,
      dt,
    );
    this.smoothedManual.brake = approach(
      this.smoothedManual.brake,
      targetBrake,
      targetBrake > this.smoothedManual.brake ? INPUT_RISE_RATE : INPUT_FALL_RATE,
      dt,
    );
    this.smoothedManual.steer = approach(this.smoothedManual.steer, targetSteer, STEER_INPUT_RATE, dt);

    const throttle = Math.abs(this.smoothedManual.throttle) < 0.02 ? 0 : this.smoothedManual.throttle;
    const brake = Math.abs(this.smoothedManual.brake) < 0.02 ? 0 : this.smoothedManual.brake;
    const steer = Math.abs(this.smoothedManual.steer) < 0.02 ? 0 : this.smoothedManual.steer;
    this.lastManual = { throttle, brake, steer };
    return this.lastManual;
  }

  setVirtualControl(code: 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight', pressed: boolean): void {
    if (pressed) this.keyState.add(code);
    else this.keyState.delete(code);
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    this.keyState.clear();
  }

  update(dt: number, command: ControlCommand, colliders: { box: THREE.Box3 }[]): void {
    if (dt <= 0 || dt > 0.1) return;
    this.previous.copy(this.position);
    this.events = {
      collision: false,
      offRoute: false,
      redLightViolation: false,
      goalReached: this.events.goalReached,
      episodeDone: this.events.episodeDone,
    };

    const targetSteer = clamp(command.steer, -1, 1);
    const steerBlend = 1 - Math.exp(-STEER_RESPONSE * dt);
    this.steering += (targetSteer - this.steering) * steerBlend;

    if (command.throttle > 0) this.speed += ACCEL * command.throttle * dt;
    if (command.brake > 0) {
      if (this.speed > 0.5) this.speed -= BRAKE * command.brake * dt;
      else this.speed -= REVERSE_ACCEL * command.brake * dt;
    }
    if (command.throttle === 0 && command.brake === 0) {
      const drag = DRAG * dt;
      this.speed = Math.abs(this.speed) <= drag ? 0 : this.speed - Math.sign(this.speed) * drag;
    }

    this.speed = clamp(this.speed, -MAX_REVERSE, MAX_SPEED);
    const speedFactor = clamp(Math.abs(this.speed) / 8, 0, 1);
    const lowSpeedTurn = 0.28 + speedFactor * 0.72;
    this.heading = wrapPi(this.heading - this.steering * MAX_STEER_RATE * lowSpeedTurn * dt);
    this.position.x += Math.sin(this.heading) * this.speed * dt;
    this.position.z += Math.cos(this.heading) * this.speed * dt;

    this.updateCollisionBox();
    for (const collider of colliders) {
      if (this.collisionBox.intersectsBox(collider.box)) {
        this.events.collision = true;
        this.position.copy(this.previous);
        this.speed *= -0.2;
        break;
      }
    }

    const nearest = nearestRouteIndex(this.graph.route, this.position);
    this.events.offRoute = nearest.distance > 12;
    this.currentLaneId = `route_${Math.floor(nearest.index / 20)}`;
    this.syncMesh();
  }

  setTaskEvents(goalReached: boolean, episodeDone: boolean): void {
    this.events.goalReached = goalReached;
    this.events.episodeDone = episodeDone;
  }

  getState(): EgoState {
    return {
      x: this.position.x,
      z: this.position.z,
      heading: this.heading,
      speed: this.speed / MAX_SPEED,
      steering: this.steering,
      laneId: this.currentLaneId,
    };
  }

  commandToActions(command: ControlCommand) {
    return {
      forward: command.throttle > 0.2 ? 1 : 0,
      backward: this.speed < -0.5 ? 1 : 0,
      left: command.steer < -0.18 ? 1 : 0,
      right: command.steer > 0.18 ? 1 : 0,
    } as const;
  }

  getFrontCameraPose(): { position: THREE.Vector3; lookAt: THREE.Vector3 } {
    const eye = new THREE.Vector3(0, 2.2, 2.3).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.heading).add(this.position);
    const target = new THREE.Vector3(0, 0.9, 20).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.heading).add(this.position);
    return { position: eye, lookAt: target };
  }

  getChaseCameraPose(): { position: THREE.Vector3; lookAt: THREE.Vector3 } {
    const pos = new THREE.Vector3(0, 12, -20).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.heading).add(this.position);
    const target = new THREE.Vector3(0, 1, 8).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.heading).add(this.position);
    return { position: pos, lookAt: target };
  }

  getAutonomyCameraPose(): { position: THREE.Vector3; lookAt: THREE.Vector3 } {
    const routeIndex = nearestRouteIndex(this.graph.route, this.position).index;
    const plannedHeading = pointHeadingAtRoute(this.graph.route, routeIndex);
    const pos = new THREE.Vector3(0, 42, -42).applyAxisAngle(new THREE.Vector3(0, 1, 0), plannedHeading).add(this.position);
    const lookAt = new THREE.Vector3(0, 1.1, 14).applyAxisAngle(new THREE.Vector3(0, 1, 0), plannedHeading).add(this.position);
    return { position: pos, lookAt };
  }

  getBevCameraPose(): { position: THREE.Vector3; lookAt: THREE.Vector3 } {
    return { position: this.position.clone().add(new THREE.Vector3(0, 115, 0.01)), lookAt: this.position.clone() };
  }

  private buildMesh(): void {
    const assetCar = this.assets?.createVehicle('sedan', 'ego');
    if (assetCar) {
      this.group.add(assetCar);
      const marker = new THREE.Mesh(
        new THREE.CircleGeometry(0.34, 24),
        new THREE.MeshBasicMaterial({ color: 0x22c7ff, transparent: true, opacity: 0.85 }),
      );
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(0, 0.08, 2.55);
      this.group.add(marker);
      return;
    }

    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.38, metalness: 0.08 });
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x1d2731, roughness: 0.16, metalness: 0.25 });
    const accentMat = new THREE.MeshBasicMaterial({ color: 0x111827 });
    const blueMat = new THREE.MeshBasicMaterial({ color: 0x22c7ff, transparent: true, opacity: 0.8 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(2.35, 0.82, 4.75), bodyMat);
    body.position.y = 0.62;
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.72, 2.15), glassMat);
    cabin.position.set(0, 1.22, -0.35);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(1.38, 0.08, 1.1), accentMat);
    roof.position.set(0, 1.64, -0.38);
    const pathEmitter = new THREE.Mesh(new THREE.CircleGeometry(0.32, 18), blueMat);
    pathEmitter.rotation.x = -Math.PI / 2;
    pathEmitter.position.set(0, 0.08, 2.45);
    this.group.add(body, cabin, roof, pathEmitter);

    const wheelMat = new THREE.MeshBasicMaterial({ color: 0x222831 });
    for (const x of [-1.28, 1.28]) {
      for (const z of [-1.45, 1.45]) {
        const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.22, 16), wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(x, 0.34, z);
        this.group.add(wheel);
      }
    }
  }

  private updateCollisionBox(): void {
    this.collisionBox.setFromCenterAndSize(
      new THREE.Vector3(this.position.x, 0.9, this.position.z),
      new THREE.Vector3(2.12, 1.35, 4.35),
    );
  }

  private syncMesh(): void {
    this.group.position.copy(this.position);
    this.group.rotation.y = this.heading;
    this.updateCollisionBox();
  }

  private clearControlState(): void {
    this.smoothedManual = { throttle: 0, brake: 0, steer: 0 };
    this.lastManual = { throttle: 0, brake: 0, steer: 0 };
    this.lastManualTime = 0;
  }
}

function approach(current: number, target: number, rate: number, dt: number): number {
  const delta = target - current;
  const maxStep = rate * dt;
  if (Math.abs(delta) <= maxStep) return target;
  return current + Math.sign(delta) * maxStep;
}
