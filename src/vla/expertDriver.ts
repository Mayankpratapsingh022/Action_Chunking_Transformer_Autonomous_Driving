import * as THREE from 'three';
import type { ActorState, ControlCommand, RoadGraph, TrafficLightState } from '../types';
import { clamp, wrapPi } from '../utils/rng';
import { pointHeadingAtRoute } from '../world/roadGraph';

export interface ExpertDriverOptions {
  speedScale?: number;
  followingDistanceScale?: number;
  steeringGain?: number;
}

export class ExpertDriver {
  private progressIndex = 0;
  private recovering = false;
  private stopReason: 'forward_hazard' | 'traffic_light' | 'crossing_conflict' | null = null;

  constructor(
    private readonly graph: RoadGraph,
    private readonly options: ExpertDriverOptions = {},
  ) {}

  reset(routeIndex = 0): void {
    this.progressIndex = Math.max(0, Math.min(this.graph.route.length - 1, Math.round(routeIndex)));
    this.recovering = false;
  }

  get routeIndex(): number {
    return this.progressIndex;
  }

  get isRecovering(): boolean {
    return this.recovering;
  }

  get lastStopReason(): string | null {
    return this.stopReason;
  }

  compute(
    egoPosition: THREE.Vector3,
    egoHeading: number,
    egoSpeed: number,
    actors: ActorState[],
    lights: TrafficLightState[],
  ): ControlCommand {
    this.stopReason = null;
    const nearest = this.nearestProgressPoint(egoPosition);
    if (nearest.index > this.progressIndex) this.progressIndex = nearest.index;
    const currentHeading = pointHeadingAtRoute(this.graph.route, nearest.index);
    const routeHeadingError = Math.abs(wrapPi(currentHeading - egoHeading));
    this.recovering = nearest.distance > 3.2 || routeHeadingError > 0.38;
    const lookaheadSteps = this.recovering
      ? Math.floor(7 + Math.abs(egoSpeed) * 1.2)
      : Math.floor(12 + Math.abs(egoSpeed) * 2.2);
    const lookaheadIndex = Math.min(this.graph.route.length - 1, nearest.index + lookaheadSteps);
    const target = this.graph.route[lookaheadIndex];
    const desiredHeading = pointHeadingAtRoute(this.graph.route, lookaheadIndex);
    const angleToTarget = Math.atan2(target.x - egoPosition.x, target.z - egoPosition.z);
    const headingError = wrapPi(angleToTarget - egoHeading) * 0.65 + wrapPi(desiredHeading - egoHeading) * 0.35;
    const steeringGain = this.options.steeringGain ?? 1;
    let steer = clamp(-headingError * 1.75 * steeringGain, -1, 1);

    const hazard = this.closestForwardHazard(egoPosition, egoHeading, actors);
    const crossingConflict = this.crossingConflict(egoPosition, egoHeading, egoSpeed, actors);
    const redLight = this.redLightAhead(egoPosition, egoHeading, lights);
    const routeError = nearest.distance;

    const speedScale = this.options.speedScale ?? 1;
    const followingScale = this.options.followingDistanceScale ?? 1;
    let targetSpeed = 10.5 * speedScale;
    if (routeError > 5) targetSpeed = 5.2 * speedScale;
    if (routeError > 9 || routeHeadingError > 0.75) targetSpeed = 3.4 * speedScale;
    if (Math.abs(headingError) > 0.55) targetSpeed = Math.min(targetSpeed, 6.2 * speedScale);
    if (hazard.distance < 15 * followingScale) targetSpeed = Math.min(targetSpeed, 3.2);
    if (hazard.distance < 8 * followingScale) {
      targetSpeed = 0;
      this.stopReason = 'forward_hazard';
    }
    if (redLight) {
      targetSpeed = 0;
      this.stopReason = 'traffic_light';
    }
    if (crossingConflict < 4.5) targetSpeed = Math.min(targetSpeed, 3.2);
    if (crossingConflict < 2.8) {
      targetSpeed = 0;
      this.stopReason = 'crossing_conflict';
    }

    if (hazard.side !== 0 && hazard.distance < 22) {
      steer += hazard.side > 0 ? 0.35 : -0.35;
    }

    const speedError = targetSpeed - egoSpeed;
    return {
      throttle: speedError > 0.8 ? clamp(speedError / 8, 0, 1) : 0,
      brake: speedError < -0.8 ? clamp(-speedError / 8, 0, 1) : 0,
      steer: clamp(steer, -1, 1),
    };
  }

  private nearestProgressPoint(position: THREE.Vector3): { index: number; distance: number } {
    const route = this.graph.route;
    const start = Math.max(0, this.progressIndex - 14);
    const end = Math.min(route.length - 1, this.progressIndex + 90);
    let bestIndex = this.progressIndex;
    let bestDistance = Infinity;
    for (let index = start; index <= end; index++) {
      const distance = route[index].distanceTo(position);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    return { index: bestIndex, distance: bestDistance };
  }

  private closestForwardHazard(
    egoPosition: THREE.Vector3,
    egoHeading: number,
    actors: ActorState[],
  ): { distance: number; side: number } {
    const forward = new THREE.Vector3(Math.sin(egoHeading), 0, Math.cos(egoHeading));
    const right = new THREE.Vector3(Math.cos(egoHeading), 0, -Math.sin(egoHeading));
    let best = { distance: Infinity, side: 0 };
    for (const actor of actors) {
      const delta = actor.position.clone().sub(egoPosition);
      const longitudinal = delta.dot(forward);
      const lateral = delta.dot(right);
      // A vehicle that has already moved beside or behind the ego must not
      // deadlock the expert after an overtake.
      if (longitudinal < 1.5 || longitudinal > 38) continue;
      if (Math.abs(lateral) > actor.radius + 2.7) continue;
      if (longitudinal < best.distance) best = { distance: longitudinal, side: Math.sign(lateral) };
    }
    return best;
  }

  private redLightAhead(
    egoPosition: THREE.Vector3,
    egoHeading: number,
    lights: TrafficLightState[],
  ): boolean {
    const forward = new THREE.Vector3(Math.sin(egoHeading), 0, Math.cos(egoHeading));
    const right = new THREE.Vector3(Math.cos(egoHeading), 0, -Math.sin(egoHeading));
    let closest: { distance: number; state: TrafficLightState['state'] } | null = null;
    for (const light of lights) {
      const delta = light.position.clone().sub(egoPosition);
      const longitudinal = delta.dot(forward);
      const lateral = Math.abs(delta.dot(right));
      if (longitudinal < 3 || longitudinal > 30 || lateral > 7.5) continue;
      if (!closest || longitudinal < closest.distance) closest = { distance: longitudinal, state: light.state };
    }
    return closest !== null && closest.state !== 'green';
  }

  private crossingConflict(
    egoPosition: THREE.Vector3,
    egoHeading: number,
    egoSpeed: number,
    actors: ActorState[],
  ): number {
    if (egoSpeed < 0.2) return Infinity;
    const egoForward = new THREE.Vector3(Math.sin(egoHeading), 0, Math.cos(egoHeading));
    let earliest = Infinity;
    for (const actor of actors) {
      if (actor.id.startsWith('npc_') || actor.speed <= 0 || actor.type === 'pedestrian' || actor.type === 'cyclist') continue;
      const actorForward = new THREE.Vector3(Math.sin(actor.heading), 0, Math.cos(actor.heading));
      for (let time = 0.5; time <= 5; time += 0.25) {
        const predictedEgo = egoPosition.clone().addScaledVector(egoForward, Math.max(0, egoSpeed) * time);
        const predictedActor = actor.position.clone().addScaledVector(actorForward, actor.speed * time);
        if (predictedEgo.distanceTo(predictedActor) < actor.radius + 3.4) {
          earliest = Math.min(earliest, time);
          break;
        }
      }
    }
    return earliest;
  }
}
