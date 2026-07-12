const wrappers = [
  (clause) => `${capitalize(clause)}.`,
  (clause) => `Please ${clause}.`,
  (clause) => `Safely ${clause}.`,
  (clause) => `Your driving instruction is to ${clause}.`,
];

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function phrases(clauses) {
  const values = clauses.flatMap((clause) => wrappers.map((wrap) => wrap(clause)));
  if (values.length !== 20 || new Set(values).size !== 20) {
    throw new Error('Each task must produce exactly 20 unique paraphrases');
  }
  return values;
}

export const TASKS = [
  {
    id: 'turn_left_intersection',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'left',
    seedGroup: 'shared-intersection',
    paraphrases: phrases([
      'turn left at the main intersection',
      'take the left-hand route through the upcoming intersection',
      'make a left turn when you reach the city intersection',
      'follow the road left at the central junction',
      'use the left exit from the intersection and continue to the destination',
    ]),
  },
  {
    id: 'continue_straight_intersection',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'straight',
    seedGroup: 'shared-intersection',
    paraphrases: phrases([
      'continue straight through the main intersection',
      'take the straight route at the upcoming intersection',
      'cross the city junction without turning',
      'keep going forward through the central intersection',
      'stay on the straight path at the intersection and continue to the destination',
    ]),
  },
  {
    id: 'turn_right_intersection',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'right',
    seedGroup: 'shared-intersection',
    paraphrases: phrases([
      'turn right at the main intersection',
      'take the right-hand route through the upcoming intersection',
      'make a right turn when you reach the city intersection',
      'follow the road right at the central junction',
      'use the right exit from the intersection and continue to the destination',
    ]),
  },
  {
    id: 'overtake_slow_vehicle',
    scenario: 'lane_change_overtake',
    routeVariant: 'default',
    paraphrases: phrases([
      'overtake the slow vehicle and return to your lane',
      'change lanes to pass the slower car before merging back',
      'pass the vehicle ahead while maintaining a safe gap',
      'move around the slow lead car and continue north',
      'complete a safe overtake and settle back into the travel lane',
    ]),
  },
  {
    id: 'yield_cut_in',
    scenario: 'cut_in_vehicle',
    routeVariant: 'default',
    paraphrases: phrases([
      'yield smoothly to the vehicle cutting into your lane',
      'make space for the car merging in front of you',
      'slow down safely when the nearby vehicle cuts in',
      'maintain your lane and yield to the merging car',
      'allow the cut-in vehicle to enter before continuing',
    ]),
  },
  {
    id: 'detour_blocked_lane',
    scenario: 'blocked_lane_detour',
    routeVariant: 'default',
    paraphrases: phrases([
      'detour around the blocked lane and merge back safely',
      'steer around the cones and barriers before returning to your lane',
      'avoid the road blockage and continue along the route',
      'change lanes to pass the construction obstruction',
      'navigate around the closed section without striking any barriers',
    ]),
  },
  {
    id: 'stop_pedestrians',
    scenario: 'pedestrian_crossing',
    routeVariant: 'default',
    paraphrases: phrases([
      'stop for pedestrians in the crossing before continuing',
      'yield to people crossing the road ahead',
      'bring the car to a safe stop at the occupied crosswalk',
      'wait for the pedestrians to clear your path',
      'approach the crossing carefully and continue only when it is clear',
    ]),
  },
  {
    id: 'obey_traffic_lights',
    scenario: 'traffic_light_stop_go',
    routeVariant: 'default',
    paraphrases: phrases([
      'obey the traffic signal and continue when it is safe',
      'stop for the red or yellow light before proceeding on green',
      'follow the signal through the upcoming intersection',
      'wait at the traffic light when required and then continue straight',
      'drive through the signal-controlled junction without violating the light',
    ]),
  },
  {
    id: 'follow_curved_loop',
    scenario: 'curved_loop_drive',
    routeVariant: 'default',
    paraphrases: phrases([
      'follow the curved road smoothly to the destination',
      'stay on the looping road and maintain a safe speed',
      'continue around the bend without leaving the lane',
      'track the curved route until you reach the goal',
      'drive around the loop while keeping the car centred',
    ]),
  },
];

export const EXPERT_PROFILES = [
  { id: 'cautious', speedScale: 0.86, followingDistanceScale: 1.15, steeringGain: 0.94 },
  { id: 'normal', speedScale: 1, followingDistanceScale: 1, steeringGain: 1 },
  { id: 'assertive', speedScale: 1.08, followingDistanceScale: 0.92, steeringGain: 1.04 },
];

export const RECOVERY_TYPES = [
  'road_edge',
  'lateral_offset',
  'heading_error',
  'overspeed',
  'late_steering',
  'aborted_lane_change',
  'close_following',
  'late_braking',
  'steering_oscillation',
  'intersection_misalignment',
];

export const FAILURE_LABELS = [
  'lane_departure',
  'wrong_route',
  'unsafe_overspeed',
  'offroad_timeout',
  'recovery_failed',
  'episode_timeout',
  'collision_vehicle',
  'collision_obstacle',
  'red_light_violation',
  'collision_pedestrian',
];

export const DATASET_COUNTS = {
  nominalPerTask: 100,
  recoveryPerTask: 20,
  failurePerTask: 10,
};
