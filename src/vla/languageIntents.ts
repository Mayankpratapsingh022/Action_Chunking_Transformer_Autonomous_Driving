import type { RouteVariant, ScenarioKind } from '../types';

export interface LanguageIntentOption {
  id: string;
  label: string;
  text: string;
  scenario: ScenarioKind;
  routeVariant: RouteVariant;
}

export const LANGUAGE_INTENTS: LanguageIntentOption[] = [
  {
    id: 'turn_left_intersection',
    label: 'Turn left at intersection',
    text: 'Proceed through the city and make the protected left turn at the main intersection.',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'left',
  },
  {
    id: 'continue_straight_intersection',
    label: 'Continue straight at intersection',
    text: 'Continue straight through the main intersection and proceed to the destination.',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'straight',
  },
  {
    id: 'turn_right_intersection',
    label: 'Turn right at intersection',
    text: 'Approach the main intersection and make a safe right turn toward the destination.',
    scenario: 'intersection_unprotected_left',
    routeVariant: 'right',
  },
  {
    id: 'overtake_slow_vehicle',
    label: 'Overtake slow vehicle',
    text: 'Change lanes, overtake the slow vehicle, return to lane, and continue north.',
    scenario: 'lane_change_overtake',
    routeVariant: 'default',
  },
  {
    id: 'yield_cut_in',
    label: 'Yield to cut-in',
    text: 'Maintain lane and yield smoothly if a vehicle cuts into your path.',
    scenario: 'cut_in_vehicle',
    routeVariant: 'default',
  },
  {
    id: 'detour_blocked_lane',
    label: 'Detour blocked lane',
    text: 'Detour around the blocked lane, avoid cones and barriers, then merge back safely.',
    scenario: 'blocked_lane_detour',
    routeVariant: 'default',
  },
  {
    id: 'stop_pedestrians',
    label: 'Stop for pedestrians',
    text: 'Drive south and stop for pedestrians crossing before continuing toward the destination.',
    scenario: 'pedestrian_crossing',
    routeVariant: 'default',
  },
  {
    id: 'obey_traffic_lights',
    label: 'Obey traffic lights',
    text: 'Follow traffic lights, stop on red or yellow when required, and continue straight when safe.',
    scenario: 'traffic_light_stop_go',
    routeVariant: 'default',
  },
  {
    id: 'follow_curved_loop',
    label: 'Follow curved loop',
    text: 'Follow the curved loop road smoothly and drive to the marked destination.',
    scenario: 'curved_loop_drive',
    routeVariant: 'default',
  },
];

export function languageIntentById(id: string): LanguageIntentOption {
  return LANGUAGE_INTENTS.find((intent) => intent.id === id) ?? LANGUAGE_INTENTS[0];
}

export function languageIntentForScenario(kind: ScenarioKind): LanguageIntentOption {
  return LANGUAGE_INTENTS.find((intent) => intent.scenario === kind) ?? LANGUAGE_INTENTS[0];
}

export function languageIntentIndex(id: string): number {
  return Math.max(0, LANGUAGE_INTENTS.findIndex((intent) => intent.id === id));
}
