import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  DATASET_COUNTS,
  EXPERT_PROFILES,
  FAILURE_LABELS,
  RECOVERY_TYPES,
  TASKS,
} from './lib/dataset-config.mjs';

const args = parseArgs(process.argv.slice(2));
const root = resolve(args.root ?? process.env.VLA_DATASET_ROOT ?? 'datasets/urban-vla-expert-v1');
const manifestDir = resolve(root, 'manifests');
await mkdir(manifestDir, { recursive: true });

const nominal = TASKS.flatMap((task) => Array.from(
  { length: DATASET_COUNTS.nominalPerTask },
  (_, index) => makeEpisode(task, 'nominal', index),
));
const recovery = TASKS.flatMap((task) => Array.from(
  { length: DATASET_COUNTS.recoveryPerTask },
  (_, index) => makeEpisode(task, 'recovery', index),
));
const failures = TASKS.flatMap((task) => Array.from(
  { length: DATASET_COUNTS.failurePerTask },
  (_, index) => makeEpisode(task, 'failure', index),
));

await Promise.all([
  writeJsonl(resolve(manifestDir, 'nominal.jsonl'), nominal),
  writeJsonl(resolve(manifestDir, 'recovery.jsonl'), recovery),
  writeJsonl(resolve(manifestDir, 'failures.jsonl'), failures),
]);

console.log(`Generated ${nominal.length + recovery.length + failures.length} episodes in ${manifestDir}`);
console.log(`  nominal: ${nominal.length}`);
console.log(`  recovery: ${recovery.length}`);
console.log(`  failures: ${failures.length}`);

function makeEpisode(task, kind, index) {
  const split = splitFor(kind, index);
  const seedIndex = splitSeedIndex(kind, split, index);
  const seedKey = `${task.seedGroup ?? task.id}:${kind}:${split}:${seedIndex}`;
  const worldSeed = stableSeed(seedKey);
  const allowedPhrases = split === 'train'
    ? task.paraphrases.slice(0, 12)
    : split === 'validation'
      ? task.paraphrases.slice(12, 16)
      : task.paraphrases.slice(16, 20);
  const instructionIndex = allowedPhrases.length === 0 ? 0 : index % allowedPhrases.length;
  const instruction = allowedPhrases[instructionIndex] ?? task.paraphrases[index % task.paraphrases.length];
  const conditionIndex = stableSeed(`${seedKey}:conditions`);
  const weather = ['clear', 'rain', 'fog'][conditionIndex % 3];
  const trafficDensity = ['low', 'medium', 'high'][Math.floor(conditionIndex / 3) % 3];
  const expertProfile = EXPERT_PROFILES[Math.floor(conditionIndex / 9) % EXPERT_PROFILES.length];
  const id = `${kind}-${task.id}-${String(index).padStart(4, '0')}`;
  const base = {
    id,
    kind,
    split,
    taskId: task.id,
    instructionId: `${task.id}-${split}-${instructionIndex}`,
    instruction,
    scenario: task.scenario,
    routeVariant: task.routeVariant,
    worldSeed,
    trafficSeed: stableSeed(`${seedKey}:traffic`),
    weather,
    trafficDensity,
    expertProfile,
    maxDurationSeconds: kind === 'failure' ? 10 : kind === 'recovery' ? 52 : 60,
  };
  if (kind === 'recovery') {
    return {
      ...base,
      perturbation: {
        type: RECOVERY_TYPES[index % RECOVERY_TYPES.length],
        direction: index % 2 === 0 ? 'left' : 'right',
        severity: ['mild', 'medium', 'severe'][index % 3],
        triggerProgress: Number((0.14 + (index % 6) * 0.075).toFixed(3)),
      },
    };
  }
  if (kind === 'failure') {
    return {
      ...base,
      split: 'analysis',
      trafficDensity: 'low',
      failureLabel: FAILURE_LABELS[index % FAILURE_LABELS.length],
      perturbation: {
        type: 'road_edge',
        direction: index % 2 === 0 ? 'left' : 'right',
        severity: 'severe',
        triggerProgress: Number((0.08 + (index % 4) * 0.04).toFixed(3)),
      },
    };
  }
  return base;
}

function splitFor(kind, index) {
  if (kind === 'failure') return 'analysis';
  const total = kind === 'nominal' ? DATASET_COUNTS.nominalPerTask : DATASET_COUNTS.recoveryPerTask;
  if (index < Math.round(total * 0.7)) return 'train';
  if (index < Math.round(total * 0.85)) return 'validation';
  return 'test';
}

function splitSeedIndex(kind, split, index) {
  const offsets = kind === 'nominal'
    ? { train: 0, validation: 10_000, test: 20_000, analysis: 30_000 }
    : { train: 40_000, validation: 50_000, test: 60_000, analysis: 70_000 };
  return offsets[split] + index;
}

function stableSeed(value) {
  const hash = createHash('sha256').update(value).digest();
  return hash.readUInt32BE(0) & 0x7fffffff;
}

function parseArgs(values) {
  const parsed = {};
  for (let index = 0; index < values.length; index++) {
    if (values[index] === '--root') parsed.root = values[++index];
  }
  return parsed;
}

async function writeJsonl(path, rows) {
  await writeFile(path, `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`);
}
