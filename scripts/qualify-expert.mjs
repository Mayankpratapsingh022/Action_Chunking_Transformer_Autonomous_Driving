import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { parseArgs, integerArg } from './lib/cli.mjs';
import { TASKS } from './lib/dataset-config.mjs';
import { CollectionState } from './lib/state-db.mjs';

const args = parseArgs(process.argv.slice(2));
const perTask = integerArg(args.episodesPerTask, 5, { min: 1, max: 50 });
const workers = integerArg(args.workers, 2, { min: 1, max: 4 });
const projectRoot = resolve(import.meta.dirname, '..');
const root = await mkdtemp(resolve(tmpdir(), 'vla-expert-qualification-'));

const generation = spawnSync(process.execPath, [
  resolve(import.meta.dirname, 'generate-dataset-manifest.mjs'), '--root', root,
], { cwd: projectRoot, encoding: 'utf8' });
if (generation.status !== 0) throw new Error(generation.stderr || generation.stdout);
const nominalPath = resolve(root, 'manifests/nominal.jsonl');
const allNominal = (await readFile(nominalPath, 'utf8')).split('\n').filter(Boolean).map(JSON.parse);
const selectedTasks = args.task ? TASKS.filter((task) => task.id === args.task) : TASKS;
if (selectedTasks.length === 0) throw new Error(`Unknown task: ${args.task}`);
const selected = selectedTasks.flatMap((task) => allNominal.filter((row) => row.taskId === task.id).slice(0, perTask));
await writeFile(nominalPath, `${selected.map((row) => JSON.stringify(row)).join('\n')}\n`);
await writeFile(resolve(root, 'manifests/recovery.jsonl'), '');
await writeFile(resolve(root, 'manifests/failures.jsonl'), '');

const result = spawnSync(process.execPath, [
  '--no-warnings', resolve(import.meta.dirname, 'dataset-supervisor.mjs'),
  '--root', root, '--workers', String(workers), '--kinds', 'nominal',
], { cwd: projectRoot, stdio: 'inherit' });
const state = new CollectionState(resolve(root, 'state/collection.sqlite'));
const summary = state.summary();
state.close();
const successRate = selected.length ? summary.accepted / selected.length : 0;
console.log(`Expert qualification: ${summary.accepted}/${selected.length} (${(successRate * 100).toFixed(1)}%)`);
if (summary.failed > 0 || successRate < 0.98 || result.status !== 0) {
  console.error(`Qualification did not reach 98%. Artifacts retained at ${root}`);
  process.exitCode = 1;
} else if (args.keep) {
  console.log(`Qualification artifacts retained at ${root}`);
} else {
  await rm(root, { recursive: true, force: true });
}
