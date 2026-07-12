import { spawnSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { parseArgs, integerArg } from './lib/cli.mjs';
import { CollectionState } from './lib/state-db.mjs';

const args = parseArgs(process.argv.slice(2));
const episodes = integerArg(args.episodes, 3, { min: 1, max: 20 });
const projectRoot = resolve(import.meta.dirname, '..');
const results = [];

for (const workers of [1, 2]) {
  const root = await mkdtemp(resolve(tmpdir(), `vla-benchmark-${workers}-`));
  const start = Date.now();
  const result = spawnSync(process.execPath, [
    '--no-warnings', resolve(import.meta.dirname, 'dataset-supervisor.mjs'),
    '--root', root, '--workers', String(workers), '--limit', String(episodes), '--kinds', 'nominal',
  ], { cwd: projectRoot, stdio: 'inherit' });
  const wallMs = Date.now() - start;
  const state = new CollectionState(resolve(root, 'state/collection.sqlite'));
  const summary = state.summary();
  const episodeTimes = state.recentWallTimes(episodes);
  state.close();
  results.push({
    workers,
    requested: episodes,
    accepted: summary.accepted,
    totalWallSeconds: wallMs / 1000,
    averageEpisodeSeconds: episodeTimes.length
      ? episodeTimes.reduce((sum, value) => sum + value, 0) / episodeTimes.length / 1000
      : null,
    projectedHoursFor1170: summary.accepted
      ? (wallMs / summary.accepted) * 1170 / 3_600_000
      : null,
    exitCode: result.status,
  });
  if (!args.keep) await rm(root, { recursive: true, force: true });
}

console.log(JSON.stringify(results, null, 2));
const [one, two] = results;
if (one.accepted && two.accepted) {
  const speedup = one.totalWallSeconds / two.totalWallSeconds;
  console.log(`Two-worker speedup: ${speedup.toFixed(2)}x`);
  console.log(speedup >= 1.35
    ? 'Recommendation: use two workers when the Mac is idle.'
    : 'Recommendation: use one worker; the second worker adds contention.');
}
