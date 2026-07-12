import { spawn } from 'node:child_process';
import { closeSync, existsSync, openSync, readFileSync, statSync } from 'node:fs';
import { readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs, integerArg } from './lib/cli.mjs';
import { atomicWriteJson, datasetRoot, ensureDatasetDirectories, pathsFor, readJson } from './lib/dataset-paths.mjs';
import { formatProgress, progressSnapshot } from './lib/progress.mjs';
import { CollectionState } from './lib/state-db.mjs';

const [command = 'status', ...values] = process.argv.slice(2);
const args = parseArgs(values);
const root = datasetRoot(args.root);
const paths = pathsFor(root);

if (command === 'start') await start();
else if (command === 'status') await status(Boolean(args.watch));
else if (command === 'logs') await logs(Boolean(args.follow));
else if (command === 'stop') await stop();
else throw new Error(`Unknown dataset command: ${command}`);

async function start() {
  if (args.fresh) throw new Error('Refusing to remove existing progress. Use a different --root for a fresh collection.');
  await ensureDatasetDirectories(paths);
  await unlink(paths.stopFile).catch(() => {});
  const current = await readJson(paths.runFile, null);
  if (current && isProcessAlive(current.supervisorPid ?? current.launcherPid) && heartbeatFresh(current)) {
    throw new Error(`Collector is already running with PID ${current.supervisorPid ?? current.launcherPid}`);
  }
  await rotateLog();
  const workers = integerArg(args.workers, 1, { min: 1, max: 4 });
  const supervisor = resolve(import.meta.dirname, 'dataset-supervisor.mjs');
  const supervisorArgs = ['--no-warnings', supervisor, '--root', root, '--workers', String(workers)];
  if (args.limit) supervisorArgs.push('--limit', String(args.limit));
  if (args.kinds) supervisorArgs.push('--kinds', String(args.kinds));
  if (args.retryFailed) supervisorArgs.push('--retry-failed');
  const executable = process.platform === 'darwin' ? '/usr/bin/caffeinate' : process.execPath;
  const childArgs = process.platform === 'darwin'
    ? ['-dimsu', process.execPath, ...supervisorArgs]
    : supervisorArgs;
  const output = openSync(paths.logFile, 'a');
  const child = spawn(executable, childArgs, {
    cwd: resolve(import.meta.dirname, '..'),
    detached: true,
    stdio: ['ignore', output, output],
  });
  closeSync(output);
  child.unref();
  await atomicWriteJson(paths.runFile, {
    launcherPid: child.pid,
    state: 'starting',
    workers,
    startedAt: new Date().toISOString(),
    heartbeatAt: new Date().toISOString(),
    activeElapsedMs: 0,
    current: [],
  });
  console.log(`Dataset collector started in the background.`);
  console.log(`PID: ${child.pid}`);
  console.log(`Root: ${root}`);
  console.log(`Log: ${paths.logFile}`);
  console.log(`Workers: ${workers}`);
  console.log(`Status: npm run dataset:status -- --watch`);
  console.log(`Logs: npm run dataset:logs -- --follow`);
}

async function status(watch) {
  const render = async () => {
    const run = await readJson(paths.runFile, null);
    if (!existsSync(paths.database)) {
      return `Dataset has not been initialized.\nRoot: ${root}`;
    }
    const state = new CollectionState(paths.database);
    const summary = state.summary();
    const elapsedMs = Number(run?.activeElapsedMs ?? state.getMetadata('active_elapsed_ms', 0));
    const workers = Number(run?.workers ?? 1);
    const snapshot = progressSnapshot(summary, state.recentWallTimes(), workers, elapsedMs);
    state.close();
    const pid = run?.supervisorPid ?? run?.launcherPid;
    const alive = isProcessAlive(pid) && heartbeatFresh(run);
    const current = run?.current?.map((entry) => `${entry.worker}: ${entry.episode}`).join(', ') || null;
    const detail = formatProgress({
      summary,
      snapshot,
      elapsedMs,
      workers,
      current,
      heartbeatAt: run?.heartbeatAt,
      state: alive ? run?.state ?? 'running' : run?.state ?? 'stopped',
    });
    return `State: ${alive ? run?.state ?? 'running' : run?.state === 'complete' ? 'complete' : 'not running'}\nPID: ${pid ?? 'n/a'}\n${detail}`;
  };
  if (!watch) {
    console.log(await render());
    return;
  }
  const draw = async () => {
    process.stdout.write(`\x1b[2J\x1b[H${await render()}\n\nPress Ctrl+C to stop watching; collection will continue.\n`);
  };
  await draw();
  const timer = setInterval(() => void draw(), 5_000);
  process.on('SIGINT', () => {
    clearInterval(timer);
    process.exit(0);
  });
  await new Promise(() => {});
}

async function logs(follow) {
  if (!existsSync(paths.logFile)) {
    console.log(`No collector log exists at ${paths.logFile}`);
    return;
  }
  const printTail = async (from = null) => {
    const content = await readFile(paths.logFile, 'utf8');
    if (from === null) {
      const lines = content.split('\n');
      console.log(lines.slice(-100).join('\n'));
    } else if (content.length > from) {
      process.stdout.write(content.slice(from));
    }
    return content.length;
  };
  let offset = await printTail();
  if (!follow) return;
  const timer = setInterval(async () => { offset = await printTail(offset); }, 1_000);
  process.on('SIGINT', () => {
    clearInterval(timer);
    process.exit(0);
  });
  await new Promise(() => {});
}

async function stop() {
  const run = await readJson(paths.runFile, null);
  const pid = run?.supervisorPid ?? run?.launcherPid;
  if (!isProcessAlive(pid)) {
    console.log('Dataset collector is not running. Starting it again will resume pending episodes.');
    return;
  }
  if (args.force) {
    process.kill(pid, 'SIGTERM');
    console.log(`Force signal sent to PID ${pid}. The active partial episode will be retried on resume.`);
    return;
  }
  await writeFile(paths.stopFile, `${new Date().toISOString()}\n`);
  await atomicWriteJson(paths.runFile, { ...run, state: 'stopping', heartbeatAt: new Date().toISOString() });
  console.log(`Graceful stop requested for PID ${pid}. Waiting for the active episode to finish...`);
  const deadline = Date.now() + 60_000;
  let stopped = false;
  while (Date.now() < deadline) {
    const latest = await readJson(paths.runFile, {});
    if (['stopped', 'complete', 'idle', 'failed', 'completed_with_failures'].includes(latest.state)) {
      stopped = true;
      break;
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 500));
  }
  console.log(stopped
    ? 'Collector stopped. Run dataset:start later to continue remaining episodes.'
    : 'Collector is still finishing work. Check dataset:status, or use dataset:stop -- --force if it is unresponsive.');
}

function isProcessAlive(pid) {
  if (!Number.isInteger(Number(pid)) || Number(pid) <= 0) return false;
  try {
    process.kill(Number(pid), 0);
    return true;
  } catch (error) {
    return error?.code === 'EPERM';
  }
}

function heartbeatFresh(run) {
  if (!run?.heartbeatAt) return false;
  return Date.now() - Date.parse(run.heartbeatAt) < 60_000;
}

async function rotateLog() {
  if (!existsSync(paths.logFile) || statSync(paths.logFile).size < 10 * 1024 * 1024) return;
  for (let index = 4; index >= 1; index--) {
    const source = `${paths.logFile}.${index}`;
    if (existsSync(source)) await rename(source, `${paths.logFile}.${index + 1}`);
  }
  await rename(paths.logFile, `${paths.logFile}.1`);
}
