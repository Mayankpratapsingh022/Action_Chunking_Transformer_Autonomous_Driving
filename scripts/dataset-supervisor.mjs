import { spawn, spawnSync } from 'node:child_process';
import { access, appendFile, mkdir, rename, stat, writeFile } from 'node:fs/promises';
import { constants as fsConstants, existsSync } from 'node:fs';
import { createServer as createNetServer } from 'node:net';
import { resolve } from 'node:path';
import { chromium } from 'playwright';
import { parseArgs, integerArg } from './lib/cli.mjs';
import { datasetRoot, pathsFor, ensureDatasetDirectories, atomicWriteJson, readJson } from './lib/dataset-paths.mjs';
import { FrameSinkServer } from './lib/episode-sink.mjs';
import { formatProgress, progressSnapshot } from './lib/progress.mjs';
import { CollectionState } from './lib/state-db.mjs';

const args = parseArgs(process.argv.slice(2));
const root = datasetRoot(args.root);
const paths = pathsFor(root);
const workerCount = integerArg(args.workers, 1, { min: 1, max: 4 });
const runLimit = args.limit ? integerArg(args.limit, null, { min: 1 }) : null;
const allowedKinds = args.kinds ? String(args.kinds).split(',').filter(Boolean) : null;
const projectRoot = resolve(import.meta.dirname, '..');
const startedAtMs = Date.now();
let stopRequested = false;
let claimedThisRun = 0;
let viteProcess = null;
let browser = null;
let sink = null;
let state = null;
let heartbeatTimer = null;
let stopPollTimer = null;
let runWrite = Promise.resolve();
let collectorUrl = null;
const currentEpisodes = new Map();

await ensureDatasetDirectories(paths);
await ensureManifests();
state = new CollectionState(paths.database);
const inserted = state.importManifests(paths.manifests);
const recovered = state.recoverInterrupted();
if (args.retryFailed) state.retryFailed();
const previousRun = await readJson(paths.runFile, {});
const activeBeforeRun = Number(state.getMetadata('active_elapsed_ms', 0));
const runState = {
  launcherPid: previousRun?.launcherPid ?? process.pid,
  supervisorPid: process.pid,
  state: 'starting',
  workers: workerCount,
  startedAt: new Date(startedAtMs).toISOString(),
  heartbeatAt: new Date().toISOString(),
  activeElapsedMs: activeBeforeRun,
  current: [],
  lastError: null,
};

process.on('SIGTERM', requestStop);
process.on('SIGINT', requestStop);

try {
  console.log(`[collector] dataset root: ${root}`);
  console.log(`[collector] imported ${inserted} new manifest rows; recovered ${recovered} interrupted rows`);
  preflight();
  await persistRunState();
  buildCollector();
  const collectorPort = await availablePort();
  collectorUrl = `http://127.0.0.1:${collectorPort}/collector.html`;
  runState.collectorUrl = collectorUrl;
  viteProcess = startVite(collectorPort);
  await waitForUrl(collectorUrl, 30_000);
  sink = new FrameSinkServer();
  await sink.start();
  browser = await launchBrowser();
  runState.state = 'running';
  await persistRunState();
  heartbeatTimer = setInterval(() => void heartbeat(), 10_000);
  stopPollTimer = setInterval(() => {
    if (existsSync(paths.stopFile)) requestStop('cooperative stop file');
  }, 250);

  await Promise.all(Array.from({ length: workerCount }, (_, index) => runWorker(index + 1)));
  const finalSummary = state.summary();
  runState.state = stopRequested
    ? 'stopped'
    : finalSummary.failed > 0
      ? 'completed_with_failures'
      : finalSummary.accepted === finalSummary.total
        ? 'complete'
        : 'idle';
  await heartbeat();
  console.log(`[collector] ${runState.state}: ${finalSummary.accepted}/${finalSummary.total} complete`);
} catch (error) {
  runState.state = 'failed';
  runState.lastError = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error('[collector] fatal error', error);
  await persistRunState();
  process.exitCode = 1;
} finally {
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  if (stopPollTimer) clearInterval(stopPollTimer);
  await browser?.close().catch(() => {});
  await sink?.close().catch(() => {});
  if (viteProcess && !viteProcess.killed) viteProcess.kill('SIGTERM');
  const activeElapsedMs = activeBeforeRun + (Date.now() - startedAtMs);
  state?.setMetadata('active_elapsed_ms', activeElapsedMs);
  runState.activeElapsedMs = activeElapsedMs;
  runState.heartbeatAt = new Date().toISOString();
  await persistRunState();
  state?.close();
}

function requestStop(source = 'signal') {
  if (stopRequested) return;
  stopRequested = true;
  runState.state = 'stopping';
  console.log(`[collector] stop requested via ${source}; finishing active episodes before exit`);
  void persistRunState();
}

async function runWorker(number) {
  const workerId = `worker-${number}`;
  const page = await browser.newPage({ viewport: { width: 256, height: 256 } });
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error'
      && !text.includes("Couldn't load texture Textures/colormap.png")
      && !text.includes('Failed to load resource: the server responded with a status of 404')) {
      console.error(`[${workerId}:browser] ${text}`);
    }
  });
  page.on('pageerror', (error) => console.error(`[${workerId}:page] ${error.message}`));
  await loadCollectorPage(page);
  let episodesOnPage = 0;

  try {
    while (!stopRequested) {
      if (runLimit !== null && claimedThisRun >= runLimit) break;
      claimedThisRun++;
      const episode = state.claimNext(workerId, allowedKinds);
      if (!episode) {
        claimedThisRun--;
        break;
      }
      currentEpisodes.set(workerId, `${episode.kind}/${episode.taskId} ${episode.id}`);
      await heartbeat();
      await collectEpisode(page, workerId, episode);
      currentEpisodes.delete(workerId);
      episodesOnPage++;
      if (episodesOnPage >= 100 && !stopRequested) {
        await loadCollectorPage(page);
        episodesOnPage = 0;
      }
    }
  } finally {
    currentEpisodes.delete(workerId);
    await page.close().catch(() => {});
  }
}

async function loadCollectorPage(page) {
  await page.goto(collectorUrl, { waitUntil: 'networkidle', timeout: 60_000 });
  await page.waitForFunction(() => window.__VLA_COLLECTOR__?.health().ready === true, null, { timeout: 60_000 });
}

async function collectEpisode(page, workerId, episode) {
  const wallStart = Date.now();
  const partialDirectory = resolve(paths.partial, episode.id);
  await preparePartialDirectory(partialDirectory, episode);
  const { token, endpoint } = sink.begin(partialDirectory);
  let result;
  let encoded;
  try {
    result = await page.evaluate(
      (config) => window.__VLA_COLLECTOR__.runEpisode(config),
      { ...episode, frameEndpoint: endpoint },
    );
    encoded = await sink.finish(token);
    validateEpisodeResult(result, encoded);
    await writeEpisodeFiles(partialDirectory, episode, result);
  } catch (error) {
    await sink.abort(token).catch(() => {});
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    await writeFile(resolve(partialDirectory, 'error.txt'), `${message}\n`).catch(() => {});
    const destination = await rejectedDestination(episode);
    await rename(partialDirectory, destination).catch(() => {});
    const nextStatus = state.reject(episode.id, message);
    logEvent({ type: 'episode_rejected', episodeId: episode.id, workerId, status: nextStatus, error: message });
    console.error(`[collector] ${episode.id} rejected (${nextStatus}): ${message}`);
    await logProgress();
    return;
  }

  if (!result.valid) {
    const destination = await rejectedDestination(episode);
    await rename(partialDirectory, destination);
    const nextStatus = state.reject(episode.id, result.error ?? result.outcome);
    logEvent({ type: 'episode_rejected', episodeId: episode.id, workerId, status: nextStatus, outcome: result.outcome });
    console.error(`[collector] ${episode.id} rejected (${nextStatus}): ${result.error ?? result.outcome}`);
    await logProgress();
    return;
  }

  const destination = resolve(episode.kind === 'failure' ? paths.failures : paths.accepted, episode.id);
  await rename(partialDirectory, destination);
  const telemetryBytes = Buffer.byteLength(result.telemetry.map((row) => JSON.stringify(row)).join('\n'));
  const wallMs = Date.now() - wallStart;
  state.accept(episode.id, {
    frames: result.frames,
    simulatedSeconds: result.simulatedSeconds,
    wallMs,
    bytes: encoded.bytes + telemetryBytes,
  });
  logEvent({
    type: 'episode_accepted',
    episodeId: episode.id,
    workerId,
    kind: episode.kind,
    frames: result.frames,
    simulatedSeconds: result.simulatedSeconds,
    wallMs,
    bytes: encoded.bytes + telemetryBytes,
  });
  console.log(`[collector] accepted ${episode.id}: ${result.frames} frames in ${(wallMs / 1000).toFixed(1)}s`);
  await logProgress();
}

function validateEpisodeResult(result, encoded) {
  if (!result || !Array.isArray(result.telemetry)) throw new Error('Browser returned an invalid episode result');
  if (result.error && result.frames === 0) throw new Error(result.error);
  if (result.frames < 2) throw new Error(`Episode has only ${result.frames} frames`);
  if (result.frames !== result.telemetry.length) throw new Error(`Telemetry mismatch: ${result.frames} != ${result.telemetry.length}`);
  if (result.frames !== encoded.frames) throw new Error(`Video mismatch: ${result.frames} telemetry frames != ${encoded.frames} encoded frames`);
  for (let index = 0; index < result.telemetry.length; index++) {
    const row = result.telemetry[index];
    if (row.frameIndex !== index) throw new Error(`Unexpected frame index ${row.frameIndex} at ${index}`);
    if (Math.abs(row.timestamp - index * 0.1) > 0.001) throw new Error(`Timestamp drift at frame ${index}: ${row.timestamp}`);
    const [throttle, brake, steer] = row.action;
    if (![throttle, brake, steer].every(Number.isFinite)) throw new Error(`Non-finite action at frame ${index}`);
    if (throttle < 0 || throttle > 1 || brake < 0 || brake > 1 || steer < -1 || steer > 1) {
      throw new Error(`Out-of-range action at frame ${index}`);
    }
  }
}

async function writeEpisodeFiles(directory, episode, result) {
  const telemetry = `${result.telemetry.map((row) => JSON.stringify(row)).join('\n')}\n`;
  const metadata = {
    schemaVersion: 'urban-vla-expert-v1',
    fps: 10,
    width: 256,
    height: 256,
    episode,
    result: { ...result, telemetry: undefined },
    collectedAt: new Date().toISOString(),
  };
  await Promise.all([
    writeFile(resolve(directory, 'telemetry.jsonl'), telemetry),
    writeFile(resolve(directory, 'episode.json'), `${JSON.stringify(metadata, null, 2)}\n`),
  ]);
}

async function preparePartialDirectory(directory, episode) {
  try {
    await access(directory, fsConstants.F_OK);
    const destination = await rejectedDestination({ ...episode, attempts: `${episode.attempts}-interrupted` });
    await rename(directory, destination);
  } catch {}
  await mkdir(directory, { recursive: true });
}

async function rejectedDestination(episode) {
  const suffix = `${episode.id}-attempt-${episode.attempts}-${Date.now()}`;
  return resolve(paths.rejected, suffix);
}

async function logProgress() {
  const summary = state.summary();
  const activeElapsedMs = activeBeforeRun + (Date.now() - startedAtMs);
  const snapshot = progressSnapshot(summary, state.recentWallTimes(), workerCount, activeElapsedMs);
  const text = formatProgress({
    summary,
    snapshot,
    elapsedMs: activeElapsedMs,
    workers: workerCount,
    current: [...currentEpisodes.values()].join(', ') || null,
    heartbeatAt: runState.heartbeatAt,
  });
  console.log(`\n${text}\n`);
  runState.summary = { ...summary, snapshot };
  await heartbeat();
}

function logEvent(event) {
  void appendFile(paths.eventLog, `${JSON.stringify({ timestamp: new Date().toISOString(), ...event })}\n`);
}

async function heartbeat() {
  const activeElapsedMs = activeBeforeRun + (Date.now() - startedAtMs);
  runState.heartbeatAt = new Date().toISOString();
  runState.activeElapsedMs = activeElapsedMs;
  runState.current = [...currentEpisodes.entries()].map(([worker, episode]) => ({ worker, episode }));
  state.setMetadata('active_elapsed_ms', activeElapsedMs);
  await persistRunState();
}

function persistRunState() {
  runWrite = runWrite.then(() => atomicWriteJson(paths.runFile, runState));
  return runWrite;
}

async function ensureManifests() {
  try {
    await Promise.all(['nominal.jsonl', 'recovery.jsonl', 'failures.jsonl'].map((name) => access(resolve(paths.manifests, name))));
  } catch {
    const result = spawnSync(process.execPath, [
      resolve(projectRoot, 'scripts/generate-dataset-manifest.mjs'),
      '--root', root,
    ], { cwd: projectRoot, encoding: 'utf8' });
    if (result.status !== 0) throw new Error(`Manifest generation failed: ${result.stderr || result.stdout}`);
    console.log(result.stdout.trim());
  }
}

function startVite(port) {
  const child = spawn(process.execPath, [
    resolve(projectRoot, 'node_modules/vite/bin/vite.js'),
    'preview',
    '--host', '127.0.0.1', '--port', String(port), '--strictPort',
  ], { cwd: projectRoot, stdio: ['ignore', 'inherit', 'inherit'] });
  child.once('exit', (code) => {
    if (!stopRequested && code !== 0) console.error(`[collector] Vite exited with code ${code}`);
  });
  return child;
}

function preflight() {
  const nodeMajor = Number.parseInt(process.versions.node.split('.')[0], 10);
  if (nodeMajor < 22) throw new Error(`Node.js 22 or newer is required; found ${process.version}`);
  const ffmpeg = spawnSync('ffmpeg', ['-version'], { encoding: 'utf8' });
  if (ffmpeg.status !== 0) throw new Error('FFmpeg is required but was not found on PATH');
}

async function availablePort() {
  const server = createNetServer();
  server.listen(0, '127.0.0.1');
  await new Promise((resolveListen, reject) => {
    server.once('listening', resolveListen);
    server.once('error', reject);
  });
  const address = server.address();
  const port = address.port;
  await new Promise((resolveClose) => server.close(resolveClose));
  return port;
}

function buildCollector() {
  console.log('[collector] building deterministic collector bundle');
  const result = spawnSync('npm', ['run', 'build'], { cwd: projectRoot, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`Collector build failed:\n${result.stdout}\n${result.stderr}`);
}

async function waitForUrl(url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (stopRequested) throw new Error('Stopped while waiting for collector server');
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {}
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

async function launchBrowser() {
  try {
    return await chromium.launch({ channel: 'chrome', headless: true });
  } catch (chromeError) {
    try {
      return await chromium.launch({ headless: true });
    } catch (playwrightError) {
      throw new Error(`Could not launch Chrome or Playwright Chromium. Chrome: ${chromeError}. Chromium: ${playwrightError}`);
    }
  }
}
