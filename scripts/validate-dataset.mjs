import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { parseArgs, integerArg } from './lib/cli.mjs';
import { datasetRoot, pathsFor } from './lib/dataset-paths.mjs';

const args = parseArgs(process.argv.slice(2));
const root = datasetRoot(args.root);
const paths = pathsFor(root);
const visualSampleCount = integerArg(args.visualSamples, 20, { min: 0, max: 100 });
const errors = [];
const warnings = [];

const manifests = await loadManifests();
validateManifestCounts(manifests);
validateSplitLeakage(manifests);

const expertIds = new Set((await listDirectories(paths.accepted)).map((name) => name));
const failureIds = new Set((await listDirectories(paths.failures)).map((name) => name));
const expectedExpert = manifests.filter((episode) => episode.kind !== 'failure');
const expectedFailures = manifests.filter((episode) => episode.kind === 'failure');

if (!args.allowIncomplete) {
  for (const episode of expectedExpert) if (!expertIds.has(episode.id)) errors.push(`Missing expert episode: ${episode.id}`);
  for (const episode of expectedFailures) if (!failureIds.has(episode.id)) errors.push(`Missing failure episode: ${episode.id}`);
}
for (const id of failureIds) {
  if (expertIds.has(id)) errors.push(`Failure episode also exists in accepted expert data: ${id}`);
}

const collected = [
  ...[...expertIds].map((id) => ({ id, directory: resolve(paths.accepted, id), expectedKind: 'expert' })),
  ...[...failureIds].map((id) => ({ id, directory: resolve(paths.failures, id), expectedKind: 'failure' })),
];

let totalFrames = 0;
let totalBytes = 0;
for (const item of collected) {
  const result = await validateEpisode(item);
  totalFrames += result.frames;
  totalBytes += result.bytes;
}

for (const item of collected.slice(0, visualSampleCount)) validateVideoPixels(item);

const report = {
  valid: errors.length === 0,
  generatedAt: new Date().toISOString(),
  root: basename(root),
  manifestEpisodes: manifests.length,
  collectedEpisodes: collected.length,
  expertEpisodes: expertIds.size,
  failureEpisodes: failureIds.size,
  totalFrames,
  totalBytes,
  errors,
  warnings,
};
await mkdir(paths.reports, { recursive: true });
await writeFile(resolve(paths.reports, 'validation-summary.json'), `${JSON.stringify(report, null, 2)}\n`);

console.log(JSON.stringify(report, null, 2));
if (errors.length) process.exitCode = 1;

async function loadManifests() {
  const rows = [];
  for (const name of ['nominal.jsonl', 'recovery.jsonl', 'failures.jsonl']) {
    const path = resolve(paths.manifests, name);
    if (!existsSync(path)) {
      errors.push(`Missing manifest: ${path}`);
      continue;
    }
    const content = await readFile(path, 'utf8');
    for (const line of content.split('\n')) if (line.trim()) rows.push(JSON.parse(line));
  }
  return rows;
}

function validateManifestCounts(rows) {
  const expected = { nominal: 900, recovery: 180, failure: 90 };
  const ids = new Set();
  for (const row of rows) {
    if (ids.has(row.id)) errors.push(`Duplicate manifest ID: ${row.id}`);
    ids.add(row.id);
  }
  for (const [kind, count] of Object.entries(expected)) {
    const actual = rows.filter((row) => row.kind === kind).length;
    if (actual !== count) errors.push(`Expected ${count} ${kind} episodes, found ${actual}`);
  }
}

function validateSplitLeakage(rows) {
  const seedSplits = new Map();
  const instructionSplits = new Map();
  for (const row of rows.filter((episode) => episode.split !== 'analysis')) {
    const seedKey = `${row.kind}:${row.worldSeed}`;
    const previousSeedSplit = seedSplits.get(seedKey);
    if (previousSeedSplit && previousSeedSplit !== row.split) errors.push(`Seed leakage: ${seedKey} in ${previousSeedSplit} and ${row.split}`);
    seedSplits.set(seedKey, row.split);
    const instructionKey = `${row.taskId}:${row.instruction}`;
    const previousInstructionSplit = instructionSplits.get(instructionKey);
    if (previousInstructionSplit && previousInstructionSplit !== row.split) {
      errors.push(`Instruction leakage for ${row.taskId}: ${row.instruction}`);
    }
    instructionSplits.set(instructionKey, row.split);
  }
}

async function validateEpisode(item) {
  const metadataPath = resolve(item.directory, 'episode.json');
  const telemetryPath = resolve(item.directory, 'telemetry.jsonl');
  const videoPath = resolve(item.directory, 'front.mp4');
  if (!existsSync(metadataPath) || !existsSync(telemetryPath) || !existsSync(videoPath)) {
    errors.push(`Incomplete episode directory: ${item.id}`);
    return { frames: 0, bytes: 0 };
  }
  const metadata = JSON.parse(await readFile(metadataPath, 'utf8'));
  const rows = (await readFile(telemetryPath, 'utf8')).split('\n').filter(Boolean).map(JSON.parse);
  if (metadata.episode.id !== item.id) errors.push(`Metadata ID mismatch: ${item.id}`);
  if (item.expectedKind === 'failure' && metadata.episode.kind !== 'failure') errors.push(`Unlabelled failure directory: ${item.id}`);
  if (item.expectedKind === 'expert' && metadata.episode.kind === 'failure') errors.push(`Failure in expert directory: ${item.id}`);
  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    if (row.frameIndex !== index) errors.push(`${item.id}: frame index mismatch at ${index}`);
    if (Math.abs(row.timestamp - index * 0.1) > 0.001) errors.push(`${item.id}: timestamp drift at ${index}`);
    const [throttle, brake, steer] = row.action;
    if (throttle < 0 || throttle > 1 || brake < 0 || brake > 1 || steer < -1 || steer > 1) {
      errors.push(`${item.id}: invalid action at ${index}`);
      break;
    }
  }
  const probe = spawnSync('ffprobe', [
    '-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', videoPath,
  ], { encoding: 'utf8' });
  if (probe.status !== 0) {
    errors.push(`${item.id}: ffprobe failed: ${probe.stderr}`);
    return { frames: rows.length, bytes: 0 };
  }
  const stream = JSON.parse(probe.stdout).streams?.[0];
  if (stream?.width !== 256 || stream?.height !== 256) errors.push(`${item.id}: expected 256x256 video`);
  if (stream?.r_frame_rate !== '10/1') errors.push(`${item.id}: expected 10 FPS, found ${stream?.r_frame_rate}`);
  if (Number(stream?.nb_read_frames) !== rows.length) errors.push(`${item.id}: video/telemetry frame mismatch`);
  const bytes = (await stat(videoPath)).size;
  return { frames: rows.length, bytes };
}

function validateVideoPixels(item) {
  const videoPath = resolve(item.directory, 'front.mp4');
  const frame = spawnSync('ffmpeg', [
    '-hide_banner', '-loglevel', 'error', '-i', videoPath,
    '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1',
  ], { encoding: null, maxBuffer: 256 * 256 * 3 + 4096 });
  if (frame.status !== 0 || frame.stdout.length < 256 * 256 * 3) {
    errors.push(`${item.id}: could not decode visual sample`);
    return;
  }
  let min = 255;
  let max = 0;
  let routeBlue = 0;
  for (let index = 0; index < frame.stdout.length; index += 3) {
    const red = frame.stdout[index];
    const green = frame.stdout[index + 1];
    const blue = frame.stdout[index + 2];
    min = Math.min(min, red, green, blue);
    max = Math.max(max, red, green, blue);
    if (blue > 145 && blue > green + 35 && blue > red + 45) routeBlue++;
  }
  if (max - min < 20) errors.push(`${item.id}: first video frame appears blank`);
  if (routeBlue > 500) errors.push(`${item.id}: presentation route may be visible (${routeBlue} saturated blue pixels)`);
}

async function listDirectories(path) {
  if (!existsSync(path)) return [];
  const entries = await readdir(path, { withFileTypes: true });
  return entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort();
}
