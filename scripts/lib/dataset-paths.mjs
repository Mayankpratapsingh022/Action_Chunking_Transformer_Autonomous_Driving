import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

export function datasetRoot(value) {
  return resolve(value ?? process.env.VLA_DATASET_ROOT ?? 'datasets/urban-vla-expert-v1');
}

export function pathsFor(root) {
  return {
    root,
    manifests: resolve(root, 'manifests'),
    state: resolve(root, 'state'),
    raw: resolve(root, 'raw'),
    partial: resolve(root, 'raw/partial'),
    accepted: resolve(root, 'raw/accepted'),
    failures: resolve(root, 'raw/failures'),
    rejected: resolve(root, 'raw/rejected'),
    lerobot: resolve(root, 'lerobot'),
    reports: resolve(root, 'reports'),
    logs: resolve(root, 'logs'),
    runFile: resolve(root, 'state/run.json'),
    stopFile: resolve(root, 'state/stop.requested'),
    database: resolve(root, 'state/collection.sqlite'),
    logFile: resolve(root, 'logs/collector.log'),
    eventLog: resolve(root, 'logs/events.jsonl'),
  };
}

export async function ensureDatasetDirectories(paths) {
  await Promise.all([
    paths.manifests,
    paths.state,
    paths.partial,
    paths.accepted,
    paths.failures,
    paths.rejected,
    paths.lerobot,
    paths.reports,
    paths.logs,
  ].map((path) => mkdir(path, { recursive: true })));
}

export async function atomicWriteJson(path, value) {
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`);
  await rename(temporary, path);
}

export async function readJson(path, fallback = null) {
  try {
    return JSON.parse(await readFile(path, 'utf8'));
  } catch {
    return fallback;
  }
}
