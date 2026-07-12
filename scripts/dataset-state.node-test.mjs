import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import test from 'node:test';
import { CollectionState } from './lib/state-db.mjs';
import { progressSnapshot } from './lib/progress.mjs';

test('collection state resumes interrupted episodes without duplicating accepted work', async () => {
  const root = await mkdtemp(resolve(tmpdir(), 'vla-state-test-'));
  const manifests = resolve(root, 'manifests');
  const stateDirectory = resolve(root, 'state');
  await mkdir(manifests, { recursive: true });
  await mkdir(stateDirectory, { recursive: true });
  const rows = [
    { id: 'n-1', kind: 'nominal', split: 'train', taskId: 'drive' },
    { id: 'r-1', kind: 'recovery', split: 'train', taskId: 'drive' },
    { id: 'f-1', kind: 'failure', split: 'analysis', taskId: 'drive' },
  ];
  await writeFile(resolve(manifests, 'nominal.jsonl'), `${JSON.stringify(rows[0])}\n`);
  await writeFile(resolve(manifests, 'recovery.jsonl'), `${JSON.stringify(rows[1])}\n`);
  await writeFile(resolve(manifests, 'failures.jsonl'), `${JSON.stringify(rows[2])}\n`);
  const state = new CollectionState(resolve(stateDirectory, 'collection.sqlite'));
  assert.equal(state.importManifests(manifests), 3);
  const first = state.claimNext('worker-1');
  state.accept(first.id, { frames: 10, simulatedSeconds: 1, wallMs: 500, bytes: 100 });
  const second = state.claimNext('worker-1');
  assert.equal(second.kind, 'recovery');
  assert.equal(state.recoverInterrupted(), 1);
  const resumed = state.claimNext('worker-2');
  assert.equal(resumed.id, second.id);
  assert.notEqual(resumed.id, first.id);
  state.close();
});

test('progress ETA waits for five durations and scales across workers', () => {
  const summary = { total: 100, accepted: 20, bytes: 1_000, kinds: {} };
  assert.equal(progressSnapshot(summary, [1000, 1000], 1, 20_000).etaMs, null);
  assert.equal(progressSnapshot(summary, [1000, 1000, 1000, 1000, 1000], 2, 20_000).etaMs, 40_000);
});

test('progress ETA excludes exhausted failures from runnable work', () => {
  const summary = { total: 100, accepted: 80, pending: 0, running: 0, failed: 20, bytes: 1_000, kinds: {} };
  const snapshot = progressSnapshot(summary, [1000, 1000, 1000, 1000, 1000], 2, 20_000);
  assert.equal(snapshot.remaining, 0);
  assert.equal(snapshot.etaMs, 0);
});
