import { readFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { resolve } from 'node:path';

export class CollectionState {
  constructor(databasePath) {
    this.db = new DatabaseSync(databasePath);
    this.db.exec(`
      PRAGMA journal_mode = WAL;
      PRAGMA synchronous = NORMAL;
      PRAGMA busy_timeout = 5000;
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        split TEXT NOT NULL,
        task_id TEXT NOT NULL,
        config_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        worker_id TEXT,
        started_at TEXT,
        completed_at TEXT,
        frames INTEGER NOT NULL DEFAULT 0,
        simulated_seconds REAL NOT NULL DEFAULT 0,
        wall_ms INTEGER NOT NULL DEFAULT 0,
        bytes INTEGER NOT NULL DEFAULT 0,
        last_error TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status, kind, id);
      CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );
    `);
    this.insertEpisode = this.db.prepare(`
      INSERT OR IGNORE INTO episodes (id, kind, split, task_id, config_json)
      VALUES (?, ?, ?, ?, ?)
    `);
  }

  importManifests(manifestDirectory) {
    let inserted = 0;
    for (const name of ['nominal.jsonl', 'recovery.jsonl', 'failures.jsonl']) {
      const content = readFileSync(resolve(manifestDirectory, name), 'utf8');
      for (const line of content.split('\n')) {
        if (!line.trim()) continue;
        const episode = JSON.parse(line);
        const result = this.insertEpisode.run(
          episode.id,
          episode.kind,
          episode.split,
          episode.taskId,
          JSON.stringify(episode),
        );
        inserted += Number(result.changes);
      }
    }
    return inserted;
  }

  recoverInterrupted() {
    return Number(this.db.prepare(`
      UPDATE episodes
      SET status = 'pending', worker_id = NULL, started_at = NULL,
          last_error = CASE
            WHEN last_error IS NULL THEN 'Interrupted before completion'
            ELSE last_error
          END
      WHERE status = 'running'
    `).run().changes);
  }

  retryFailed() {
    return Number(this.db.prepare(`
      UPDATE episodes SET status = 'pending', worker_id = NULL
      WHERE status = 'failed'
    `).run().changes);
  }

  claimNext(workerId, allowedKinds = null) {
    const placeholders = allowedKinds?.length ? allowedKinds.map(() => '?').join(',') : null;
    const query = placeholders
      ? `SELECT * FROM episodes WHERE status = 'pending' AND kind IN (${placeholders}) ORDER BY CASE kind WHEN 'nominal' THEN 0 WHEN 'recovery' THEN 1 ELSE 2 END, id LIMIT 1`
      : `SELECT * FROM episodes WHERE status = 'pending' ORDER BY CASE kind WHEN 'nominal' THEN 0 WHEN 'recovery' THEN 1 ELSE 2 END, id LIMIT 1`;
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const row = this.db.prepare(query).get(...(allowedKinds ?? []));
      if (!row) {
        this.db.exec('COMMIT');
        return null;
      }
      this.db.prepare(`
        UPDATE episodes
        SET status = 'running', attempts = attempts + 1, worker_id = ?, started_at = ?
        WHERE id = ?
      `).run(workerId, new Date().toISOString(), row.id);
      this.db.exec('COMMIT');
      return { ...JSON.parse(row.config_json), attempts: Number(row.attempts) + 1 };
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
  }

  accept(id, { frames, simulatedSeconds, wallMs, bytes }) {
    this.db.prepare(`
      UPDATE episodes
      SET status = 'accepted', completed_at = ?, frames = ?, simulated_seconds = ?,
          wall_ms = ?, bytes = ?, last_error = NULL
      WHERE id = ?
    `).run(new Date().toISOString(), frames, simulatedSeconds, wallMs, bytes, id);
  }

  reject(id, error, maxAttempts = 3) {
    const row = this.db.prepare('SELECT attempts FROM episodes WHERE id = ?').get(id);
    const status = Number(row?.attempts ?? maxAttempts) >= maxAttempts ? 'failed' : 'pending';
    this.db.prepare(`
      UPDATE episodes
      SET status = ?, worker_id = NULL, completed_at = ?, last_error = ?
      WHERE id = ?
    `).run(status, new Date().toISOString(), String(error), id);
    return status;
  }

  summary() {
    const rows = this.db.prepare(`
      SELECT kind, status, COUNT(*) AS count, COALESCE(SUM(bytes), 0) AS bytes
      FROM episodes GROUP BY kind, status
    `).all();
    const result = {
      total: 0,
      accepted: 0,
      pending: 0,
      running: 0,
      failed: 0,
      bytes: 0,
      kinds: {},
    };
    for (const row of rows) {
      const count = Number(row.count);
      result.total += count;
      result[row.status] = (result[row.status] ?? 0) + count;
      result.bytes += Number(row.bytes);
      result.kinds[row.kind] ??= { total: 0, accepted: 0, pending: 0, running: 0, failed: 0 };
      result.kinds[row.kind].total += count;
      result.kinds[row.kind][row.status] = count;
    }
    return result;
  }

  recentWallTimes(limit = 20) {
    return this.db.prepare(`
      SELECT wall_ms FROM episodes
      WHERE status = 'accepted' AND wall_ms > 0
      ORDER BY completed_at DESC LIMIT ?
    `).all(limit).map((row) => Number(row.wall_ms));
  }

  setMetadata(key, value) {
    this.db.prepare(`
      INSERT INTO metadata (key, value) VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value
    `).run(key, String(value));
  }

  getMetadata(key, fallback = null) {
    return this.db.prepare('SELECT value FROM metadata WHERE key = ?').get(key)?.value ?? fallback;
  }

  close() {
    this.db.close();
  }
}
