export function progressSnapshot(summary, wallTimes, workers, elapsedMs) {
  const completed = summary.accepted;
  const hasQueueCounts = Number.isFinite(summary.pending) && Number.isFinite(summary.running);
  const remaining = Math.max(0, hasQueueCounts
    ? summary.pending + summary.running
    : summary.total - completed);
  const averageWallMs = wallTimes.length
    ? wallTimes.reduce((total, value) => total + value, 0) / wallTimes.length
    : null;
  const etaMs = remaining === 0
    ? 0
    : averageWallMs === null || wallTimes.length < 5
    ? null
    : (averageWallMs * remaining) / Math.max(1, workers);
  const ratePerMinute = elapsedMs > 0 ? completed / (elapsedMs / 60_000) : 0;
  const projectedBytes = completed > 0 ? Math.round((summary.bytes / completed) * summary.total) : 0;
  return {
    completed,
    remaining,
    percent: summary.total > 0 ? (completed / summary.total) * 100 : 0,
    etaMs,
    ratePerMinute,
    projectedBytes,
  };
}

export function formatProgress({ summary, snapshot, elapsedMs, workers, current = null, heartbeatAt = null, state = 'RUNNING' }) {
  const eta = snapshot.remaining === 0
    ? 'ETA: no queued episodes'
    : `ETA: ${snapshot.etaMs === null ? 'calculating' : formatDuration(snapshot.etaMs)} | Expected finish: ${snapshot.etaMs === null ? 'calculating' : new Date(Date.now() + snapshot.etaMs).toISOString()}`;
  const lines = [
    `${new Date().toISOString()}  ${state.toUpperCase()}  ${snapshot.percent.toFixed(1)}%`,
    `Overall:   ${summary.accepted} / ${summary.total}`,
    kindLine('Nominal', summary.kinds.nominal),
    kindLine('Recovery', summary.kinds.recovery),
    kindLine('Failures', summary.kinds.failure),
    `Failed/requires review: ${summary.failed}`,
    current ? `Current: ${current}` : snapshot.remaining > 0 ? 'Current: waiting for next episode' : 'Current: no runnable episodes',
    `Elapsed: ${formatDuration(elapsedMs)} | Rate: ${snapshot.ratePerMinute.toFixed(2)} episodes/min`,
    eta,
    `Storage: ${formatBytes(summary.bytes)} | Estimated final: ${formatBytes(snapshot.projectedBytes)}`,
    `Workers: ${workers}${heartbeatAt ? ` | Last heartbeat: ${Math.max(0, Math.round((Date.now() - Date.parse(heartbeatAt)) / 1000))}s ago` : ''}`,
  ];
  return lines.join('\n');
}

function kindLine(label, value = {}) {
  const total = value.total ?? 0;
  const accepted = value.accepted ?? 0;
  const percent = total ? ((accepted / total) * 100).toFixed(1) : '0.0';
  return `${label.padEnd(10)} ${String(accepted).padStart(4)} / ${String(total).padEnd(4)} [${percent}%]`;
}

export function formatDuration(milliseconds) {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remaining = seconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remaining).padStart(2, '0')}`;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}
