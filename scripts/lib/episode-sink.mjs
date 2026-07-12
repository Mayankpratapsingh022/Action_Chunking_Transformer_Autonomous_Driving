import { randomBytes } from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { rename, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { once } from 'node:events';

const FRAME_BYTES = 256 * 256 * 4;

export class FrameSinkServer {
  constructor() {
    this.encoders = new Map();
    this.server = createServer((request, response) => void this.handle(request, response));
  }

  async start() {
    this.server.listen(0, '127.0.0.1');
    await once(this.server, 'listening');
    const address = this.server.address();
    this.baseUrl = `http://127.0.0.1:${address.port}`;
    return this.baseUrl;
  }

  begin(directory) {
    const token = randomBytes(18).toString('hex');
    const encoder = new EpisodeEncoder(directory);
    this.encoders.set(token, encoder);
    return { token, endpoint: `${this.baseUrl}/frame/${token}`, encoder };
  }

  async finish(token) {
    const encoder = this.encoders.get(token);
    if (!encoder) throw new Error('Unknown frame encoder token');
    this.encoders.delete(token);
    return encoder.finish();
  }

  async abort(token) {
    const encoder = this.encoders.get(token);
    this.encoders.delete(token);
    await encoder?.abort();
  }

  async close() {
    await Promise.all([...this.encoders.values()].map((encoder) => encoder.abort()));
    this.encoders.clear();
    this.server.close();
    await once(this.server, 'close').catch(() => {});
  }

  async handle(request, response) {
    response.setHeader('access-control-allow-origin', '*');
    response.setHeader('access-control-allow-methods', 'POST, OPTIONS');
    response.setHeader('access-control-allow-headers', 'content-type');
    if (request.method === 'OPTIONS') {
      response.writeHead(204).end();
      return;
    }
    const match = request.url?.match(/^\/frame\/([a-f0-9]+)$/);
    const encoder = match ? this.encoders.get(match[1]) : null;
    if (request.method !== 'POST' || !encoder) {
      response.writeHead(404).end();
      return;
    }
    try {
      let received = 0;
      for await (const chunk of request) {
        received += chunk.length;
        await encoder.write(chunk);
      }
      if (received !== FRAME_BYTES) throw new Error(`Expected ${FRAME_BYTES} frame bytes, received ${received}`);
      encoder.frames++;
      response.writeHead(204).end();
    } catch (error) {
      response.writeHead(500, { 'content-type': 'text/plain' }).end(String(error));
    }
  }
}

class EpisodeEncoder {
  constructor(directory) {
    this.frames = 0;
    this.bytesIn = 0;
    this.partialPath = resolve(directory, 'front.partial.mp4');
    this.finalPath = resolve(directory, 'front.mp4');
    const encoder = chooseEncoder();
    const outputArgs = encoder === 'h264_videotoolbox'
      ? ['-c:v', encoder, '-q:v', '58', '-allow_sw', '1']
      : ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20'];
    this.process = spawn('ffmpeg', [
      '-hide_banner', '-loglevel', 'error', '-y',
      '-f', 'rawvideo', '-pixel_format', 'rgba', '-video_size', '256x256', '-framerate', '10',
      '-i', 'pipe:0', '-vf', 'vflip', '-an', ...outputArgs,
      '-pix_fmt', 'yuv420p', '-movflags', '+faststart', this.partialPath,
    ], { stdio: ['pipe', 'ignore', 'pipe'] });
    this.stderr = '';
    this.process.stderr.setEncoding('utf8');
    this.process.stderr.on('data', (chunk) => { this.stderr += chunk; });
    this.exit = new Promise((resolveExit) => {
      this.process.once('exit', (code, signal) => resolveExit({ code, signal }));
      this.process.once('error', (error) => resolveExit({ code: -1, signal: null, error }));
    });
  }

  async write(chunk) {
    this.bytesIn += chunk.length;
    if (!this.process.stdin.write(chunk)) await once(this.process.stdin, 'drain');
  }

  async finish() {
    this.process.stdin.end();
    const result = await this.exit;
    if (result.code !== 0) throw new Error(`FFmpeg failed (${result.code ?? result.signal}): ${this.stderr.trim()}`);
    await rename(this.partialPath, this.finalPath);
    const info = await stat(this.finalPath);
    return { frames: this.frames, bytes: info.size, path: this.finalPath };
  }

  async abort() {
    if (!this.process.killed) this.process.kill('SIGTERM');
    await this.exit;
  }
}

let selectedEncoder;
function chooseEncoder() {
  if (selectedEncoder) return selectedEncoder;
  if (process.platform === 'darwin') {
    const probe = spawnSync('ffmpeg', ['-hide_banner', '-encoders'], { encoding: 'utf8' });
    if (`${probe.stdout}${probe.stderr}`.includes('h264_videotoolbox')) {
      selectedEncoder = 'h264_videotoolbox';
      return selectedEncoder;
    }
  }
  selectedEncoder = 'libx264';
  return selectedEncoder;
}
