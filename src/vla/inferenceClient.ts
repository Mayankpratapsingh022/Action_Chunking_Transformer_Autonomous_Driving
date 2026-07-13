import type { ControlCommand } from '../types';

export interface InferenceRequest {
  image: string;
  instruction: string;
  state: [speed: number, steering: number, previousThrottle: number, previousBrake: number];
}

export interface InferenceResult {
  action: ControlCommand;
  rawAction: ControlCommand;
  latencyMs: number;
}

export class InferenceClient {
  private ws: WebSocket | null = null;
  private lastResult: InferenceResult | null = null;
  private pendingRequestId: number | null = null;
  private nextRequestId = 1;
  private minimumRequestId = 1;

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get latest(): InferenceResult | null {
    return this.lastResult;
  }

  connect(url: string): Promise<void> {
    const normalized = url.startsWith('ws')
      ? url
      : url.replace(/^http/, 'ws').replace(/\/$/, '') + '/ws';
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(normalized);
      ws.onopen = () => {
        this.ws = ws;
        resolve();
      };
      ws.onerror = () => reject(new Error('Could not connect to inference server'));
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data) as ServerMessage;
        if (message.type === 'prediction') {
          if (message.request_id < this.minimumRequestId) return;
          this.pendingRequestId = null;
          this.lastResult = {
            action: message.action,
            rawAction: message.raw_action,
            latencyMs: message.latency_ms,
          };
        } else if (message.type === 'error') {
          this.pendingRequestId = null;
          console.error(`Inference request failed: ${message.error}`);
        }
      };
      ws.onclose = () => {
        if (this.ws === ws) {
          this.ws = null;
          this.pendingRequestId = null;
        }
      };
    });
  }

  predict(request: InferenceRequest): InferenceResult | null {
    if (this.connected && this.pendingRequestId === null) {
      const requestId = this.nextRequestId++;
      this.pendingRequestId = requestId;
      this.ws?.send(JSON.stringify({ type: 'predict', request_id: requestId, ...request }));
    }
    return this.lastResult;
  }

  reset(): void {
    this.lastResult = null;
    this.pendingRequestId = null;
    this.minimumRequestId = this.nextRequestId;
    if (this.connected) this.ws?.send(JSON.stringify({ type: 'reset' }));
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
    this.lastResult = null;
    this.pendingRequestId = null;
  }
}

interface PredictionMessage {
  type: 'prediction';
  request_id: number;
  action: ControlCommand;
  raw_action: ControlCommand;
  latency_ms: number;
}

interface ErrorMessage {
  type: 'error';
  request_id: number | null;
  error: string;
}

interface ResetMessage {
  type: 'reset_ack';
}

type ServerMessage = PredictionMessage | ErrorMessage | ResetMessage;
