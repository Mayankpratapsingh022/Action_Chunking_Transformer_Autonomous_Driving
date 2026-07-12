import type { BinaryActions } from '../types';

export interface InferenceResult {
  actions: BinaryActions;
  confidences?: Record<string, number>;
}

export class InferenceClient {
  private ws: WebSocket | null = null;
  private lastResult: InferenceResult | null = null;

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
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
        this.lastResult = JSON.parse(event.data) as InferenceResult;
      };
      ws.onclose = () => {
        if (this.ws === ws) this.ws = null;
      };
    });
  }

  predict(image: string, languageId: number): InferenceResult | null {
    if (this.connected) {
      this.ws?.send(JSON.stringify({ image, language_id: languageId }));
    }
    return this.lastResult;
  }

  disconnect(): void {
    this.ws?.close();
    this.ws = null;
  }
}
