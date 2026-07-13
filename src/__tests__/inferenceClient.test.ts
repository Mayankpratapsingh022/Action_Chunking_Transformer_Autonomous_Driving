import { afterEach, describe, expect, it, vi } from 'vitest';
import { InferenceClient } from '../vla/inferenceClient';

class FakeWebSocket {
  static readonly OPEN = 1;
  static instances: FakeWebSocket[] = [];

  readyState = 0;
  sent: string[] = [];
  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(payload: string): void {
    this.sent.push(payload);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.(new Event('open'));
  }

  receive(payload: object): void {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent);
  }

  close(): void {
    this.readyState = 3;
    this.onclose?.({} as CloseEvent);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
  FakeWebSocket.instances = [];
});

describe('InferenceClient', () => {
  it('sends synchronized ACT observations and allows only one in-flight request', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
    const client = new InferenceClient();
    const connecting = client.connect('http://localhost:8000');
    const socket = FakeWebSocket.instances[0];
    socket.open();
    await connecting;

    const request = {
      image: 'data:image/png;base64,abc',
      instruction: 'Turn left at the intersection.',
      state: [7.5, 0.1, 0.6, 0] as [number, number, number, number],
    };
    expect(client.predict(request)).toBeNull();
    expect(client.predict(request)).toBeNull();
    expect(socket.sent).toHaveLength(1);
    expect(JSON.parse(socket.sent[0])).toMatchObject({
      type: 'predict',
      request_id: 1,
      instruction: request.instruction,
      state: request.state,
    });

    socket.receive({
      type: 'prediction',
      request_id: 1,
      action: { throttle: 0.7, brake: 0, steer: -0.2 },
      raw_action: { throttle: 0.7, brake: 0.02, steer: -0.2 },
      latency_ms: 42.5,
    });

    expect(client.predict(request)).toEqual({
      action: { throttle: 0.7, brake: 0, steer: -0.2 },
      rawAction: { throttle: 0.7, brake: 0.02, steer: -0.2 },
      latencyMs: 42.5,
    });
    expect(socket.sent).toHaveLength(2);
  });

  it('clears cached controls and ignores stale predictions after reset', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
    const client = new InferenceClient();
    const connecting = client.connect('ws://localhost:8000/ws');
    const socket = FakeWebSocket.instances[0];
    socket.open();
    await connecting;

    const request = {
      image: 'data:image/png;base64,abc',
      instruction: 'Drive straight.',
      state: [0, 0, 0, 0] as [number, number, number, number],
    };
    client.predict(request);
    client.reset();
    socket.receive({
      type: 'prediction',
      request_id: 1,
      action: { throttle: 1, brake: 0, steer: 0 },
      raw_action: { throttle: 1, brake: 0, steer: 0 },
      latency_ms: 10,
    });

    expect(client.predict(request)).toBeNull();
    expect(JSON.parse(socket.sent[1])).toEqual({ type: 'reset' });
    expect(JSON.parse(socket.sent[2])).toMatchObject({ request_id: 2 });
  });
});
