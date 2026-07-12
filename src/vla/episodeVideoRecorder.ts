export class EpisodeVideoRecorder {
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private chunks: Blob[] = [];
  private blob: Blob | null = null;
  private mimeType = 'video/webm';

  get recording(): boolean {
    return this.recorder?.state === 'recording';
  }

  get ready(): boolean {
    return this.blob !== null && this.blob.size > 0;
  }

  start(canvas: HTMLCanvasElement): boolean {
    if (this.recording) return true;
    if (typeof MediaRecorder === 'undefined' || typeof canvas.captureStream !== 'function') return false;

    this.stopStream();
    this.chunks = [];
    this.blob = null;
    this.mimeType = pickMimeType();

    try {
      this.stream = canvas.captureStream(30);
      this.recorder = new MediaRecorder(
        this.stream,
        this.mimeType ? { mimeType: this.mimeType } : undefined,
      );
      this.recorder.addEventListener('dataavailable', (event) => {
        if (event.data.size > 0) this.chunks.push(event.data);
      });
      this.recorder.addEventListener('stop', () => {
        this.blob = new Blob(this.chunks, { type: this.mimeType || 'video/webm' });
        this.stopStream();
      });
      this.recorder.start(500);
      return true;
    } catch (error) {
      console.warn('Could not start episode video recording', error);
      this.recorder = null;
      this.stopStream();
      return false;
    }
  }

  stop(): Promise<boolean> {
    const recorder = this.recorder;
    if (!recorder || recorder.state === 'inactive') return Promise.resolve(this.ready);

    return new Promise((resolve) => {
      recorder.addEventListener('stop', () => resolve(this.ready), { once: true });
      recorder.stop();
      this.recorder = null;
    });
  }

  clear(): void {
    if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop();
    this.recorder = null;
    this.stopStream();
    this.chunks = [];
    this.blob = null;
  }

  download(filename: string): boolean {
    if (!this.blob) return false;
    const url = URL.createObjectURL(this.blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename.endsWith('.webm') ? filename : `${filename}.webm`;
    link.click();
    URL.revokeObjectURL(url);
    return true;
  }

  private stopStream(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }
}

function pickMimeType(): string {
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}
