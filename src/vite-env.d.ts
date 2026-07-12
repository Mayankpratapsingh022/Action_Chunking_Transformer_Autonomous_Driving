/// <reference types="vite/client" />

import type { BrowserCollectorApi } from './collection/protocol';

declare global {
  interface Window {
    __VLA_COLLECTOR__: BrowserCollectorApi;
  }
}

export {};
