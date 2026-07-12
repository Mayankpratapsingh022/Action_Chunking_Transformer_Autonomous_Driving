import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        app: resolve(import.meta.dirname, 'index.html'),
        collector: resolve(import.meta.dirname, 'collector.html'),
      },
    },
  },
});
