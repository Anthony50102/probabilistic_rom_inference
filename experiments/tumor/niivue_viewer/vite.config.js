import { defineConfig } from 'vite';

// The read-only local server serves generated data without copying it into dist.
// figure.html is the render target for scripts/render-figure.js (paper panels).
export default defineConfig({
  publicDir: false,
  build: { rollupOptions: { input: { main: 'index.html', figure: 'figure.html' } } },
});
