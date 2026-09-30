import { defineConfig } from 'vite';

// The read-only local server serves generated data without copying it into dist.
// figure.html renders the paper panels and compose.html lays out the MRI-style figure (scripts/render-figure.js).
export default defineConfig({
  publicDir: false,
  build: { rollupOptions: { input: { main: 'index.html', figure: 'figure.html', compose: 'compose.html' } } },
});
