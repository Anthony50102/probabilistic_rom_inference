import { defineConfig } from 'vite';

// The read-only local server serves generated data without copying it into dist.
export default defineConfig({ publicDir: false });
