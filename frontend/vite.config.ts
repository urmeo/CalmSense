import { defineConfig, normalizePath } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const plotlyBundle = createRequire(import.meta.url).resolve('plotly.js/dist/plotly-basic.min.js');
const plotlyNotice = readFileSync(plotlyBundle, 'utf8')
  .match(/^\/\*\*[\s\S]*?\*\//)![0].replace('/**', '/*!');

export default defineConfig({
  base: '/CalmSense/',
  plugins: [react(), tailwindcss()],
  esbuild: { legalComments: 'eof' },
  build: {
    license: { fileName: 'licenses.txt' },
    rollupOptions: {
      output: {
        banner: (chunk) => chunk.moduleIds.includes(normalizePath(plotlyBundle)) ? plotlyNotice : '',
      },
    },
  },
});
