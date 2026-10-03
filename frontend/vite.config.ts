import { defineConfig, normalizePath, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import manifest from './config/manifest.mjs';

const plotlyBundle = createRequire(import.meta.url).resolve('plotly.js/dist/plotly-basic.min.js');
const plotlyNotice = readFileSync(plotlyBundle, 'utf8')
  .match(/^\/\*\*[\s\S]*?\*\//)![0].replace('/**', '/*!');

function webManifest(): Plugin {
  const source = JSON.stringify(manifest, null, 2) + '\n';
  const fileName = 'manifest.webmanifest';
  return {
    name: 'calmsense-web-manifest',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const path = new URL(request.url ?? '/', 'http://localhost').pathname;
        if (path !== `${server.config.base}${fileName}` && path !== `/${fileName}`) {
          return next();
        }
        response.setHeader('Content-Type', 'application/manifest+json');
        response.end(request.method === 'HEAD' ? undefined : source);
      });
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName, source });
    },
  };
}

export default defineConfig({
  base: '/CalmSense/',
  plugins: [react(), tailwindcss(), webManifest()],
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
