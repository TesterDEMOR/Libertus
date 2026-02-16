import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [viteSingleFile()],
  base: './',
  build: {
    target: 'esnext',
    assetsInlineLimit: Infinity,
    cssCodeSplit: false,
    modulePreload: false,
  },
  server: {
    port: 7749,
    host: true,
    // Allow serving .litertlm model file from project root in dev mode
    fs: {
      allow: ['.'],
    },
  },
});
