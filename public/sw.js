const APP_VERSION = '1.4.0';
const CACHE_NAME = `libertus-v${APP_VERSION}`;

const PRECACHE = [
  './',
  './index.html',
  './manifest.json',
  './favicon.svg',
  './wasm/genai_wasm_internal.js',
  './wasm/genai_wasm_internal.wasm',
  './wasm/genai_wasm_nosimd_internal.js',
  './wasm/genai_wasm_nosimd_internal.wasm',
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith('libertus-') && k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.pathname.endsWith('.litertlm') || url.hostname === 'huggingface.co') {
    return;
  }

  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;

      return fetch(event.request)
        .then((response) => {
          if (!response || response.status !== 200) return response;
          return response;
        })
        .catch(() => {
          if (event.request.mode === 'navigate') {
            return caches.match('./index.html');
          }
          return new Response('Offline', { status: 503 });
        });
    })
  );
});