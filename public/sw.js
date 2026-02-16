// ============================================================
//  Libertus — Service Worker
//  Version-based cache invalidation for PWA updates
//  This file lives in public/ and is NOT bundled by Vite
// ============================================================

const APP_VERSION = '1.0.0';
const CACHE_NAME = `libertus-v${APP_VERSION}`;

// Files to precache (the single index.html + manifest + favicon)
const PRECACHE = [
  './',
  './index.html',
  './manifest.json',
  './favicon.svg',
];

// Install: precache app shell
self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
});

// Activate: delete ALL old versioned caches
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

// Fetch strategy:
// - Model files (.litertlm) and HuggingFace: pass through (handled by app + IndexedDB)
// - CDN (jsdelivr): cache first, then network (WASM files for MediaPipe)
// - App shell: cache first, then network fallback
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Don't intercept model downloads
  if (url.pathname.endsWith('.litertlm') || url.hostname === 'huggingface.co') {
    return;
  }

  // Only handle GET requests
  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;

      return fetch(event.request)
        .then((response) => {
          if (!response || response.status !== 200) return response;

          // Cache CDN resources (MediaPipe WASM)
          const shouldCache =
            url.hostname === 'cdn.jsdelivr.net' ||
            url.hostname === 'unpkg.com';

          if (shouldCache) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }

          return response;
        })
        .catch(() => {
          // Offline fallback for navigation
          if (event.request.mode === 'navigate') {
            return caches.match('./index.html');
          }
          return new Response('Offline', { status: 503 });
        });
    })
  );
});
